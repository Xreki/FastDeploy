"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import paddle
from paddle import nn

from fastdeploy.model_executor.ops.gpu import tritonmoe_preprocess

from .fused_moe_method_base import FusedMoEMethodBase
from .triton_moe_kernels import fused_moe_kernel_paddle


def ceil_div(a, b):
    """
    ceil(a / b)
    """
    return (a + b - 1) // b


class TritonFusedMoeMethod(FusedMoEMethodBase):
    """
    Use Triton Group Gemm to compute Fused MoE.
    """

    def __init__(self, moe_compute_params):
        """
        Triton Group Gemm to compute Fused MoE.
        """

        self.num_local_experts = moe_compute_params.num_local_experts
        self.moe_quant_type = moe_compute_params.moe_quant_type
        self.hidden_size = moe_compute_params.hidden_size
        self.moe_intermediate_size = moe_compute_params.moe_intermediate_size
        self.top_k = moe_compute_params.top_k
        self.tp_size = moe_compute_params.tp_size

    def create_weights(
            self,
            layer: nn.Layer,
            ffn1_tensor,
            ffn2_tensor,
            ffn1_bias=None,
            ffn2_bias=None,
            # belows only used in w4a8.
            moe_ffn1_weight_scale=None,
            moe_ffn2_weight_scale=None,
            moe_ffn1_in_scale=None,
            moe_ffn2_in_scale=None):
        """
        Triton MoE create weight process.
        """

        assert len(ffn1_tensor) == self.num_local_experts
        assert len(ffn2_tensor) == self.num_local_experts
        assert self.moe_quant_type == "weight_only_int8"
        assert len(ffn2_tensor) == self.num_local_experts
        assert ffn1_tensor[0].shape == [
            self.hidden_size, self.moe_intermediate_size * 2
        ]
        assert ffn2_tensor[0].shape == [
            self.moe_intermediate_size, self.hidden_size
        ]

        ffn1_tensor = paddle.stack(ffn1_tensor, axis=0)
        ffn2_tensor = paddle.stack(ffn2_tensor, axis=0)

        if self.moe_quant_type == "weight_only_int8":
            max_bound = 127
        elif self.moe_quant_type == "weight_only_int4":
            max_bound = 7

        added_weight_attrs = ["moe_ffn1_weight", "moe_ffn2_weight"]
        added_scale_attrs = ["moe_ffn1_weight_scale", "moe_ffn2_weight_scale"]

        for idx, weight_tensor in enumerate([ffn1_tensor, ffn2_tensor]):
            weight_name = added_weight_attrs[idx]
            scale_name = added_scale_attrs[idx]

            quanted_weight_scale = weight_tensor.abs().max(axis=1)
            quanted_weight = weight_tensor / quanted_weight_scale[:,
                                                                  None, :] * max_bound
            quanted_weight = paddle.round(quanted_weight).astype("int8")
            quanted_weight_scale = quanted_weight_scale / max_bound

            setattr(
                layer, weight_name,
                layer.create_parameter(
                    shape=quanted_weight.shape,
                    dtype=quanted_weight.dtype,
                    default_initializer=paddle.nn.initializer.Constant(0),
                ))
            getattr(layer, weight_name).set_value(quanted_weight)

            setattr(
                layer, scale_name,
                layer.create_parameter(
                    shape=quanted_weight_scale.shape,
                    dtype=quanted_weight_scale.dtype,
                ))
            getattr(layer, scale_name).set_value(quanted_weight_scale)

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Triton compute Fused MoE.
        """
        token_num = x.shape[0]
        top_k = self.top_k
        num_local_experts = self.num_local_experts
        top_k = self.top_k
        moe_intermediate_size = self.moe_intermediate_size
        hidden_size = self.hidden_size

        gate_out = paddle.matmul(x.cast("float32"), layer.gate_weight)
        scores = paddle.nn.functional.softmax(gate_out, axis=-1)

        topk_weights, topk_ids = paddle.topk(scores,
                                             k=top_k,
                                             axis=-1,
                                             sorted=False)
        topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdim=True)

        intermediate_cache1 = paddle.empty(
            [token_num * top_k, moe_intermediate_size * 2],
            dtype=x.dtype,
        )
        intermediate_cache2 = paddle.empty(
            (token_num * top_k, moe_intermediate_size),
            dtype=x.dtype,
        )
        intermediate_cache3 = paddle.empty(
            (token_num * top_k, hidden_size),
            dtype=x.dtype,
        )

        config = {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
        }

        sorted_token_ids, expert_ids, num_tokens_post_padded = tritonmoe_preprocess(
            topk_ids, num_local_experts, config["BLOCK_SIZE_M"])
        max_num_tokens_padded = sorted_token_ids.shape[0]
        grid = (ceil_div(max_num_tokens_padded, config["BLOCK_SIZE_M"]) *
                ceil_div(moe_intermediate_size * 2, config["BLOCK_SIZE_N"]), )

        fused_moe_kernel_paddle[grid](
            x,
            layer.moe_ffn1_weight,
            intermediate_cache1,
            None,
            layer.moe_ffn1_weight_scale,
            None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            moe_intermediate_size * 2,
            hidden_size,
            max_num_tokens_padded,
            token_num * top_k,
            stride_am=x.strides[0],
            stride_ak=x.strides[1],
            stride_be=layer.moe_ffn1_weight.strides[0],
            stride_bk=layer.moe_ffn1_weight.strides[1],
            stride_bn=layer.moe_ffn1_weight.strides[2],
            stride_cm=intermediate_cache1.strides[0],
            stride_cn=intermediate_cache1.strides[1],
            #
            stride_asm=-1,
            stride_ask=-1,
            stride_bse=layer.moe_ffn1_weight_scale.strides[0],
            stride_bsk=-1,
            stride_bsn=layer.moe_ffn1_weight_scale.strides[1],
            group_n=-1,
            group_k=-1,
            # Meta-parameters
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            MUL_ROUTED_WEIGHT=False,
            top_k=top_k,
            compute_type_enum=1,
            use_fp8_w8a8=False,
            use_int8_w8a16=True,
            even_Ks=hidden_size % config["BLOCK_SIZE_K"] == 0,
        )

        intermediate_cache2 = paddle.incubate.nn.functional.swiglu(
            intermediate_cache1)

        grid = (ceil_div(max_num_tokens_padded, config["BLOCK_SIZE_M"]) *
                ceil_div(hidden_size, config["BLOCK_SIZE_N"]), )
        fused_moe_kernel_paddle[grid](
            intermediate_cache2,
            layer.moe_ffn2_weight,
            intermediate_cache3,
            None,
            layer.moe_ffn2_weight_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            hidden_size,
            moe_intermediate_size,
            max_num_tokens_padded,
            token_num * top_k,
            stride_am=intermediate_cache2.strides[0],
            stride_ak=intermediate_cache2.strides[1],
            stride_be=layer.moe_ffn2_weight.strides[0],
            stride_bk=layer.moe_ffn2_weight.strides[1],
            stride_bn=layer.moe_ffn2_weight.strides[2],
            stride_cm=intermediate_cache3.strides[0],
            stride_cn=intermediate_cache3.strides[1],
            stride_asm=-1,
            stride_ask=-1,
            stride_bse=layer.moe_ffn2_weight_scale.strides[0],
            stride_bsk=-1,
            stride_bsn=layer.moe_ffn2_weight_scale.strides[1],
            group_n=-1,
            group_k=-1,
            # Meta-parameters
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            MUL_ROUTED_WEIGHT=True,
            top_k=1,
            compute_type_enum=1,
            use_fp8_w8a8=False,
            use_int8_w8a16=True,
            even_Ks=moe_intermediate_size % config["BLOCK_SIZE_K"] == 0,
        )

        intermediate_cache3.reshape_([token_num, top_k, hidden_size])
        out = intermediate_cache3.sum(axis=1)
        return out

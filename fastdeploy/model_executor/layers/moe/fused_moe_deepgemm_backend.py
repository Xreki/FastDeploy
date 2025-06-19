"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import fastdeploy
import fastdeploy.model_executor.ops.gpu.deep_gemm as deep_gemm
from fastdeploy.model_executor.ops.gpu import count_tokens_per_expert_func

from ..quantization.quant_base import QuantMethodBase
from .fused_moe_cutlass_backend import create_and_set_parameter


class DeepGemmFusedMoeMethod(QuantMethodBase):
    """
    DeepGemmFusedMoeMethod is a class that implements the FusedMoEMethodBase interface for DeepGemm backend.
    """

    def __init__(self, ) -> None:
        '''
        Nothing need to comment.
        '''
        super().__init__()
        self.added_weight_attrs = ["moe_ffn1_weight", "moe_ffn2_weight"]
        self.added_scale_attrs = [
            "moe_ffn1_weight_scale", "moe_ffn2_weight_scale"
        ]

    def create_weights(self, layer: nn.Layer, state_dict):
        """
        deepgemm create weight process.
        """

        ffn1_weights, ffn2_weights = layer.extract_moe_ffn_weights(state_dict)

        for idx, weight_tensor in enumerate([ffn1_weights, ffn2_weights]):
            weight_name = self.added_weight_attrs[idx]
            scale_name = self.added_scale_attrs[idx]

            weight_list = []
            weight_scale_list = []
            for i in range(layer.num_local_experts):
                from fastdeploy.model_executor.layers.utils import \
                    per_block_cast_to_fp8
                quant_weight, scale = per_block_cast_to_fp8(weight_tensor[i])

                weight_list.append(quant_weight)
                weight_scale_list.append(scale)
            quanted_weight = paddle.stack(weight_list, axis=0)
            quanted_weight = quanted_weight.transpose([0, 2, 1]).contiguous()
            create_and_set_parameter(layer, weight_name, quanted_weight)

            quanted_weight_scale = paddle.stack(weight_scale_list, axis=0)
            quanted_weight_scale = quanted_weight_scale.transpose(
                [0, 2, 1]).contiguous()
            create_and_set_parameter(layer, scale_name, quanted_weight_scale)

    def process_loaded_weights(self, layer, weights) -> None:
        '''
        Nothing need to comment.
        '''
        raise NotImplementedError

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Paddle Use DeepGemm compute Fused MoE.
        below is TP compute method.
        """

        topk_ids, topk_weights = fastdeploy.model_executor.ops.gpu.moe_topk_select(
            gate_out,
            layer.gate_correction_bias,
            layer.top_k,
            True,  # apply_norm_weight
            False,
        )

        tmp = count_tokens_per_expert_func(topk_ids,
                                      layer.num_experts).numpy().tolist()
        recv_num_tokens_per_expert_list = tmp[0]
        recv_num_tokens_per_expert_list_padded = tmp[1]

        token_padded_all = sum(recv_num_tokens_per_expert_list_padded)
        token_all_num = x.shape[0] * layer.top_k

        recv_x, recv_x_scale = fastdeploy.model_executor.ops.gpu.per_token_quant(
            x, 128)

        (
            permute_input,
            permute_scale,
            permute_indices_per_token,
            recv_num_tokens_per_expert_list_cumsum,
            recv_num_tokens_per_expert_list_padded_cumsum,
            dst_weights,
            dst_indices,
            cumsum_idx_gpu,
            m_indices,
        ) = fastdeploy.model_executor.ops.gpu.ep_moe_expert_dispatch_fp8(
            recv_x,
            recv_x_scale,
            topk_ids,
            topk_weights,
            recv_num_tokens_per_expert_list,
            recv_num_tokens_per_expert_list_padded,
            token_all_num,
            token_padded_all,
        )

        permute_scale = permute_scale.transpose([1, 0]).contiguous()
        permute_scale = permute_scale.transpose([1, 0])

        # ffn1
        ffn_out = paddle.empty(
            (permute_input.shape[0], layer.moe_ffn1_weight.shape[1]),
            dtype=paddle.bfloat16,
        )
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            (permute_input, permute_scale),
            (layer.moe_ffn1_weight, layer.moe_ffn1_weight_scale),
            ffn_out,
            m_indices,
        )
        # swiglu
        ffn_out = paddle.incubate.nn.functional.swiglu(ffn_out)

        # ffn2
        ffn_in_x, ffn_in_x_scale_tensor = fastdeploy.model_executor.ops.gpu.per_token_quant(
            ffn_out, 128)

        ffn_in_x_scale_tensor = ffn_in_x_scale_tensor.transpose(
            [1, 0]).contiguous()
        ffn_in_x_scale_tensor = ffn_in_x_scale_tensor.transpose([1, 0])

        ffn_out = paddle.empty(
            (ffn_out.shape[0], layer.moe_ffn2_weight.shape[1]),
            dtype=paddle.bfloat16)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            (ffn_in_x, ffn_in_x_scale_tensor),
            (layer.moe_ffn2_weight, layer.moe_ffn2_weight_scale),
            ffn_out,
            m_indices,
        )
        # prmt back per rank
        tmp_ffn_out = fastdeploy.model_executor.ops.gpu.ep_moe_expert_combine(
            ffn_out,
            dst_weights,
            permute_indices_per_token,
            dst_indices,
            None,
            False,  # norm_topk_prob
            1.0,
        )[0]

        return tmp_ffn_out

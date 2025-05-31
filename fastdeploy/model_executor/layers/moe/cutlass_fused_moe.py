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

from abc import abstractmethod

import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.framework import in_dynamic_or_pir_mode
from paddle.nn.quant import weight_quantize

from fastdeploy.model_executor.layers.quantization.quant_base import \
    QuantMethodBase
from fastdeploy.model_executor.ops.gpu import (moe_expert_dispatch,
                                               moe_expert_ffn,
                                               moe_expert_reduce)


class FusedMoEMethodBase(QuantMethodBase):
    """
    Use Cutlass Group Gemm to compute Fused MoE.
    """

    @abstractmethod
    def create_weights(self,
                       layer: nn.Layer,
                       moe_compute_params,
                       ffn1_tensor,
                       ffn2_tensor,
                       ffn1_bias=None,
                       ffn2_bias=None):
        """
        How to create weights, you should implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: nn.Layer,
        moe_compute_params,
        x: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Compute methods, you should implement this method.
        """

        raise NotImplementedError


class CutlassFusedMoeMethod(FusedMoEMethodBase):
    """
    Use Cutlass Group Gemm to compute Fused MoE.
    """

    def create_weights(self,
                       layer: nn.Layer,
                       moe_compute_params,
                       ffn1_tensor,
                       ffn2_tensor,
                       ffn1_bias=None,
                       ffn2_bias=None):

        num_local_experts = moe_compute_params.num_local_experts
        moe_quant_type = moe_compute_params.moe_quant_type

        assert len(ffn1_tensor) == num_local_experts
        assert len(ffn2_tensor) == num_local_experts

        if moe_quant_type in ["weight_only_int4", "weight_only_int8", "w4a8"]:

            added_weight_attrs = ["moe_ffn1_weight", "moe_ffn2_weight"]
            added_scale_attrs = [
                "moe_ffn1_weight_scale", "moe_ffn2_weight_scale"
            ]

            for idx, weight_tensor in enumerate([ffn1_tensor, ffn2_tensor]):
                weight_name = added_weight_attrs[idx]
                scale_name = added_scale_attrs[idx]

                weight_tensor_list = []
                weight_scale_tensor_list = []
                for i in range(num_local_experts):
                    quant_weight, scale = weight_quantize(
                        weight_tensor[i],
                        algo=moe_quant_type,
                        arch=80,
                    )
                    weight_tensor_list.append(quant_weight)
                    weight_scale_tensor_list.append(scale)
                quanted_weight = paddle.stack(weight_tensor_list, axis=0)
                quanted_weight_scale = paddle.stack(weight_scale_tensor_list,
                                                    axis=0)

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
                        attr=paddle.ParamAttr(
                            name=f"{layer.layer_name}.{scale_name}"),
                        dtype=quanted_weight_scale.dtype,
                    ))
                getattr(layer, scale_name).set_value(quanted_weight_scale)

        else:
            # not need any process!
            pass

    def apply(
        self,
        layer: nn.Layer,
        moe_compute_params,
        x: paddle.Tensor,
    ) -> paddle.Tensor:

        gate_out = paddle.matmul(x.cast("float32"), layer.gate_weight)

        (
            permute_input,
            token_nums_per_expert,
            permute_indices_per_token,
            topk_weights,
            topk_idx,
        ) = moe_expert_dispatch(
            x,
            gate_out,
            layer.gate_correction_bias,
            moe_compute_params.top_k,
            False,
            topk_only_mode=False,
        )

        ffn_out = moe_expert_ffn(
            permute_input,
            token_nums_per_expert,
            layer.moe_ffn1_weight,
            layer.moe_ffn2_weight,
            None,
            (layer.moe_ffn1_weight_scale
             if hasattr(layer, "moe_ffn1_weight_scale") else None),
            (layer.moe_ffn2_weight_scale
             if hasattr(layer, "moe_ffn2_weight_scale") else None),
            (layer.moe_ffn2_in_scale
             if hasattr(layer, "moe_ffn2_in_scale") else None),
            None,  # expert_idx_per_token
            moe_compute_params.moe_quant_type,
            False,  # used_in_ep_low_latency
        )

        if False:
            if in_dynamic_or_pir_mode():
                hcg = fleet.get_hybrid_communicate_group()
                mp_group = hcg.get_model_parallel_group()
                paddle.distributed.all_reduce(ffn_out, group=mp_group)
            else:
                paddle.distributed.all_reduce(ffn_out, group=mp_group)

        # reduce 中会做 topk 个 weight 的 norm 和 routed_scaling_factor
        fused_moe_out = moe_expert_reduce(
            ffn_out,
            topk_weights,
            permute_indices_per_token,
            topk_idx,
            None,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
        )
        return fused_moe_out

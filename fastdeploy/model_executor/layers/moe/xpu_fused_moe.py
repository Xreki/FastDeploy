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
from paddlenlp.utils.log import logger

from fastdeploy.model_executor.ops.xpu import weight_quantize_xpu

from .fused_moe_method_base import FusedMoEMethodBase


class XPUFusedMoeMethod(FusedMoEMethodBase):
    """
    XPU Fused MoE Method.
    """

    def __init__(self, moe_compute_params):
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
        Paddle cutlass create weight process.
        """
        assert len(ffn1_tensor) == self.num_local_experts
        assert len(ffn2_tensor) == self.num_local_experts
        assert ffn1_tensor[0].shape == [
            self.hidden_size, self.moe_intermediate_size * 2
        ]
        assert ffn2_tensor[0].shape == [
            self.moe_intermediate_size, self.hidden_size
        ]

        added_weight_attrs = ["moe_ffn1_weight", "moe_ffn2_weight"]
        added_scale_attrs = ["moe_ffn1_weight_scale", "moe_ffn2_weight_scale"]

        if self.moe_quant_type == "w4a8":
            raise NotImplementedError("XPU not support w4a8 now")

        if self.moe_quant_type in [
                "weight_only_int4", "weight_only_int8", "w4a8"
        ]:

            for idx, weight_tensor in enumerate([ffn1_tensor, ffn2_tensor]):
                logger.info("Quantizing {}th tensor".format(idx + 1))
                weight_name = added_weight_attrs[idx]
                scale_name = added_scale_attrs[idx]

                weight_list = []
                weight_scale_list = []
                for i in range(self.num_local_experts):
                    quant_weight, scale = weight_quantize_xpu(
                        weight_tensor[i], self.moe_quant_type, -1,
                        -1)  # weight is [k,n]
                    weight_list.append(quant_weight.transpose(
                        [1, 0]))  # transpose weight to [n,k]
                    if self.moe_quant_type != "w4a8":
                        # scale holds no memoty in w4a8, don't touch it!
                        weight_scale_list.append(scale)
                quanted_weight = paddle.stack(weight_list, axis=0)
                setattr(
                    layer, weight_name,
                    layer.create_parameter(
                        shape=quanted_weight.shape,
                        dtype=quanted_weight.dtype,
                        default_initializer=paddle.nn.initializer.Constant(0),
                    ))
                getattr(layer, weight_name).set_value(quanted_weight)

                # this scale only useful for wint8/4.
                if self.moe_quant_type != "w4a8":
                    quanted_weight_scale = paddle.stack(weight_scale_list,
                                                        axis=0)
                    setattr(
                        layer, scale_name,
                        layer.create_parameter(
                            shape=quanted_weight_scale.shape,
                            dtype=quanted_weight_scale.dtype,
                        ))
                    getattr(layer, scale_name).set_value(quanted_weight_scale)

        if self.moe_quant_type == "w4a8":
            raise NotImplementedError("XPU not support w4a8 now")

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        XPU compute Fused MoE.
        """
        from fastdeploy.model_executor.ops.xpu import xpu_moe_layer

        return xpu_moe_layer(
            x,
            layer.gate_weight.transpose([1, 0]),
            layer.gate_correction_bias,
            layer.moe_ffn1_weight,
            layer.moe_ffn2_weight,
            None,  # ffn1 bias
            None,  # ffn2 bias
            (layer.moe_ffn1_weight_scale
             if hasattr(layer, "moe_ffn1_weight_scale") else None),
            (layer.moe_ffn2_weight_scale
             if hasattr(layer, "moe_ffn2_weight_scale") else None),
            (layer.moe_ffn2_in_scale
             if hasattr(layer, "moe_ffn2_in_scale") else None),
            self.moe_quant_type,
            self.top_k,
            False,  # moe group, used in deepseek
        )

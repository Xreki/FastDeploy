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
from paddle.nn.quant import weight_quantize

import fastdeploy
from fastdeploy.model_executor.layers.utils import get_tensor

from .fused_moe_method_base import FusedMoEMethodBase, create_and_set_parameter


class CutlassFusedMoeMethod(FusedMoEMethodBase):
    """
    Use Cutlass Group Gemm to compute Fused MoE.
    This method is the oldest way to compute MoE in Paddle.
    """

    def create_weights(self, layer: nn.Layer, weight_key_map: dict,
                       state_dict: dict):
        """
        Paddle cutlass create weight process.
        """
        ffn1_weights, ffn2_weights = self.extract_moe_ffn_weights(
            weight_key_map, state_dict)

        pack_num = 1
        if self.moe_quant_type == "w4a8":
            pack_num = 2

        assert ffn1_weights[0].shape == [
            self.hidden_size // pack_num, self.moe_intermediate_size * 2
        ]
        assert ffn2_weights[0].shape == [
            self.moe_intermediate_size // pack_num, self.hidden_size
        ]

        added_weight_attrs = ["moe_ffn1_weight", "moe_ffn2_weight"]
        added_scale_attrs = ["moe_ffn1_weight_scale", "moe_ffn2_weight_scale"]

        if self.moe_quant_type in [
                "weight_only_int4", "weight_only_int8", "w4a8"
        ]:

            for idx, weight_tensor in enumerate([ffn1_weights, ffn2_weights]):
                weight_name = added_weight_attrs[idx]
                scale_name = added_scale_attrs[idx]

                weight_list = []
                weight_scale_list = []
                for i in range(self.num_local_experts):
                    quant_weight, scale = weight_quantize(
                        weight_tensor[i], algo=self.moe_quant_type, arch=80)
                    weight_list.append(quant_weight)
                    if self.moe_quant_type != "w4a8":
                        # scale holds no memory in w4a8, don't touch it!
                        weight_scale_list.append(scale)
                quanted_weight = paddle.stack(weight_list, axis=0)
                create_and_set_parameter(layer, weight_name, quanted_weight)

                # this scale only useful for wint8/4.
                if self.moe_quant_type != "w4a8":
                    quanted_weight_scale = paddle.stack(weight_scale_list,
                                                        axis=0)
                    create_and_set_parameter(layer, scale_name,
                                             quanted_weight_scale)

        if self.moe_quant_type == "w4a8":
            self.create_w4a8_scale_weights(layer, weight_key_map, state_dict)

    def create_w4a8_scale_weights(self, layer: nn.Layer, weight_key_map: dict,
                                  state_dict: dict):
        """
        Get w4a8 weights from state dict and process them.
        Args:
            layer (nn.Layer): The layer to add parameters to.
            weight_key_map (dict): The weight key map.
            state_dict (dict): The state dict.
        """

        def _extract_scale_tensor(state_dict, key_template, expert_idx):
            return get_tensor(
                state_dict.pop(key_template.format(self.layer_idx,
                                                   expert_idx)))

        def _process_in_scale(name: str, in_scales: list[paddle.Tensor]):
            processed_in_scale = 1 / paddle.concat(in_scales)
            create_and_set_parameter(layer, name, processed_in_scale)
            return processed_in_scale

        def _process_weight_scale(name: str,
                                  weight_scales: list[paddle.Tensor],
                                  processed_in_scale: paddle.Tensor):
            processed_weight_scale = (paddle.stack(weight_scales, axis=0) /
                                      (127 * 112) /
                                      processed_in_scale[:, None]).cast(
                                          paddle.get_default_dtype())
            create_and_set_parameter(layer, name, processed_weight_scale)

        # 1. Init scale containers and maps
        moe_ffn1_weight_scales = []
        moe_ffn2_weight_scales = []
        moe_ffn1_in_scales = []
        moe_ffn2_in_scales = []

        scale_weight_map = {
            "moe_ffn1_weight_scale": moe_ffn1_weight_scales,
            "moe_ffn2_weight_scale": moe_ffn2_weight_scales,
            "moe_ffn1_in_scale": moe_ffn1_in_scales,
            "moe_ffn2_in_scale": moe_ffn2_in_scales,
        }
        scale_key_map = {
            "moe_ffn1_weight_scale":
            weight_key_map.get("ffn1_expert_weight_scale_key", None),
            "moe_ffn2_weight_scale":
            weight_key_map.get("ffn2_expert_weight_scale_key", None),
            "moe_ffn1_in_scale":
            weight_key_map.get("ffn1_expert_in_scale_key", None),
            "moe_ffn2_in_scale":
            weight_key_map.get("ffn2_expert_in_scale_key", None),
        }
        for name, value in scale_key_map.items():
            if value is None:
                raise ValueError(
                    f"scale {name} should not be none in w4a8 mode.")

        # 2. Extract scale tensor from state dict
        if layer.ep_size > 1:
            expert_id_offset = self.ep_rank * self.local_num_experts
        else:
            expert_id_offset = 0

        for local_expert_idx in range(self.local_num_experts):
            expert_idx = local_expert_idx + expert_id_offset * self.local_num_experts
            for name, scale_key_template in scale_key_map.items():
                scale_tensor = _extract_scale_tensor(state_dict,
                                                     scale_key_template,
                                                     expert_idx)
                scale_weight_map[name].append(scale_tensor)

        # 3. Process scale tensor and set to layer
        in_scales = []
        for in_scale_name in ["moe_ffn1_in_scale", "moe_ffn2_in_scale"]:
            in_scales.append(
                _process_in_scale(in_scale_name,
                                  scale_weight_map[in_scale_name]))

        for i, weight_scale_name in enumerate(
            ["moe_ffn1_weight_scale", "moe_ffn2_weight_scale"]):
            _process_weight_scale(weight_scale_name,
                                  scale_weight_map[weight_scale_name],
                                  in_scales[i])

    def compute_ffn(
        self,
        layer: nn.Layer,
        permute_input: paddle.Tensor,
        token_nums_per_expert: paddle.Tensor,
        expert_idx_per_token: paddle.Tensor,
        used_in_ep_low_latency: bool = False,
    ):
        """
        Paddle Cutlass compute Fused MoE.
        """
        return fastdeploy.model_executor.ops.gpu.moe_expert_ffn(
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
            expert_idx_per_token,
            self.moe_quant_type,
            used_in_ep_low_latency,
        )

    @abstractmethod
    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Paddle Cutlass compute Fused MoE.
        """
        raise NotImplementedError

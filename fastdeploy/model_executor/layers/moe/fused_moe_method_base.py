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

from fastdeploy.model_executor.layers.quantization.quant_base import \
    QuantMethodBase
from fastdeploy.model_executor.layers.utils import get_tensor


def load_experts_weight(state_dict: dict, ffn1_expert_weight_key: str,
                        ffn2_expert_weight_key: str, expert_id_offset: int,
                        num_experts: int, ep_size: int, ep_rank: int):
    """
    Load experts weight from state_dict.
    Args:
        state_dict (dict): The state_dict of model.
        ffn1_expert_weight_key (str): The key of ffn1 expert weight.
        ffn2_expert_weight_key (str): The key of ffn2 expert weight.
    """
    ffn1_weights = []
    ffn2_weights = []
    is_ffn_merged = ffn1_expert_weight_key.format(expert_id_offset) in state_dict

    if is_ffn_merged:
        for i in range(num_experts):
            expert_idx = expert_id_offset + i
            ffn1_weights.append(
                get_tensor(
                    state_dict.pop(ffn1_expert_weight_key.format(expert_idx))))
            ffn2_weights.append(
                get_tensor(
                    state_dict.pop(ffn2_expert_weight_key.format(expert_idx))))
    else:
        gate_expert_weight_key = ffn1_expert_weight_key.replace("up_gate_proj", "gate_proj")
        up_expert_weight_key = ffn1_expert_weight_key.replace("up_gate_proj", "up_proj")
        for j in range(num_experts):
            expert_idx = expert_id_offset + j
            gate = get_tensor(
                    state_dict.pop(gate_expert_weight_key.format(expert_idx)))
            up = get_tensor(
                    state_dict.pop(up_expert_weight_key.format(expert_idx)))
            ffn1_weights.append(paddle.concat([gate, up], axis=-1))
            ffn2_weights.append(
                get_tensor(
                    state_dict.pop(ffn2_expert_weight_key.format(expert_idx))))
    return ffn1_weights, ffn2_weights


def create_and_set_parameter(layer: nn.Layer, name: str,
                             tensor: paddle.Tensor):
    """
    Create a parameter with the given name and set its value to the given tensor.
    """
    setattr(
        layer, name,
        layer.create_parameter(
            shape=tensor.shape,
            dtype=tensor.dtype,
            default_initializer=paddle.nn.initializer.Constant(0),
        ))
    getattr(layer, name).set_value(tensor)


class FusedMoEMethodBase(QuantMethodBase):
    """
    All MoE Method should inherit this class.
    and must implement following methods!

    """

    def __init__(self, moe_compute_params):
        self.layer_idx = moe_compute_params.layer_idx
        self.num_local_experts = moe_compute_params.num_local_experts
        self.expert_id_offset = moe_compute_params.expert_id_offset
        self.moe_quant_type = moe_compute_params.moe_quant_type
        self.hidden_size = moe_compute_params.hidden_size
        self.moe_intermediate_size = moe_compute_params.moe_intermediate_size
        self.top_k = moe_compute_params.top_k
        self.tp_size = moe_compute_params.tp_size
        self.ep_size = moe_compute_params.ep_size
        self.ep_rank = moe_compute_params.ep_rank

        self.global_num_experts = moe_compute_params.global_num_experts
        self.local_num_experts = moe_compute_params.num_local_experts
        self.use_method = moe_compute_params.use_method

    def extract_moe_ffn_weights(self, weight_key_map: dict, state_dict: dict):
        """
        Extract MoE FFN weights from state dict based on weight key mapping.

        Args:
            weight_key_map (dict): Dictionary mapping weight names to state dict keys.
                Expected keys: "ffn1_expert_weight_key" and "ffn2_expert_weight_key"
            state_dict (dict): Model state dictionary containing the weights.

        Returns:
            tuple: A tuple containing two lists:
                - ffn1_weights: List of tensors for first FFN layer weights
                - ffn2_weights: List of tensors for second FFN layer weights

        Raises:
            AssertionError: If required weight keys are missing or number of weights
                doesn't match number of local experts.
        """
        ffn1_expert_weight_key = weight_key_map.get("ffn1_expert_weight_key",
                                                    None)
        ffn2_expert_weight_key = weight_key_map.get("ffn2_expert_weight_key",
                                                    None)
        assert ffn1_expert_weight_key is not None, "ffn1_expert_weight_key should not be none."
        assert ffn2_expert_weight_key is not None, "ffn2_expert_weight_key should not be none."

        ffn1_weights, ffn2_weights = load_experts_weight(
            state_dict, ffn1_expert_weight_key, ffn2_expert_weight_key,
            self.expert_id_offset, self.local_num_experts, self.ep_size,
            self.ep_rank)
        assert len(
            ffn1_weights
        ) == self.num_local_experts, "ffn1_weights length should be equal to num_local_experts."
        assert len(
            ffn2_weights
        ) == self.num_local_experts, "ffn2_weights length should be equal to num_local_experts."

        return ffn1_weights, ffn2_weights

    @abstractmethod
    def create_weights(self, layer: nn.Layer, weight_key_map: dict,
                       state_dict: dict):
        """
        How to create weights, you must implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Compute methods, you must implement this method.
        """

        raise NotImplementedError

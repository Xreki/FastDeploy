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

from dataclasses import dataclass

from paddle import nn
from paddlenlp.utils.log import logger

from fastdeploy.model_executor.layers.utils import get_tensor

from .cutlass_fused_moe import CutlassFusedMoeMethod


@dataclass
class UniversalMoEComputeParams:
    """
    some params for computing MoE.
    it is given to different compute methods.
    """
    global_num_experts: int = -1
    top_k: int = -1
    hidden_size: int = -1
    num_local_experts: int = -1
    moe_intermediate_size: int = -1

    tp_size: int = -1
    ep_size: int = -1
    dp_size: int = -1

    moe_quant_type: str = ""


class FusedMoE(nn.Layer):
    """
    FusedMoE is a layer that performs MoE (Mixture of Experts) computation.
    """

    def __init__(
        self,
        llm_config,
        layer_name,
        layer_idx=-1,
    ):
        """
        Initialize the Moe layer with given parameters.
        Args:
            llm_config (LLMConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.

            layer_name (str): Unique name of the layer.
        """
        super().__init__()

        self.llm_config = llm_config
        self.layer_name = layer_name
        self.layer_idx = layer_idx

        self.tp_size = llm_config.parallel_config.mp_size
        self.ep_size = llm_config.parallel_config.ep_size

        self.hidden_size = llm_config.model_config.hidden_size
        self.moe_config = llm_config.moe_config
        self.moe_quant_type = self.moe_config.moe_quant_type
        self.use_offline_quant = llm_config.tmp_config.use_offline_quant
        moe_tag = self.llm_config.moe_config.moe_tag
        logger.info(f"{moe_tag}MoE is running in {self.moe_quant_type} mode")
        
        self.num_experts = self.moe_config.num_experts
        self.num_local_experts = self.num_experts // self.ep_size

        if self.ep_size >= 2:
            logger.debug("MoE is running in ep mode")
            self.moe_intermediate_size = self.moe_config.moe_intermediate_size
        else:
            logger.debug(f"MoE is running in tp{self.tp_size} mode")
            self.moe_intermediate_size = (
                self.moe_config.moe_intermediate_size // self.tp_size)

        weight_keys = llm_config.load_config.weight_keys
        self.gate_weight_key = weight_keys.moe_gate_weight_keys.format(
            layer_idx)
        self.gate_correction_bias_key = weight_keys.moe_gate_correction_bias_keys.format(
            layer_idx)

        self.ffn1_expert_weight_key = weight_keys.moe_ffn1_weight_keys
        self.ffn2_expert_weight_key = weight_keys.moe_ffn2_weight_keys
        self.ffn1_bias_key = weight_keys.moe_ffn1_bias_keys
        self.ffn2_bias_key = weight_keys.moe_ffn2_bias_keys

        if self.moe_quant_type == "w4a8":
            # below keys are only used in MoE W4A8!
            self.ffn1_expert_weight_scale_key = weight_keys.moe_ffn1_weight_scale_key
            self.ffn2_expert_weight_scale_key = weight_keys.moe_ffn2_weight_scale_key
            self.ffn1_expert_in_scale_key = weight_keys.moe_ffn1_expert_in_scale_key
            self.ffn2_expert_in_scale_key = weight_keys.moe_ffn2_expert_in_scale_key

        self.compute_method = CutlassFusedMoeMethod()

        self.moe_compute_params = UniversalMoEComputeParams()
        self.moe_compute_params.global_num_experts = self.num_experts
        self.moe_compute_params.top_k = self.moe_config.top_k
        self.moe_compute_params.hidden_size = self.hidden_size
        self.moe_compute_params.num_local_experts = self.num_local_experts
        self.moe_compute_params.moe_quant_type = self.moe_quant_type
        self.moe_compute_params.moe_intermediate_size = self.moe_intermediate_size
        self.moe_compute_params.ep_size = self.ep_size
        self.moe_compute_params.tp_size = self.tp_size

    def load_gate_state_dict(self, state_dict):
        """
        load_gate_state_dict function.
        """
        logger.info("Load TP FFN1")
        up_gate_proj_weight = []
        up_gate_proj_weight_scale = []
        down_proj_weight = []
        down_proj_weight_scale = []
        for j in range(self.num_experts):
            up_gate_proj_weight.append(
                get_tensor(
                    state_dict.pop(
                        self.ffn1_expert_weight_key.format(self.layer_idx,
                                                           j))))
            down_proj_weight.append(
                get_tensor(
                    state_dict.pop(
                        self.ffn2_expert_weight_key.format(self.layer_idx,
                                                           j))))
        return up_gate_proj_weight, down_proj_weight

    def load_state_dict(self, state_dict, is_update: bool = False):
        """
        load_state_dict function.
        """
        # gate
        if not is_update:
            gate_weight_tensor = get_tensor(state_dict.pop(self.gate_weight_key))
            self.gate_weight = self.create_parameter(
                shape=gate_weight_tensor.shape,
                dtype="float32",
            )
            self.gate_weight.set_value(gate_weight_tensor)

        # gate_correction_bias
        if self.moe_config.moe_use_gate_correction_bias:
            gate_correction_bias_tensor = get_tensor(
                state_dict.pop(self.gate_correction_bias_key))

            self.gate_correction_bias = self.create_parameter(
                shape=gate_correction_bias_tensor.shape,
                dtype="float32",
            )

            self.gate_correction_bias.set_value(gate_correction_bias_tensor)
        else:
            self.gate_correction_bias = None

        up_gate_proj_weight, down_proj_weight = self.load_gate_state_dict(
            state_dict)

        # other weight is with compute_method
        # different method may have different way to create weights
        self.compute_method.create_weights(self, self.moe_compute_params,
                                           up_gate_proj_weight,
                                           down_proj_weight)

    def forward(self, x, **kwargs):
        """
        Defines the forward computation of the moe layer.

        Args:
            x (Tensor): Input tensor to the moe layer.

        Returns:
            Tensor: Output tensor.

        """

        out = self.compute_method.apply(self, self.moe_compute_params, x)
        return out

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

import paddle
from paddle import nn

from fastdeploy.config import MoEPhase
from fastdeploy.model_executor.layers.utils import get_tensor

from .fused_moe_method_tp import TPFusedMoeMethod


@dataclass
class MoEComputeParams:
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
        fd_config,
        moe_intermediate_size: int = -1,
        num_experts: int = -1,
        top_k: int = -1,
        moe_use_gate_correction_bias: bool = False,
        moe_quant_type: str = "weight_only_int4",
        layer_idx: int = -1,
        weight_key_map: dict = {},
        use_method="cutlass",
    ):
        """
        Initialize the Moe layer with given parameters.
        Args:
            fd_config (FDConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
        """
        super().__init__()

        self.fd_config = fd_config
        self.layer_idx = layer_idx

        self.tp_size = fd_config.parallel_config.tensor_parallel_degree
        self.ep_size = fd_config.parallel_config.expert_parallel_degree
        self.ep_rank = fd_config.parallel_config.expert_parallel_rank

        assert (self.tp_size > 1 and self.ep_size == 1) or \
                (self.tp_size == 1 and self.ep_size > 1), \
            'MoE only support parallelism on TP or EP dimension.'

        self.moe_use_gate_correction_bias = moe_use_gate_correction_bias

        self.hidden_size = fd_config.model_config.hidden_size
        self.moe_config = fd_config.moe_config
        self.use_offline_quant = fd_config.tmp_config.use_offline_quant

        self.moe_quant_type = moe_quant_type
        self.num_experts = num_experts
        self.num_local_experts = self.num_experts // self.ep_size

        self.moe_intermediate_size = moe_intermediate_size // self.tp_size
        self.weight_key_map = weight_key_map
        self.use_method = use_method

        moe_compute_params = MoEComputeParams()
        moe_compute_params.layer_idx = self.layer_idx
        moe_compute_params.global_num_experts = self.num_experts
        moe_compute_params.top_k = top_k
        moe_compute_params.hidden_size = self.hidden_size
        moe_compute_params.num_local_experts = self.num_local_experts
        moe_compute_params.moe_quant_type = self.moe_quant_type
        moe_compute_params.moe_intermediate_size = self.moe_intermediate_size
        moe_compute_params.ep_size = self.ep_size
        moe_compute_params.tp_size = self.tp_size
        moe_compute_params.ep_rank = self.ep_rank
        moe_compute_params.num_max_dispatch_tokens_per_rank = fd_config.moe_config.num_max_dispatch_tokens_per_rank
        moe_compute_params.use_method = self.use_method

        if self.ep_size > 1:
            # Lazy import
            from .fused_moe_method_ep import (EPDecoderFusedMoeMethod,
                                              EPPrefillFusedMoeMethod)

            if fd_config.parallel_config.moe_phase == MoEPhase.PREFILL:
                self.compute_method = EPPrefillFusedMoeMethod(
                    moe_compute_params)
            else:
                self.compute_method = EPDecoderFusedMoeMethod(
                    moe_compute_params)
        else:
            self.compute_method = TPFusedMoeMethod(moe_compute_params)

    def load_state_dict(self, state_dict, is_update: bool = False):
        """
        load_state_dict function.
        """
        # gate
        if not is_update:
            gate_weight_key = self.weight_key_map.get("gate_weight_key", None)
            assert gate_weight_key is not None, "gate_weight_key should not be None, please check model checkpoints"

            gate_weight_tensor = get_tensor(state_dict.pop(gate_weight_key))

            self.gate_weight = self.create_parameter(
                shape=gate_weight_tensor.shape,
                dtype="float32",
            )
            self.gate_weight.set_value(gate_weight_tensor)

        # gate_correction_bias
        if self.moe_use_gate_correction_bias:
            gate_correction_bias_key = self.weight_key_map.get(
                "gate_correction_bias_key", None)
            assert gate_weight_key is not None, "gate_correction_bias_key should \
                not be None when moe_use_gate_correction_bias is True, please check model checkpoints"

            gate_correction_bias_tensor = get_tensor(
                state_dict.pop(gate_correction_bias_key).astype("float32"))

            self.gate_correction_bias = self.create_parameter(
                shape=gate_correction_bias_tensor.shape,
                dtype="float32",
            )

            self.gate_correction_bias.set_value(gate_correction_bias_tensor)

        # other weight is with compute_method
        # different method may have different way to create weights
        self.compute_method.create_weights(self, self.weight_key_map,
                                           state_dict)

    def forward(self, x: paddle.Tensor):
        """
        Defines the forward computation of the moe layer.

        Args:
            x (Tensor): Input tensor to the moe layer.

        Returns:
            Tensor: Output tensor.

        """
        gate_out = paddle.matmul(x.cast("float32"), self.gate_weight)
        out = self.compute_method.apply(self, x, gate_out)
        return out

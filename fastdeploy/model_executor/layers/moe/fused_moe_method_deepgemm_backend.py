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

from abc import abstractmethod

import paddle
from paddle import nn

from .fused_moe_method_base import FusedMoEMethodBase


class DeepGemmFusedMoeMethod(FusedMoEMethodBase):
    """
    DeepGemmFusedMoeMethod is a class that implements the FusedMoEMethodBase interface for DeepGemm backend.
    """

    def create_weights(self, layer: nn.Layer, weight_key_map: dict,
                       state_dict: dict):
        """
        deepgemm create weight process.
        """
        raise NotImplementedError

    def compute_ffn(
        self,
        layer: nn.Layer,
        permute_input: paddle.Tensor,
        token_nums_per_expert: paddle.Tensor,
        expert_idx_per_token: paddle.Tensor,
    ):
        """
        Compute the feed-forward network (FFN) part of MoE using DeepGemm backend.

        Args:
            layer (nn.Layer): The layer containing MoE weights and parameters.
            permute_input (paddle.Tensor): Permuted input tensor for expert computation.
            token_nums_per_expert (paddle.Tensor): Number of tokens assigned to each expert.
            expert_idx_per_token (paddle.Tensor): Expert indices for each token.

        Returns:
            paddle.Tensor: Output tensor after FFN computation.

        Note:
            This is an abstract method that should be implemented by concrete subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: nn.Layer,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Paddle Cutlass compute Fused MoE.
        """
        raise NotImplementedError

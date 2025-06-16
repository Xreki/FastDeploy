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

from fastdeploy.model_executor.ops.gpu import (moe_expert_dispatch,
                                               moe_expert_reduce)

from .fused_moe_method_cutlass_backend import CutlassFusedMoeMethod
from .fused_moe_method_deepgemm_backend import DeepGemmFusedMoeMethod
from .fused_moe_method_triton_backend import TritonFusedMoeMethod


class TPFusedMoeMethod(CutlassFusedMoeMethod, DeepGemmFusedMoeMethod,
                       TritonFusedMoeMethod):
    """
    """

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Paddle Cutlass compute Fused MoE.
        """
        if self.use_method == "triton":
            return TritonFusedMoeMethod.apply(layer, x, gate_out)

        (
            permute_input,
            token_nums_per_expert,
            permute_indices_per_token,
            topk_weights,
            topk_idx,
            expert_idx_per_token,
        ) = moe_expert_dispatch(
            x,
            gate_out,
            getattr(layer, "gate_correction_bias", None),
            getattr(layer, "moe_ffn1_in_scale", None), # if set, permute_input will be int8_t
            self.top_k,
            False,
            topk_only_mode=False,
        )

        if self.moe_quant_type != "w4a8":
            # only w4a8 need expert_idx_per_token
            # Other need not this tensor, so we make it None.
            expert_idx_per_token = None
        else:
            expert_idx_per_token = expert_idx_per_token.cast("int64")

        if self.moe_quant_type == "fp8":
            raise NotImplementedError
        elif self.moe_quant_type in [
                "weight_only_int4", "weight_only_int8", "w4a8"
        ]:
            ffn_out = self.compute_ffn(layer, permute_input,
                                       token_nums_per_expert,
                                       expert_idx_per_token)
        else:
            raise NotImplementedError

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

        if self.tp_size > 1:
            from fastdeploy.distributed.communication_op import \
                tensor_model_parallel_all_reduce
            tensor_model_parallel_all_reduce(fused_moe_out)

        return fused_moe_out

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

import fastdeploy

from .ep import EPDecoderRunner, EPPrefillRunner
from .fused_moe_method_cutlass_backend import CutlassFusedMoeMethod
from .fused_moe_method_deepgemm_backend import DeepGemmFusedMoeMethod


class EPPrefillFusedMoeMethod(CutlassFusedMoeMethod, DeepGemmFusedMoeMethod):
    """
    """

    def __init__(self, moe_compute_params):
        """
        Initialize the EP prefill method.
        """
        super().__init__(moe_compute_params)
        self.ep_prefill_runner = EPPrefillRunner(self.top_k, self.hidden_size,
                                                 self.global_num_experts,
                                                 self.ep_size, self.ep_rank)

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Apply the EP prefill method.
        """
        # 1. Select topk experts and weights
        topk_idx, topk_weights = self.ep_prefill_runner.moe_select(
            layer, gate_out)
        # 2. EP Dispatch
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_num_tokens_per_expert_list,
            handle,
            _,
        ) = self.ep_prefill_runner.dispatch(x, topk_idx, topk_weights)
        token_all_num = sum(recv_num_tokens_per_expert_list)
        logger.info(f"token_all_num {token_all_num}")

        # 3. Compute ffn
        if token_all_num > 0:
            (
                permute_input,
                permute_indices_per_token,
                recv_num_tokens_per_expert_list_cumsum,
                dst_weights,
                dst_indices,
                cumsum_idx_gpu,
                expert_idx_per_token,
            ) = fastdeploy.model_executor.ops.gpu.ep_moe_expert_dispatch(
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                (self.moe_ffn1_in_scale
                 if hasattr(self, "moe_ffn1_in_scale") else None),
                recv_num_tokens_per_expert_list,
                token_all_num,
                self.moe_quant_type,
            )
            if self.moe_quant_type == "fp8":
                raise NotImplementedError
            elif self.moe_quant_type in [
                    "weight_only_int4", "weight_only_int8", "w4a8"
            ]:
                ffn_out = self.compute_ffn(
                    layer, permute_input,
                    recv_num_tokens_per_expert_list_cumsum,
                    expert_idx_per_token)
            else:
                raise NotImplementedError

            # prmt back per rank
            tmp_ffn_out = fastdeploy.model_executor.ops.gpu.ep_moe_expert_combine(
                ffn_out,
                dst_weights,
                permute_indices_per_token,
                dst_indices,
                None,  # moe_ffn2_bias,
                False,  # norm_topk_prob
                1.0,
            )[0]
        else:
            tmp_ffn_out = recv_x

        # 4. EP combine
        return self.ep_prefill_runner.combine(tmp_ffn_out, handle,
                                              recv_topk_weights)


class EPDecoderFusedMoeMethod(CutlassFusedMoeMethod, DeepGemmFusedMoeMethod):
    """
    """

    def __init__(self, moe_compute_params):
        """
        Initialize the EP decoder method.
        """
        super().__init__(moe_compute_params)
        self.num_max_dispatch_tokens_per_rank = moe_compute_params.num_max_dispatch_tokens_per_rank
        self.ep_decoder_runner = EPDecoderRunner(
            self.top_k, self.hidden_size, self.global_num_experts,
            self.num_max_dispatch_tokens_per_rank, self.ep_size, self.ep_rank)

    def apply(
        self,
        layer: nn.Layer,
        x: paddle.Tensor,
        gate_out: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Apply the EP decoder method.
        """
        if self.use_method == "triton":
            raise NotImplementedError

        # 1. Select topk experts and weights
        topk_idx, topk_weights = self.ep_decoder_runner.moe_select(
            layer, gate_out)
        # 2. EP Dispatch
        permute_input, token_nums_per_expert, handle = self.ep_decoder_runner.dispatch(
            x, topk_idx, topk_weights)
        # 3. Compute ffn

        if self.moe_quant_type == "fp8":
            raise NotImplementedError
        elif self.moe_quant_type in [
                "weight_only_int4", "weight_only_int8", "w4a8"
        ]:
            if self.moe_quant_type == "w4a8":
                num_local_experts, max_num, _ = permute_input.shape
                expert_idx_per_token = paddle.arange(
                    num_local_experts)[:, None].tile([1, max_num])
            else:
                expert_idx_per_token = None

            ffn_out = CutlassFusedMoeMethod.compute_ffn(
                layer, permute_input, token_nums_per_expert,
                expert_idx_per_token, True)
        else:
            raise NotImplementedError

        # 4. EP combine
        return self.ep_decoder_runner.combine(ffn_out, handle, topk_idx,
                                              topk_weights)

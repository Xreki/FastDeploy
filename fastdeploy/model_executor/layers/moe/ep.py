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

import os
import paddle
import fastdeploy

from paddle.base.core import Config
from paddle.distributed.communication.group import Group
from paddle.distributed.communication import deep_ep
from paddlenlp.utils.log import logger

from .moe import MoELayer
from fastdeploy.inference_args import GenerationPhase
from .utils import get_tensor
import fastdeploy.model_executor.ops.gpu.deep_gemm as deep_gemm

import numpy as np


class DeepEPEngine:
    """
    A wrapper class for DeepEP engine.
    """

    def __init__(
        self,
        group: Group,
        num_ranks: int,
        rank_id: int,
        num_max_dispatch_tokens_per_rank: int,
        hidden: int,
        num_experts: int,
        generation_phase: GenerationPhase,
        async_finish: bool = False,
    ):
        """
        Initialize the DeepEP engine.
        Args:
            group: The MPI group object.
            num_ranks: The number of ranks.
            rank_id: The rank id.
            num_max_dispatch_tokens_per_rank: The maximum number of tokens per rank to dispatch.
            hidden: The hidden dimension of the model.
            num_experts: The number of experts.
        """
        self.group = group
        self.num_ranks = num_ranks
        self.rank_id = rank_id
        self.hidden = hidden
        self.num_experts = num_experts
        self.num_local_experts = num_experts // num_ranks
        self.generation_phase = generation_phase
        self.async_finish = async_finish

        self.deepep_engine = None

        if generation_phase == GenerationPhase.DECODER:
            logger.info("Initializing Low Latency Buffer")
            self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
            self.get_low_latency_buffer()
        elif generation_phase == GenerationPhase.PREFILL:
            self.deepep_engine = deep_ep.Buffer(
                group,
                int(1e9),
                0,
                low_latency_mode=False,
                num_qps_per_rank=1,
            )
            self.ep_config = Config(24, 6, 256)
        else:
            raise ValueError(f"Unknown generation phase {generation_phase}")

    def get_low_latency_buffer(self) -> deep_ep.Buffer:
        """
        Get the DeepEP buffer.
        Args:
            group: The MPI group object.
            num_max_dispatch_tokens_per_rank: The maximum number of tokens per rank to dispatch.
            hidden: The hidden dimension of the model.
        """
        # NOTES: the low-latency mode will consume much more space than the normal mode
        # So we recommend that `num_max_dispatch_tokens_per_rank`
        #   (the actual batch size in the decoding engine) should be less than 256
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            self.num_max_dispatch_tokens_per_rank,
            self.hidden,
            self.num_ranks,
            self.num_experts,
        )
        # Allocate a buffer if not existed or not enough buffer size
        if (
            self.deepep_engine is None
            or self.deepep_engine.group != self.group
            or not self.deepep_engine.low_latency_mode
            or self.deepep_engine.num_rdma_bytes < num_rdma_bytes
        ):
            # NOTES: for best performance, the QP number **must** be equal to the number of the local experts
            assert self.num_experts % self.num_ranks == 0
            self.deepep_engine = deep_ep.Buffer(
                self.group,
                0,
                num_rdma_bytes,
                low_latency_mode=True,
                num_qps_per_rank=self.num_experts // self.num_ranks,
            )

    def low_latency_dispatch(
        self,
        hidden_states: paddle.Tensor,
        topk_idx: paddle.Tensor,
        use_fp8: bool = False,
    ):
        """
        Args:
            hidden_states: [token_num, hidden] 'bfloat16/int8'
            topk_idx: [token_num, num_topk] 'int64'

        Returns:
            recv_hidden_states: [num_local_experts,
                                 num_max_dispatch_tokens_per_rank * num_ranks, hidden]
                                 num_ranks * num_local_experts = num_experts
            recv_count: [num_local_experts]
            recv_count: a tensor shaped `[num_local_experts]` with type `torch.int`, indicating how many tokens each
                expert receive. As mentioned before, all not tokens are valid in `recv_x`.
            handle: the communication handle to be used in the `low_latency_combine` function.
            event: the event after executing the kernel (valid only if `async_finish` is set).
            hook: the receiving hook function (valid only if `return_recv_hook` is set).
        """
        (
            packed_recv_x,
            recv_expert_count,
            handle,
            _,
            dispatch_hook,
        ) = self.deepep_engine.low_latency_dispatch(
            hidden_states,
            topk_idx,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            use_fp8=use_fp8,
            async_finish=False,
            return_recv_hook=True,
        )

        return packed_recv_x, recv_expert_count, handle, dispatch_hook

    def low_latency_combine(
        self,
        hidden_states: paddle.Tensor,
        topk_idx: paddle.Tensor,
        topk_weights: paddle.Tensor,
        handle,
    ):
        """

        Return:
            combined_hidden_states: [num_tokens, hidden]
        """

        combined_hidden_states, _, combine_hook = (
            self.deepep_engine.low_latency_combine(
                hidden_states,
                topk_idx,
                topk_weights,
                handle,
                async_finish=False,
                return_recv_hook=True,
            )
        )
        return combined_hidden_states, combine_hook

    def clean_low_latency_buffer(self):
        """
        clean_low_latency_buffer
        """
        self.deepep_engine.clean_low_latency_buffer(
            self.num_max_dispatch_tokens_per_rank, self.hidden, self.num_experts
        )

    def barrier_all(self):
        """
        barrier_all
        """
        self.deepep_engine.barrier_all()


class MoeEPLayer(MoELayer):
    """
    MOE EP Layer
    """

    def __init__(
        self, ep_engine: DeepEPEngine, num_local_experts: int, *args, **kwargs
    ):
        """
        Initialize MOE EP Layer
        """
        kwargs["num_local_experts"] = num_local_experts
        kwargs["nranks"] = 1  # Only support 1 rank for  EP MOE
        super().__init__(*args, **kwargs)
        self.ep_engine = ep_engine
        self.ep_size = self.ep_engine.num_ranks
        self.ep_rank = self.ep_engine.rank_id

    def load_scale_state_dict(self):
        """
        load_scale_state_dict function.
        """
        up_gate_proj_weight_scale = []
        down_proj_weight_scale = []
        up_gate_proj_in_scale = []
        down_proj_in_scale = []

        for j in range(
            self.num_experts_start_offset,
            self.num_experts_start_offset + self.num_local_experts,
        ):
            up_gate_proj_in_scale_value = self.inference_args.act_scale_dict.pop(
                self.ffn1_expert_in_scale_key.format(j)
            )
            up_gate_proj_weight_scale_np = np.array(
                self.inference_args.weight_scale_dict.pop(
                    self.ffn1_expert_weight_scale_key.format(j)
                )
            )
            up_gate_proj_weight_scale_np = up_gate_proj_weight_scale_np / (
                127.0 * 112.0 * up_gate_proj_in_scale_value
            )
            up_gate_proj_in_scale.append(up_gate_proj_in_scale_value)
            up_gate_proj_weight_scale.append(
                paddle.to_tensor(up_gate_proj_weight_scale_np, dtype="float32")
            )

            down_proj_in_scale_value = self.inference_args.act_scale_dict.pop(
                self.ffn2_expert_in_scale_key.format(j)
            )
            down_proj_weight_scale_np = np.array(
                self.inference_args.weight_scale_dict.pop(
                    self.ffn2_expert_weight_scale_key.format(j)
                )
            )
            down_proj_weight_scale_np = down_proj_weight_scale_np / (
                127.0 * 112.0 * down_proj_in_scale_value
            )
            down_proj_in_scale.append(down_proj_in_scale_value)
            down_proj_weight_scale.append(
                paddle.to_tensor(down_proj_weight_scale_np, dtype="float32")
            )
        return (
            up_gate_proj_weight_scale,
            down_proj_weight_scale,
            up_gate_proj_in_scale,
            down_proj_in_scale,
        )

    def load_gate_state_dict(self, state_dict):
        """
        Load Gate State Dict from state_dict
        Args:
            state_dict: state dict
        """
        up_gate_proj_weight = []
        down_proj_weight = []
        for j in range(
            self.num_experts_start_offset,
            self.num_experts_start_offset + self.num_local_experts,
        ):
            up_gate_proj_weight.append(
                get_tensor(state_dict.pop(self.ffn1_expert_weight_key.format(j)))
            )
            down_proj_weight.append(
                get_tensor(state_dict.pop(self.ffn2_expert_weight_key.format(j)))
            )
        return up_gate_proj_weight, down_proj_weight

    def forward(self, x, **kwargs):
        """
        MoeEPLayer Forward Function
        """
        raise NotImplementedError


class PrefillMoeEPLayer(MoeEPLayer):
    """
    Prefill MOE EP Layer
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug("Init Prefill EP Layer")
        self.ep_async_finish = False

    def micro_batch_gate(self, x):
        """
        Run the micro-batch's gate and select topk's export.

        Args:
            x (Tensor): The index of micro-batch. The shape is
                `[token, num_export]`. The data type should be bfloat16,
                float16 or float32.

        Returns:
            topk_idx (Tensor): The index of getting highest score's exports.
                The shape is `[token, topk]`. The data type should
                be int64.
            topk_weights (Tensor): The scores of getting highest score's exports.
                The shape is `[token, topk]`. The data type should be float32.
        """
        gate_out = paddle.matmul(x.cast("float32"), self.gate_weight)
        topk_idx, topk_weights = fastdeploy.model_executor.ops.gpu.moe_topk_select(
            gate_out,
            (
                self.gate_correction_bias
                if self.moe_config.moe_use_gate_correction_bias
                else None
            ),
            self.top_k,
            True,
            False,
        )
        return topk_idx, topk_weights

    def micro_batch_dispatch(self, x, topk_idx, topk_weights, event):
        """
        Run the micro-batch's all to all dispatch.

        Args:
            x (Tensor): The index of micro-batch. The shape is
                `[token, num_export]`. The data type should be bfloat16,
                float16 or float32.
            topk_idx (Tensor): The index of getting highest score's exports.
                The shape is `[token, topk]`. The data type should
                be int64.
            topk_weights (Tensor): The scores of getting highest score's exports.
                The shape is `[token, topk]`. The data type should be float32.
            event (EventOverlap): The event of execute dispatch communication
        """
        (num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _) = (
            self.ep_engine.deepep_engine.get_dispatch_layout(
                topk_idx,
                self.num_experts,
                previous_event=event,
                async_finish=self.ep_engine.async_finish,
                allocate_on_comm_stream=self.ep_engine.async_finish,
            )
        )
        dispatch_args = {
            "x": x,
            "num_tokens_per_rank": num_tokens_per_rank,
            "is_token_in_rank": is_token_in_rank,
            "num_tokens_per_expert": num_tokens_per_expert,
            "config": self.ep_engine.ep_config,
            "async_finish": self.ep_engine.async_finish,
            "topk_idx": topk_idx,
            "topk_weights": topk_weights,
            "previous_event": event,
            "allocate_on_comm_stream": self.ep_engine.async_finish,
        }
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_num_tokens_per_expert_list,
            handle,
            event,
        ) = self.ep_engine.deepep_engine.dispatch(**dispatch_args)
        return (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            recv_num_tokens_per_expert_list,
            handle,
            event,
        )

    def micro_batch_ffn(
        self,
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        recv_num_tokens_per_expert_list,
        handle,
    ):
        r"""
        Run the micro-batch's moe ffn.
        """
        (
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            is_token_in_rank,
            send_head,
        ) = handle
        token_all_num = sum(recv_num_tokens_per_expert_list)
        if self.moe_quant_type == "fp8":
            if token_all_num > 0:
                recv_num_tokens_per_expert_list_np = np.array(
                    recv_num_tokens_per_expert_list
                )
                recv_num_tokens_per_expert_list_padded = (
                    128
                    - recv_num_tokens_per_expert_list_np % 128
                    + recv_num_tokens_per_expert_list_np
                ).tolist()
                token_padded_all = sum(recv_num_tokens_per_expert_list_padded)
                (recv_x, recv_x_scale) = recv_x
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
                    recv_topk_idx,
                    recv_topk_weights,
                    recv_num_tokens_per_expert_list,
                    recv_num_tokens_per_expert_list_padded,
                    token_all_num,
                    token_padded_all,
                )
                # ffn1
                ffn_out = paddle.empty(
                    (permute_input.shape[0], self.ffn1_weight_shape[1]),
                    dtype=paddle.bfloat16,
                )
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (permute_input, permute_scale),
                    (self.moe_ffn1_weight, self.moe_ffn1_weight_scale),
                    ffn_out,
                    m_indices,
                )
                # swiglu
                ffn_out = paddle.incubate.nn.functional.swiglu(ffn_out, None)
                # ffn2
                ffn_in_x, ffn_in_x_scale_tensor = (
                    fastdeploy.model_executor.ops.gpu.per_token_quant(
                        ffn_out, self.inference_args.weight_block_size[0]
                    )
                )
                ffn_out = paddle.empty(
                    (ffn_out.shape[0], self.ffn2_weight_shape[1]), dtype=paddle.bfloat16
                )
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (ffn_in_x, ffn_in_x_scale_tensor),
                    (self.moe_ffn2_weight, self.moe_ffn2_weight_scale),
                    ffn_out,
                    m_indices,
                )
                # prmt back per rank
                tmp_ffn_out = fastdeploy.model_executor.ops.gpu.ep_moe_expert_combine(
                    ffn_out,
                    dst_weights,
                    permute_indices_per_token,
                    dst_indices,
                    self.moe_ffn2_bias,
                    False,  # norm_topk_prob
                    1.0,
                )[0]
            else:
                tmp_ffn_out = paddle.cast(recv_x, self._dtype)
        else:
            if token_all_num > 0:
                # token个数为0时不能走自定义算子
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
                    (
                        self.moe_ffn1_in_scale
                        if hasattr(self, "moe_ffn1_in_scale")
                        else None
                    ),
                    recv_num_tokens_per_expert_list,
                    token_all_num,
                    self.moe_quant_type,
                )

                # moe ffn per rank
                ffn_out = fastdeploy.model_executor.ops.gpu.moe_expert_ffn(
                    permute_input,
                    recv_num_tokens_per_expert_list_cumsum,
                    self.moe_ffn1_weight,
                    self.moe_ffn2_weight,
                    self.moe_ffn1_bias,
                    (
                        self.moe_ffn1_weight_scale
                        if hasattr(self, "moe_ffn1_weight_scale")
                        else None
                    ),
                    (
                        self.moe_ffn2_weight_scale
                        if hasattr(self, "moe_ffn2_weight_scale")
                        else None
                    ),
                    (
                        self.moe_ffn2_in_scale
                        if hasattr(self, "moe_ffn2_in_scale")
                        else None
                    ),
                    expert_idx_per_token,
                    self.moe_quant_type,
                    False,  # used_in_ep_low_latency
                )
                # prmt back per rank
                tmp_ffn_out = fastdeploy.model_executor.ops.gpu.ep_moe_expert_combine(
                    ffn_out,
                    dst_weights,
                    permute_indices_per_token,
                    dst_indices,
                    self.moe_ffn2_bias,
                    False,  # norm_topk_prob
                    1.0,
                )[0]
            else:
                tmp_ffn_out = recv_x
        return tmp_ffn_out

    def micro_batch_combine(self, tmp_ffn_out, recv_topk_weights, handle, event):
        """
        Run the micro-batch's all to all dispatch.
        """
        combine_args = {
            "x": tmp_ffn_out,
            "handle": handle,
            "config": self.ep_engine.ep_config,
            "async_finish": self.ep_engine.async_finish,
            "topk_weights": recv_topk_weights,
            "previous_event": event,
            "allocate_on_comm_stream": self.ep_engine.async_finish,
        }
        before_norm_fused_moe_out, combined_topk_weights, event = (
            self.ep_engine.deepep_engine.combine(**combine_args)
        )
        return before_norm_fused_moe_out, combined_topk_weights, event

    def forward(self, x, **kwargs):
        """
        PrefillMoeEPLayer Forward Function
        Args:
            x: [token_num, hidden_dim]
        """
        gate_out = paddle.matmul(x.cast("float32"), self.gate_weight)
        # get topk
        topk_idx, topk_weights = fastdeploy.model_executor.ops.gpu.moe_topk_select(
            gate_out,
            (
                self.gate_correction_bias
                if self.moe_config.moe_use_gate_correction_bias
                else None
            ),
            self.top_k,
            True,  # apply_norm_weight,
            False,
        )
        # dispatch intranode
        (num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _) = (
            self.ep_engine.deepep_engine.get_dispatch_layout(topk_idx, self.num_experts)
        )
        if self.moe_quant_type == "fp8":
            x, x_scale_tensor = fastdeploy.model_executor.ops.gpu.per_token_quant(
                x, self.inference_args.weight_block_size[0]
            )
            # dispatch intranode
            dispatch_args = {
                "x": (x, x_scale_tensor),
                "num_tokens_per_rank": num_tokens_per_rank,
                "is_token_in_rank": is_token_in_rank,
                "num_tokens_per_expert": num_tokens_per_expert,
                "config": self.ep_engine.ep_config,
                "async_finish": self.ep_engine.async_finish,
                "topk_idx": topk_idx,
                "topk_weights": topk_weights,
            }
            (
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                recv_num_tokens_per_expert_list,
                handle,
                event,
            ) = self.ep_engine.deepep_engine.dispatch(**dispatch_args)
            (
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                recv_src_idx,
                is_token_in_rank,
                send_head,
            ) = handle
            # prmt per rank
            token_all_num = sum(recv_num_tokens_per_expert_list)
            if token_all_num > 0:
                recv_num_tokens_per_expert_list_np = np.array(
                    recv_num_tokens_per_expert_list
                )
                recv_num_tokens_per_expert_list_padded = (
                    128
                    - recv_num_tokens_per_expert_list_np % 128
                    + recv_num_tokens_per_expert_list_np
                ).tolist()
                token_padded_all = sum(recv_num_tokens_per_expert_list_padded)
                (recv_x, recv_x_scale) = recv_x
                # token个数为0时不能走自定义算子
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
                    recv_topk_idx,
                    recv_topk_weights,
                    recv_num_tokens_per_expert_list,
                    recv_num_tokens_per_expert_list_padded,
                    token_all_num,
                    token_padded_all,
                )
                # ffn1
                ffn_out = paddle.empty(
                    (permute_input.shape[0], self.ffn1_weight_shape[1]),
                    dtype=paddle.bfloat16,
                )
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (permute_input, permute_scale),
                    (self.moe_ffn1_weight, self.moe_ffn1_weight_scale),
                    ffn_out,
                    m_indices,
                )
                # swiglu
                ffn_out = paddle.incubate.nn.functional.swiglu(ffn_out, None)
                # ffn2
                ffn_in_x, ffn_in_x_scale_tensor = (
                    fastdeploy.model_executor.ops.gpu.per_token_quant(
                        ffn_out, self.inference_args.weight_block_size[0]
                    )
                )
                ffn_out = paddle.empty(
                    (ffn_out.shape[0], self.ffn2_weight_shape[1]), dtype=paddle.bfloat16
                )
                deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                    (ffn_in_x, ffn_in_x_scale_tensor),
                    (self.moe_ffn2_weight, self.moe_ffn2_weight_scale),
                    ffn_out,
                    m_indices,
                )
                # prmt back per rank
                tmp_ffn_out = fastdeploy.model_executor.ops.gpu.ep_moe_expert_combine(
                    ffn_out,
                    dst_weights,
                    permute_indices_per_token,
                    dst_indices,
                    self.moe_ffn2_bias,
                    False,  # norm_topk_prob
                    1.0,
                )[0]
            else:
                tmp_ffn_out = paddle.cast(recv_x, self._dtype)
            # intranode combine
            combine_args = {
                "x": tmp_ffn_out,
                "handle": handle,
                "config": self.ep_engine.ep_config,
                "async_finish": self.ep_engine.async_finish,
                "topk_weights": recv_topk_weights,
            }
            fused_moe_out, combined_topk_weights, event = (
                self.ep_engine.deepep_engine.combine(**combine_args)
            )
        else:
            # dispatch intranode
            dispatch_args = {
                "x": x,
                "num_tokens_per_rank": num_tokens_per_rank,
                "is_token_in_rank": is_token_in_rank,
                "num_tokens_per_expert": num_tokens_per_expert,
                "config": self.ep_engine.ep_config,
                "async_finish": self.ep_engine.async_finish,
                "topk_idx": topk_idx,
                "topk_weights": topk_weights,
            }
            (
                recv_x,
                recv_topk_idx,
                recv_topk_weights,
                recv_num_tokens_per_expert_list,
                handle,
                event,
            ) = self.ep_engine.deepep_engine.dispatch(**dispatch_args)
            (
                rank_prefix_matrix,
                channel_prefix_matrix,
                recv_channel_prefix_matrix,
                recv_src_idx,
                is_token_in_rank,
                send_head,
            ) = handle
            # prmt per rank
            token_all_num = sum(recv_num_tokens_per_expert_list)
            if token_all_num > 0:
                # token个数为0时不能走自定义算子
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
                    (
                        self.moe_ffn1_in_scale
                        if hasattr(self, "moe_ffn1_in_scale")
                        else None
                    ),
                    recv_num_tokens_per_expert_list,
                    token_all_num,
                    self.moe_quant_type,
                )
                # moe ffn per rank
                ffn_out = fastdeploy.model_executor.ops.gpu.moe_expert_ffn(
                    permute_input,
                    recv_num_tokens_per_expert_list_cumsum,
                    self.moe_ffn1_weight,
                    self.moe_ffn2_weight,
                    self.moe_ffn1_bias,
                    (
                        self.moe_ffn1_weight_scale
                        if hasattr(self, "moe_ffn1_weight_scale")
                        else None
                    ),
                    (
                        self.moe_ffn2_weight_scale
                        if hasattr(self, "moe_ffn2_weight_scale")
                        else None
                    ),
                    (
                        self.moe_ffn2_in_scale
                        if hasattr(self, "moe_ffn2_in_scale")
                        else None
                    ),
                    expert_idx_per_token,
                    self.moe_quant_type,
                    False,  # used_in_ep_low_latency
                )
                # prmt back per rank
                tmp_ffn_out = fastdeploy.model_executor.ops.gpu.ep_moe_expert_combine(
                    ffn_out,
                    dst_weights,
                    permute_indices_per_token,
                    dst_indices,
                    self.moe_ffn2_bias,
                    False,  # norm_topk_prob
                    1.0,
                )[0]
            else:
                tmp_ffn_out = recv_x
            # intranode combine
            combine_args = {
                "x": tmp_ffn_out,
                "handle": handle,
                "config": self.ep_engine.ep_config,
                "async_finish": self.ep_engine.async_finish,
                "topk_weights": recv_topk_weights,
            }
            fused_moe_out, combined_topk_weights, event = (
                self.ep_engine.deepep_engine.combine(**combine_args)
            )

        return fused_moe_out


class DecoderMoeEPLayer(MoeEPLayer):
    """
    DecoderMoeEPLayer
    """

    def __init__(self, *args, **kwargs):
        """
        DecoderMoeEPLayer Init
        """
        super().__init__(*args, **kwargs)

    def gate(self, x):
        """
        Calculate gate
        """
        gate_out = paddle.matmul(x.cast("float32"), self.gate_weight)

        if os.getenv("EP_DECODER_PERF_TEST", "False") == "True":
            gate_out = paddle.rand(shape=gate_out.shape, dtype=gate_out.dtype)

        topk_idx, topk_weights = fastdeploy.model_executor.ops.gpu.moe_topk_select(
            gate_out,
            (
                self.gate_correction_bias
                if self.moe_config.moe_use_gate_correction_bias
                else None
            ),
            self.top_k,
            True,  # apply_norm_weight
            False,
        )
        return topk_idx, topk_weights

    def ffn(self, permute_input, token_nums_per_expert, expert_idx_per_token=None):
        """
        Calculate moe
        """
        if self.moe_quant_type == "fp8":
            assert isinstance(permute_input, tuple)

            ffn1_out = paddle.empty(
                [
                    self.num_local_experts,
                    self.ep_engine.num_ranks
                    * self.ep_engine.num_max_dispatch_tokens_per_rank,
                    self.moe_intermediate_size * 2,
                ],
                dtype=self._dtype,
            )

            ffn_out = paddle.empty(
                [
                    self.num_local_experts,
                    self.ep_engine.num_ranks
                    * self.ep_engine.num_max_dispatch_tokens_per_rank,
                    self.ep_engine.hidden,
                ],
                dtype=self._dtype,
            )

            expected_m = 128
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                permute_input,
                (
                    self.moe_ffn1_weight,
                    self.moe_ffn1_weight_scale,
                ),
                ffn1_out,
                token_nums_per_expert,
                expected_m,
            )

            act_out = fastdeploy.model_executor.ops.gpu.group_swiglu_with_masked(
                ffn1_out, token_nums_per_expert
            )

            act_out_fp8, scale = fastdeploy.model_executor.ops.gpu.masked_per_token_quant(
                act_out, token_nums_per_expert, 128
            )

            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                (act_out_fp8, scale),
                (
                    self.moe_ffn2_weight,
                    self.moe_ffn2_weight_scale,
                ),
                ffn_out,
                token_nums_per_expert,
                expected_m,
            )
        else:
            ffn_out = fastdeploy.model_executor.ops.gpu.moe_expert_ffn(
                permute_input,
                token_nums_per_expert.cast("int64"),
                self.moe_ffn1_weight,
                self.moe_ffn2_weight,
                self.moe_ffn1_bias,
                (
                    self.moe_ffn1_weight_scale
                    if hasattr(self, "moe_ffn1_weight_scale")
                    else None
                ),
                (
                    self.moe_ffn2_weight_scale
                    if hasattr(self, "moe_ffn2_weight_scale")
                    else None
                ),
                (
                    self.moe_ffn2_in_scale
                    if hasattr(self, "moe_ffn2_in_scale")
                    else None
                ),
                expert_idx_per_token,
                self.moe_quant_type,
                True,  # used_in_ep_low_latency
            )
        return ffn_out

    def forward(self, x, **kwargs):
        """
        DecoderMoeEPLayer Forward (Not micro-batch)
        """
        topk_idx, topk_weights = self.gate(x)

        recv_hidden_states, recv_expert_count, handle, dispatch_hook = (
            self.ep_engine.low_latency_dispatch(
                x, topk_idx, self.moe_quant_type == "fp8"
            )
        )
        if dispatch_hook is not None:
            dispatch_hook()

        # TODO: dispatch 获取expert_idx_per_token
        # expert_idx_per_token = []
        # for expert_id in range(len(token_nums_per_expert)):
        #     for i in range(token_nums_per_expert[expert_id]):
        #         expert_idx_per_token.append(expert_id)
        # expert_idx_per_token = paddle.to_tensor(expert_idx_per_token, "int64")
        ffn_out = self.ffn(recv_hidden_states, recv_expert_count)

        combined_hidden_states, combine_hook = self.ep_engine.low_latency_combine(
            ffn_out, topk_idx, topk_weights, handle
        )
        if combine_hook is not None:
            combine_hook()

        return combined_hidden_states


class DecoderEPMicroBatchRunner:
    """
    DecoderEPMicroBatchRunner
    """

    def __init__(self, moe_layers: list, ep_engine: DeepEPEngine):
        """ """
        self.moe_layers = moe_layers
        self.ep_engine = ep_engine

        self.recv_hidden_states = None
        self.recv_expert_count = None
        self.combined_hidden_states = None
        self.handle = None

        self.dispatch_hook = None
        self.combine_hook = None
        self.topk_idx = None
        self.topk_weights = None
        self.ffn_out = None

    def dispatch_issue(self, x, topk_idx, topk_weights, layer_idx):
        """
        issue dispatch
        """
        self.topk_idx = topk_idx
        self.topk_weights = topk_weights
        (
            self.recv_hidden_states,
            self.recv_expert_count,
            self.handle,
            self.dispatch_hook,
        ) = self.ep_engine.low_latency_dispatch(
            x, self.topk_idx, self.moe_layers[layer_idx].moe_quant_type == "fp8"
        )

    def dispatch_hook_wrap(self):
        """ """
        self.dispatch_hook()
        self.dispatch_hook = None

    def ffn(self, layer_idx):
        """ """
        self.ffn_out = self.moe_layers[layer_idx].ffn(
            self.recv_hidden_states, self.recv_expert_count
        )

        self.recv_hidden_states = None
        self.recv_expert_count = None

    def combine_issue(self):
        """ """
        self.combined_hidden_states, self.combine_hook = (
            self.ep_engine.low_latency_combine(
                self.ffn_out, self.topk_idx, self.topk_weights, self.handle
            )
        )

    def combine_hook_wrap(self):
        """ """
        self.combine_hook()

        self.combine_hook = None
        self.ffn_out = None
        self.topk_idx = None
        self.topk_weights = None
        self.handle = None

        combine_out = self.combined_hidden_states
        self.combined_hidden_states = None

        return combine_out

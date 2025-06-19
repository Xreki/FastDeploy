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

from typing import List

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.ops import (
    apply_penalty_multi_scores, apply_speculative_penalty_multi_scores,
    top_p_sampling)
from fastdeploy.model_executor.ops.gpu import (speculate_verify,
                                               top_p_candidates)
from fastdeploy.platforms import current_platform


class Sampler(nn.Layer):
    """
    """

    def __init__(self):
        """
        """
        super().__init__()
        if current_platform.is_cuda() or current_platform.is_xpu():
            self.forward = self.forward_cuda
        else:
            raise NotImplementedError()

    def forward_cuda(
        self,
        logits: paddle.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> paddle.Tensor:
        """
        """

        logits = apply_penalty_multi_scores(
            sampling_metadata.prompt_token_ids,
            logits,
            sampling_metadata.repetition_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.presence_penalties,
            sampling_metadata.temperature,
            sampling_metadata.bad_words_token_ids,
            sampling_metadata.step_idx,
            sampling_metadata.min_dec_lens,
            sampling_metadata.eos_token_ids,
        )

        probs = F.softmax(logits)

        _, next_tokens = top_p_sampling(probs, sampling_metadata.top_p)

        return next_tokens


class SpeculativeSampler(nn.Layer):
    """
    """

    def __init__(self, cfg: FDConfig):
        """
        """
        super().__init__()
        if current_platform.is_cuda():
            self.forward = self.forward_cuda
        else:
            raise NotImplementedError()
        self.cfg = cfg
        self.speculative_verify_window = cfg.speculative_config.verify_window
        self.speculative_max_candidate_len = cfg.speculative_config.max_candidate_len

    def forward_cuda(
        self,
        logits: paddle.Tensor,
        sampling_metadata: SamplingMetadata,
        max_len: int,
        model_kwargs: List[paddle.Tensor],
    ) -> paddle.Tensor:
        """
        """

        logits = apply_speculative_penalty_multi_scores(
            sampling_metadata.prompt_token_ids,
            logits,
            sampling_metadata.repetition_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.presence_penalties,
            sampling_metadata.temperature,
            sampling_metadata.bad_words_token_ids,
            sampling_metadata.step_idx,
            sampling_metadata.min_dec_lens,
            sampling_metadata.eos_token_ids,
            model_kwargs["seq_lens_this_time"],
            model_kwargs["seq_lens_encoder"],
            model_kwargs["seq_lens_decoder"],
            max_len,
        )

        probs = F.softmax(logits)

        verify_scores, verify_tokens, actual_candidate_len = top_p_candidates(
            probs,
            sampling_metadata.top_p,
            model_kwargs["output_padding_offset"],
            self.speculative_max_candidate_len,
            max_len,
        )

        speculate_verify(
            model_kwargs["accept_tokens"],
            model_kwargs["accept_num"],
            model_kwargs["step_idx"],
            model_kwargs["stop_flags"],
            model_kwargs["seq_lens_encoder"],
            model_kwargs["seq_lens_decoder"],
            model_kwargs[
                "draft_tokens"],  # Both input and output, need to write the last 1 token accepted to position 0.
            model_kwargs["seq_lens_this_time"],
            verify_tokens,
            verify_scores,
            model_kwargs["max_dec_len"],
            sampling_metadata.eos_token_ids,
            model_kwargs["is_block_step"],
            model_kwargs["output_cum_offsets"],
            actual_candidate_len,
            model_kwargs["actual_draft_token_num"],
            sampling_metadata.top_p,
            max_len,
            self.speculative_verify_window,
            True,  # enable_topp
        )

        return None

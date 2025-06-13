"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import random
import time
from typing import List, Optional

import numpy as np
import paddle
import paddle.nn as nn

from fastdeploy.config import FDConfig, KVCacheConfig
from fastdeploy.engine.request import Request
from fastdeploy.model_executor.layers.attention import get_attention_backend
from fastdeploy.model_executor.layers.attention.base_attention_backend import \
    AttentionBackend
from fastdeploy.model_executor.layers.rotary_embedding import get_rope
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.sampler import Sampler
from fastdeploy.model_executor.model_loader import get_model_from_loader
from fastdeploy.model_executor.ops.gpu import rebuild_padding
from fastdeploy.model_executor.pre_and_post_process import (post_process,
                                                            pre_process,
                                                            step_cuda)
from fastdeploy.utils import get_logger
from fastdeploy.worker.model_runner.forward_meta import ForwardMeta
from fastdeploy.worker.output import ModelOutputData, ModelRunnerOutput
from fastdeploy.worker.V1.model_runner_base import ModelRunnerBase

logger = get_logger("gpu_model_runner", "gpu_model_runner.log")


class GPUModelRunner(ModelRunnerBase):
    """ """

    def __init__(self, fd_config: FDConfig, device: str, rank: int,
                 local_rank: int):
        super().__init__(fd_config=fd_config, device=device)
        self.rank = rank
        self.local_rank = local_rank

        #  Sampler
        self.sampler = Sampler()

        # Lazy initialize kv cache after model loading
        # self.kv_caches: list[paddle.Tensor] = []

        # Cuda Graph
        self.use_cuda_grpah = False
        self.input_ids = paddle.zeros(self.parallel_config.max_num_seqs,
                                      dtype='int32')

        # Initialize share inputs
        self._init_share_inputs(self.fd_config.parallel_config.max_num_seqs)
        self.infer_seed_increment = paddle.full(
            shape=[self.parallel_config.max_num_seqs, 1],
            fill_value=4,
            dtype="int64")

        # Initialize attention Backend
        # Note(gonshaotian): Currently, all attention layers share one attention backend instance.
        # In the future, we will expand it as a list.
        self.attn_backends: list[AttentionBackend] = []
        self.forward_meta: ForwardMeta = None
        # self.attn_metadatas: list[AttentionMetadata] = []
        self.initialize_attn_backend()

        # Forward meta store the global meta information of the forward
        self.forward_meta: ForwardMeta = None

    def process_prefill_inputs(self, req_dicts: List[Request]):
        """ Process inputs for prefill tasks and update share_inputs buffer """
        req_len = len(req_dicts)
        for i in range(req_len):
            request = req_dicts[i]
            idx = request.idx
            length = request.prompt_token_ids_len
            self.share_inputs["input_ids"][idx:idx + 1, :length] = np.array(
                request.prompt_token_ids)
            if len(request.eos_token_ids
                   ) < self.parallel_config.eos_tokens_lens:
                request.eos_token_ids.append(request.eos_token_ids[0])
            self.share_inputs["eos_token_id"][:] = np.array(
                request.eos_token_ids, dtype="int64").reshape(-1, 1)
            self.share_inputs["pre_ids"][idx:idx + 1] = -1
            self.share_inputs["top_p"][idx:idx + 1] = request.get("top_p", 0.7)
            self.share_inputs["temperature"][idx:idx + 1] = request.get(
                "temperature", 0.95)
            self.share_inputs["penalty_score"][idx:idx + 1] = request.get(
                "repetition_penalty", 1.0)
            self.share_inputs["frequency_score"][idx:idx + 1] = request.get(
                "frequency_penalty", 0.0)
            self.share_inputs["presence_score"][idx:idx + 1] = request.get(
                "presence_penalty", 0.0)
            self.share_inputs["seq_lens_this_time"][idx:idx + 1] = length
            self.share_inputs["step_seq_lens_encoder"][idx:idx + 1] = length
            self.share_inputs["seq_lens_encoder"][idx:idx + 1] = length
            self.share_inputs["seq_lens_decoder"][idx:idx + 1] = 0
            self.share_inputs["step_idx"][idx:idx + 1] = 0
            self.share_inputs["min_dec_len"][idx:idx + 1] = request.get(
                "min_tokens", 1)

            self.share_inputs["max_dec_len"][idx:idx + 1] = request.get(
                "max_tokens", self.model_config.max_length)
            self.share_inputs["stop_flags"][idx:idx + 1] = False

            self.share_inputs["first_token_ids"][
                idx:idx + 1] = self.share_inputs["input_ids"][idx:idx + 1, :1]
            self.share_inputs["ori_seq_lens_encoder"][idx:idx + 1] = length

            if request.get("seed") is not None:
                self.share_inputs["infer_seed"][idx:idx +
                                                1] = request.get("seed")
            encoder_block_num = len(request.get("block_tables"))
            self.share_inputs["encoder_block_lens"][idx:idx +
                                                    1] = encoder_block_num
            self.share_inputs["block_tables"][idx:idx + 1, :] = -1
            self.share_inputs["block_tables"][
                idx:idx + 1, :encoder_block_num] = np.array(
                    request.block_tables, dtype="int32")

            if request.get("stop_token_ids") is not None and request.get(
                    "stop_seqs_len") is not None:
                stop_seqs_num = len(request.get("stop_seqs_len"))
                for i in range(stop_seqs_num,
                               self.model_config.max_stop_seqs_num):
                    request.stop_seqs_len.append(0)
                self.share_inputs["stop_seqs_len"][:] = np.array(
                    request.stop_seqs_len, dtype="int32")
                self.share_inputs["stop_seqs"][:stop_seqs_num, :len(
                    request.get("stop_token_ids")[0])] = np.array(
                        request.get("stop_token_ids"), dtype="int64")

        self.share_inputs["not_need_stop"][0] = True

    def _dummy_prefill_inputs(self, num_tokens: int, batch_size: int):
        """ Set dummy prefill inputs to share_inputs """
        full_length = num_tokens // batch_size
        input_length = int(full_length * self.parallel_config.kv_cache_ratio)
        block_num = (input_length + self.parallel_config.block_size - 1 +
                     self.parallel_config.enc_dec_block_num
                     ) // self.parallel_config.block_size

        for i in range(batch_size):
            idx = i
            self.share_inputs["input_ids"][idx:idx +
                                           1, :input_length] = np.array(
                                               [5] * input_length)
            self.share_inputs["eos_token_id"][:] = np.array(
                [2], dtype="int64").reshape(-1, 1)
            self.share_inputs["seq_lens_this_time"][idx:idx + 1] = input_length
            self.share_inputs["step_seq_lens_encoder"][idx:idx +
                                                       1] = input_length
            self.share_inputs["seq_lens_encoder"][idx:idx + 1] = input_length
            self.share_inputs["seq_lens_decoder"][idx:idx + 1] = 0
            self.share_inputs["step_idx"][idx:idx + 1] = 0
            self.share_inputs["max_dec_len"][idx:idx + 1] = 10
            self.share_inputs["stop_flags"][idx:idx + 1] = False

            self.share_inputs["first_token_ids"][
                idx:idx + 1] = self.share_inputs["input_ids"][idx:idx + 1, :1]
            self.share_inputs["ori_seq_lens_encoder"][idx:idx +
                                                      1] = input_length

            self.share_inputs["infer_seed"][idx:idx + 1] = random.randint(
                0, 922337203685477580)
            self.share_inputs["encoder_block_lens"][idx:idx + 1] = block_num
            self.share_inputs["block_tables"][idx : idx + 1, :block_num] = np.arange(idx * block_num, \
                                                                                (idx + 1) * block_num, 1)

    def _init_share_inputs(self, max_num_seqs: int):
        """Initialize all share buffers for model inputs.
        Note: In the future, we may abandon share buffers.
        """
        self.MAX_INFER_SEED = 9223372036854775806
        self.share_inputs = {}

        self.share_inputs["pre_ids"] = paddle.full(
            [max_num_seqs, self.parallel_config.max_model_len],
            -1,
            dtype='int64')
        self.share_inputs["input_ids"] = paddle.full(
            [max_num_seqs, self.parallel_config.max_model_len],
            self.parallel_config.pad_token_id,
            dtype='int64')
        self.share_inputs["eos_token_id"] = paddle.full(
            [self.parallel_config.eos_tokens_lens, 1], 0, dtype='int64')
        self.share_inputs["top_p"] = paddle.full([max_num_seqs, 1],
                                                 self.model_config.top_p,
                                                 dtype='float32')
        self.share_inputs["temperature"] = paddle.full(
            [max_num_seqs, 1], self.model_config.temperature, dtype='float32')
        self.share_inputs["penalty_score"] = paddle.full(
            [max_num_seqs, 1],
            self.model_config.penalty_score,
            dtype='float32')
        self.share_inputs["frequency_score"] = paddle.full(
            [max_num_seqs, 1],
            self.model_config.frequency_score,
            dtype='float32')
        self.share_inputs["presence_score"] = paddle.full(
            [max_num_seqs, 1],
            self.model_config.presence_score,
            dtype='float32')

        self.share_inputs["min_dec_len"] = paddle.full(
            [max_num_seqs, 1], self.model_config.min_length, dtype='int64')
        self.share_inputs["max_dec_len"] = paddle.full(
            [max_num_seqs, 1], self.model_config.max_length, dtype='int64')
        self.share_inputs["min_length"] = paddle.full(
            [max_num_seqs, 1], self.model_config.min_length, dtype='int64')
        self.share_inputs["max_length"] = paddle.full(
            [max_num_seqs, 1], self.model_config.max_length, dtype='int64')
        self.share_inputs["seq_lens_this_time"] = paddle.full(max_num_seqs,
                                                              0,
                                                              dtype='int32')
        self.share_inputs["seq_lens_encoder"] = paddle.full([max_num_seqs, 1],
                                                            0,
                                                            dtype='int32')
        self.share_inputs["seq_lens_decoder"] = paddle.full([max_num_seqs, 1],
                                                            0,
                                                            dtype='int32')
        self.share_inputs["step_seq_lens_encoder"] = paddle.full(
            [max_num_seqs, 1], 0, dtype='int32')
        self.share_inputs["step_idx"] = paddle.full([max_num_seqs, 1],
                                                    0,
                                                    dtype='int64')
        self.share_inputs["not_need_stop"] = paddle.full(
            [1], False,
            dtype='bool').cpu()  # TODO(gongshaotian): move to pinnd memory
        self.share_inputs["stop_flags"] = paddle.full([max_num_seqs, 1],
                                                      True,
                                                      dtype='bool')
        self.share_inputs["stop_nums"] = paddle.full([1],
                                                     max_num_seqs,
                                                     dtype='int64')

        self.share_inputs["bad_tokens"] = paddle.full([1], -1, dtype='int64')
        self.share_inputs["next_tokens"] = paddle.full([max_num_seqs, 1],
                                                       -1,
                                                       dtype='int64')
        self.share_inputs["is_block_step"] = paddle.full([max_num_seqs],
                                                         False,
                                                         dtype='bool')
        self.share_inputs["encoder_block_lens"] = paddle.full([max_num_seqs],
                                                              0,
                                                              dtype='int32')
        self.share_inputs["step_block_list"] = paddle.full([max_num_seqs],
                                                           -1,
                                                           dtype='int32')
        self.share_inputs["step_lens"] = paddle.full([1], 0, dtype='int32')
        self.share_inputs["recover_block_list"] = paddle.full([max_num_seqs],
                                                              -1,
                                                              dtype='int32')
        self.share_inputs["recover_lens"] = paddle.full([1], 0, dtype='int32')
        self.share_inputs["need_block_list"] = paddle.full([max_num_seqs],
                                                           -1,
                                                           dtype='int32')
        self.share_inputs["need_block_len"] = paddle.full([1],
                                                          0,
                                                          dtype='int32')
        self.share_inputs["used_list_len"] = paddle.full([max_num_seqs],
                                                         0,
                                                         dtype='int32')
        self.share_inputs["infer_seed"] = paddle.full([max_num_seqs, 1],
                                                      0,
                                                      dtype='int64')
        self.share_inputs["first_token_ids"] = paddle.full([max_num_seqs, 1],
                                                           -1,
                                                           dtype='int64')
        self.share_inputs["ori_seq_lens_encoder"] = paddle.full(
            [max_num_seqs, 1], 0, dtype='int32')
        self.share_inputs["system_lens"] = paddle.full([max_num_seqs, 1],
                                                       0,
                                                       dtype='int32')
        self.share_inputs["system_ids"] = paddle.full([max_num_seqs, 1],
                                                      -1,
                                                      dtype='int32')

        # Initialize rotary position embedding
        tmp_position_ids = paddle.arange(
            self.parallel_config.max_model_len).reshape((1, -1))
        # TODO(gongshaotian): move to models
        self.share_inputs["rope_emb"] = get_rope(
            rotary_dim=self.model_config.head_dim,
            position_ids=tmp_position_ids,
            base=self.model_config.rope_theta,
            model_config=self.model_config)

        # Set block tables
        pre_max_block_num = (
            self.parallel_config.max_model_len +
            self.parallel_config.block_size - 1
        ) // self.parallel_config.block_size + self.parallel_config.enc_dec_block_num
        self.share_inputs["block_tables"] = paddle.full(
            [max_num_seqs, pre_max_block_num], -1, dtype='int32')

        # Initialize free list
        free_list = list(
            range(
                self.parallel_config.max_block_num - 1,
                int(self.parallel_config.max_block_num *
                    self.parallel_config.kv_cache_ratio) - 1, -1))
        self.free_list_len = len(free_list)
        self.share_inputs["free_list"] = paddle.to_tensor(free_list,
                                                          dtype="int32")
        self.share_inputs["free_list_len"] = paddle.full([1],
                                                         self.free_list_len,
                                                         dtype="int32")

        # Initialize stop seqs
        self.share_inputs["stop_seqs_len"] = paddle.full(
            [self.model_config.max_stop_seqs_num], 0, dtype="int32")
        self.share_inputs["stop_seqs"] = paddle.full([
            self.model_config.max_stop_seqs_num,
            self.model_config.stop_seqs_max_len
        ],
                                                     -1,
                                                     dtype="int32")

    def _prepare_inputs(self) -> None:
        """ prepare the model inputs """
        # Remove padding
        (
            ids_remove_padding,
            cum_offsets,
            padding_offset,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = pre_process(self.parallel_config.max_model_len,
                        self.share_inputs["input_ids"],
                        self.share_inputs["seq_lens_this_time"],
                        use_speculate_method=False)

        # Initialize forward meta data
        self.share_inputs["ids_remove_padding"] = ids_remove_padding
        self.share_inputs["cum_offsets"] = cum_offsets
        self.share_inputs["padding_offset"] = padding_offset
        self.share_inputs["cu_seqlens_q"] = cu_seqlens_q
        self.share_inputs["cu_seqlens_k"] = cu_seqlens_k
        self.initialize_forward_meta()

        # Get sampling metadata
        self.sampling_metadata = SamplingMetadata(
            temperature=self.share_inputs["temperature"],
            top_p=self.share_inputs["top_p"],
            step_idx=self.share_inputs["step_idx"],
            prompt_token_ids=self.share_inputs["input_ids"],
            frequency_penalties=self.share_inputs["frequency_score"],
            presence_penalties=self.share_inputs["presence_score"],
            repetition_penalties=self.share_inputs["penalty_score"],
            min_dec_lens=self.share_inputs["min_dec_len"],
            bad_words_token_ids=self.share_inputs["bad_tokens"],
            eos_token_ids=self.share_inputs["eos_token_id"],
        )

    def load_model(self) -> None:
        """ load or download model """
        logger.info(
            f"Starting to load model {self.model_config.architectures[0]}")
        time_before_load = time.perf_counter()
        # 1. Load original model
        self.model = get_model_from_loader(fd_config=self.fd_config)

        # 2. Load lora model

        # 3. Load drafter model(for speculative decoding)

        time_after_load = time.perf_counter()
        logger.info(
            f"Model loading took {time_after_load - time_before_load} seconds")

    def get_model(self) -> nn.Layer:
        """ get current model """
        return self.model

    def initialize_forward_meta(self):
        """
        Initialize forward meta and attention meta data
        """
        # Initialize forward meta
        self.forward_meta = ForwardMeta.init_forward_meta(
            self.share_inputs, self.attn_backends[0])

        # Initialzie attention meta data
        for attn_backend in self.attn_backends:
            attn_backend.init_attention_metadata(self.forward_meta)

    def initialize_kv_cache(self,
                            kv_cache_config: KVCacheConfig = None) -> None:
        """
        Initialize kv cache
        Args:
            kv_cache_config:
        """
        cache_kvs = {}
        max_block_num = self.num_gpu_blocks

        kv_cache_shape = self.attn_backends[0].get_kv_cache_shape(
            max_num_blocks=max_block_num)

        for i in range(self.model_config.num_layers):
            cache_type = self.parallel_config.dtype

            if self.fd_config.kv_cache_config.cache_quant_dtype == "cache_int8":
                cache_type = 'uint8'

            cache_kvs["key_caches_{}".format(i)] = paddle.full(
                shape=kv_cache_shape,
                fill_value=0,
                dtype=cache_type,
            )
            cache_kvs["value_caches_{}".format(i)] = paddle.full(
                shape=kv_cache_shape,
                fill_value=0,
                dtype=cache_type,
            )
        self.share_inputs["caches"] = list(cache_kvs.values())
        for value in cache_kvs.values():
            del value
        paddle.device.cuda.empty_cache()

    def initialize_attn_backend(self,
                                kv_cache_config: Optional[KVCacheConfig] = None
                                ) -> None:
        """
        Initialize attention backends and forward metadata
        Args:
            kv_cache_config:
        """
        assert len(self.attn_backends) == 0

        # TODO(gongshaotian): Get rank from config
        num_heads = self.model_config.num_attention_heads // self.parallel_config.mp_size
        self.model_config.kv_num_heads = int(
            self.model_config.num_key_value_heads
        ) // self.parallel_config.mp_size
        head_dim = self.model_config.head_dim

        # Get the attention backend
        attn_cls = get_attention_backend(
            self.parallel_config.attention_backend)
        attn_backend = attn_cls(self.fd_config,
                                kv_num_heads=self.model_config.kv_num_heads,
                                num_heads=num_heads,
                                head_dim=head_dim)
        if attn_backend is None:
            raise NotImplementedError(
                f"{ self.parallel_config.attention_backend} attention backend is not support by GPUModelRunner"
            )
        self.attn_backends.append(attn_backend)

    def _dummy_run(self, num_tokens, batch_size) -> paddle.Tensor:
        """
        Use dummy inputs to run before formal execution.
        Args:
            num_tokens: Expected number of tokens generated
        """
        self._dummy_prefill_inputs(num_tokens, batch_size)

        while True:

            # 1. Compute real num_tokens

            self._prepare_inputs()

            # 2. Initialize attention backend and forward meta data

            # 3. Prepare lora

            # 4. Run model
            model_output = self.model(
                ids_remove_padding=self.share_inputs["ids_remove_padding"],
                forward_meta=self.forward_meta)
            hiddden_states = rebuild_padding(
                model_output,
                self.share_inputs["cum_offsets"],
                self.share_inputs["seq_lens_this_time"],
                self.share_inputs["seq_lens_decoder"],
                self.share_inputs["seq_lens_encoder"],
                None,  # speculative decoding requires
                self.parallel_config.max_model_len,
            )

            # 5. Execute spec decode
            logits = self.model.compute_logits(hiddden_states)
            sampled_token_ids = self.sampler(logits, self.sampling_metadata)
            # self._dummy_sampler_run()

            # 6. post process
            model_output_data = ModelOutputData(
                next_tokens=self.share_inputs["next_tokens"],
                stop_flags=self.share_inputs["stop_flags"],
                step_idx=self.share_inputs["step_idx"],
                max_dec_len=self.share_inputs["max_dec_len"],
                pre_ids=self.share_inputs["pre_ids"],
                seq_lens_this_time=self.share_inputs["seq_lens_this_time"],
                eos_token_id=self.share_inputs["eos_token_id"],
                not_need_stop=self.share_inputs["not_need_stop"],
                input_ids=self.share_inputs["input_ids"],
                stop_nums=self.share_inputs["stop_nums"],
                seq_lens_encoder=self.share_inputs["seq_lens_encoder"],
                seq_lens_decoder=self.share_inputs["seq_lens_decoder"],
                is_block_step=self.share_inputs["is_block_step"],
                output_via_mq=self.model_config.output_via_mq,
                msg_queue_id=self.parallel_config.msg_queue_id,
                mp_rank=self.local_rank,
                use_ep=self.parallel_config.use_ep)

            post_process(
                sampled_token_ids=sampled_token_ids,
                model_output=model_output_data,
            )

            # 7. Updata 'infer_seed' and step_cuda()
            self.share_inputs["infer_seed"].add_(self.infer_seed_increment)
            self.share_inputs["infer_seed"][:] %= self.MAX_INFER_SEED
            step_cuda(self.share_inputs, self.parallel_config.block_size,
                      self.parallel_config.enc_dec_block_num)

            if int((self.share_inputs['seq_lens_this_time'] > 0).sum()) == 0:
                break

    def _dummy_sampler_run(self) -> paddle.Tensor:
        """ """
        pass

    def capture_model(self) -> None:
        """
        Trigger CUDA Graph capture for all shapes in 'CudaGraphConfig.cudagraph_capture_sizes'
        """
        pass

    def execute_model(
        self,
        model_forward_batch: Optional[List[Request]] = None,
    ) -> Optional[ModelRunnerOutput]:
        """
        The Entrance of model execute.
        Args:
            model_forward_batch: 'Request' contains information related to prompt and is an abstract
            class at the server level, which is too granular for ModelRunner.
            We plan to replace it with 'ModelForwardBatch'.
            intermediate_tensors:
        """
        # 1. Prepare inputs of model and decoder.
        self._prepare_inputs()

        # 2. Padding inputs for cuda grph

        # 3. Execute model
        model_output = self.model(self.share_inputs["ids_remove_padding"],
                                  self.forward_meta)
        hiddden_states = rebuild_padding(
            model_output,
            self.share_inputs["cum_offsets"],
            self.share_inputs["seq_lens_this_time"],
            self.share_inputs["seq_lens_decoder"],
            self.share_inputs["seq_lens_encoder"],
            None,  #self.share_inputs["padding_offset"],
            self.parallel_config.max_model_len,
        )

        # 4. Compute logits, Sample
        logits = self.model.compute_logits(hiddden_states)

        sampled_token_ids = self.sampler(logits, self.sampling_metadata)

        # 5. Speculative decode

        # 6. Post Process
        model_output_data = ModelOutputData(
            next_tokens=self.share_inputs["next_tokens"],
            stop_flags=self.share_inputs["stop_flags"],
            step_idx=self.share_inputs["step_idx"],
            max_dec_len=self.share_inputs["max_dec_len"],
            pre_ids=self.share_inputs["pre_ids"],
            seq_lens_this_time=self.share_inputs["seq_lens_this_time"],
            eos_token_id=self.share_inputs["eos_token_id"],
            not_need_stop=self.share_inputs["not_need_stop"],
            input_ids=self.share_inputs["input_ids"],
            stop_nums=self.share_inputs["stop_nums"],
            seq_lens_encoder=self.share_inputs["seq_lens_encoder"],
            seq_lens_decoder=self.share_inputs["seq_lens_decoder"],
            is_block_step=self.share_inputs["is_block_step"],
            output_via_mq=self.model_config.output_via_mq,
            msg_queue_id=self.parallel_config.msg_queue_id,
            mp_rank=self.local_rank,
            use_ep=self.parallel_config.use_ep)
        post_process(sampled_token_ids=sampled_token_ids,
                     model_output=model_output_data)

        # 7. Updata 'infer_seed' and step_cuda()
        self.share_inputs["infer_seed"].add_(self.infer_seed_increment)
        self.share_inputs["infer_seed"][:] %= self.MAX_INFER_SEED
        step_cuda(self.share_inputs, self.parallel_config.block_size,
                  self.parallel_config.enc_dec_block_num)

        return None

    def profile_run(self) -> None:
        """Execute a forward pass with dummy inputs to profile the memory usage of the model."""

        # Initialize kv cache for profile run. After profile run kv cache will be reset.
        # TODO(gongshaotian): Optimize the management logic of kvcache
        self.num_gpu_blocks = self.parallel_config.max_block_num
        self.initialize_kv_cache()

        # 1. Profile with multimodal encoder & encoder cache

        # 2. Dummy run
        self._dummy_run(num_tokens=self.parallel_config.max_model_len,
                        batch_size=self.parallel_config.max_num_seqs)

        # 3. gc
        del self.share_inputs["caches"]
        if self.forward_meta is not None:
            del self.forward_meta.caches
        del self.share_inputs["block_tables"]
        # paddle.device.cuda.synchronize()
        paddle.device.cuda.empty_cache()
        # gc.collect()

    def update_share_input_block_num(self, num_gpu_blocks: int) -> None:
        """
        Set a globally unified block number and update the model's shared input.
        Args:
            num_gpu_blocks:
        """
        self.num_gpu_blocks = num_gpu_blocks

        # Reset block table and kv cache with global block num
        self.initialize_kv_cache()

        self.share_inputs["block_tables"] = paddle.full(
            [self.parallel_config.max_num_seqs, self.num_gpu_blocks],
            -1,
            dtype="int32")

        # Reset free list
        free_list = list(
            range(
                self.num_gpu_blocks - 1,
                int(self.num_gpu_blocks * self.parallel_config.kv_cache_ratio)
                - 1, -1))
        self.free_list_len = len(free_list)
        self.share_inputs.update({
            "free_list":
            paddle.to_tensor(free_list, dtype="int32"),
            "free_list_len":
            paddle.full([1], self.free_list_len, dtype="int32"),
        })

    def cal_theortical_kvcache(self):
        """
        Calculate the total block memory required at the model level
        TODO(gongshaotian): Move to Attention Backend
        """
        """
        Byte of dtype:
        - default(bf16): 2
        - cache_int8: 1
        - cache_int4:
        """
        cache_quant_dtype = self.kv_cache_config.cache_quant_dtype
        print(
            f"parallel_config.dtype: {self.kv_cache_config.cache_quant_dtype}")
        print(f"cache_quant_dtype: {cache_quant_dtype}")

        if cache_quant_dtype == "cache_int8":
            byte_of_dtype = 1
        elif self.parallel_config.dtype == "wint4":
            byte_of_dtype = 0.5
        else:  # default
            byte_of_dtype = 2

        hidden_dim = self.model_config.head_dim * self.model_config.kv_num_heads
        required_memory = (
            byte_of_dtype * 2 *  # k + v
            (self.parallel_config.block_size * hidden_dim) *
            self.model_config.num_layers)
        return required_memory

    def not_need_stop(self) -> bool:
        """ """
        return self.share_inputs["not_need_stop"][0]

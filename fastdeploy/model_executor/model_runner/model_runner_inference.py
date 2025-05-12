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
import math
import os
import numpy as np
import random
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet

from fastdeploy.model_executor.model_runner.model_runner_base import ModelRunnerBase

class ModelRunner(ModelRunnerBase):
    def __init__(self, config, args, nranks, rank):
        self.nranks = nranks
        self.rank = rank
        super().__init__(config,args)
        self._reset_paddle_env()



    def _reset_paddle_env(self):
        #FLAGS_gqa_use_tensorcore
        #FLAGS_ffn2_use_hardamard
        # gqa .etc paddle Flags set
        pass

    def _load_model(self, model_name):
        from efficientllm.models.export_model import build_stream_line_model
        from efficientllm.models.tokenizer import ErnieBotTokenizer
        vocab_file_names = ["tokenizer.model", "spm.model", "ernie_token_100k.model"]
        for i in range(len(vocab_file_names)):
            if os.path.exists(os.path.join(self.args.model_name_or_path, vocab_file_names[i])):
                ErnieBotTokenizer.resource_files_names["vocab_file"] = vocab_file_names[i]
                break
        config, tokenizer, model = build_stream_line_model(
            os.path.join(self.args.model_name_or_path, os.getenv("CONFIG_JSON_FILE", "config.json")),
            self.args.model_name_or_path,
            self.args.dtype,
            block_size=self.args.block_size,
            max_len=self.args.max_model_len,
            stage_flag="msgid-1 predict",
            export_model_type="default",
            use_fake_parameter=False,
            use_stop_seqs=self.model_cfg.ellm_dynamic_use_stop_seqs,
            use_beam_search=False,
            speculate_method=None,
            speculate_max_draft_token_num=5,
            return_all_hidden_states=False,
        )
        model.eval()
        self.model = model

    def init_rotary_position_embedding(self, max_model_len):
        tmp_position_ids = paddle.arange(max_model_len).reshape((1, -1))
        self.share_inputs["rope_emb"] = self.get_rotary_position_embedding(
            tmp_position_ids,
            self.model_cfg.hidden_size // self.model_cfg.num_attention_heads,
            model_type=self.model_cfg.model_type,
            rope_scaling=self.rope_scaling,
        )

    def _init_kvcache(self, max_block_num):
        """
        分享不拷贝数据
        """

        self.cache_kvs = {}

        if (
            hasattr(self.model_cfg, "num_key_value_heads")
            and hasattr(self.model_cfg, "num_key_value_heads")
            and self.model_cfg.num_key_value_heads is not None
            and int(self.model_cfg.num_key_value_heads) > 0
        ):
            kv_num_head = int(self.model_cfg.num_key_value_heads) // self.nranks
        else:
            kv_num_head = self.model_cfg.num_attention_heads // self.nranks

        self.model_cfg.kv_num_head = kv_num_head

        for i in range(self.model_cfg.num_layers):
            cache_type = self.args.dtype
            self.cache_kvs["key_caches_{}".format(i)] = paddle.full(
                shape=[
                    max_block_num,
                    kv_num_head,
                    self.args.block_size,
                    self.model_cfg.hidden_size // self.model_cfg.num_attention_heads,
                ],
                fill_value=0,
                dtype=cache_type,
            )
            self.cache_kvs["value_caches_{}".format(i)] = paddle.full(
                shape=[
                    max_block_num,
                    kv_num_head,
                    self.args.block_size,
                    self.model_cfg.hidden_size // self.model_cfg.num_attention_heads,
                ],
                fill_value=0,
                dtype=cache_type,
            )

        self.share_inputs["caches"] = list(self.cache_kvs.values())
        for value in self.cache_kvs.values():
            del value

    def dy_input_preprocess(self, tasks):
        """
        dynamic insertion
        """

        for i in range(len(tasks)):
            task = tasks[i]
            idx = task.idx
            length = task.prompt_token_ids_len
            self.share_inputs["input_ids"][idx : idx + 1, :length] = np.array(task.prompt_token_ids)
            if len(task.eos_token_ids) < self.args.eos_tokens_lens:
                task.eos_token_ids.append(task.eos_token_ids[0])
            self.share_inputs["eos_token_id"][:] = np.array(task.eos_token_ids, dtype="int64").reshape(-1, 1)
            self.share_inputs["pre_ids"][idx : idx + 1] = -1
            self.share_inputs["top_p"][idx : idx + 1] = task.get("topp", 0.7)
            self.share_inputs["temperature"][idx : idx + 1] = task.get("temperature", 0.95)
            self.share_inputs["penalty_score"][idx : idx + 1] = task.get("repetition_penalty", 1.0)
            self.share_inputs["frequency_score"][idx : idx + 1] = task.get("frequency_penalty", 0.0)
            self.share_inputs["presence_score"][idx : idx + 1] = task.get("presence_penalty", 0.0)
            self.share_inputs["seq_lens_this_time"][idx : idx + 1] = length
            self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = length
            self.share_inputs["seq_lens_encoder"][idx : idx + 1] = length
            self.share_inputs["seq_lens_decoder"][idx : idx + 1] = 0
            self.share_inputs["step_idx"][idx : idx + 1] = 0
            self.share_inputs["min_dec_len"][idx : idx + 1] = task.get("min_tokens", 1)

            self.share_inputs["max_dec_len"][idx : idx + 1] = task.get("max_tokens", self.max_length)
            self.share_inputs["stop_flags"][idx : idx + 1] = False

            self.share_inputs["first_token_ids"][idx : idx + 1] = self.share_inputs["input_ids"][idx : idx + 1, :1]
            self.share_inputs["ori_seq_lens_encoder"][idx : idx + 1] = length

            if task.get("seed") is not None:
                self.share_inputs["infer_seed"][idx : idx + 1] = task.get("seed")
            encoder_block_num = len(task.get("block_tables"))
            self.share_inputs["encoder_block_lens"][idx : idx + 1] = encoder_block_num
            self.share_inputs["block_tables"][idx : idx + 1, :] = -1
            self.share_inputs["block_tables"][idx : idx + 1, :encoder_block_num] = np.array(
                task.block_tables, dtype="int32"
            )

            # TODO 待确认正确性
            if task.get("stop_token_ids") is not None and task.get("stop_seqs_len") is not None:
                stop_seqs_num = len(task.get("stop_seqs_len"))
                for i in range(stop_seqs_num, self.model_cfg.max_stop_seqs_num):
                    task.stop_seqs_len.append(0)
                self.share_inputs["stop_seqs_len"][:] = np.array(task.stop_seqs_len, dtype="int32")
                self.share_inputs["stop_seqs"][:stop_seqs_num, : len(task.get("stop_token_ids")[0])] = np.array(
                    task.get("stop_token_ids"), dtype="int64"
                )



    def get_rotary_position_embedding(self,position_ids, head_dim, rope_theta=160000,model_type="ernie_bot",rope_scaling=None):
        """
        Pre-calculate rotary position embedding for position_ids.

        Args:
            position_ids: [1, S]
            head_dim: D

        Returns:
            rot_emb: [2, 1, S, 1, D // 2] or [2, 1, S, 1, D], cos + sin
        """
        bsz, max_model_len = position_ids.shape[:2]
        if model_type=="ernie_bot":
            inv_freq = rope_theta ** (-paddle.arange(0, head_dim, 2, dtype="float32") / head_dim)

            # shape: [B, S, D/2]
            # eblite should divide compression_ratio, default 1.0 for eb3.5 or eb4
            compression_ratio = 1.0
            compressed_position_ids = position_ids / compression_ratio
            freqs = paddle.einsum("ij,k->ijk", compressed_position_ids.cast("float32"),
                                inv_freq)

            rot_emb = paddle.zeros((2, bsz, max_model_len, 1, head_dim // 2), dtype="float32")
            emb = paddle.stack([freqs], axis=-1).reshape((bsz, max_model_len, head_dim // 2))
            # shape: [B, S, 1, D]
            emb = paddle.unsqueeze(emb, 2)

            rot_emb[0] = paddle.cos(emb)
            rot_emb[1] = paddle.sin(emb)

            return rot_emb
        elif model_type=="llama":
            rope_theta=10000.0
            rot_emb = paddle.zeros((2, bsz, max_model_len, 1, head_dim), dtype="float32")
            inv_freq = rope_theta ** (-paddle.arange(0, head_dim, 2, dtype="float32") / head_dim)

            if rope_scaling is not None:
                rope_type = rope_scaling.get("rope_type", None)
                if rope_type is not None and rope_type == "llama3":
                    factor = rope_scaling.get("factor", 8.0)
                    low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
                    high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
                    original_max_position_embeddings = rope_scaling.get("original_max_position_embeddings", 8192)

                    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
                    high_freq_wavelen = original_max_position_embeddings / high_freq_factor
                    new_freqs = []
                    for freq in inv_freq:
                        wavelen = 2 * math.pi / freq
                        if wavelen < high_freq_wavelen:
                            new_freqs.append(freq)
                        elif wavelen > low_freq_wavelen:
                            new_freqs.append(freq / factor)
                        else:
                            assert low_freq_wavelen != high_freq_wavelen
                            smooth = (original_max_position_embeddings / wavelen - low_freq_factor) / (
                                high_freq_factor - low_freq_factor
                            )
                            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
                    inv_freq = paddle.to_tensor(new_freqs, dtype=inv_freq.dtype)

            # shape: [B, S, D/2]
            freqs = paddle.einsum("ij,k->ijk", position_ids.cast("float32"), inv_freq)
            # shape: [B, S, 1, D]
            emb = paddle.concat([freqs, freqs], axis=-1).reshape((bsz, max_model_len, 1, head_dim))

            rot_emb[0] = paddle.cos(emb)
            rot_emb[1] = paddle.sin(emb)
            return rot_emb

    def generate(self):
        self.model(**self.share_inputs)

    def _cal_theortical_kvcache(self):
        """
        计算理论的kvcache大小
        """
        num_layers = self.model_cfg.num_layers
        byte_of_cache = 2
        #TODO
        # 支持c8 c4

        hidden_size = self.model_cfg.hidden_size
        attention_heads = self.model_cfg.num_attention_heads
        hidden_dim = hidden_size / attention_heads * self.model_cfg.kv_num_head
        theoretical_kv_cache_memory = (2 * byte_of_cache * self.args.block_size * num_layers * hidden_dim)
        return theoretical_kv_cache_memory


    def _update_share_input_block_num(self, num_gpu_blocks):
        del self.share_inputs["caches"]
        self._init_kvcache(num_gpu_blocks)

        del self.share_inputs["block_tables"]
        self.share_inputs["block_tables"] = paddle.full(
            [self.args.max_num_seqs, num_gpu_blocks], -1, dtype="int32"
        )

        # 初始化free list
        free_list = list(
            range(num_gpu_blocks - 1, int(num_gpu_blocks * self.args.kv_cache_ratio) - 1, -1)
        )
        self.free_list_len = len(free_list)
        self.share_inputs.update({
            "free_list": paddle.to_tensor(free_list, dtype="int32"),
            "free_list_len": paddle.full([1], self.free_list_len, dtype="int32"),
        })

    def dummy_input(self, num_total_tokens, number_of_tasks):
        """
        fake input to profile
        """
        full_length = num_total_tokens // number_of_tasks
        input_length = int(full_length * self.args.kv_cache_ratio)
        block_num = (input_length + self.args.block_size - 1 + self.args.enc_dec_block_num) // self.args.block_size

        for i in range(number_of_tasks):
            idx = i
            self.share_inputs["input_ids"][idx : idx + 1, :input_length] = np.array([5] * input_length)
            self.share_inputs["eos_token_id"][:] = np.array([2], dtype="int64").reshape(-1, 1)
            self.share_inputs["seq_lens_this_time"][idx : idx + 1] = input_length
            self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = input_length
            self.share_inputs["seq_lens_encoder"][idx : idx + 1] = input_length
            self.share_inputs["seq_lens_decoder"][idx : idx + 1] = 0
            self.share_inputs["step_idx"][idx : idx + 1] = 0
            self.share_inputs["max_dec_len"][idx : idx + 1] = 10
            self.share_inputs["stop_flags"][idx : idx + 1] = False

            self.share_inputs["first_token_ids"][idx : idx + 1] = self.share_inputs["input_ids"][idx : idx + 1, :1]
            self.share_inputs["ori_seq_lens_encoder"][idx : idx + 1] = input_length

            self.share_inputs["infer_seed"][idx : idx + 1] = random.randint(0, 922337203685477580)
            self.share_inputs["encoder_block_lens"][idx : idx + 1] = block_num
            self.share_inputs["block_tables"][idx : idx + 1, :block_num] = np.arange(idx * block_num, \
                                                                                (idx + 1) * block_num, 1)

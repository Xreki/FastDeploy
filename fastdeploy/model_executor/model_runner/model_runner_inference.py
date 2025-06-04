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
import os
import random

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet

from fastdeploy.model_executor.model_runner.model_runner_base import ModelRunnerBase


class ModelRunner(ModelRunnerBase):

    def __init__(self, config, args, nranks, rank):
        self.nranks = nranks
        self.rank = rank
        super().__init__(config, args)
        self._reset_paddle_env()
        self.init_local_params()

    def _reset_paddle_env(self):
        #FLAGS_gqa_use_tensorcore
        #FLAGS_ffn2_use_hardamard
        # gqa .etc paddle Flags set
        pass

    def init_local_params(self):
        if self.args.enable_chunked_prefill:
            self.chunked_prefill_seq_lens = paddle.full(
                shape=[self.args.max_num_seqs], 
                fill_value=0, 
                dtype='int32',
            )
            self.chunked_prefill_cur_seq_lens = paddle.full(
                shape=[self.args.max_num_seqs], 
                fill_value=0, 
                dtype='int32',
            )
            self.chunked_prefill_cur_input_ids = paddle.full(
                shape=[self.args.max_num_seqs, self.args.max_model_len], 
                fill_value=0, 
                dtype='int64',
            )
    
    def update_chunked_prefill(self, token_chunk_size=384):
        """
        更新chunked prefill相关参数
        """
        if not self.args.enable_chunked_prefill:
            return
        
        from fastdeploy.model_executor.ops.gpu import update_split_fuse_inputs
        update_split_fuse_inputs(
            self.chunked_prefill_seq_lens,
            self.chunked_prefill_cur_seq_lens,
            self.chunked_prefill_cur_input_ids,
            self.share_inputs['input_ids'],
            self.share_inputs['seq_lens_this_time'],
            self.share_inputs["seq_lens_encoder"],
            self.share_inputs["seq_lens_decoder"],
            self.share_inputs["step_idx"],
            self.args.max_model_len,
            self.args.max_num_seqs,
            token_chunk_size,
        )

    def _load_model(self, model_name, dynamic_load_weight):
        use_pip_eff_llm = os.getenv('USE_PIP_EFF_LLM')
        
        local_test = False
        if os.getenv("RUN_MODE", "") == "test":
            local_test = True

        if dynamic_load_weight:
            if use_pip_eff_llm:
                from efficientllm.models.efficientllm_model import EfficientModel
                dynamic_load_model = EfficientModel(
                    model_name_or_path=self.args.model_name_or_path,
                    dtype=self.args.dtype,
                    block_size=self.args.block_size,
                    max_len=self.args.max_model_len,
                    gemm_method="weight_only_int8",
                    moe_quant_type="weight_only_int8",
                    load_model_from_ipc=dynamic_load_weight,
                    embeddings_column_cut=False,
                    nranks=self.nranks,
                    rank=self.rank,
                    local_test=local_test,
                )
            else:
                from ..models.dynamic_load_model import DynamicLoadModel
                dynamic_load_model = DynamicLoadModel(
                    model_name_or_path=self.args.model_name_or_path,
                    dtype=self.args.dtype,
                    block_size=self.args.block_size,
                    max_len=self.args.max_model_len,
                    moe_quant_type="weight_only_int8",
                    load_model_from_ipc=dynamic_load_weight,
                    nranks=self.nranks,
                    rank=self.rank,
                    embeddings_column_cut=False,
                    local_test=local_test
                )
            self.model = dynamic_load_model
        else:
            if use_pip_eff_llm is None:
                from ..models.export_model import build_stream_line_model
                from ..models.tokenizer import ErnieBotTokenizer
            else:
                from efficientllm.models.export_model import build_stream_line_model
                from efficientllm.models.tokenizer import ErnieBotTokenizer
            vocab_file_names = [
                "tokenizer.model", "spm.model", "ernie_token_100k.model"
            ]
            for i in range(len(vocab_file_names)):
                if os.path.exists(
                        os.path.join(self.args.model_name_or_path,
                                    vocab_file_names[i])):
                    ErnieBotTokenizer.resource_files_names[
                        "vocab_file"] = vocab_file_names[i]
                    break
            config, tokenizer, model, _ = build_stream_line_model(
                os.path.join(self.args.model_name_or_path,
                            os.getenv("CONFIG_JSON_FILE", "config.json")),
                self.args.model_name_or_path,
                self.args.dtype,
                block_size=self.args.block_size,
                max_len=self.args.max_model_len,
                stage_flag="msgid-1 predict",
                export_model_type="weight_only_int8",
                use_fake_parameter=False,
                use_stop_seqs=self.model_cfg.ellm_dynamic_use_stop_seqs,
                use_beam_search=False,
                speculate_method=None,
                speculate_max_draft_token_num=5,
                return_all_hidden_states=False,
                moe_quant_type="weight_only_int4",
                use_safetensors=self.model_cfg.is_unified_ckpt,
            )
            model.eval()
            self.model = model

    def init_rotary_position_embedding(self, max_model_len):
        tmp_position_ids = paddle.arange(max_model_len).reshape((1, -1))
        self.share_inputs["rope_emb"] = self.get_rotary_position_embedding(
            tmp_position_ids,
            self.model_cfg.hidden_size // self.model_cfg.num_attention_heads,
            self.rope_theta)

    def _init_kvcache(self):
        """
        分享不拷贝数据
        """

        cache_kvs = {}
        total_block_num = self.num_gpu_blocks

        if (hasattr(self.model_cfg, "num_key_value_heads")
                and hasattr(self.model_cfg, "num_key_value_heads")
                and self.model_cfg.num_key_value_heads is not None
                and int(self.model_cfg.num_key_value_heads) > 0):
            kv_num_head = int(
                self.model_cfg.num_key_value_heads) // self.nranks
        else:
            kv_num_head = self.model_cfg.num_attention_heads // self.nranks
        self.model_cfg.kv_num_head = kv_num_head

        for i in range(self.model_cfg.num_layers):
            cache_type = self.args.dtype
            cache_kvs["key_caches_{}".format(i)] = paddle.full(
                shape=[
                    total_block_num,
                    kv_num_head,
                    self.args.block_size,
                    self.model_cfg.hidden_size //
                    self.model_cfg.num_attention_heads,
                ],
                fill_value=0,
                dtype=cache_type,
            )
            cache_kvs["value_caches_{}".format(i)] = paddle.full(
                shape=[
                    total_block_num,
                    kv_num_head,
                    self.args.block_size,
                    self.model_cfg.hidden_size //
                    self.model_cfg.num_attention_heads,
                ],
                fill_value=0,
                dtype=cache_type,
            )

        self.share_inputs["caches"] = list(cache_kvs.values())
        for value in cache_kvs.values():
            del value
        paddle.device.cuda.empty_cache()
    
    def prefill_finished(self):
        """
        判断是否已经完成了prefill操作
        """
        prefill_statue = (self.share_inputs["seq_lens_this_time"] != 0) & (self.share_inputs["seq_lens_this_time"] != 1)
        return not paddle.any(prefill_statue).numpy()

    def dy_input_preprocess(self, tasks):
        """
        dynamic insertion
        """
        for i in range(len(tasks)):
            task = tasks[i]
            idx = task.idx
            length = task.prompt_token_ids_len
            
            if self.args.enable_chunked_prefill:
                if task.token_chunk_size > length:
                    self.share_inputs["seq_lens_this_time"][idx] = length
                    self.share_inputs['input_ids'][idx, :length] = np.array(task.prompt_token_ids)
                    self.share_inputs['step_seq_lens_encoder'][idx] = task.token_chunk_size
                    self.share_inputs['seq_lens_encoder'][idx] = length
                    self.chunked_prefill_seq_lens[idx] = length
                    self.chunked_prefill_cur_seq_lens[idx] = length
                else:
                    self.chunked_prefill_cur_input_ids[idx, :length] = np.array(task.prompt_token_ids)
                    self.chunked_prefill_cur_seq_lens[idx] = task.token_chunk_size
                    self.chunked_prefill_seq_lens[idx] = length
                    self.share_inputs["seq_lens_this_time"][idx] = task.token_chunk_size
                    self.share_inputs['input_ids'][idx, :task.token_chunk_size] = np.array(
                        self.chunked_prefill_cur_input_ids[idx, :task.token_chunk_size]
                    )
                    self.share_inputs['step_seq_lens_encoder'][idx] = task.token_chunk_size
                    self.share_inputs['seq_lens_encoder'][idx] = task.token_chunk_size
            else:
                self.share_inputs["input_ids"][idx:idx + 1, :length] = np.array(
                    task.prompt_token_ids)
                self.share_inputs["seq_lens_this_time"][idx:idx + 1] = length
                self.share_inputs["step_seq_lens_encoder"][idx:idx + 1] = length
                self.share_inputs["seq_lens_encoder"][idx:idx + 1] = length

            if len(task.eos_token_ids) < self.args.eos_tokens_lens:
                task.eos_token_ids.append(task.eos_token_ids[0])
            self.share_inputs["eos_token_id"][:] = np.array(
                task.eos_token_ids, dtype="int64").reshape(-1, 1)
            self.share_inputs["pre_ids"][idx:idx + 1] = -1
            self.share_inputs["top_p"][idx:idx + 1] = task.get("top_p", 0.7)
            self.share_inputs["temperature"][idx:idx + 1] = task.get(
                "temperature", 0.95)
            self.share_inputs["penalty_score"][idx:idx + 1] = task.get(
                "repetition_penalty", 1.0)
            self.share_inputs["frequency_score"][idx:idx + 1] = task.get(
                "frequency_penalty", 0.0)
            self.share_inputs["presence_score"][idx:idx + 1] = task.get(
                "presence_penalty", 0.0)
            self.share_inputs["seq_lens_decoder"][idx:idx + 1] = 0
            self.share_inputs["step_idx"][idx:idx + 1] = 0
            self.share_inputs["min_dec_len"][idx:idx + 1] = task.get(
                "min_tokens", 1)

            self.share_inputs["max_dec_len"][idx:idx + 1] = task.get(
                "max_tokens", self.max_length)
            self.share_inputs["stop_flags"][idx:idx + 1] = False

            self.share_inputs["first_token_ids"][
                idx:idx + 1] = self.share_inputs["input_ids"][idx:idx + 1, :1]
            self.share_inputs["ori_seq_lens_encoder"][idx:idx + 1] = length

            if task.get("seed") is not None:
                self.share_inputs["infer_seed"][idx:idx + 1] = task.get("seed")
            encoder_block_num = len(task.get("block_tables"))
            self.share_inputs["encoder_block_lens"][idx:idx +
                                                    1] = encoder_block_num
            self.share_inputs["block_tables"][idx:idx + 1, :] = -1
            self.share_inputs["block_tables"][
                idx:idx + 1, :encoder_block_num] = np.array(task.block_tables,
                                                            dtype="int32")

            # TODO 待确认正确性
            if task.get("stop_token_ids") is not None and task.get(
                    "stop_seqs_len") is not None:
                stop_seqs_num = len(task.get("stop_seqs_len"))
                for i in range(stop_seqs_num,
                               self.model_cfg.max_stop_seqs_num):
                    task.stop_seqs_len.append(0)
                self.share_inputs["stop_seqs_len"][:] = np.array(
                    task.stop_seqs_len, dtype="int32")
                self.share_inputs["stop_seqs"][:stop_seqs_num, :len(
                    task.get("stop_token_ids")[0])] = np.array(
                        task.get("stop_token_ids"), dtype="int64")

    def get_rotary_position_embedding(self,
                                      position_ids,
                                      head_dim,
                                      rope_theta=160000):
        """
        Pre-calculate rotary position embedding for position_ids.

        Args:
            position_ids: [1, S]
            head_dim: D

        Returns:
            rot_emb: [2, 1, S, 1, D // 2] or [2, 1, S, 1, D], cos + sin
        """
        bsz, max_model_len = position_ids.shape[:2]
        inv_freq = rope_theta ** (
            -paddle.arange(0, head_dim, 2, dtype="float32") / head_dim)

        # shape: [B, S, D/2]
        # eblite should divide compression_ratio, default 1.0 for eb3.5 or eb4
        compression_ratio = 1.0
        compressed_position_ids = position_ids / compression_ratio
        freqs = paddle.einsum("ij,k->ijk",
                              compressed_position_ids.cast("float32"),
                              inv_freq)

        rot_emb = paddle.zeros((2, bsz, max_model_len, 1, head_dim // 2),
                               dtype="float32")
        emb = paddle.stack([freqs], axis=-1).reshape(
            (bsz, max_model_len, head_dim // 2))
        # shape: [B, S, 1, D]
        emb = paddle.unsqueeze(emb, 2)

        rot_emb[0] = paddle.cos(emb)
        rot_emb[1] = paddle.sin(emb)

        return rot_emb

    def generate(self):
        self.model(**self.share_inputs)

    def clear_parameters(self, pid):
        if "caches" in self.share_inputs:
            self.model.clear_parameters(pid)
            del self.share_inputs["caches"]
            paddle.device.cuda.empty_cache()
            self.model.log_memory_usage("clear all memory")

    def update_parameters(self, pid):
        if "caches" not in self.share_inputs:
            self.model.update_parameters(pid)
            self._init_kvcache()
            self.model.log_memory_usage("update all memory")

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
        theoretical_kv_cache_memory = (2 * byte_of_cache *
                                       self.args.block_size * num_layers *
                                       hidden_dim)
        return theoretical_kv_cache_memory

    def _update_share_input_block_num(self):
        del self.share_inputs["caches"]
        self._init_kvcache()

        del self.share_inputs["block_tables"]
        self.share_inputs["block_tables"] = paddle.full(
            [self.args.max_num_seqs, self.num_gpu_blocks], -1, dtype="int32")

        # 初始化free list
        free_list = list(
            range(self.num_gpu_blocks - 1,
                  int(self.num_gpu_blocks * self.args.kv_cache_ratio) - 1, -1))
        self.free_list_len = len(free_list)
        self.share_inputs.update({
            "free_list":
            paddle.to_tensor(free_list, dtype="int32"),
            "free_list_len":
            paddle.full([1], self.free_list_len, dtype="int32"),
        })

    def dummy_input(self, num_total_tokens, number_of_tasks):
        """
        fake input to profile
        """
        input_length = num_total_tokens // number_of_tasks

        block_num = (input_length + self.args.block_size - 1 +
                     self.args.enc_dec_block_num) // self.args.block_size
        self.share_inputs["free_list"] = paddle.to_tensor([], dtype="int32")
        self.share_inputs["free_list_len"][0] = 0


        for i in range(number_of_tasks):
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

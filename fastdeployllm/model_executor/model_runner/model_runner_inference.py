import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet

import os
import numpy as np
from fastdeployllm.model_executor.model_runner.model_runner_base import ModelRunnerBase

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
        config, tokenizer, model = build_stream_line_model(
            os.path.join(self.args.model_name_or_path, os.getenv("CONFIG_JSON_FILE", "config.json")),
            self.args.model_name_or_path,
            self.args.dtype,
            block_size=self.args.block_size,
            max_len=self.args.max_seq_len,
            stage_flag="msgid-1 predict",
            export_model_type="wint8",
            use_fake_parameter=False,
            use_stop_seqs=self.model_cfg.ellm_dynamic_use_stop_seqs,
            use_beam_search=False,
            speculate_method=None,
            speculate_max_draft_token_num=5,
            return_all_hidden_states=False,
            is_int4_moe=True,
        )
        model.eval()
        self.model = model

    def init_rotary_position_embedding(self, max_seq_len):
        tmp_position_ids = paddle.arange(max_seq_len).reshape((1, -1))
        self.share_inputs["rope_emb"] = self.get_rotary_position_embedding(
            tmp_position_ids,
            self.model_cfg.hidden_size // self.model_cfg.num_attention_heads
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

        for i in range(self.model_cfg.num_layers):
            cache_type = self.args.dtype
            self.cache_kvs["key_caches_{}".format(i)] = paddle.full(
                shape=[
                    self.args.max_block_num,
                    kv_num_head,
                    self.args.block_size,
                    self.model_cfg.hidden_size // self.model_cfg.num_attention_heads,
                ],
                fill_value=0,
                dtype=cache_type,
            )
            self.cache_kvs["value_caches_{}".format(i)] = paddle.full(
                shape=[
                    self.args.max_block_num,
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

            self.share_inputs["max_dec_len"][idx : idx + 1] = task.get("max_tokens", self.args.max_dec_len)
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
            if task.get("stop_token_ids") is not None:
                stop_seqs_num = len(task.get("stop_seqs_len"))
                for i in range(stop_seqs_num, self.model_cfg.max_stop_seqs_num):
                    task.stop_seqs_len.append(0)
                self.share_inputs["stop_seqs_len"][:] = np.array(task.stop_seqs_len, dtype="int32")
                self.share_inputs["stop_seqs"][:stop_seqs_num, : len(task.get("stop_token_ids")[0])] = np.array(
                    task.get("stop_token_ids"), dtype="int64"
                )

    def get_rotary_position_embedding(self,position_ids, head_dim, rope_theta=160000):
        """
        Pre-calculate rotary position embedding for position_ids.

        Args:
            position_ids: [1, S]
            head_dim: D

        Returns:
            rot_emb: [2, 1, S, 1, D // 2] or [2, 1, S, 1, D], cos + sin 
        """
        bsz, max_seq_len = position_ids.shape[:2]
        inv_freq = rope_theta ** (-paddle.arange(0, head_dim, 2, dtype="float32") / head_dim)

        # shape: [B, S, D/2]
        # eblite should divide compression_ratio, default 1.0 for eb3.5 or eb4
        compression_ratio = 1.0
        compressed_position_ids = position_ids / compression_ratio
        freqs = paddle.einsum("ij,k->ijk", compressed_position_ids.cast("float32"),
                            inv_freq)

        rot_emb = paddle.zeros((2, bsz, max_seq_len, 1, head_dim // 2), dtype="float32")
        emb = paddle.stack([freqs], axis=-1).reshape((bsz, max_seq_len, head_dim // 2))
        # shape: [B, S, 1, D]
        emb = paddle.unsqueeze(emb, 2)

        rot_emb[0] = paddle.cos(emb)
        rot_emb[1] = paddle.sin(emb)

        return rot_emb

    def generate(self):
        self.model(**self.share_inputs)

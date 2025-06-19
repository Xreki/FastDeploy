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

import json
import os

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.rotary_embedding import get_rope
from fastdeploy.model_executor.models.export_model import \
    build_stream_line_model
from fastdeploy.utils import spec_logger

try:
    from fastdeploy.model_executor.ops.gpu import (
        draft_model_postprocess, draft_model_preprocess,
        eagle_get_hidden_states, eagle_get_self_hidden_states, ngram_match,
        speculate_update_seq_lens_this_time)
except ImportError:
    pass


class Proposer:
    """
    Proposer Base Class.

    Used to provide an extensible interface for draft tokens within
    the speculative decoding framework
    """

    def __init__(self, cfg: FDConfig):
        self.cfg = cfg
        self.spec_cfg = cfg.speculative_config
        self.max_num_seqs = cfg.parallel_config.max_num_seqs
        self.max_draft_token_num = cfg.speculative_config.num_speculative_tokens

        spec_logger.info(f"Speculate config: {self.spec_cfg}")

    def run(self, share_inputs):
        """
        run
        """
        raise NotImplementedError

    def insert_query(self, preprocessed_inputs):
        """
        insert query
        """
        pass

    def postprocess(self, base_model_inputs):
        """
        postprocess
        """
        pass


class AutogressiveProposer(Proposer):
    """
    Proposer for Autoregressive Decoding.

    Without draft tokens, simply utilizing the framework
    to place the last autoregressively generated token in
    the first position of draft_tokens.
    """

    def run(self, share_inputs, **kargs):
        speculate_update_seq_lens_this_time(
            kargs["seq_lens_this_time"],
            share_inputs["seq_lens_encoder"],
            share_inputs["seq_lens_decoder"],
            kargs["real_batch_size"],
            1,
        )


class NgramProposer(Proposer):
    """
    Proposer for Ngram match method.

    Matching corresponding tokens in input and output as draft tokens.
    """

    def __init__(self, cfg: FDConfig):
        super().__init__(cfg)
        self.max_ngram_size = self.spec_cfg.max_ngram_size
        self.input_ids_len = paddle.zeros(shape=[self.max_num_seqs, 1],
                                          dtype="int64").cpu()

    def update(self, bid: int, seq_len: int):
        """
        update
        """
        self.input_ids_len[bid] = seq_len

    def run(self, share_inputs, **kargs):
        """
        run
        """
        draft_tokens = share_inputs["draft_tokens"].cpu()
        seq_lens_this_time = share_inputs["seq_lens_this_time"].cpu()
        seq_lens_encoder = share_inputs["seq_lens_encoder"].cpu()
        seq_lens_decoder = share_inputs["seq_lens_decoder"].cpu()

        ngram_match(
            share_inputs["input_ids_cpu"],
            self.input_ids_len.cpu(),
            share_inputs["pre_ids"].cpu(),
            share_inputs["step_idx"].cpu(),
            share_inputs["actual_draft_token_num"].cpu(),
            draft_tokens,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            share_inputs["max_dec_len"].cpu(),
            self.max_ngram_size,
            self.max_draft_token_num,
        )
        share_inputs["draft_tokens"][:] = draft_tokens.cuda()
        share_inputs["seq_lens_encoder"][:] = seq_lens_encoder.cuda()
        share_inputs["seq_lens_this_time"][:] = seq_lens_this_time.cuda()


class ModelProposer(Proposer):
    """
    Proposer for model-based method. Like Draft Model/Eagle/MTP.
    """

    def __init__(self, args, max_draft_tokens, batch_size):
        super().__init__()
        print(f"Initialize {args.speculative_method} proposer")
        self.args = args
        self.draft_type = args.speculative_method
        assert self.draft_type in (
            "draft_model",
            "eagle",
            "mtp",
        ), f"draft_type support [draft_model, eagle], but get {self.draft_type}"
        self.max_batch_size = batch_size
        self.max_draft_tokens = max_draft_tokens
        self.actual_draft_token_num = max_draft_tokens

        self.init_predictor(args)

    def init_predictor(self, args):
        """
        init_predictor
        """
        self.use_beam_search = False
        tensor_parallel_degree = dist.get_world_size()
        self.rank: int = dist.get_rank()
        self.nranks = dist.get_world_size()

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": tensor_parallel_degree,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

        self.beam_batch_size = args.batch_size * args.beam_width
        self.use_beam_search = True if args.beam_width > 1 else False
        self.speculative_max_draft_tokens = args.speculative_max_draft_tokens
        self.show_topk = False

        # 2. build model
        paddle.set_default_dtype(self.args.dtype)
        config_path = os.path.join(args.draft_model_path, "config.json")
        with open(config_path) as model_config_file:
            model_config = json.load(model_config_file)

        use_cache_kv_int8 = False
        # use_cache_kv_int4 = False
        if "quant_type" in model_config:
            if "C8" in model_config["quant_type"]:
                use_cache_kv_int8 = True
            # if "C4" in model_config["quant_type"]:
            #     use_cache_kv_int4 = True

            args.draft_model_type = model_config["quant_type"]

        config, tokenizer, model, _ = build_stream_line_model(
            config_path,
            args.draft_model_path,
            args.dtype,
            block_size=args.block_size,
            max_len=args.max_seq_len,
            stage_flag="msgid-speculate-draft predict",
            min_dec_len=args.min_dec_len,
            max_dec_len=args.max_dec_len,
            temperature=args.temperature,  # not use
            top_k=args.top_k,  # not use
            top_p=args.top_p,  # not use
            export_model_type=args.draft_model_type,
            speculative_method=self.draft_type,
            pad_vocab=False,
            draft_type=self.draft_type,
        )
        model.eval()

        self.model = model
        self.model_config = config
        self.tokenizer = tokenizer
        # init cache_kvs
        num_layers = self.model_config.get("num_layers",
                                           None) or self.model_config.get(
                                               "num_hidden_layers", None)
        self.cache_kvs = []
        self.free_list = list(range(args.max_num_blocks))
        self.used_list = [[] for _ in range(self.beam_batch_size)]
        head_dim = self.model_config["head_dim"]
        self.pre_ids = paddle.to_tensor(
            np.zeros((self.beam_batch_size,
                      args.max_dec_len)).astype("int64") - 1)
        tmp_position_ids = paddle.arange(args.max_seq_len).reshape((1, -1))
        compression_ratio = self.model_config.get("compression_ratio", 1)
        rope_theta = self.model_config.get("rope_theta", 10000.0)
        self.rope_emb = get_rope(
            rotary_dim=head_dim,
            base=rope_theta,
            position_ids=tmp_position_ids,
            partial_rotary_factor=compression_ratio,
        )
        # （liuzichang）: eliminate rope effect
        # self.rope_emb[0,:,:,:,:] = 1
        # self.rope_emb[1,:,:,:,:] = 0

        self.input_ids = paddle.full(
            shape=[self.beam_batch_size, args.max_seq_len],
            fill_value=self.tokenizer.pad_id,
            dtype="int64",
        )
        num_key_value_heads = self.model_config.get(
            "num_key_value_heads", self.model_config["num_attention_heads"])
        if num_key_value_heads is None:
            num_key_value_heads = self.model_config["num_attention_heads"]
        num_key_value_heads = num_key_value_heads // self.nranks
        if use_cache_kv_int8:
            cache_type = "uint8"
        else:
            cache_type = args.dtype

        self.vocab_size = int(self.model_config["vocab_size"])
        self.hidden_size = int(self.model_config["hidden_size"])
        for i in range(num_layers):
            self.cache_kvs.append(
                paddle.to_tensor(
                    np.zeros([
                        args.max_num_blocks,
                        num_key_value_heads,
                        args.block_size,
                        head_dim,
                    ]).astype("float32")).astype(cache_type))
            self.cache_kvs.append(
                paddle.to_tensor(
                    np.zeros([
                        args.max_num_blocks,
                        num_key_value_heads,
                        args.block_size,
                        head_dim,
                    ]).astype("float32")).astype(cache_type))

    def insert_query(self, preprocessed_inputs):
        self.model_inputs = {}
        base_model_inputs = preprocessed_inputs["inputs"]
        seq_len = preprocessed_inputs["seq_len"]
        real_bs = preprocessed_inputs["real_bs"]

        max_sec_len = self.args.max_seq_len
        self.model_inputs["block_tables"] = paddle.full_like(
            base_model_inputs["block_tables"], fill_value=-1, dtype="int32")
        for i in range(real_bs):
            real_len = seq_len[i] + self.args.max_dec_len
            if real_len > max_sec_len:
                self.free_list = list(range(self.args.max_num_blocks))
                self.used_list = [[] for _ in range(self.beam_batch_size)]
                raise ValueError(f"input_len({seq_len[i]}) + \
max_dec_len({self.args.max_dec_len}) > max_seq_len({max_sec_len})")
            for j in range(
                (real_len + self.args.block_size - 1) // self.args.block_size):
                used_block_id = self.free_list.pop()
                self.used_list[i].append(used_block_id)
                self.model_inputs["block_tables"][i, j] = used_block_id
        self.model_inputs["input_ids"] = paddle.clone(
            base_model_inputs["input_ids"])
        self.model_inputs["seq_lens_this_time"] = paddle.clone(
            base_model_inputs["seq_lens_this_time"])
        self.model_inputs["seq_lens_encoder"] = paddle.clone(
            base_model_inputs["seq_lens_encoder"])
        self.model_inputs["seq_lens_decoder"] = paddle.clone(
            base_model_inputs["seq_lens_decoder"])
        self.model_inputs["step_idx"] = paddle.clone(
            base_model_inputs["step_idx"])
        self.model_inputs["stop_flags"] = paddle.clone(
            base_model_inputs["stop_flags"])
        self.model_inputs["stop_nums"] = paddle.clone(
            base_model_inputs["stop_nums"])
        self.model_inputs["not_need_stop"] = paddle.to_tensor([False],
                                                              dtype="bool",
                                                              place="cpu")
        self.model_inputs["pre_ids"] = self.pre_ids
        self.model_inputs["rope_emb"] = self.rope_emb
        self.model_inputs["caches"] = self.cache_kvs

        self.model_inputs["top_p"] = base_model_inputs["top_p"]
        self.model_inputs["temperature"] = base_model_inputs["temperature"]
        self.model_inputs["eos_token_id"] = base_model_inputs["eos_token_id"]
        self.model_inputs["penalty_score"] = base_model_inputs["penalty_score"]
        self.model_inputs["frequency_score"] = base_model_inputs[
            "frequency_score"]
        self.model_inputs["presence_score"] = base_model_inputs[
            "presence_score"]
        self.model_inputs["max_dec_len"] = base_model_inputs["max_dec_len"]
        self.model_inputs["min_dec_len"] = base_model_inputs["min_dec_len"]
        self.model_inputs["bad_tokens"] = base_model_inputs["bad_tokens"]
        self.model_inputs["next_tokens"] = paddle.full(
            shape=[self.beam_batch_size, 1], fill_value=-1, dtype="int64")
        self.model_inputs["base_model_draft_tokens"] = base_model_inputs[
            "draft_tokens"]
        self.model_inputs["draft_tokens"] = paddle.full(
            shape=[self.args.batch_size, 2], fill_value=-1, dtype="int64")

        self.seq_lens_encoder_record = paddle.full(
            shape=[self.beam_batch_size, 1], fill_value=-1, dtype="int32")
        self.seq_lens_decoder_record = paddle.full(
            shape=[self.max_batch_size, 1], fill_value=0, dtype="int32")

        self.model_inputs["substep"] = 0
        for i in range(real_bs):
            self.model_inputs["pre_ids"][
                i, 0] = self.model_inputs["input_ids"][i, -1]
            self.seq_lens_encoder_record[i:i + 1] = seq_len[i]

        self.model_inputs["batch_drop"] = paddle.full(
            shape=[self.max_batch_size, 1], fill_value=False, dtype="bool")

    def run_preprocess(self, share_inputs):
        """
        update draft model parameteds
        """
        draft_model_preprocess(
            self.model_inputs["draft_tokens"],
            self.model_inputs["input_ids"],
            self.model_inputs["stop_flags"],
            self.model_inputs["seq_lens_this_time"],
            self.model_inputs["seq_lens_encoder"],
            self.model_inputs["seq_lens_decoder"],
            self.model_inputs["step_idx"],
            self.seq_lens_encoder_record,
            self.seq_lens_decoder_record,
            self.model_inputs["not_need_stop"],
            self.model_inputs["batch_drop"],
            share_inputs["accept_tokens"],
            share_inputs["accept_num"],
            share_inputs["seq_lens_encoder"],
            share_inputs["seq_lens_decoder"],
            share_inputs["step_idx"],
            share_inputs["stop_flags"],
            share_inputs["is_block_step"],
            share_inputs["draft_tokens"],
            self.actual_draft_token_num,
            self.draft_type in ["eagle", "mtp"],
        )

    def run_infer(self, share_inputs):
        """
        Should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses mut implement this function")

    def run_postprocess(self, share_inputs):
        """
        Update base model draft_tokens
        """
        draft_model_postprocess(
            share_inputs["draft_tokens"],
            share_inputs["seq_lens_this_time"],
            share_inputs["seq_lens_encoder"],
            share_inputs["stop_flags"],
        )

    def run(self, share_inputs, **kargs):
        self.run_preprocess(share_inputs)
        self.run_infer(share_inputs)
        self.run_postprocess(share_inputs)

    def postprocess(self, base_model_inputs):
        for i in range(self.max_batch_size):
            if not base_model_inputs["stop_flags"][i]:
                break
        self.pre_ids[:] = -1
        self.free_list = list(range(self.args.max_num_blocks))
        self.used_list = [[] for _ in range(self.beam_batch_size)]


class MTPProposer(ModelProposer):
    """
    用于 Eagle 的 Proposer
    在输入输出中匹配符合的tokens作为 draft tokens
    """

    def insert_query(self, preprocessed_inputs):
        super().insert_query(preprocessed_inputs)
        # real_bs = preprocessed_inputs["real_bs"]
        # seq_len = preprocessed_inputs["seq_len"]
        base_model_inputs = preprocessed_inputs["inputs"]

        self.model_inputs["input_ids"][:, :-1] = base_model_inputs[
            "input_ids"][:, 1:]
        self.last_seq_lens_this_time = paddle.full_like(
            base_model_inputs["seq_lens_this_time"],
            fill_value=-1,
            dtype="int32")

    def run_infer(self, share_inputs):
        if self.model_inputs["not_need_stop"]:
            base_model_hidden_states = eagle_get_hidden_states(
                share_inputs["all_hidden_states"],
                self.model_inputs["seq_lens_this_time"],
                self.model_inputs["seq_lens_encoder"],
                self.model_inputs["seq_lens_decoder"],
                self.model_inputs["stop_flags"],
                share_inputs["accept_num"],
                share_inputs["seq_lens_this_time"],
                share_inputs["seq_lens_encoder"],
                self.actual_draft_token_num,
            )
            self.model_inputs["hidden_states"] = base_model_hidden_states

        with paddle.no_grad():
            self.model_inputs["substep"] = 0
            while (self.model_inputs["not_need_stop"]
                   and self.model_inputs["substep"] < self.max_draft_tokens):
                self.last_seq_lens_this_time[:] = self.model_inputs[
                    "seq_lens_this_time"][:]
                output_hidden_states = self.model(**self.model_inputs)

                self.model_inputs["substep"] += 1
                if (self.model_inputs["not_need_stop"]
                        and self.model_inputs["substep"]
                        < self.actual_draft_token_num):
                    self.model_inputs[
                        "hidden_states"] = eagle_get_self_hidden_states(
                            output_hidden_states,
                            self.last_seq_lens_this_time,
                            self.model_inputs["seq_lens_this_time"],
                            self.model_inputs["step_idx"],
                        )
                else:
                    self.model_inputs["hidden_states"] = None

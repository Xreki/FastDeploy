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

"""infer generation under static graph"""

from __future__ import annotations

import os
import sys

import argparse
import contextlib
import copy
import json
import multiprocessing as mp
import struct
import time

import numpy as np
from paddlenlp.trainer.argparser import strtobool
from paddlenlp.utils.log import logger
from tqdm import tqdm

import paddle
import paddle.distributed as dist

from efficientllm.models.tokenizer import ErnieBotTokenizer
from paddle import profiler
from paddle.distributed import fleet
from efficientllm.models.utils import (
    get_infer_model_path,
    get_rotary_position_embedding,
    infer_save_test_case,
    load_prefix_weights,
    load_sharded_checkpoint,
)

from efficientllm.platform import current_platform

if current_platform.is_cuda() and current_platform.available():
    from efficientllm.ops.gpu import reset_stop_value, speculate_update_input_ids_cpu
elif paddle.is_compiled_with_xpu():
    from custom_setup_ops import reset_stop_value
elif paddle.is_compiled_with_custom_device("npu"):
    from paddle_custom_device.npu import reset_stop_value
else:  # CPU
    from efficientllm.ops.cpu import reset_stop_value

from efficientllm.models.data_utils import (
    convert_fc_infer_data,
    convert_to_input_ids,
    get_infer_data_type,
    insert_fc_instruction,
)
from efficientllm.models.token_utils import TokenTimer, check_output

from efficientllm.models.speculate_proposers import (
    AutogressiveProposer,
    DraftModelProposer,
    EagleProposer,
    HydraProposer,
    InferenceWithReferenceProposer,
    MTPProposer,
)


should_check_python_safety = False
if should_check_python_safety:
    if not os.path.isfile("./utils/token_utils.pyxes"):
        print("Some toolkit files have been modified, check failed.")
        sys.exit(-15)
    from utils.token_utils import check_python_safe

    check_python_safe()


def deserialize_from_file(fp):
    """Deserialize a numpy array from file."""
    x_type = fp.read(1)
    x_type_out = struct.unpack("c", x_type)[0]
    # data
    data_list = []
    if x_type_out == b"0":
        data = fp.read(4)
        data_out = struct.unpack("f", data)[0]
        while data:
            data_out = struct.unpack("f", data)[0]
            data_list.append(data_out)
            data = fp.read(4)
    elif x_type_out == b"1":
        data = fp.read(8)
        while data:
            data_out = struct.unpack("l", data)[0]
            data_list.append(data_out)
            data = fp.read(8)
    elif x_type_out == b"2":
        data = fp.read(4)
        while data:
            data_out = struct.unpack("i", data)[0]
            data_list.append(data_out)
            data = fp.read(4)
    else:
        print("type error")
    data_arr = np.array(data_list)
    return data_arr


def get_parser():
    """Setup inference arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_num_blocks", type=int, default=2000)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument(
        "--append_bos_token",
        default="True",
        type=strtobool,
        help="whether predict with bos token",
    )
    parser.add_argument("--model_name_or_path", type=str, default="./inference")
    parser.add_argument(
        "--data_format",
        type=str,
        default="sft",
        choices=["pt", "sft", "ec2_completion", "ec3_completion", "rm"],
        help="The data format.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--model_prefix", type=str, default="model")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--penalty_score", type=float, default=1)
    parser.add_argument("--frequency_score", type=float, default=0)
    parser.add_argument("--presence_score", type=float, default=0)
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--beam_group_num", type=int, default=1)
    parser.add_argument("--beam_length_penalty", type=float, default=0.0)
    parser.add_argument("--beam_diversity_penalty", type=float, default=0.0)
    parser.add_argument("--lora_num", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument(
        "--max_seq_len", type=int, default=8192, help="max input + output len"
    )
    parser.add_argument("--min_dec_len", type=int, default=1)
    parser.add_argument("--max_dec_len", type=int, default=2048, help="max output len")
    parser.add_argument("--brenchmark_steps", type=int, default=5)
    parser.add_argument("--use_cache_kv_int8", type=int, default=0)
    parser.add_argument("--use_cache_kv_int4", type=int, default=0)
    parser.add_argument(
        "--input_file", type=str, default="./data/query-answers-list.jsonl"
    )
    parser.add_argument("--output_file", type=str, default="infer.jsonl")
    parser.add_argument("--save_output_file_flush", type=int, default=0)
    parser.add_argument("--begin_infer_idx", type=int, default=0)
    parser.add_argument(
        "--use_system",
        default="False",
        type=strtobool,
        help="use system or not",
    )
    parser.add_argument(
        "--prefix_path",
        default=None,
        type=str,
        help="The directory of Prefix Tuning parameters. Default to None",
    )
    parser.add_argument(
        "--device",
        default="GPU",
        type=str,
        choices=["GPU", "XPU", "NPU", "CPU"],
        help="The device for program execution. e.g. GPU, XPU, NPU, CPU",
    )
    parser.add_argument(
        "--use_stop_seqs",
        default="False",
        type=strtobool,
        help="whether use stop seqs",
    )
    parser.add_argument(
        "--enf_gen_input_file",
        type=str,
        default="",
        help="Enforced generated contents",
    )
    parser.add_argument(
        "--enf_gen_context_cache_dir",
        type=str,
        default="",
        help="If not set, it will default to a folder with the same name as the input file.",
    )
    parser.add_argument(
        "--speculate_method",
        default=None,
        type=str,
        choices=[
            "autoregressive",
            "inference_with_reference",
            "draft_model",
            "hydra",
            "eagle",
        ],
    )
    parser.add_argument("--draft_model_path", default="None", type=str)
    parser.add_argument(
        "--draft_model_type",
        default="default",
        type=str,
        choices=["default", "WINT8", "W8A8C8", "W8A8C16"],
    )

    parser.add_argument("--speculate_max_draft_tokens", default=1, type=int)
    parser.add_argument("--speculate_max_ngram_size", default=1, type=int)
    parser.add_argument("--speculate_hydra_ckpt", default="", type=str)
    parser.add_argument(
        "--perf_input_len",
        default=-1,
        type=int,
        help="Default is -1, means normal mode.",
    )
    parser.add_argument("--perf_warm_up_times", default=2, type=int)
    parser.add_argument(
        "--perf_repeat_times", default=10, type=int, help="Repeat times in perf mode"
    )
    parser.add_argument(
        "--perf_output_file",
        default="perf_result",
        type=str,
        help="The output file of perf result",
    )
    parser.add_argument(
        "--rm_use_cls",
        type=strtobool,
        default="True",
        help="Whether to use cls to predict RM score.",
    )
    parser.add_argument(
        "--use_efficientllm",
        default="False",
        type=strtobool,
        help="use efficientllm or not",
    )
    parser.add_argument(
        "--msg_queue_id",
        type=int,
        default=1,
        help="The msg queue id for get output.",
    )
    parser.add_argument(
        "--use_ep",
        default="False",
        type=strtobool,
        help="use ep or not",
    )
    parser.add_argument(
        "--ep_just_for_test",
        default="False",
        type=strtobool,
        help="ep just for test or not",
    )

    return parser


def setup_args():
    """Setup inference arguments."""
    parser = get_parser()
    args = parser.parse_args()

    assert args.msg_queue_id >= 0
    return args


class CpuGuard:
    """CpuGuard"""

    def __init__(self):
        """init"""
        pass

    def __enter__(self):
        """enter"""
        self.ori_device = paddle.device.get_device()
        paddle.device.set_device("cpu")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """exit"""
        paddle.device.set_device(self.ori_device)


class _StaticGuard:
    """_StaticGuard"""

    def __init__(self):
        """init"""
        pass

    def __enter__(self):
        """enter"""
        paddle.enable_static()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """exit"""
        paddle.disable_static()


def init_dist_env(world_size, seed=20):
    """
    初始化分布式环境。

    Args:
        world_size (int): 分布式训练任务的机器数量。
        seed (int): 随机种子。默认值为20。

    Returns:
        None。
    """
    context = contextlib.nullcontext()
    # xpu device will hang if use dynamic mode
    if paddle.is_compiled_with_xpu():
        context = _StaticGuard()
    with context:
        # start to init distributed env
        strategy = fleet.DistributedStrategy()

        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": world_size,
            "pp_degree": 1,
            "sharding_degree": 1,
        }

        # Set control in tensor parallel
        strategy.tensor_parallel_configs = {"tensor_init_seed": seed}

        fleet.init(is_collective=True, strategy=strategy)


class Predictor:
    """
    Predictor for inference.
    """

    def __init__(self, args):
        """
        初始化模型。

        Args:
            args (dict): 参数字典，包含以下参数：
                vocab_path (str): 分词器文件路径。
                config_path (str): 模型配置文件路径。
                batch_size (int): 最大批大小。
                max_num_blocks (int): 当前可用的最大块数量。
                block_size (int): 每个块的最大序列长度。
                max_seq_len (int): 输入序列的最大长度。
                use_cache_kv_int8 (bool): 是否使用缓存密钥为整数的INT8类型。
                dtype (str): 数据类型。
                compression_ratio (float): 压缩比率。

        """
        self.args = args
        self.num_input_tokens = 0
        self.num_output_tokens = 0

        paddle.set_default_dtype(args.dtype)

        if dist.get_world_size() > 1:
            init_dist_env(dist.get_world_size())
            self.nranks = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            strategy = fleet.DistributedStrategy()
            strategy.hybrid_configs = {
                "dp_degree": 1,
                "mp_degree": 1,
                "pp_degree": 1,
            }
            fleet.init(is_collective=True, strategy=strategy)
            self.nranks = 1
            self.rank = 0

        config_path = os.path.join(args.model_name_or_path, "config.json")
        with open(config_path) as model_config_file:
            model_config = json.load(model_config_file)
        self.model_config = model_config
        self.use_beam_search = model_config.get("infer_model_use_beam_search", False)
        gqa_use_tensorcore = model_config.get("gqa_use_tensorcore", False)
        self.speculate_enable = model_config.get("infer_model_speculate_enable", False)
        self.return_all_hidden_states = model_config.get(
            "infer_model_return_all_hidden_states", False
        )
        if gqa_use_tensorcore:
            paddle.set_flags({"FLAGS_gqa_use_tensorcore": 1})
        ffn2_use_hardamard = model_config.get("ffn2_use_hardamard", False)
        if ffn2_use_hardamard:
            paddle.set_flags({"FLAGS_ffn2_use_hardamard": 1})
            logger.info("ffn2_use_hardamard is set to True.")
            hardamard_block_size = model_config.get("hardamard_block_size", None)
            if hardamard_block_size is not None:
                paddle.set_flags({"FLAGS_hardamard_block_size": hardamard_block_size})
                logger.info(f"hardamard_block_size is set to {hardamard_block_size}")
                paddle.set_flags({"FLAGS_hardamard_use_diagonal_block_matrix": 1})
                logger.info("FLAGS_hardamard_use_diagonal_block_matrix is set to True")
        layernorm_only_std = model_config.get("layernorm_only_std", False)
        if layernorm_only_std:
            paddle.set_flags({"FLAGS_only_std_norm": 1})
            logger.info("FLAGS_only_std_norm is set to True")
        self.beam_width = args.beam_width
        self.beam_group_num = args.beam_group_num
        self.enf_gen = model_config.get("infer_model_enf_gen", False)
        if self.use_beam_search:
            assert (
                self.beam_width > 1 and self.beam_width <= 16
            ), f"Beam width must be greater than one and less equal than 16. But get {self.beam_width}"
            assert (
                self.beam_group_num <= self.beam_width
            ), f"beam_group_num must be less than or equal to beam_width. \
                  But get beam_width:{self.beam_width} and beam_group_num:{self.beam_group_num}"
            assert (
                self.beam_width % self.beam_group_num == 0
            ), f"beam_width must be divisible by beam_group_num. \
                  But get beam_width:{self.beam_width} and beam_group_num:{self.beam_group_num}"
        if self.speculate_enable:
            assert (
                args.speculate_method is not None
            ), "Speculate Model must be set a speculate_method"
        else:
            assert (
                args.speculate_method is None
            ), "Non-Speculate Model should set speculate_method=None"

        if self.beam_width <= 1:
            self.result_queue = mp.Queue()

            from efficientllm.models.utils import MAX_BSZ, MAX_DRAFT_TOKENS

            if args.speculate_method is not None:
                from efficientllm.models.utils import speculate_read_res

                output_tensor_max_shape = [MAX_BSZ * MAX_DRAFT_TOKENS + MAX_BSZ + 2]
                self.read_res_process = mp.Process(
                    target=speculate_read_res,
                    args=[
                        self.args.model_name_or_path,
                        output_tensor_max_shape,
                        self.result_queue,
                        self.args.msg_queue_id,
                    ],
                )
            else:
                from efficientllm.models.utils import read_res

                output_tensor_max_shape = [MAX_BSZ + 2, 1]
                self.read_res_process = mp.Process(
                    target=read_res,
                    args=[
                        self.args.model_name_or_path,
                        output_tensor_max_shape,
                        self.result_queue,
                        self.args.msg_queue_id,
                        self.args.use_ep,
                        self.args.ep_just_for_test,
                    ],
                )

            if self.rank == 0 or (
                self.args.use_ep and (not self.args.ep_just_for_test)
            ):
                self.read_res_process.start()
        self.tokenizer = ErnieBotTokenizer.from_pretrained(args.model_name_or_path)

        if (
            "infer_model_mp_num" in model_config
            and self.nranks != model_config["infer_model_mp_num"]
        ):
            raise ValueError(
                f"Error infer mp rank is {self.nranks}, "
                + f"but load inference model with mp num:{model_config['infer_model_mp_num']}"
            )
        if "infer_model_block_size" in model_config:
            logger.info("Using block_size in infer model config ")
            args.block_size = model_config["infer_model_block_size"]
        if "infer_model_dtype" in model_config:
            logger.info("Using dtype in infer model config ")
            args.dtype = model_config["infer_model_dtype"]
        if "infer_model_max_seq_len" in model_config:
            logger.info("Using max_seq_len in infer model config ")
            args.max_seq_len = model_config["infer_model_max_seq_len"]
        if "infer_model_cache_kv_type" in model_config:
            logger.info("Using cache_kv_type in infer model config ")
            if model_config["infer_model_cache_kv_type"] == "cachekv_quant_int8":
                args.use_cache_kv_int8 = 1
                args.use_cache_kv_int4 = 0
            elif model_config["infer_model_cache_kv_type"] == "cachekv_quant_int4":
                args.use_cache_kv_int8 = 0
                args.use_cache_kv_int4 = 1
            else:
                args.use_cache_kv_int8 = 0
                args.use_cache_kv_int4 = 0

        if self.beam_width > 1:
            args.max_num_blocks = args.batch_size * (
                (args.max_seq_len + args.block_size - 1) // args.block_size
            ) + args.batch_size * (self.beam_width - 1) * (
                (args.max_dec_len + args.block_size - 1) // args.block_size
            )
        else:
            args.max_num_blocks = (
                args.batch_size
                * args.beam_width
                * ((args.max_seq_len + args.block_size - 1) // args.block_size)
            )

        head_dim = model_config["hidden_size"] // model_config["num_attention_heads"]
        self.beam_batch_size = args.batch_size * self.beam_width

        self.free_list = list(range(args.max_num_blocks))
        self.used_list = [[] for _ in range(self.beam_batch_size)]
        self.use_cache_kv_int8 = True if args.use_cache_kv_int8 == 1 else False
        self.use_cache_kv_int4 = True if args.use_cache_kv_int4 == 1 else False
        if (
            self.use_cache_kv_int8
            or "C8" in model_config.get("quant_type", "")
            or "C4" in model_config.get("quant_type", "")
            or self.use_cache_kv_int4
        ):
            self.cache_dtype = "uint8"
        else:
            self.cache_dtype = args.dtype
        num_key_value_heads = model_config.get(
            "num_key_value_heads", model_config["num_attention_heads"]
        )
        if num_key_value_heads is None:
            num_key_value_heads = model_config["num_attention_heads"]
        num_key_value_heads = num_key_value_heads // self.nranks
        self.cache_kvs = {}
        cur_head_dim = head_dim
        if self.use_cache_kv_int4:
            cur_head_dim = head_dim // 2
        for i in range(model_config["num_layers"]):
            self.cache_kvs[f"key_caches_{i}"] = paddle.full(
                shape=[
                    args.max_num_blocks,
                    num_key_value_heads,
                    args.block_size,
                    cur_head_dim,
                ],
                fill_value=0,
                dtype=self.cache_dtype,
            )
            self.cache_kvs[f"value_caches_{i}"] = paddle.full(
                shape=[
                    args.max_num_blocks,
                    num_key_value_heads,
                    args.block_size,
                    cur_head_dim,
                ],
                fill_value=0,
                dtype=self.cache_dtype,
            )

        self.pre_ids = paddle.to_tensor(
            np.zeros((self.beam_batch_size, args.max_dec_len)).astype("int64") - 1
        )

        if self.args.prefix_path:
            prefix_cache = load_prefix_weights(
                self.args.prefix_path,
                batch_size=self.beam_batch_size,
                dtype=self.args.dtype,
            )
            prefix_cache = [
                item.squeeze_(0)
                for item in paddle.split(prefix_cache, len(prefix_cache), axis=0)
            ]
            self.prefix_cache = {}
            self.prefix_cache_length = prefix_cache[0].shape[-2]

            for i in range(len(prefix_cache)):
                self.prefix_cache[f"pre_caches_{i}"] = prefix_cache[i]
        else:
            self.prefix_cache_length = 0

        self._static_guard = _StaticGuard()
        with self._static_guard:
            self.inner_predictor = self.create_predictor(args)

        tmp_position_ids = paddle.arange(args.max_seq_len).reshape((1, -1))
        compression_ratio = model_config.get("compression_ratio", 1)
        rope_theta = model_config.get("rope_theta", 10000.0)
        if self.args.device == "NPU":
            # NOTE(duanyanhui): FLAGS_npu_use_compressed_mask is used for NPU long text scenes like speed-128k.
            # It will be set to True by default in the next version.
            if int(os.getenv("FLAGS_npu_use_compressed_mask", "0")) > 0:
                self.attention_mask = np.tril(np.ones([128, 128])).astype("float32")
            else:
                self.attention_mask = np.tril(
                    np.ones([args.max_seq_len, args.max_seq_len])
                ).astype("float32")
            self.rope_emb = get_rotary_position_embedding(
                tmp_position_ids.repeat_interleave(args.batch_size, axis=0),
                head_dim=head_dim,
                compression_ratio=compression_ratio,
                rope_theta=rope_theta,
            )
        elif self.args.device == "XPU":
            # XPU 计算长文 rope_emb 有精度问题，故使用 CPU 计算
            with CpuGuard():
                tmp_position_ids = paddle.arange(args.max_seq_len).reshape((1, -1))
                self.rope_emb = get_rotary_position_embedding(
                    tmp_position_ids,
                    head_dim=head_dim,
                    compression_ratio=compression_ratio,
                    rope_theta=rope_theta,
                )
            self.rope_emb = paddle.to_tensor(self.rope_emb.numpy())
        else:
            self.rope_emb = get_rotary_position_embedding(
                tmp_position_ids,
                head_dim=head_dim,
                compression_ratio=compression_ratio,
                rope_theta=rope_theta,
            )
        self.input_ids = paddle.full(
            shape=[self.beam_batch_size, args.max_seq_len],
            fill_value=self.tokenizer.pad_token_id,
            dtype="int64",
        )
        self.qkv_weights_lora_A = None
        self.qkv_weights_lora_B = None
        self.linear_weights_lora_A = None
        self.linear_weights_lora_B = None
        self.ffn1_weights_lora_A = None
        self.ffn1_weights_lora_A = None
        self.ffn2_weights_lora_A = None
        self.ffn2_weights_lora_B = None
        if args.lora_num > 0:
            lora_states = load_sharded_checkpoint(args.lora_dir, return_numpy=True)
            lora_config_path = os.path.join(args.lora_dir, "lora_config.json")
            with open(lora_config_path) as config_file:
                lora_config = json.load(config_file)
            lora_scale = lora_config.get("scaling", 1.0)
            qkv_weights_lora_A = []
            qkv_weights_lora_B = []
            linear_weights_lora_A = []
            linear_weights_lora_B = []
            ffn1_weights_lora_A = []
            ffn1_weights_lora_B = []
            ffn2_weights_lora_A = []
            ffn2_weights_lora_B = []
            for i in range(model_config["num_layers"]):
                qkv_weights_lora_A.append(
                    lora_states[
                        f"gpt.decoder.layers.{i}.self_attn.qkv_proj.lora_A"
                    ].transpose((1, 0))
                )
                qkv_weights_lora_B.append(
                    lora_states[
                        f"gpt.decoder.layers.{i}.self_attn.qkv_proj.lora_B"
                    ].transpose((1, 0))
                )
                linear_weights_lora_A.append(
                    lora_states[
                        f"gpt.decoder.layers.{i}.self_attn.out_proj.lora_A"
                    ].transpose((1, 0))
                )
                linear_weights_lora_B.append(
                    lora_states[
                        f"gpt.decoder.layers.{i}.self_attn.out_proj.lora_B"
                    ].transpose((1, 0))
                )
                ffn1_weights_lora_A.append(
                    lora_states[f"gpt.decoder.layers.{i}.linear1.lora_A"].transpose(
                        (1, 0)
                    )
                )
                # for ffn1
                value = lora_states[f"gpt.decoder.layers.{i}.linear1.lora_B"]
                convert_value = np.zeros_like(value)
                out_dim = value.shape[-1]
                convert_value[:, : out_dim // 2] = value[:, ::2]
                convert_value[:, out_dim // 2 :] = value[:, 1::2]
                ffn1_weights_lora_B.append(convert_value.transpose((1, 0)))

                ffn2_weights_lora_A.append(
                    lora_states[f"gpt.decoder.layers.{i}.linear2.lora_A"].transpose(
                        (1, 0)
                    )
                )
                ffn2_weights_lora_B.append(
                    lora_states[f"gpt.decoder.layers.{i}.linear2.lora_B"].transpose(
                        (1, 0)
                    )
                )
            self.qkv_weights_lora_A = (
                paddle.to_tensor(
                    np.expand_dims(np.stack(qkv_weights_lora_A), 0).repeat(
                        args.lora_num, 0
                    ),
                    dtype=args.dtype,
                )
                * lora_scale
            )
            self.qkv_weights_lora_B = paddle.to_tensor(
                np.expand_dims(np.stack(qkv_weights_lora_B), 0).repeat(
                    args.lora_num, 0
                ),
                dtype=args.dtype,
            )
            self.linear_weights_lora_A = (
                paddle.to_tensor(
                    np.expand_dims(np.stack(linear_weights_lora_A), 0).repeat(
                        args.lora_num, 0
                    ),
                    dtype=args.dtype,
                )
                * lora_scale
            )
            self.linear_weights_lora_B = paddle.to_tensor(
                np.expand_dims(np.stack(linear_weights_lora_B), 0).repeat(
                    args.lora_num, 0
                ),
                dtype=args.dtype,
            )
            self.ffn1_weights_lora_A = (
                paddle.to_tensor(
                    np.expand_dims(np.stack(ffn1_weights_lora_A), 0).repeat(
                        args.lora_num, 0
                    ),
                    dtype=args.dtype,
                )
                * lora_scale
            )
            self.ffn1_weights_lora_B = paddle.to_tensor(
                np.expand_dims(np.stack(ffn1_weights_lora_B), 0).repeat(
                    args.lora_num, 0
                ),
                dtype=args.dtype,
            )
            self.ffn2_weights_lora_A = (
                paddle.to_tensor(
                    np.expand_dims(np.stack(ffn2_weights_lora_A), 0).repeat(
                        args.lora_num, 0
                    ),
                    dtype=args.dtype,
                )
                * lora_scale
            )
            self.ffn2_weights_lora_B = paddle.to_tensor(
                np.expand_dims(np.stack(ffn2_weights_lora_B), 0).repeat(
                    args.lora_num, 0
                ),
                dtype=args.dtype,
            )
        if self.enf_gen:
            try:
                from tools.enforce_generation import (
                    build_transformers_prefix_allowed_tokens_fn,
                    generate_enf_gen_context,
                )

                logger.info(
                    f"Loading enforce generation context from {args.enf_gen_input_file}"
                )
                parser = generate_enf_gen_context(
                    args.enf_gen_input_file, args.enf_gen_context_cache_dir
                )
                self.enf_gen_func = build_transformers_prefix_allowed_tokens_fn(
                    self.tokenizer, parser
                )
                self.enf_gen_first_allowed_tokens = self.enf_gen_func(0, [])
                self.enf_gen_first_allowed_tokens_len = len(
                    self.enf_gen_first_allowed_tokens
                )
                logger.info("Load enforce generation context success")
            except BaseException as e:
                logger.error(e)
                logger.error("Load enforce generation context failed, exit")
                sys.exit(-1)
        else:
            self.enf_gen_func = None

        # init speculate components
        if args.speculate_method == "inference_with_reference":
            self.proposer = InferenceWithReferenceProposer(
                args.speculate_max_draft_tokens,
                args.speculate_max_ngram_size,
                args.batch_size,
            )
        elif args.speculate_method == "autoregressive":
            self.proposer = AutogressiveProposer()
        elif args.speculate_method == "hydra":
            hidden_size = self.model_config["hidden_size"]
            vocab_size = self.model_config["vocab_size"]
            self.proposer = HydraProposer(
                hidden_size,
                self.nranks,
                self.rank,
                vocab_size,
                args.speculate_hydra_ckpt,
                args.batch_size,
                args.max_seq_len,
            )
            args.speculate_max_draft_tokens = self.proposer.hydra_num_heads
        elif args.speculate_method == "draft_model":
            self.proposer = DraftModelProposer(
                args, args.speculate_max_draft_tokens, args.batch_size
            )
        elif args.speculate_method == "eagle" or args.speculate_method == "mtp":
            self.proposer = EagleProposer(
                args, args.speculate_max_draft_tokens, args.batch_size
            )
        elif args.speculate_method == "mtp":
            self.proposer = MTPProposer(
                args, args.speculate_max_draft_tokens, args.batch_size
            )
        else:
            self.proposer = None

        if self.args.perf_input_len > 0:
            self.first_token_time_list = []
            self.batch_time_list = []

    def create_predictor(self, args):
        """
        根据给定的参数创建预测器。

        Args:
            args: 包含模型路径和模型前缀的 argparse 模型。

        Returns:
            返回创建好的预测器对象。
        """
        infer_model_path = get_infer_model_path(
            args.model_name_or_path, args.model_prefix
        )
        full_model_path = infer_model_path + ".pdmodel"
        if not os.path.exists(full_model_path):
            # In PIR mode, the file name suffix is ​​json.
            full_model_path = infer_model_path + ".json"

        config = paddle.inference.Config(
            full_model_path, infer_model_path + ".pdiparams"
        )
        # config.enable_memory_optim()
        # NOTE(ZhenyuLi):Open ir_optim while cause ERRORS.
        pir_flag = int(os.environ.get("FLAGS_enable_pir_api", 0))
        if args.device == "GPU":
            config.switch_ir_optim(False)
            device_id = int(os.environ.get("FLAGS_selected_gpus", 0))
            config.enable_use_gpu(100, device_id)
            if args.use_efficientllm and pir_flag == 1:
                config.enable_new_executor()
                config.enable_new_ir()
        elif args.device == "XPU":
            config.enable_xpu()
            device_id = int(os.environ.get("FLAGS_selected_xpus", 0))
            xpu_config = paddle.inference.XpuConfig()
            xpu_config.device_id = device_id
            xpu_config.context_gm_size = int(
                os.getenv("XPU_CONTEXT_GM_SIZE", default=1)
            )
            xpu_config.l3_size = 50327552
            config.set_xpu_config(xpu_config)

            if pir_flag == 1:
                config.enable_new_executor()
                config.enable_new_ir()
                custom_passes = [
                    "fused_multi_transformer_int8_dybatch_xpu_pass",
                    "fused_multi_transformer_dyquant_dybatch_xpu_pass",
                    "top_p_sampling_xpu_pass",
                    "fc_xpu_fuse_v2_pass",
                ]
                config.enable_custom_passes(custom_passes, False)
                # To support fp16, temporarily use fc_xpu_fuse_v2_pass instead of fc_xpu_fuse_pass
                config.delete_pass("fc_xpu_fuse_pass")
                # This Pass does not provide any inference acceleration and may increase XPU memory usage
                config.delete_pass("add_shadow_output_after_dead_parameter_pass")

        elif args.device == "NPU":
            device_id = int(os.environ.get("FLAGS_selected_npus", 0))
            config.enable_custom_device("npu", device_id)
            config.switch_ir_optim(True)
            config.enable_memory_optim()
        elif args.device == "CPU":
            config.disable_gpu()
            config.enable_new_executor(True)
            config.enable_new_ir(True)
            config.disable_mkldnn()

        if self.nranks > 1:
            trainer_endpoints = fleet.worker_endpoints()
            current_endpoint = trainer_endpoints[self.rank]
            if hasattr(config, "dist_config"):
                dist_config = config.dist_config()
                dist_config.set_ranks(self.nranks, self.rank)
                dist_config.set_endpoints(trainer_endpoints, current_endpoint)
                dist_config.enable_dist_model(True)

                dist_config.set_comm_init_config(
                    os.path.join(args.model_name_or_path, "rank_mapping.csv")
                )
                config.set_dist_config(dist_config)

        predictor = paddle.inference.create_predictor(config)

        return predictor

    def pad_batch_data(self, insts):
        """Pad the instances to the max sequence length in batch."""
        seq_len = []
        for i, inst in enumerate(insts):
            length = len(inst)
            seq_len.append(length)
            self.input_ids[i, :length] = np.array(inst)
        return seq_len

    def convert_tokens_to_input_ids(self, input_ids, num_input_tokens):
        """
        Wrapper for derived class
        """
        self.num_input_tokens += num_input_tokens
        return input_ids

    def preprocess(self, dials, extra_infos=None):
        """Pre-process generation inputs."""
        # construct inputs
        if self.args.perf_input_len > 0:
            input_ids = [
                [10002] * self.args.perf_input_len for i in range(self.args.batch_size)
            ]
            num_input_tokens = self.args.batch_size * self.args.perf_input_len
        else:
            input_ids, num_input_tokens = convert_to_input_ids(
                dials,
                self.tokenizer,
                self.args.data_format,
                self.args.append_bos_token,
                self.args.max_seq_len - self.args.max_dec_len,
                extra_infos=extra_infos,
                system_prompt_version=self.model_config.get(
                    "system_prompt_version", "V1"
                ),
                rm_use_cls=self.args.rm_use_cls,
            )

        input_ids = self.convert_tokens_to_input_ids(input_ids, num_input_tokens)

        inputs = {}
        if self.beam_width > 1:
            for i in range(len(input_ids)):
                if i % self.beam_width > 0:
                    input_ids[i] = []
        seq_len = self.pad_batch_data(input_ids)

        seq_lens = [
            0,
        ] * self.beam_batch_size
        inputs["input_ids"] = self.input_ids
        bs = len(dials)
        self.bsz = bs
        seq_lens[:bs] = seq_len
        step_idx = [
            0,
        ] * self.beam_batch_size

        max_sec_len = self.args.max_seq_len
        inputs["block_tables"] = paddle.full(
            shape=[
                self.beam_batch_size,
                (max_sec_len + self.args.block_size - 1) // self.args.block_size,
            ],
            fill_value=-1,
            dtype="int32",
        )
        if self.beam_width > 1:
            for i in range(bs // self.beam_width):
                query_seq_len = seq_len[i * self.beam_width]
                input_block_ids = []
                for j in range(query_seq_len):
                    used_block_id = self.free_list.pop()
                    input_block_ids.append(used_block_id)
                for beam_id in range(self.beam_width):
                    self.used_list[i * self.beam_width + beam_id].extend(
                        input_block_ids
                    )
                    inputs["block_tables"][
                        i * self.beam_width + beam_id, :query_seq_len
                    ] = paddle.to_tensor(input_block_ids, dtype="int32")
                    for k in range(self.args.max_dec_len):
                        used_block_id = self.free_list.pop()
                        self.used_list[i * self.beam_width + beam_id].append(
                            used_block_id
                        )
                        inputs["block_tables"][
                            i * self.beam_width + beam_id, query_seq_len + k
                        ] = used_block_id
        else:
            for i in range(bs):
                for j in range(
                    (seq_len[i] + self.args.max_dec_len + self.args.block_size - 1)
                    // self.args.block_size
                ):
                    used_block_id = self.free_list.pop()
                    self.used_list[i].append(used_block_id)
                    inputs["block_tables"][i, j] = used_block_id
        # inputs["src_mask"] = (self.attention_mask - 1) * 1e6
        if self.args.device == "NPU":
            inputs["attention_mask"] = ((self.attention_mask - 1) * 1e6).astype(
                self.args.dtype
            )
        inputs["top_p"] = (
            np.array(
                [
                    self.args.top_p,
                ]
                * self.beam_batch_size
            )
            .reshape(-1, 1)
            .astype("float32")
        )
        inputs["temperature"] = (
            np.array(
                [
                    self.args.temperature,
                ]
                * self.beam_batch_size
            )
            .reshape(-1, 1)
            .astype("float32")
        )
        inputs["eos_token_id"] = np.array(
            [self.tokenizer.eos_token_id, self.tokenizer.cls_token_id]
        ).astype("int64")
        inputs["penalty_score"] = (
            np.array(
                [
                    self.args.penalty_score,
                ]
                * self.beam_batch_size
            )
            .reshape(-1, 1)
            .astype("float32")
        )
        inputs["frequency_score"] = (
            np.array(
                [
                    self.args.frequency_score,
                ]
                * self.beam_batch_size
            )
            .reshape(-1, 1)
            .astype("float32")
        )
        inputs["presence_score"] = (
            np.array(
                [
                    self.args.presence_score,
                ]
                * self.beam_batch_size
            )
            .reshape(-1, 1)
            .astype("float32")
        )
        inputs["seq_lens_this_time"] = np.array(seq_len).astype("int32").reshape(-1, 1)
        inputs["seq_lens_encoder"] = np.array(seq_lens).astype("int32").reshape(-1, 1)
        inputs["seq_lens_decoder"] = (
            np.array(
                [
                    0,
                ]
                * self.beam_batch_size
            )
            .astype("int32")
            .reshape(-1, 1)
        )
        inputs["step_idx"] = np.array(step_idx).astype("int64").reshape(-1, 1)
        # inputs["tgt_mask"] = self.tgt_generation_mask
        inputs["min_dec_len"] = (
            np.array(
                [
                    0,
                ]
                * self.beam_batch_size
            )
            .astype("int64")
            .reshape(-1, 1)
        )
        inputs["max_dec_len"] = (
            np.array(
                [
                    0,
                ]
                * self.beam_batch_size
            )
            .astype("int64")
            .reshape(-1, 1)
        )
        inputs["not_need_stop"] = np.array([True]).astype("bool")
        inputs["stop_flags"] = (
            np.array(
                [
                    1,
                ]
                * self.beam_batch_size
            )
            .astype("bool")
            .reshape(-1, 1)
        )
        inputs["stop_nums"] = np.array([self.beam_batch_size]).astype("int64")
        inputs["pre_ids"] = self.pre_ids
        inputs["rope_emb"] = self.rope_emb
        inputs["bad_tokens"] = np.array(
            [
                -1,
            ]
        ).astype("int64")

        inputs["next_tokens"] = paddle.full(
            shape=[self.beam_batch_size, 1], fill_value=-1, dtype="int64"
        )
        inputs["is_block_step"] = paddle.full(
            shape=[self.beam_batch_size], fill_value=False, dtype="bool"
        )
        if self.args.use_stop_seqs:
            # NOTE(Zhenyu Li): just for test
            inputs["stop_seqs"] = np.array(
                [
                    [
                        2,
                        -1,
                        -1,
                    ],
                    [
                        51989,
                        18425,
                        4538,
                    ],
                ]
            ).astype("int64")
            inputs["stop_seqs_len"] = np.array([1, 3]).astype("int32")
        if self.args.lora_num > 0:
            # NOTE(Zhenyu Li): just for test
            inputs["w_offsets"] = np.array(
                [i % self.args.lora_num for i in range(self.beam_batch_size)]
            ).astype("int32")
        if self.args.use_system:
            # NOTE(Zhenyu Li): just for test
            inputs["system_ids"] = paddle.full(
                shape=[self.args.batch_size, 1], fill_value=-1, dtype="int32"
            )
            inputs["system_lens"] = paddle.full(
                shape=[self.args.batch_size, 1], fill_value=0, dtype="int32"
            )
            for i in range(bs):
                inputs["system_lens"][i] = 0  # system_len
                inputs["system_ids"][i] = i

        for i in range(bs):
            inputs["min_dec_len"][i : i + 1] = self.args.min_dec_len
            inputs["max_dec_len"][i : i + 1] = self.args.max_dec_len
            inputs["stop_flags"][i : i + 1] = 0

        if self.beam_width > 1:
            inputs["beam_offset"] = -1 * np.ones(
                shape=(
                    self.args.batch_size,
                    self.beam_width,
                    self.args.max_seq_len,
                ),
                dtype="int32",
            )
            inputs["beam_cache_ids"] = np.full(
                (self.beam_batch_size, self.args.max_dec_len), -1, dtype="int32"
            )
            inputs["cum_score"] = np.zeros(
                shape=(self.beam_batch_size, 1), dtype="float32"
            )
            inputs["beam_hyps"] = np.full(
                (self.beam_batch_size, self.args.max_dec_len), -1, "int32"
            )
            inputs["beam_hyps_score"] = np.full((self.beam_batch_size, 1), -1e8).astype(
                "float32"
            )
            inputs["beam_finished"] = (
                np.array(
                    [
                        0,
                    ]
                    * self.beam_batch_size
                )
                .astype("bool")
                .reshape(-1, 1)
            )
            inputs["beam_width"] = (
                np.array([self.beam_width]).astype("int32").reshape(1, 1)
            )
            inputs["beam_group_num"] = (
                np.array([self.beam_group_num]).astype("int32").reshape(1, 1)
            )
            inputs["beam_length_penalty"] = np.full(
                (self.args.batch_size, 1), self.args.beam_length_penalty
            ).astype("float32")
            inputs["beam_diversity_penalty"] = np.full(
                (self.args.batch_size, 1), self.args.beam_diversity_penalty
            ).astype("float32")
            for i in range(bs):
                inputs["beam_hyps"][i, :] = -1
                inputs["cum_score"][i, :] = 0
                inputs["beam_offset"][i // self.beam_width, :, : seq_len[i]] = 0

        if self.enf_gen:
            vocab_size = self.model_config["vocab_size"]
            inputs["enf_gen_status_and_tokens"] = np.full(
                (self.beam_batch_size, vocab_size + 3), -1, dtype="int32"
            )
            inputs["enf_gen_logit_mask"] = np.full(
                (self.beam_batch_size, vocab_size), True, dtype=bool
            )
            self.token_sequence = [[] for _ in range(self.beam_batch_size)]

        if self.args.speculate_method is not None:
            inputs["accept_tokens"] = np.full(
                shape=[
                    self.args.batch_size,
                    self.args.speculate_max_draft_tokens + 1,
                ],
                fill_value=0,
                dtype="int64",
            )
            inputs["accept_num"] = np.full(
                shape=[self.args.batch_size], fill_value=0, dtype="int32"
            )
            inputs["draft_tokens"] = np.full(
                shape=[
                    self.args.batch_size,
                    self.args.speculate_max_draft_tokens + 1,
                ],
                fill_value=0,
                dtype="int64",
            )
            inputs["actual_draft_token_num"] = np.full(
                shape=[self.args.batch_size],
                fill_value=self.args.speculate_max_draft_tokens,
                dtype="int32",
            )
            inputs["input_ids_cpu"] = paddle.full(
                shape=[self.args.batch_size, self.args.max_seq_len],
                fill_value=1,
                dtype="int64",
            ).cpu()

            if self.return_all_hidden_states:
                inputs["all_hidden_states"] = None

            hidden_size = self.model_config["hidden_size"]
            inputs["output_hidden_states"] = paddle.full(
                shape=[
                    self.args.batch_size * (self.args.speculate_max_draft_tokens + 1),
                    hidden_size,
                ],
                fill_value=0.0,
                dtype=self.args.dtype,
            )
            inputs["output_padding_offset"] = paddle.full(
                shape=[
                    self.args.batch_size * (self.args.speculate_max_draft_tokens + 1)
                ],
                fill_value=0,
                dtype="int32",
            )

            for bid in range(bs):
                speculate_update_input_ids_cpu(
                    inputs["input_ids_cpu"],
                    input_ids[bid],
                    bid,
                    self.args.max_seq_len,
                )
                inputs["pre_ids"][bid, 0] = input_ids[bid][-1]

            if self.args.speculate_method == "inference_with_reference":
                for bid in range(bs):
                    self.proposer.update(bid, seq_lens[bid])
        inputs_info = {"inputs": inputs, "real_bs": bs, "seq_len": seq_len}

        return inputs_info

    def postprocess(self, infer_data):
        """Post-process generation outputs."""
        result = []
        for x in infer_data:
            sentence = self.tokenizer.decode(x, skip_special_tokens=True)
            result.append(sentence)
        out_dict = {"result": result}
        return out_dict

    def enf_gen_init(self, model_inputs):
        """"""
        model_inputs["enf_gen_status_and_tokens"][
            :, 2
        ] = self.enf_gen_first_allowed_tokens_len
        model_inputs["enf_gen_status_and_tokens"][
            :, 3 : self.enf_gen_first_allowed_tokens_len + 3
        ] = np.array(self.enf_gen_first_allowed_tokens)

    def enf_gen_step_process(self, model_inputs, batch_size):
        """"""
        if self.beam_width > 1:
            for i in range(batch_size):
                cur_step = model_inputs["step_idx"][i]
                cur_token_sequence = model_inputs["beam_cache_ids"][
                    i, :cur_step
                ].tolist()
                cur_allow_tokens = self.enf_gen_func(i, cur_token_sequence)
                cur_allow_tokens_len = len(cur_allow_tokens)
                model_inputs["enf_gen_status_and_tokens"][i, 2] = cur_allow_tokens_len
                if cur_allow_tokens_len > 0:
                    model_inputs["enf_gen_status_and_tokens"][
                        i, 3 : cur_allow_tokens_len + 3
                    ] = np.array(cur_allow_tokens)
        else:
            step_out_name = self.inner_predictor.get_output_names()[0]
            step_out_list = (
                self.inner_predictor.get_output_handle(step_out_name)
                .copy_to_cpu()
                .tolist()
            )
            for i in range(batch_size):
                token = step_out_list[i][0]
                if token != -1:
                    self.token_sequence[i].append(token)
                    next_step_allowed_tokens = self.enf_gen_func(
                        i, self.token_sequence[i]
                    )
                    next_step_allowed_tokens_len = len(next_step_allowed_tokens)
                    model_inputs["enf_gen_status_and_tokens"][
                        i, 2
                    ] = next_step_allowed_tokens_len
                    if next_step_allowed_tokens_len > 0:
                        model_inputs["enf_gen_status_and_tokens"][
                            i, 3 : next_step_allowed_tokens_len + 3
                        ] = np.array(next_step_allowed_tokens)

    @paddle.no_grad()
    def predict(self, dials: list[list[dict]], extra_infos=None):
        """
        Predict the response for a given input.
        """
        global sum_accept_num, sum_draft_num
        """Get inference sequence."""
        self.inputs_info = self.preprocess(dials, extra_infos=extra_infos)
        model_inputs = self.inputs_info["inputs"]
        for k, v in model_inputs.items():
            if isinstance(v, np.ndarray):
                model_inputs[k] = paddle.to_tensor(v)
            else:
                model_inputs[k] = copy.deepcopy(v)
        if self.args.lora_num > 0:
            model_inputs["qkv_weights_lora_A"] = self.qkv_weights_lora_A
            model_inputs["qkv_weights_lora_B"] = self.qkv_weights_lora_B
            model_inputs["linear_weights_lora_A"] = self.linear_weights_lora_A
            model_inputs["linear_weights_lora_B"] = self.linear_weights_lora_B
            model_inputs["ffn1_weights_lora_A"] = self.ffn1_weights_lora_A
            model_inputs["ffn1_weights_lora_B"] = self.ffn1_weights_lora_B
            model_inputs["ffn2_weights_lora_A"] = self.ffn2_weights_lora_A
            model_inputs["ffn2_weights_lora_B"] = self.ffn2_weights_lora_B
        model_inputs["not_need_stop"] = model_inputs["not_need_stop"].cpu()

        if self.return_all_hidden_states:
            for i in range(self.model_config["num_layers"]):
                model_inputs[f"value_caches_{i}"] = self.cache_kvs[f"value_caches_{i}"]
                model_inputs[f"key_caches_{i}"] = self.cache_kvs[f"key_caches_{i}"]
            model_inputs["pre_ids"] = self.pre_ids
        else:
            for name in self.inner_predictor.get_input_names():
                input_tensor = self.inner_predictor.get_input_handle(name)
                if "pre_caches" in name:
                    input_tensor.share_external_data(self.prefix_cache[name])
                elif "key_caches" in name:
                    input_tensor.share_external_data(self.cache_kvs[name])
                elif "value_caches" in name:
                    input_tensor.share_external_data(self.cache_kvs[name])
                elif "mask" in name or "position" in name:
                    input_tensor.share_external_data(model_inputs[name])
                # elif "pre_id" in name:
                #     input_tensor.share_external_data(self.pre_ids)
                elif "lora" in name:
                    # NOTE(Zhenyu Li): actually used in serving.
                    # input_tensor.share_external_data_by_ptr_name(
                    #     name,
                    #     list(model_inputs[name].shape),
                    #     int(model_inputs[name].dtype),
                    #     int(model_inputs[name].place._type())
                    # )
                    input_tensor.share_external_data(model_inputs[name])
                else:
                    input_tensor.share_external_data(model_inputs[name])

        if self.model_config["architectures"][0] == "ErnieBotRM":
            self.inner_predictor.run()
            self.pre_ids[:] = -1
            self.free_list = list(range(self.args.max_num_blocks))
            self.used_list = [[] for _ in range(self.beam_batch_size)]
            return

        reset_stop_value(model_inputs["not_need_stop"])
        if self.proposer is not None:
            self.proposer.insert_query(self.inputs_info)

        if self.enf_gen:
            self.enf_gen_init(model_inputs)

        if self.args.device == "CPU":
            model_inputs.pop("block_tables")
            model_inputs.pop("rope_emb")

        if self.args.perf_input_len > 0:
            time_begin = time.time()
            first_token_time = None

        while model_inputs["not_need_stop"]:
            if self.proposer is not None:
                self.proposer.run(
                    model_inputs,
                    real_batch_size=self.args.batch_size,
                    seq_lens_this_time=model_inputs["seq_lens_this_time"],
                )
            if self.return_all_hidden_states:
                input_tensors = []
                model_input_names = model_inputs.keys()
                for k in self.inner_predictor.get_input_names():
                    assert k in model_input_names, f"Input {k} must be created."
                    v = model_inputs[k]
                    v.name = k
                    input_tensors.append(v)
                outputs = self.inner_predictor.run(input_tensors)
                model_inputs["all_hidden_states"] = outputs[0]
            else:
                self.inner_predictor.run()
                if self.args.perf_input_len > 0:
                    if first_token_time is None:
                        first_token_time = time.time() - time_begin

            if self.enf_gen:
                self.enf_gen_step_process(model_inputs, len(dials))

        if self.args.perf_input_len > 0:
            self.first_token_time_list.append(first_token_time)
            self.batch_time_list.append(time.time() - time_begin)
        self.pre_ids[:] = -1
        self.free_list = list(range(self.args.max_num_blocks))
        self.used_list = [[] for _ in range(self.beam_batch_size)]
        if self.proposer is not None:
            self.proposer.postprocess(model_inputs)

    def compute_perf_result(self):
        """
        Compute and log the results from the performance data
        """
        self.first_token_time_list = self.first_token_time_list[
            self.args.perf_warm_up_times :
        ]
        self.batch_time_list = self.batch_time_list[self.args.perf_warm_up_times :]
        avg_first_token_time = np.mean(self.first_token_time_list)
        avg_batch_time = np.mean(self.batch_time_list)
        with open(self.args.perf_output_file, "a") as file:
            print(
                f"BSZ: {self.args.batch_size}. Input length: {self.args.perf_input_len}."
                f" Output length: {self.args.max_dec_len}",
                file=file,
            )
            print(
                f"QPS: {1.0 / avg_batch_time * self.args.batch_size}. "
                f"Avg first_token_time: {avg_first_token_time}. "
                f"Avg non-first_token_time {(avg_batch_time - avg_first_token_time) / (self.args.max_dec_len - 1)}. "
                f"Avg batch_time {avg_batch_time}. ",
                file=file,
            )


def my_on_trace_ready(prof):  # 定义回调函数，性能分析器结束采集数据时会被调用
    """
    当性能分析器完成数据采集后被调用的回调函数

    Args:
        prof: 分析器对象，用于获取性能数据

    Returns:
        None
    """
    callback = profiler.export_chrome_tracing(
        "./profiler_demo"
    )  # 创建导出性能数据到profiler_demo文件夹的回调函数
    callback(prof)  # 执行该导出函数
    prof.summary(
        sorted_by=profiler.SortedKeys.GPUTotal
    )  # 打印表单，按GPUTotal排序表单项


p = profiler.Profiler(
    scheduler=[3, 4], on_trace_ready=my_on_trace_ready, timer_only=False
)  # 初始化Profiler对象


def main():
    """ """
    args = setup_args()
    if args.lora_num > 0:
        assert args.lora_dir is not None, "lora_dir should be set when lora_num > 0"

    predictor = Predictor(args)

    # enable_auth = False
    # if enable_auth:
    #     from encryption.auth import auth_product

    #     product_name = auth_product(args.model_name_or_path)

    check_output_dir = False
    if check_output_dir and not check_output(args.model_name_or_path):
        logger.error(
            "args.model_name_or_path is not safe."
        )  # Must before using paddlenlp, paddleslim logger
        sys.exit(-1)

    token_audit = False
    if token_audit:
        token_timer = TokenTimer("infer", args.model_name_or_path, 0)
        token_timer.start()

    # inference
    infer_dials: list[list[dict]] = []
    if args.perf_input_len > 0:
        assert (
            args.perf_repeat_times > 0
        ), f"Repeat time must > 0, but get {args.perf_repeat_times}"
        assert (
            args.perf_input_len + args.max_dec_len <= args.max_seq_len
        ), f"input_len + \
output_len must <= max_seq_len. But get {args.perf_input_len} + {args.max_dec_len} \
= {args.perf_input_len + args.max_dec_len} > {args.max_seq_len}"
        infer_dials = (
            [[{"role": "user", "utterance": "北京天安门在哪？"}]]
            * args.batch_size
            * (args.perf_warm_up_times + args.perf_repeat_times)
        )
        if args.max_dec_len != args.min_dec_len:
            args.min_dec_len = args.max_dec_len
    elif args.input_file is None or not os.path.exists(args.input_file):
        infer_dials = [
            [{"role": "user", "utterance": "北京天安门在哪？"}]
        ] * args.batch_size
    else:
        with open(args.input_file, "r") as fin:
            for i, line in enumerate(fin, start=1):
                if i < args.begin_infer_idx:
                    continue
                try:
                    cur_line = json.loads(line)
                    cur_format = get_infer_data_type(cur_line)
                    if "fc_data" == cur_format:
                        cur_dial = convert_fc_infer_data(cur_line)
                        tools = [
                            tool
                            for item in cur_dial
                            if isinstance(item, list)
                            for tool in item
                        ]
                        system = next(
                            (
                                item["utterance"]
                                for item in cur_dial
                                if "role" in item and item["role"] == "system"
                            ),
                            None,
                        )
                        if tools:
                            cur_dial = insert_fc_instruction(
                                cur_dial, {"tools": tools, "system": system}
                            )
                        infer_dials.append(cur_dial)
                    elif "qa_data" == cur_format:
                        infer_dials.append(cur_line)
                    else:
                        raise ValueError("Unknown data format")
                except Exception as e:
                    logger.warning(
                        f"Failed to parse line {i} from args.input_file (start=1): {e}, skip."
                    )

    beam_width = predictor.beam_width
    if beam_width > 1:
        infer_dials = [dials for dials in infer_dials for _ in range(beam_width)]
    test_case = []
    rank = dist.get_rank()
    beam_batch_size = args.batch_size * beam_width

    args.save_output_file_flush = (
        args.save_output_file_flush // beam_batch_size * beam_batch_size
    )

    try:
        for idx in tqdm(range(0, len(infer_dials), beam_batch_size)):
            batch_dials = infer_dials[idx : idx + beam_batch_size]
            predictor.predict(batch_dials)
            if rank == 0:
                if beam_width > 1:
                    outputs_handle = predictor.inner_predictor.get_output_handle(
                        "beam_hyps"
                    )
                    output_tensor = outputs_handle.copy_to_cpu()
                    output_tensor[output_tensor == -1] = 2
                    predictor.num_output_tokens += (
                        (
                            (output_tensor != predictor.tokenizer.eos_token_id)
                            & (output_tensor != predictor.tokenizer.cls_token_id)
                        )
                        .sum()
                        .item()
                    )
                    outputs = predictor.postprocess(output_tensor)["result"]
                else:
                    outputs = []
                    if predictor.model_config["architectures"][0] == "ErnieBotRM":
                        output_name = predictor.inner_predictor.get_output_names()[0]
                        output_handle = predictor.inner_predictor.get_output_handle(
                            output_name
                        )
                        output_tensor = output_handle.copy_to_cpu()
                        outputs.append(output_tensor.tolist())
                    else:
                        outputs = []
                        while len(outputs) < predictor.bsz:
                            queue_res = predictor.result_queue.get()
                            outputs.append(queue_res[-1])
                            predictor.num_output_tokens += queue_res[1]
                print("result ->", outputs)

                for in_dial, out_resp in zip(batch_dials, outputs):
                    if not isinstance(in_dial, list):
                        in_dial = []
                    conversation_data = in_dial + [
                        {"role": "bot", "utterance": out_resp}
                    ]
                    test_case.append(conversation_data)

                if (
                    args.save_output_file_flush > 0
                    and idx % args.save_output_file_flush == 0
                    and idx > 0
                ):
                    if rank == 0:
                        infer_save_test_case(
                            test_case[idx - args.save_output_file_flush : idx],
                            args.output_file,
                        )
        logger.info(
            f"The task is completed. Total input token: {predictor.num_input_tokens}. \
    Total output token: {predictor.num_output_tokens}"
        )
        if args.perf_input_len > 0 and rank == 0:
            predictor.compute_perf_result()

    except BaseException as e:
        logger.error(e)
        logger.info(
            f"The task is partially successful. Total success input token: \
{predictor.num_input_tokens}. Total success output token: {predictor.num_output_tokens}"
        )
        logger.info(
            f"The task is stopped/aborted on {len(test_case)} / {len(infer_dials)}. \
Please download the completion part ({len(test_case)} / {len(infer_dials)}) from \
{args.output_file} and review it. "
        )

    try:
        token_timer.set_context_tokens(predictor.num_input_tokens)
        token_timer.set_generation_tokens(predictor.num_output_tokens)
        token_timer.check_and_write()
        token_timer.stop()
    except Exception:
        pass

    if rank == 0:
        if args.save_output_file_flush == 0:
            infer_save_test_case(test_case, args.output_file)
        else:
            write_case_idx = (
                len(test_case)
                // args.save_output_file_flush
                * args.save_output_file_flush
            )
            if len(test_case) % args.save_output_file_flush == 0:
                write_case_idx -= args.save_output_file_flush
            infer_save_test_case(test_case[write_case_idx:], args.output_file)
        if beam_width <= 1:
            predictor.read_res_process.terminate()


if __name__ == "__main__":
    main()

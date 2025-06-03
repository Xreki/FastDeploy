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
"""predict_generation under dynamic graph"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import struct
import sys
import time
from glob import glob

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddlenlp.trainer import RuntimeTimer, strtobool
from paddlenlp.utils.log import logger
from tqdm import tqdm

from fastdeploy.inference_args import GenerationPhase
from fastdeploy.model_executor.layers.rotary_embedding import get_rope
from fastdeploy.model_executor.models.data_utils import (convert_fc_infer_data,
                                                         convert_to_input_ids,
                                                         get_infer_data_type,
                                                         insert_fc_instruction)
from fastdeploy.model_executor.models.token_utils import (TokenTimer,
                                                          check_output)
from fastdeploy.model_executor.models.utils import (infer_save_test_case,
                                                    load_prefix_weights,
                                                    load_sharded_checkpoint)

try:
    from fastdeploy.model_executor.ops.gpu import \
        speculate_update_input_ids_cpu
except ImportError:
    pass

from fastdeploy.model_executor.models.speculate_proposers import (
    AutogressiveProposer, EagleProposer, HydraProposer,
    InferenceWithReferenceProposer, MTPProposer)

should_check_python_safety = False
if should_check_python_safety:
    if not os.path.isfile("./utils/token_utils.pyxes"):
        print("Some toolkit files have been modified, check failed.")
        sys.exit(-15)
    from utils.token_utils import TokenTimer, check_python_safe

    check_python_safe()

assert not (os.getenv("EP_DECODER_PERF_TEST", "False") == "True"
            and os.getenv("EP_PREFILL_PERF_TEST", "False")
            == "True"), "Cannot perf PREFILL and DECODER at the same time"

if (os.getenv("EP_DECODER_PERF_TEST", "False") == "True"
        or os.getenv("EP_PREFILL_PERF_TEST", "False") == "True"):
    from paddle import profiler

    perf_step_range = ([3, 5] if os.getenv("EP_PREFILL_PERF_TEST", "False")
                       == "True" else [10, 30])

    def my_on_trace_ready(prof):  # 定义回调函数，性能分析器结束采集数据时会被调用
        """
        当性能分析器完成数据采集后被调用的回调函数

        Args:
            prof: 分析器对象，用于获取性能数据

        Returns:
            None
        """
        callback = profiler.export_chrome_tracing(
            "./profiler_demo")  # 创建导出性能数据到profiler_demo文件夹的回调函数
        callback(prof)  # 执行该导出函数
        prof.summary(
            sorted_by=profiler.SortedKeys.GPUTotal)  # 打印表单，按GPUTotal排序表单项

    p = profiler.Profiler(scheduler=perf_step_range,
                          on_trace_ready=my_on_trace_ready,
                          timer_only=False)  # 初始化Profiler对象


def deserialize_from_file(fp):
    """
    Deserialize data from a file pointer based on the first byte indicating the data type.

    Args:
        fp (file): File pointer to the data file.

    Returns:
        numpy.ndarray: Deserialized data as a NumPy array.

    Raises:
        TypeError: If the first byte of the file does not match any known data type indicator.
    """
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


def get_parser(add_input_output_file: bool = True):
    """
    解析命令行参数。

    Args:
        add_input_output_file (bool, optional): 是否添加输入和输出文件参数，默认值为True。

    Returns:
        argparse.ArgumentParser: 返回一个 ArgumentParser 对象，用于解析命令行参数。

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--data_format",
        type=str,
        default="sft",
        choices=["pt", "sft", "ec2_completion", "ec3_completion"],
        help="The data format.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=8192,
        help="The maximum length of input + output.",
    )
    parser.add_argument("--min_dec_len", type=int, default=1)
    parser.add_argument(
        "--max_dec_len",
        type=int,
        default=2048,
        help="The maximum length of output.",
    )
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument(
        "--get_topk",
        default="False",
        type=strtobool,
        help="whether get topk token/score",
    )
    parser.add_argument("--topk_value", type=int, default=5)
    parser.add_argument("--show_topk", type=int, default=0)
    parser.add_argument("--penalty_score", type=float, default=1)
    parser.add_argument("--frequency_score", type=float, default=0)
    parser.add_argument("--presence_score", type=float, default=0)
    parser.add_argument("--lora_num", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_dir", type=str, default=None)
    parser.add_argument("--scale_dir", type=str, default=None)
    parser.add_argument(
        "--append_bos_token",
        default="True",
        type=strtobool,
        help="whether predict with bos token",
    )
    parser.add_argument("--pre_caches_length", type=int, default=0)
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
        "--outputs_op",
        type=str,
        default="none",
        help=
        "the type of outputs_op, `none` op will return the decoded_idreturn the decoded_ids",
        choices=["none", "save_with_output"],
    )
    parser.add_argument("--predict_model_type", type=str, default="default")
    parser.add_argument("--quant_type", type=str, default="")
    parser.add_argument("--use_cache_kv_int8", type=strtobool, default="False")
    parser.add_argument("--use_cache_kv_int4", type=strtobool, default="False")
    parser.add_argument(
        "--use_ep",
        default="False",
        type=strtobool,
        help="whether use EP Parallel",
    )
    parser.add_argument(
        "--ep_just_for_test",
        default="True",
        type=strtobool,
        help="whether use EP Just For Test",
    )
    parser.add_argument(
        "--generation_phase",
        default=1,
        type=int,
        choices=[1, 2],
        help="1: prefill, 2: decoder",
    )
    parser.add_argument(
        "--use_micro_batch",
        default="False",
        type=strtobool,
        help="whether use micro_batch",
    )
    parser.add_argument(
        "--use_stop_seqs",
        default="False",
        type=strtobool,
        help="whether use stop seqs",
    )
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument(
        "--msg_queue_id",
        type=int,
        default=1,
        help="The msg queue id for get output.",
    )
    parser.add_argument(
        "--use_fake_parameter",
        default="False",
        type=strtobool,
        help="use fake parameter",
    )
    parser.add_argument(
        "--beam_width",
        type=int,
        default=1,
        help="if beam_width > 1, using beam_search decoding strategy",
    )
    parser.add_argument(
        "--beam_group_num",
        type=int,
        default=1,
        help=
        "The num of groups in beam search, if beam_group_num > 1, using group beam search decoding strategy",
    )
    parser.add_argument(
        "--beam_length_penalty",
        type=float,
        default=0.0,
        help="The length penalty for beam search",
    )
    parser.add_argument(
        "--beam_diversity_penalty",
        type=float,
        default=0.0,
        help="The diversity penaly for group beam search",
    )
    parser.add_argument(
        "--enf_gen",
        type=strtobool,
        default="False",
        help="Use enforce generation decoding strategy",
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
        help=
        "If not set, it will default to a folder with the same name as the input file.",
    )
    parser.add_argument(
        "--lazy_load",
        default="False",
        type=strtobool,
        help="whether set state dict for model.",
    )
    parser.add_argument(
        "--use_flash_attention",
        default="True",
        type=strtobool,
        help="whether use flash attention.",
    )

    parser.add_argument(
        "--speculate_method",
        default=None,
        type=str,
        choices=["autoregressive", "inference_with_reference", "hydra", "mtp"],
    )
    parser.add_argument("--speculate_max_draft_tokens", default=1, type=int)
    parser.add_argument("--speculate_max_ngram_size", default=1, type=int)
    parser.add_argument("--speculate_max_candidate_len", default=5, type=int)
    parser.add_argument("--speculate_verify_window", default=2, type=int)
    parser.add_argument("--speculate_hydra_ckpt", default="", type=str)
    parser.add_argument(
        "--return_all_hidden_states",
        default="False",
        type=strtobool,
        help="whether return all hidden states.",
    )
    parser.add_argument("--draft_model_path", default="None", type=str)
    parser.add_argument(
        "--draft_model_type",
        default="WINT8",
        type=str,
        choices=["default", "WINT8", "W8A8C8", "W8A8C16"],
    )

    if add_input_output_file:
        parser.add_argument("--input_file", type=str, required=True)
        parser.add_argument("--output_file",
                            type=str,
                            default="predict.json",
                            required=True)
    parser.add_argument("--save_output_file_flush", type=int, default=10)
    parser.add_argument("--embedding_only", default="False", type=strtobool)
    parser.add_argument(
        "--moe_quant_type",
        default="default",
        type=str,
        choices=[
            "weight_only_int4", "weight_only_int8", "w4a8", "fp8", "default"
        ],
        help="quant type for moe part",
    )
    parser.add_argument(
        "--use_safetensors",
        default="False",
        type=strtobool,
        help="using safetensors",
    )
    return parser


def setup_args():
    """Setup inference arguments."""
    parser = get_parser()
    args = parser.parse_args()
    if args.beam_width > 1 and args.block_size != 1:
        logger.warning(
            f"Beam Search only support block size 1. Using block_size=1 instead {args.block_size}"
        )
        args.block_size = 1
    if args.beam_width > 1:
        args.max_num_blocks = args.batch_size * (
            (args.max_seq_len + args.block_size - 1) //
            args.block_size) + args.batch_size * (args.beam_width - 1) * (
                (args.max_dec_len + args.block_size - 1) // args.block_size)
    else:
        args.max_num_blocks = (args.batch_size * args.beam_width * (
            (args.max_seq_len + args.block_size - 1) // args.block_size))
    assert args.msg_queue_id >= 0
    return args


class Predictor:
    """
    Predictor
    """

    def __init__(self, args, tokenizer=None, model=None, **kwargs):
        """
        Initialization function for Predictor.
        """
        self.runtime_timer = RuntimeTimer("Predictor")
        self.num_input_tokens = 0
        self.num_output_tokens = 0
        self.use_beam_search = False
        self.show_topk = args.show_topk
        if args is None:
            self.tokenizer = tokenizer
            self.tokenizer.padding_side = "left"
            self.model = model
        else:
            self.args = args
            if os.getenv("EP_DECODER_PERF_TEST", "False") == "True":
                self.args.min_dec_len = self.args.max_dec_len
            dp_degree = getattr(self.args, "data_parallel_degree", 1)
            tp_degree = dist.get_world_size() // dp_degree
            tp_rank = dist.get_rank()

            if dist.get_world_size() > 1:
                strategy = fleet.DistributedStrategy()
                strategy.hybrid_configs = {
                    "dp_degree": dp_degree,
                    "mp_degree": tp_degree,
                    "pp_degree": 1,
                    "sharding_degree": 1,
                }
                fleet.init(is_collective=True, strategy=strategy)
                hcg = fleet.get_hybrid_communicate_group()
                tp_rank = hcg.get_model_parallel_rank()
            else:
                strategy = fleet.DistributedStrategy()
                strategy.hybrid_configs = {
                    "dp_degree": dp_degree,
                    "mp_degree": tp_degree,
                    "pp_degree": 1,
                }
                fleet.init(is_collective=True, strategy=strategy)

            triton_dir = f"/tmp/haha/triton_ops_rank_{dist.get_rank()}"
            os.environ["TRITON_KERNEL_CACHE_DIR"] = triton_dir
            self.tp_rank = tp_rank
            self.tp_degree = tp_degree

            self.beam_batch_size = args.batch_size * args.beam_width
            self.use_beam_search = True if args.beam_width > 1 else False
            self.use_system = args.use_system
            if not self.use_beam_search:
                self.result_queue = mp.Queue()
                from fastdeploy.model_executor.models.utils import (
                    MAX_BSZ, MAX_DRAFT_TOKENS)

                if self.args.use_ep and (not self.args.ep_just_for_test):
                    self.args.msg_queue_id = self.tp_rank

                if args.speculate_method is not None:
                    from fastdeploy.model_executor.models.utils import \
                        speculate_read_res

                    output_tensor_max_shape = [
                        MAX_BSZ * MAX_DRAFT_TOKENS + MAX_BSZ + 2
                    ]
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
                    from fastdeploy.model_executor.models.utils import read_res

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

                if self.tp_rank == 0 or (self.args.use_ep and
                                         (not self.args.ep_just_for_test)):
                    self.read_res_process.start()

            self.pre_caches_length = 0
            if self.args.prefix_path:
                self.prefix_cache = load_prefix_weights(
                    self.args.prefix_path,
                    batch_size=self.args.batch_size,
                    dtype=self.args.dtype,
                )
                self.pre_caches = [
                    item.squeeze_(0) for item in paddle.split(
                        self.prefix_cache,
                        self.prefix_cache.shape[0],
                        axis=0,
                    )
                ]
                self.pre_caches_length = self.pre_caches[0].shape[-2]

            # 2. build model
            paddle.set_default_dtype(self.args.dtype)
            config_path = os.path.join(args.model_name_or_path, "config.json")
            with open(config_path) as model_config_file:
                model_config = json.load(model_config_file)

            if "quant_type" in model_config:

                if args.predict_model_type != model_config["quant_type"]:
                    if (model_config["quant_type"] == "W8A8C8"
                            and args.predict_model_type == "W8A8C16"):
                        # cache kv int8 is optional. W8A8C8 can support
                        # exporting both W8A8C8 and W8A8C16 inference models.
                        pass
                    else:
                        logger.debug(
                            f"The arg export_model_type {args.predict_model_type} \
                            != model_config['quant_type'] {model_config['quant_type']}. \
                            {model_config['quant_type']} will be used.")
                        args.predict_model_type = model_config["quant_type"]

            if args.use_cache_kv_int8 and "C8" not in args.predict_model_type:
                args.predict_model_type += "C8"
            elif args.use_cache_kv_int4 and "C4" not in args.predict_model_type:
                args.predict_model_type += "C4"
            use_cache_kv_int8 = True if "C8" in args.predict_model_type else False
            use_cache_kv_int4 = True if "C4" in args.predict_model_type else False
            use_cache_kv_fp8 = True if "Cfp8" in args.predict_model_type else False

            use_beam_search = True if args.beam_width > 1 else False
            # TODO: 动态图会cuda error 700，后续再修复

            from fastdeploy.model_executor.models.export_model import \
                build_stream_line_model

            config, tokenizer, model = build_stream_line_model(
                config_path,
                args.model_name_or_path,
                args.dtype,
                block_size=args.block_size,
                max_len=args.max_seq_len,
                stage_flag=f"msgid-{self.args.msg_queue_id} predict",
                msg_queue_id=self.args.msg_queue_id,
                min_dec_len=args.min_dec_len,
                max_dec_len=args.max_dec_len,
                pre_caches_length=args.pre_caches_length,
                temperature=args.temperature,  # not use
                top_k=args.top_k,  # not use
                top_p=args.top_p,  # not use
                export_model_type=args.predict_model_type,
                use_fake_parameter=args.use_fake_parameter,
                use_stop_seqs=args.use_stop_seqs,
                pad_vocab=False,
                use_beam_search=use_beam_search,
                speculate_method=args.speculate_method,
                speculate_max_draft_token_num=args.speculate_max_draft_tokens,
                speculate_max_candidate_len=args.speculate_max_candidate_len,
                speculate_verify_window=args.speculate_verify_window,
                return_all_hidden_states=args.return_all_hidden_states,
                moe_quant_type=args.moe_quant_type,
                use_ep=args.use_ep,
                ep_just_for_test=args.ep_just_for_test,
                generation_phase=GenerationPhase(args.generation_phase),
                use_micro_batch=args.use_micro_batch,
                scale_dir=args.scale_dir,
                use_safetensors=args.use_safetensors,
            )

            model.eval()
            self.model = model
            self.model.config = config
            self.model_config = config
            self.tokenizer = tokenizer

            # enforce_generation mode
            if args.enf_gen:
                try:
                    from tools.enforce_generation import (
                        build_transformers_prefix_allowed_tokens_fn,
                        generate_enf_gen_context)

                    logger.info(
                        f"Loading enforce generation context from {args.enf_gen_input_file}"
                    )
                    parser = generate_enf_gen_context(
                        args.enf_gen_input_file,
                        args.enf_gen_context_cache_dir,
                    )
                    self.enf_gen_func = build_transformers_prefix_allowed_tokens_fn(
                        self.tokenizer, parser)
                    self.enf_gen_first_allowed_tokens = self.enf_gen_func(
                        0, [])
                    self.enf_gen_first_allowed_tokens_len = len(
                        self.enf_gen_first_allowed_tokens)
                    logger.info("Load enforce generation context success")
                except BaseException as e:
                    self.enf_gen_func = None
                    logger.error(e)
                    logger.error(
                        "Load enforce generation context failed, falling back to normal mode"
                    )
            else:
                self.enf_gen_func = None

            # init cache_kvs
            num_layers = self.model_config.get("num_layers",
                                               None) or self.model_config.get(
                                                   "num_hidden_layers", None)
            self.cache_kvs = []
            self.free_list = list(range(args.max_num_blocks))
            self.used_list = [[] for _ in range(self.beam_batch_size)]
            head_dim = (self.model_config["hidden_size"] //
                        self.model_config["num_attention_heads"])
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
            self.input_ids = paddle.full(
                shape=[self.beam_batch_size, args.max_seq_len],
                fill_value=self.tokenizer.pad_token_id,
                dtype="int64",
            )
            num_key_value_heads = self.model_config.get(
                "num_key_value_heads",
                self.model_config["num_attention_heads"],
            )
            if num_key_value_heads is None:
                num_key_value_heads = self.model_config["num_attention_heads"]
            if args.use_ep:
                num_key_value_heads = num_key_value_heads
            else:
                num_key_value_heads = num_key_value_heads // self.tp_degree
            cur_head_dim = head_dim
            if use_cache_kv_int4:
                cur_head_dim = head_dim // 2
            if use_cache_kv_int8 or use_cache_kv_int4 or use_cache_kv_fp8:
                cache_type = "uint8"
            else:
                cache_type = args.dtype

            print("cache_type", cache_type)
            for i in range(num_layers):
                for _ in ["k", "v"]:
                    self.cache_kvs.append(
                        paddle.zeros(
                            [
                                args.max_num_blocks,
                                num_key_value_heads,
                                args.block_size,
                                cur_head_dim,
                            ],
                            dtype=cache_type,
                        ))
                logger.debug(
                    f"Layer {i} memory {paddle.device.cuda.memory_allocated() / 1024 / 1024 / 1024} GB"
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
                lora_states = load_sharded_checkpoint(args.lora_dir,
                                                      return_numpy=True)
                lora_config_path = os.path.join(args.lora_dir,
                                                "lora_config.json")
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
                for i in range(num_layers):
                    qkv_weights_lora_A.append(
                        lora_states[
                            f"ernie.decoder.layers.{i}.self_attn.qkv_proj.lora_A"
                        ].transpose((1, 0))
                    )
                    qkv_weights_lora_B.append(
                        lora_states[
                            f"ernie.decoder.layers.{i}.self_attn.qkv_proj.lora_B"
                        ].transpose((1, 0))
                    )
                    linear_weights_lora_A.append(
                        lora_states[
                            f"ernie.decoder.layers.{i}.self_attn.out_proj.lora_A"
                        ].transpose((1, 0))
                    )
                    linear_weights_lora_B.append(
                        lora_states[
                            f"ernie.decoder.layers.{i}.self_attn.out_proj.lora_B"
                        ].transpose((1, 0))
                    )
                    ffn1_weights_lora_A.append(
                        lora_states[f"ernie.decoder.layers.{i}.linear1.lora_A"].transpose(
                            (1, 0)
                        )
                    )
                    # for ffn1
                    value = lora_states[f"ernie.decoder.layers.{i}.linear1.lora_B"]
                    convert_value = np.zeros_like(value)
                    out_dim = value.shape[-1]
                    convert_value[:, :out_dim // 2] = value[:, ::2]
                    convert_value[:, out_dim // 2:] = value[:, 1::2]
                    ffn1_weights_lora_B.append(convert_value.transpose((1, 0)))

                    ffn2_weights_lora_A.append(
                        lora_states[f"ernie.decoder.layers.{i}.linear2.lora_A"].transpose(
                            (1, 0)
                        )
                    )
                    ffn2_weights_lora_B.append(
                        lora_states[f"ernie.decoder.layers.{i}.linear2.lora_B"].transpose(
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
                    np.expand_dims(np.stack(qkv_weights_lora_B),
                                   0).repeat(args.lora_num, 0),
                    dtype=args.dtype,
                )
                self.linear_weights_lora_A = (paddle.to_tensor(
                    np.expand_dims(np.stack(linear_weights_lora_A), 0).repeat(
                        args.lora_num, 0),
                    dtype=args.dtype,
                ) * lora_scale)
                self.linear_weights_lora_B = paddle.to_tensor(
                    np.expand_dims(np.stack(linear_weights_lora_B),
                                   0).repeat(args.lora_num, 0),
                    dtype=args.dtype,
                )
                self.ffn1_weights_lora_A = (paddle.to_tensor(
                    np.expand_dims(np.stack(ffn1_weights_lora_A), 0).repeat(
                        args.lora_num, 0),
                    dtype=args.dtype,
                ) * lora_scale)
                self.ffn1_weights_lora_B = paddle.to_tensor(
                    np.expand_dims(np.stack(ffn1_weights_lora_B),
                                   0).repeat(args.lora_num, 0),
                    dtype=args.dtype,
                )
                self.ffn2_weights_lora_A = (paddle.to_tensor(
                    np.expand_dims(np.stack(ffn2_weights_lora_A), 0).repeat(
                        args.lora_num, 0),
                    dtype=args.dtype,
                ) * lora_scale)
                self.ffn2_weights_lora_B = paddle.to_tensor(
                    np.expand_dims(np.stack(ffn2_weights_lora_B),
                                   0).repeat(args.lora_num, 0),
                    dtype=args.dtype,
                )

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
                    self.tp_degree,
                    self.tp_rank,
                    vocab_size,
                    args.speculate_hydra_ckpt,
                    args.batch_size,
                    args.max_seq_len,
                )
                args.speculate_max_draft_tokens = self.proposer.hydra_num_heads
            elif args.speculate_method == "eagle":
                self.proposer = EagleProposer(args,
                                              args.speculate_max_draft_tokens,
                                              args.batch_size)
            elif args.speculate_method == "mtp":
                self.proposer = MTPProposer(args,
                                            args.speculate_max_draft_tokens,
                                            args.batch_size)
            else:
                self.proposer = None

    def preprocess(self, dials: list[list[dict]], extra_infos=None):
        """Pre-process generation inputs."""
        # construct inputs
        system_prompt_version = self.model_config.get("system_prompt_version",
                                                      "V1")

        input_ids, num_input_tokens = convert_to_input_ids(
            dials,
            self.tokenizer,
            self.args.data_format,
            self.args.append_bos_token,
            self.args.max_seq_len - self.args.max_dec_len,
            extra_infos=extra_infos,
            system_prompt_version=system_prompt_version,
        )

        if (os.getenv("EP_DECODER_PERF_TEST", "False") == "True"
                or os.getenv("EP_PREFILL_PERF_TEST", "False") == "True"):
            test_len = 4383
            # is_encoder = True
            input_ids_new = []
            num_input_tokens_new = 0
            for i in range(self.args.batch_size):
                old_input_ids = input_ids[i]
                len_old_input_ids = len(old_input_ids)
                input_ids_new.append([])
                for j in range(
                    (test_len + len_old_input_ids - 1) // len_old_input_ids):
                    input_ids_new[-1].extend(old_input_ids)
                input_ids_new[-1] = input_ids_new[-1][:test_len]
                num_input_tokens_new += test_len
            input_ids = input_ids_new
            num_input_tokens = num_input_tokens_new
            # print("input_ids: ", input_ids)
            print("num_input_tokens: ", num_input_tokens)
        self.num_input_tokens += num_input_tokens

        if self.args.beam_width > 1:
            for i in range(len(input_ids)):
                if i % self.args.beam_width > 0:
                    input_ids[i] = []
        inputs = {}
        seq_len = self.pad_batch_data(input_ids)
        seq_lens = [0] * self.beam_batch_size
        inputs["input_ids"] = self.input_ids
        bs = len(dials)
        self.bsz = bs
        seq_lens[:bs] = seq_len
        step_idx = [0] * self.beam_batch_size

        max_sec_len = self.args.max_seq_len
        inputs["block_tables"] = paddle.full(
            shape=[
                self.beam_batch_size,
                (max_sec_len + self.args.block_size - 1) //
                self.args.block_size,
            ],
            fill_value=-1,
            dtype="int32",
        )

        if self.args.beam_width > 1:
            for i in range(bs // self.args.beam_width):
                query_seq_len = seq_len[i * self.args.beam_width]
                if query_seq_len + self.args.max_dec_len > max_sec_len:
                    raise ValueError(f"input_len({query_seq_len}) + \
                        max_dec_len({self.args.max_dec_len}) > max_seq_len({max_sec_len})"
                                     )
                input_block_ids = []
                for j in range(query_seq_len):
                    used_block_id = self.free_list.pop()
                    input_block_ids.append(used_block_id)
                for beam_id in range(self.args.beam_width):
                    self.used_list[i * self.args.beam_width +
                                   beam_id].extend(input_block_ids)
                    inputs["block_tables"][
                        i * self.args.beam_width +
                        beam_id, :query_seq_len] = paddle.to_tensor(
                            input_block_ids, dtype="int32")
                    for k in range(self.args.max_dec_len):
                        used_block_id = self.free_list.pop()
                        self.used_list[i * self.args.beam_width +
                                       beam_id].append(used_block_id)
                        inputs["block_tables"][
                            i * self.args.beam_width + beam_id,
                            query_seq_len + k,
                        ] = used_block_id
        else:
            for i in range(bs):
                real_len = seq_len[i] + self.args.max_dec_len
                if real_len > max_sec_len:
                    raise ValueError(f"input_len({seq_len[i]}) + \
                        max_dec_len({self.args.max_dec_len}) > max_seq_len({max_sec_len})"
                                     )
                for j in range((real_len + self.args.block_size - 1) //
                               self.args.block_size):
                    used_block_id = self.free_list.pop()
                    self.used_list[i].append(used_block_id)
                    inputs["block_tables"][i, j] = used_block_id

        def get_full_array(data, dtype="float32"):
            return np.array([data] * self.beam_batch_size).reshape(
                -1, 1).astype(dtype)

        inputs["top_p"] = get_full_array(self.args.top_p)
        inputs["temperature"] = get_full_array(self.args.temperature)

        inputs["eos_token_id"] = np.array(
            [self.tokenizer.eos_token_id,
             self.tokenizer.cls_token_id]).astype("int64")

        inputs["penalty_score"] = get_full_array(self.args.penalty_score)
        inputs["frequency_score"] = get_full_array(self.args.frequency_score)
        inputs["presence_score"] = get_full_array(self.args.presence_score)

        inputs["seq_lens_this_time"] = np.array(seq_len).astype(
            "int32").reshape(-1, 1)
        inputs["seq_lens_encoder"] = np.array(seq_lens).astype(
            "int32").reshape(-1, 1)
        inputs["seq_lens_decoder"] = (np.array(
            [0] * self.beam_batch_size).astype("int32").reshape(-1, 1))
        if os.getenv("EP_PREFILL_PERF_TEST", "False") == "True":
            inputs["seq_lens_this_time"][:] = test_len
            inputs["seq_lens_encoder"][:] = test_len
            inputs["seq_lens_decoder"][:] = 0
        elif os.getenv("EP_DECODER_PERF_TEST", "False") == "True":
            inputs["seq_lens_this_time"][:] = 1
            inputs["seq_lens_encoder"][:] = 0
            inputs["seq_lens_decoder"][:] = test_len

        inputs["step_idx"] = np.array(step_idx).astype("int64").reshape(-1, 1)
        inputs["min_dec_len"] = get_full_array(0, dtype="int64")
        inputs["max_dec_len"] = get_full_array(0, dtype="int64")
        inputs["not_need_stop"] = np.array([True]).astype("bool")
        inputs["stop_flags"] = get_full_array(1, dtype="bool")
        inputs["stop_nums"] = np.array([self.beam_batch_size]).astype("int64")
        inputs["pre_ids"] = self.pre_ids
        inputs["rope_emb"] = self.rope_emb
        inputs["bad_tokens"] = np.array([
            -1,
        ]).astype("int64")
        if self.args.use_stop_seqs:
            # NOTE(Zhenyu Li): just for test
            inputs["stop_seqs"] = np.array([
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
            ]).astype("int64")
            inputs["stop_seqs_len"] = np.array([1, 3]).astype("int32")
        if self.args.lora_num > 0:
            # NOTE(Zhenyu Li): just for test
            inputs["w_offsets"] = np.array([
                i % self.args.lora_num for i in range(self.beam_batch_size)
            ]).astype("int32")
        if self.use_system:
            # NOTE(Zhenyu Li): just for test
            inputs["system_ids"] = paddle.full(
                shape=[self.args.batch_size, 1],
                fill_value=-1,
                dtype="int32",
            )
            inputs["system_lens"] = paddle.full(
                shape=[self.args.batch_size, 1], fill_value=0, dtype="int32")
            for i in range(bs):
                inputs["system_lens"][i] = 0  # system_len
                inputs["system_ids"][i] = i

        inputs["next_tokens"] = paddle.full(shape=[self.beam_batch_size, 1],
                                            fill_value=-1,
                                            dtype="int64")
        inputs["is_block_step"] = paddle.full(shape=[self.beam_batch_size],
                                              fill_value=False,
                                              dtype="bool")
        for i in range(bs):
            inputs["min_dec_len"][i:i + 1] = self.args.min_dec_len
            inputs["max_dec_len"][i:i + 1] = self.args.max_dec_len
            inputs["stop_flags"][i:i + 1] = 0

        if self.use_beam_search:
            inputs["beam_offset"] = -1 * np.ones(
                shape=(
                    self.args.batch_size,
                    self.args.beam_width,
                    self.args.max_seq_len,
                ),
                dtype="int32",
            )
            inputs["beam_cache_ids"] = np.full(
                (self.beam_batch_size, self.args.max_dec_len),
                -1,
                dtype="int32",
            )
            inputs["cum_score"] = np.zeros(shape=(self.beam_batch_size, 1),
                                           dtype="float32")
            inputs["beam_hyps"] = np.full(
                (self.beam_batch_size, self.args.max_dec_len), -1, "int32")
            inputs["beam_hyps_score"] = np.full((self.beam_batch_size, 1),
                                                -1e8).astype("float32")
            inputs["beam_finished"] = (np.array([
                0,
            ] * self.beam_batch_size).astype("bool").reshape(-1, 1))
            inputs["beam_width"] = (np.array([self.args.beam_width
                                              ]).astype("int32").reshape(1, 1))
            inputs["beam_group_num"] = (np.array(
                [self.args.beam_group_num]).astype("int32").reshape(1, 1))
            inputs["beam_length_penalty"] = np.full(
                (self.args.batch_size, 1),
                self.args.beam_length_penalty).astype("float32")
            inputs["beam_diversity_penalty"] = np.full(
                (self.args.batch_size, 1),
                self.args.beam_diversity_penalty).astype("float32")
            for i in range(bs):
                inputs["beam_hyps"][i, :] = -1
                inputs["cum_score"][i, :] = 0
                inputs["beam_offset"][i //
                                      self.args.beam_width, :, :seq_len[i]] = 0

        if self.args.enf_gen:
            vocab_size = self.model_config["vocab_size"]
            inputs["enf_gen_status_and_tokens"] = np.full(
                (self.beam_batch_size, vocab_size + 3), -1, dtype="int32")
            inputs["enf_gen_logit_mask"] = np.full(
                (self.beam_batch_size, vocab_size), True, dtype=bool)
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
            inputs["accept_num"] = np.full(shape=[self.args.batch_size],
                                           fill_value=0,
                                           dtype="int32")
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

            hidden_size = self.model_config["hidden_size"]
            inputs["output_hidden_states"] = paddle.full(
                shape=[
                    self.args.batch_size *
                    (self.args.speculate_max_draft_tokens + 1),
                    hidden_size,
                ],
                fill_value=0.0,
                dtype=self.args.dtype,
            )
            inputs["output_padding_offset"] = paddle.full(
                shape=[
                    self.args.batch_size *
                    (self.args.speculate_max_draft_tokens + 1)
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

        self.inputs_info = {
            "inputs": inputs,
            "real_bs": bs,
            "seq_len": seq_len
        }
        return inputs

    def pad_batch_data(self, insts):
        """Pad the instances to the max sequence length in batch."""
        seq_len = []
        for i, inst in enumerate(insts):
            length = len(inst)
            seq_len.append(length)
            self.input_ids[i, :length] = np.array(inst)
        return seq_len

    def postprocess(self, infer_data):
        """"""
        result = []
        for x in infer_data.tolist():
            res = self.tokenizer.decode(x, skip_special_tokens=True)
            res = res.strip("\n")
            result.append(res)
        out_dict = {"result": result}
        return out_dict

    def enf_gen_init(self, model_inputs):
        """"""
        model_inputs[
            "enf_gen_status_and_tokens"][:,
                                         2] = self.enf_gen_first_allowed_tokens_len
        model_inputs[
            "enf_gen_status_and_tokens"][:, 3:self.
                                         enf_gen_first_allowed_tokens_len +
                                         3] = np.array(
                                             self.enf_gen_first_allowed_tokens)

    def enf_gen_step_process(self, step_out, model_inputs):
        """"""
        real_batch_size = len(step_out)
        if self.use_beam_search:
            for i in range(real_batch_size):
                cur_step = model_inputs["step_idx"][i]
                cur_token_sequence = model_inputs["beam_cache_ids"][
                    i, :cur_step].tolist()
                cur_allow_tokens = self.enf_gen_func(i, cur_token_sequence)
                cur_allow_tokens_len = len(cur_allow_tokens)
                model_inputs["enf_gen_status_and_tokens"][
                    i, 2] = cur_allow_tokens_len
                if cur_allow_tokens_len > 0:
                    model_inputs["enf_gen_status_and_tokens"][
                        i, 3:cur_allow_tokens_len +
                        3] = np.array(cur_allow_tokens)
        else:
            step_out_list = step_out.tolist()
            for i in range(real_batch_size):
                token = step_out_list[i][0]
                if token != -1:
                    self.token_sequence[i].append(token)
                    next_step_allowed_tokens = self.enf_gen_func(
                        i, self.token_sequence[i])
                    next_step_allowed_tokens_len = len(
                        next_step_allowed_tokens)
                    model_inputs["enf_gen_status_and_tokens"][
                        i, 2] = next_step_allowed_tokens_len
                    if next_step_allowed_tokens_len > 0:
                        model_inputs["enf_gen_status_and_tokens"][
                            i, 3:next_step_allowed_tokens_len +
                            3] = np.array(next_step_allowed_tokens)

    def infer(self, inputs: dict) -> list[list[int]]:
        """
        Perform the prediction process of the model, where the input is a dictionary-type \
        object containing the input data required by the model.

        Args:
            inputs (dict): Contains the input data required by the model, including both \
                mandatory and optional items. For details, please refer to the model's documentation.

        Returns:
            list[list[int]]: A list-type object where each element is a list representing \
                one or more results generated during each prediction process.

        Raises:
            ValueError: This error is raised when both show_topk and use_beam_search are set to True.
        """
        model_inputs = {}
        for k, v in inputs.items():
            if paddle.is_tensor(v):
                model_inputs[k] = v
            else:
                model_inputs[k] = paddle.to_tensor(v)

        model_inputs["caches"] = self.cache_kvs
        if self.args.lora_num > 0:
            model_inputs["qkv_weights_lora_A"] = self.qkv_weights_lora_A
            model_inputs["qkv_weights_lora_B"] = self.qkv_weights_lora_B
            model_inputs["linear_weights_lora_A"] = self.linear_weights_lora_A
            model_inputs["linear_weights_lora_B"] = self.linear_weights_lora_B
            model_inputs["ffn1_weights_lora_A"] = self.ffn1_weights_lora_A
            model_inputs["ffn1_weights_lora_B"] = self.ffn1_weights_lora_B
            model_inputs["ffn2_weights_lora_A"] = self.ffn2_weights_lora_A
            model_inputs["ffn2_weights_lora_B"] = self.ffn2_weights_lora_B
        if self.args.prefix_path:
            model_inputs["pre_caches"] = self.pre_caches
        inputs["not_need_stop"][0] = True
        if self.proposer is not None:
            self.proposer.insert_query(self.inputs_info)
        if self.args.enf_gen:
            self.enf_gen_init(model_inputs)
        out_res = []
        with paddle.no_grad():
            if self.args.use_ep and (not self.args.ep_just_for_test):
                while True:
                    if self.proposer is not None:
                        self.proposer.run(
                            model_inputs,
                            real_batch_size=self.args.batch_size,
                            seq_lens_this_time=model_inputs[
                                "seq_lens_this_time"],
                        )
                    hiddden_states = self.model(**model_inputs)
                    logits = self.model.compute_logits(hiddden_states)
                    out = self.model.sample(logits, **model_inputs)
                    if self.show_topk:
                        out_res.append(out)
                    if self.args.enf_gen:
                        self.enf_gen_step_process(out, model_inputs)
            else:
                if os.getenv("EP_DECODER_PERF_TEST", "False") == "True":
                    # p.start()
                    step = 0
                while model_inputs["not_need_stop"]:
                    if os.getenv("EP_DECODER_PERF_TEST", "False") == "True":
                        s = time.time()
                    if self.proposer is not None:
                        self.proposer.run(
                            model_inputs,
                            real_batch_size=self.args.batch_size,
                            seq_lens_this_time=model_inputs[
                                "seq_lens_this_time"],
                        )
                    hiddden_states = self.model(**model_inputs)
                    logits = self.model.compute_logits(hiddden_states)
                    out = self.model.sample(logits, **model_inputs)
                    if self.args.return_all_hidden_states:
                        model_inputs["all_hidden_states"] = out[0]
                    if self.show_topk:
                        out_res.append(out)
                    if self.args.enf_gen:
                        self.enf_gen_step_process(out, model_inputs)

                    if os.getenv("EP_DECODER_PERF_TEST", "False") == "True":
                        print(f"step {step} elapse {time.time() - s} s")
                        step += 1
                #         p.step()
                # if os.getenv("EP_DECODER_PERF_TEST", "False") == "True":
                #     p.stop()
                #     exit(0)

        self.pre_ids[:] = -1
        self.free_list = list(range(self.args.max_num_blocks))
        self.used_list = [[] for _ in range(self.beam_batch_size)]
        if self.proposer is not None:
            self.proposer.postprocess(model_inputs)
        if self.show_topk and self.use_beam_search:
            raise ValueError(
                "`show_topk` and `use_beam_search` cannot be set to True at the same time!"
            )

        if self.use_beam_search:
            return model_inputs["beam_hyps"]

        if self.show_topk > 0:
            topk_tokens_all = []
            for i in range(len(out_res)):
                (next_tokens, next_probs), (topk_tokens,
                                            topk_probs) = out_res[i]
                tmp_res = {}
                tmp_res["token"] = next_tokens
                tmp_res["prob"] = next_probs
                tmp_res["topk_tokens"] = []
                for token, prob in zip(topk_tokens, topk_probs):
                    tmp_topk_res = {"token": token, "prob": prob}
                    tmp_res["topk_tokens"].append(tmp_topk_res)
                topk_tokens_all.append(tmp_res)
            res = {}
            res["topk_tokens_all"] = topk_tokens_all
            return res

    def get_result(self):
        """
        get_result function.
        """
        res = []
        files = glob(os.path.join("./real_time_save.*"))
        for j in range(1, len(files)):
            filename = f"./real_time_save.temp_ids_rank_0_step_{j}"
            if not os.path.exists(filename):
                break
            fp = open(filename, "rb+")
            fp.read(1)
            data_list = deserialize_from_file(fp)
            res.append(np.array(data_list).reshape(-1, 1))
        res = np.concatenate(res, axis=1)

        sentences = self.tokenizer.batch_decode(res.tolist(),
                                                skip_special_tokens=True)
        os.system("rm -f ./real_time_save.temp_ids_rank_*")
        return {"result": sentences}

    @paddle.no_grad()
    def predict(self, batch_dials: list[list[dict]], extra_infos=None):
        """Run batch prediction."""
        if os.getenv("EP_PREFILL_PERF_TEST", "False") == "True":
            global p
            p.start()
            for i in range(10):
                print(f"Perf PREFILL {i} time")
                model_inputs = self.preprocess(batch_dials,
                                               extra_infos=extra_infos)
                for k, v in model_inputs.items():
                    if isinstance(v, np.ndarray):
                        model_inputs[k] = paddle.to_tensor(v)
                    else:
                        model_inputs[k] = copy.deepcopy(v)
                model_inputs["not_need_stop"] = model_inputs[
                    "not_need_stop"].cpu()
                res = self.infer(model_inputs)
                p.step()
                if i == 6:
                    p.stop()
                    exit()

        model_inputs = self.preprocess(batch_dials, extra_infos=extra_infos)
        for k, v in model_inputs.items():
            if isinstance(v, np.ndarray):
                model_inputs[k] = paddle.to_tensor(v)
            else:
                model_inputs[k] = copy.deepcopy(v)
        model_inputs["not_need_stop"] = model_inputs["not_need_stop"].cpu()

        if self.use_beam_search:
            infer_result = self.infer(model_inputs)
            infer_result[infer_result == -1] = 2
            self.num_output_tokens += ((
                (infer_result != self.tokenizer.eos_token_id)
                & (infer_result != self.tokenizer.cls_token_id)).sum().item())
            output = self.postprocess(infer_result)
            return output
        else:
            if os.getenv("EP_DECODER_PERF_TEST", "False") == "True":
                s = time.time()
            res = self.infer(model_inputs)
            if os.getenv("EP_DECODER_PERF_TEST", "False") == "True":
                logger.info(f"infer elapse {time.time() - s} s")

            result = []
            if self.tp_rank == 0 or (self.args.use_ep and
                                     (not self.args.ep_just_for_test)):
                while len(result) < self.bsz:
                    queue_res = self.result_queue.get()
                    result.append(queue_res[-1])
                    self.num_output_tokens += queue_res[1]

            if self.show_topk > 0:
                res["response"] = result
                return res

            result = {"result": result}
            return result


def main():
    """
    主函数，负责参数设置和调用预测器进行对话推理。

    Args:
        无参数，返回值为None。

    Returns:
        返回值为None。

    """
    args = setup_args()

    # enable_auth = False
    # if enable_auth:
    #     from encryption.auth import auth_product

    #     product_name = auth_product(args.model_name_or_path)

    check_output_dir = False
    if check_output_dir and not check_output(args.model_name_or_path):
        logger.error("args.model_name_or_path is not safe."
                     )  # Must before using paddlenlp, paddleslim logger
        sys.exit(-1)

    token_audit = False
    if token_audit:
        token_timer = TokenTimer("predict", args.model_name_or_path, 0)
        token_timer.start()

    if args.lora_num > 0:
        assert args.lora_dir is not None, "lora_dir should be set when lora_num > 0"

    predictor = Predictor(args)
    # inference
    infer_dials: list[list[dict]] = []
    if args.input_file is None or not os.path.exists(args.input_file):
        infer_dials = [[
            {
                "role": "user",
                "utterance": "北京天安门广场在哪里?\n"
            },
        ]] * args.batch_size
    else:
        with open(args.input_file, "r") as fin:
            for i, line in enumerate(fin, start=1):
                try:
                    cur_line = json.loads(line)
                    cur_format = get_infer_data_type(cur_line)
                    if "fc_data" == cur_format:
                        cur_dial = convert_fc_infer_data(cur_line)
                        tools = [
                            tool for item in cur_dial
                            if isinstance(item, list) for tool in item
                        ]
                        system = next(
                            (item["utterance"] for item in cur_dial
                             if "role" in item and item["role"] == "system"),
                            None,
                        )
                        if tools:
                            cur_dial = insert_fc_instruction(
                                cur_dial, {
                                    "tools": tools,
                                    "system": system
                                })
                        infer_dials.append(cur_dial)
                    elif "qa_data" == cur_format:
                        infer_dials.append(cur_line)
                    else:
                        raise ValueError("Unknown data format")
                except Exception as e:
                    logger.warning(
                        f"Failed to parse line {i} from args.input_file (start=1): {e}, skip."
                    )

    if args.beam_width > 1:
        infer_dials = [
            dials for dials in infer_dials for _ in range(args.beam_width)
        ]

    beam_batch_size = args.beam_width * args.batch_size

    test_case = []

    args.save_output_file_flush = (args.save_output_file_flush //
                                   beam_batch_size * beam_batch_size)
    if predictor.args.use_ep and (not predictor.args.ep_just_for_test):
        args.output_file = (os.path.dirname(args.output_file) + "/" +
                            os.path.basename(args.output_file).split(".")[0] +
                            f"_{paddle.distributed.get_rank()}.json")

    predictor.runtime_timer.start(
        f"msgid-{args.msg_queue_id} predict stage running time")
    if predictor.args.use_ep and (not predictor.args.ep_just_for_test):
        start_idx = predictor.tp_rank * predictor.args.batch_size
        offset_now = predictor.tp_degree * predictor.args.batch_size
    else:
        start_idx = 0
        offset_now = beam_batch_size
    for idx in tqdm(range(start_idx, len(infer_dials), offset_now)):
        batch_dials = infer_dials[idx:idx + beam_batch_size]
        print("inputs ->", batch_dials)
        result = predictor.predict(batch_dials)
        print("result ->", result)

        for in_dial, out_resp in zip(batch_dials, result["result"]):
            if not isinstance(in_dial, list):
                in_dial = []
            conversation_data = in_dial + [{
                "role": "bot",
                "utterance": out_resp
            }]
            test_case.append(conversation_data)

        if (args.save_output_file_flush > 0
                and idx % args.save_output_file_flush == 0 and idx > 0):
            if paddle.distributed.get_rank() == 0 or (
                    predictor.args.use_ep and
                (not predictor.args.ep_just_for_test)):
                infer_save_test_case(
                    test_case[idx - args.save_output_file_flush:idx],
                    args.output_file,
                )
    logger.info(
        f"The task is completed. Total input token: {predictor.num_input_tokens}. \
        Total output token: {predictor.num_output_tokens}")

    logger.info(f"{predictor.runtime_timer.log()}")

    try:
        token_timer.set_context_tokens(predictor.num_input_tokens)
        token_timer.set_generation_tokens(predictor.num_output_tokens)
        token_timer.check_and_write()
        token_timer.stop()
    except Exception:
        pass

    if os.getenv("EP_DECODER_PERF_TEST", "False") == "True":
        for _ in range(5):
            for idx in tqdm(range(start_idx, len(infer_dials), offset_now)):
                batch_dials = infer_dials[idx:idx + beam_batch_size]
                print("inputs ->", batch_dials)
                result = predictor.predict(batch_dials)
                print("result ->", result)

    if paddle.distributed.get_rank() == 0 or (
            predictor.args.use_ep and (not predictor.args.ep_just_for_test)):
        if args.save_output_file_flush == 0:
            infer_save_test_case(test_case, args.output_file)
        else:
            write_case_idx = (len(test_case) // args.save_output_file_flush *
                              args.save_output_file_flush)
            if len(test_case) % args.save_output_file_flush == 0:
                write_case_idx -= args.save_output_file_flush
            infer_save_test_case(test_case[write_case_idx:], args.output_file)
        if args.beam_width <= 1:
            predictor.read_res_process.terminate()


if __name__ == "__main__":
    """
    main
    """
    main()

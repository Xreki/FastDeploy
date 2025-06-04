#!/usr/bin/env python3

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved.

Export Inference model.
"""

import os
import sys

import argparse

from paddlenlp.trainer import strtobool
from paddlenlp.utils.log import logger

import paddle
from fastdeploy.model_executor.models.configuration import ErnieBotConfig
from fastdeploy.model_executor.models.tokenizer import ErnieBotTokenizer
from paddle.distributed import fleet
from fastdeploy.model_executor.models.utils import (
    UniqueIDGenerator,
)
from fastdeploy.model_executor.models.token_utils import TokenTimer, check_output, process_index


should_check_python_safety = False
if should_check_python_safety:
    if not os.path.isfile("./utils/token_utils.pyxes"):
        print("Some toolkit files have been modified, check failed.")
        sys.exit(-15)
    from utils.token_utils import check_python_safe_export

    check_python_safe_export()


def setup_args():
    """Setup export arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="FP32 parameters file.",
    )
    parser.add_argument("--model_prefix", type=str, default="model")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--min_dec_len", type=int, default=1)
    parser.add_argument("--max_dec_len", type=int, default=8192)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument(
        "--get_topk",
        default="False",
        type=strtobool,
        help="whether get topk token/score",
    )
    parser.add_argument("--topk_value", type=int, default=5)
    parser.add_argument("--lora_num", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=0)
    parser.add_argument("--export_model_type", type=str, default="default")
    parser.add_argument("--version_file", type=str, default="export_version.txt")
    parser.add_argument("--pre_caches_length", type=int, default=0)
    parser.add_argument("--gqa_use_tensorcore", type=strtobool, default=None)
    parser.add_argument(
        "--use_system",
        default="False",
        type=strtobool,
        help="use system or not",
    )
    parser.add_argument(
        "--export_prefix",
        default="False",
        type=strtobool,
        help="export pre_cache field",
    )
    parser.add_argument(
        "--use_fake_parameter",
        default="False",
        type=strtobool,
        help="use fake parameter",
    )
    parser.add_argument(
        "--use_stop_seqs",
        default="False",
        type=strtobool,
        help="whether use stop seqs",
    )
    parser.add_argument(
        "--outputs_op",
        type=str,
        default="save_with_output",
        help="the type of outputs_op, `none` op will return the decoded_ids",
        choices=["none", "save_with_output"],
    )
    parser.add_argument(
        "--cache_quant_dtype",
        type=str,
        default="default",
        choices=["default", "float32"],
        help="the date type of CacheKScale, CacheVScale, CacheKOutScale, "
        "CacheVOutScale, CacheKZeroPoint, CacheVZeroPoint",
    )
    parser.add_argument("--pad_vocab", type=strtobool, default="True")
    parser.add_argument(
        "--use_beam_search",
        default="False",
        type=strtobool,
        help="Whether use beam search",
    )
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument(
        "--enf_gen",
        type=strtobool,
        default="False",
        help="Use enforce generation decoding strategy",
    )
    parser.add_argument("--speculate_enable", default="False", type=strtobool)
    parser.add_argument("--speculate_get_output_hidden", default="True", type=strtobool)
    parser.add_argument("--speculate_max_draft_tokens", default=1, type=int)
    parser.add_argument("--speculate_max_candidate_len", default=5, type=int)
    parser.add_argument("--speculate_verify_window", default=2, type=int)
    parser.add_argument("--return_all_hidden_states", default="False", type=strtobool)
    parser.add_argument(
        "--use_efficientllm",
        default="False",
        type=strtobool,
        help="use efficientllm or not",
    )
    parser.add_argument(
        "--normalize_for_embedding",
        default="False",
        type=strtobool,
        help="Whether do last norm for Embedding Model",
    )
    parser.add_argument(
        "--moe_quant_type",
        default="default",
        type=str,
        help="quant type for moe part",
    )
    parser.add_argument(
        "--use_safetensors",
        default="False",
        type=strtobool,
        help="using safetensors",
    )
    parser.add_argument(
        "--use_multimodality",
        default="False",
        type=strtobool,
        help="use image_features or not(only used in multi modal)",
    )
    parser.add_argument(
        "--use_offline_quant",
        default="False",
        type=strtobool,
        help="The inference uses offline-quantized weights, \
            and the script performs the offline quantization.",
    )
    args = parser.parse_args()
    return args


def add_inference_args_to_config(model_config, args):
    """Add export arguments to config."""
    model_config.infer_model_block_size = args.block_size
    model_config.infer_model_max_seq_len = args.max_seq_len
    model_config.infer_model_min_dec_len = args.min_dec_len
    model_config.infer_model_max_dec_len = args.max_dec_len
    model_config.infer_model_export_model_type = args.export_model_type
    model_config.infer_model_dtype = args.dtype
    model_config.infer_model_mp_num = args.mp_num
    model_config.infer_model_paddle_commit = paddle.version.commit
    model_config.infer_model_enf_gen = args.enf_gen
    model_config.infer_model_use_beam_search = args.use_beam_search
    model_config.infer_model_speculate_enable = args.speculate_enable
    model_config.infer_model_return_all_hidden_states = args.return_all_hidden_states
    model_config.infer_model_use_stop_seqs = args.use_stop_seqs
    model_config.model_export_id = args.unique_id
    model_config.get_topk = args.get_topk
    model_config.gqa_use_tensorcore = args.gqa_use_tensorcore
    model_config.use_system = args.use_system
    if args.get_topk:
        model_config.topk_value = args.topk_value
    if "C8" in args.export_model_type:
        model_config.infer_model_cache_kv_type = "cachekv_quant_int8"
    elif "C4" in args.export_model_type:
        model_config.infer_model_cache_kv_type = "cachekv_quant_int4"
    else:
        model_config.infer_model_cache_kv_type = "no_quant"


if __name__ == "__main__":
    args = setup_args()

    if args.export_model_type not in [
        "default",
        "WINT8",
        "W8A8C16",
        "W8A8C8",
    ]:
        raise ValueError(
            "export_model_type must be in ['default', 'WINT8', 'W8A8C16', 'W8A8C8'] when use_efficientllm is False."
        )

    enable_auth = False
    if enable_auth:
        from encryption.auth import auth_product

        product_name = auth_product(args.model_name_or_path)

    check_output_dir = False
    if check_output_dir and not check_output(args.output_path):
        print("Output dir is not safe.")
        sys.exit(-1)

    token_audit = False
    if token_audit:
        token_timer = TokenTimer("export_default", args.output_path, 0)
        token_timer.start()

    if args.device.lower() == "cpu":
        logger.info("Using CPU to export model.")
        paddle.set_device("cpu")

    strategy: fleet.DistributedStrategy = fleet.DistributedStrategy()
    args.mp_num = paddle.distributed.get_world_size()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": args.mp_num,
        "pp_degree": 1,
    }
    fleet.init(is_collective=True, strategy=strategy)

    from fastdeploy.model_executor.models.export_model import (
        export_efficientllm_model,
    )

    model = export_efficientllm_model(args)

    # Generate the unique id
    unique_id_generator = UniqueIDGenerator()
    if args.export_model_type == "default":
        export_model_type = "BF16"
    else:
        export_model_type = args.export_model_type
    unique_id = (
        export_model_type
        + "-"
        + unique_id_generator.generate_unique_id(model.state_dict())
    )
    args.unique_id = unique_id

    try:
        token_timer.set_task(f"export_{args.export_model_type}")
        token_timer.check_and_write()
        token_timer.stop()
    except Exception:
        pass

    model_config = ErnieBotConfig.from_pretrained(args.model_name_or_path)

    if enable_auth:
        model_config.product_name = product_name

    add_inference_args_to_config(model_config, args)
    if process_index() == 0:
        model_config.save_pretrained(args.output_path)
        ErnieBotTokenizer.from_pretrained(args.model_name_or_path).save_pretrained(
            args.output_path
        )

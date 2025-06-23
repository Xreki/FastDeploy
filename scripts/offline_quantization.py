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

import argparse
import importlib
import json
import os

import paddle
from paddleformers.trainer import strtobool
from paddleformers.transformers.model_utils import shard_checkpoint
from paddleformers.utils.env import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from paddleformers.utils.log import logger
from safetensors.numpy import save_file as safe_save_file

from fastdeploy.model_executor.models.utils import (
    fastsafetensors_weights_iterator, get_safetensor_file, load_ep_checkpoint)

MODEL_LIB_NAMES = [
    "fastdeploy.model_executor.models.modeling_ernie_bot",
]


def parse_arguments():
    """
    parse_arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        required=True,
        help="The directory of model.",
    )
    parser.add_argument(
        "--output_dir",
        default="merged_output",
        required=True,
        help="The directory of merged model output.",
    )
    parser.add_argument(
        "--safe_serialization",
        type=strtobool,
        default="True",
        help="Whether merge the model into safetensors format.",
    )
    parser.add_argument(
        "--predict_model_type",
        type=str,
        default="",
        help="Quantization type for the model.",
    )

    parser.add_argument(
        "--draft_type",
        type=str,
        default=None,
        choices=["autoregressive", "inference_with_reference", "hydra", "mtp"],
        help="Quantization type for the model.",
    )

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
        "--use_ep",
        type=strtobool,
        default="True",
        help="Whether merge the model into safetensors format.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--load_safetensor", type=strtobool, default="False")
    return parser.parse_args()


def get_model_cls(config):
    """
    Get model class from model configuration.
    """
    init_class = "ErnieBotFusedModel"
    for lib_name in MODEL_LIB_NAMES:
        eb_lib = importlib.import_module(lib_name)
        if hasattr(eb_lib, init_class):
            cls = getattr(eb_lib, init_class)
            return cls

    raise RuntimeError(
        f"Cannot find model architecture({init_class}) from eb_lib")


def save_safetensors(state_dict, args):
    """
    save_safetensors
    """
    logger.info("Move to numpy.")
    for k in list(state_dict.keys()):
        if isinstance(state_dict[k], paddle.Tensor):
            tensor = state_dict.pop(k)
            tensor = tensor.cpu()
            array = tensor.numpy()
            state_dict[k] = array

    logger.info("Save safetensors files.")
    shards, index = shard_checkpoint(
        state_dict,
        max_shard_size="5GB",
        weights_name=SAFE_WEIGHTS_NAME,
        shard_format="naive",
    )
    for shard_file, shard in shards.items():
        save_file = os.path.join(args.output_dir, shard_file)
        logger.info(f"Saving {save_file}")
        safe_save_file(shard, save_file, metadata={"format": "np"})

    save_index_file = os.path.join(args.output_dir, SAFE_WEIGHTS_INDEX_NAME)
    with open(save_index_file, "w", encoding="utf-8") as f:
        content = json.dumps(index, indent=2) + "\n"
        f.write(content)


def quanted_tensor(cls, config, args=None):
    """
    quanted_tensor
    """
    name_action_mappings = cls._get_tensor_quantization_mappings(config)
    state_dict_to_save = {}
    if not args.use_ep:
        loaded_state_dict_keys, safetensor_list = get_safetensor_file(
            args.model_name_or_path)
        state_keys_map = cls._resolve_prefix_keys(name_action_mappings.keys(),
                                                  loaded_state_dict_keys)
        for k, v in state_keys_map.items():
            name_action_mappings[v] = name_action_mappings.pop(k)
        weights_iterator = fastsafetensors_weights_iterator(safetensor_list)
        for key, weight in weights_iterator:
            tensor = weight
            if key in name_action_mappings:
                action = name_action_mappings.pop(key)
                quant_weight_tensor, weight_scale_tensor = action(tensor)
                if quant_weight_tensor is not None and weight_scale_tensor is not None:
                    state_dict_to_save[
                        key + ".quant_weight"] = quant_weight_tensor.cpu()
                    state_dict_to_save[
                        key + ".weight_scale"] = weight_scale_tensor.cpu()
                else:
                    state_dict_to_save[key] = quant_weight_tensor.cpu()
            else:
                state_dict_to_save[key] = tensor.cpu()
    else:
        state_dict = load_ep_checkpoint(args.model_name_or_path,
                                        config,
                                        return_numpy=True,
                                        return_key_name=True)
        state_keys_map = cls._resolve_prefix_keys(name_action_mappings.keys(),
                                                  state_dict.keys())
        for k, v in state_keys_map.items():
            name_action_mappings[v] = name_action_mappings.pop(k)
        from tqdm import tqdm

        from fastdeploy.model_executor.layers.utils import get_tensor

        for key in tqdm(state_dict.keys(), desc="process quantized weights  "):
            tensor_path = state_dict[key]
            if key in name_action_mappings:
                ret = state_dict[key]
                action = name_action_mappings.pop(key)
                quanted_weight_tensor, weight_scale_tensor = action(
                    get_tensor(ret))
                if quanted_weight_tensor is not None:
                    state_dict_to_save[
                        key + ".quant_weight"] = quanted_weight_tensor.cpu()
                if weight_scale_tensor._is_initialized():
                    state_dict_to_save[
                        key + ".weight_scale"] = weight_scale_tensor.cpu()
                else:
                    state_dict_to_save[key] = quanted_weight_tensor.cpu()
            else:
                state_dict_to_save[key] = get_tensor(tensor_path).cpu()

        if len(name_action_mappings) > 0:
            for x in name_action_mappings.keys():
                logger.debug(
                    f"key <{x}> need to merge tensor parallel but we can't find in model state."
                )
    return state_dict_to_save

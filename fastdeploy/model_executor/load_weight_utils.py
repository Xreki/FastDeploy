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

from paddleformers.transformers import PretrainedModel
from fastdeploy.config import ModelConfig
from fastdeploy.config import FDConfig
from typing import Optional
from safetensors import safe_open
from tqdm import tqdm
from paddleformers.utils.env import (
    PADDLE_WEIGHTS_INDEX_NAME,
    SAFE_MASTER_WEIGHTS_INDEX_NAME,
    SAFE_PEFT_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
)
from paddleformers.utils.log import logger
from fastdeploy.platforms import current_platform
import os
from paddleformers.transformers.model_utils import _add_variant, load_tp_checkpoint
import json
import paddle.distributed as dist
from fastsafetensors import SafeTensorsFileLoader, SingleGroup
import paddle
from paddle import nn
from typing import Dict
from functools import partial


def load_ep_checkpoint(
    model_path: str,
    config: ModelConfig,
    return_numpy: bool = False,
):
    """
    load ep checkpoint
    """
    # return_numpy=True cpu
    # return_numpy=False gpu
    with open(os.path.join(model_path, "model.safetensors.index.json"), "r") as f:
        weight_list = json.load(f)["weight_map"]
    filtered_map = {k: v for k, v in weight_list.items() if "experts" not in k}
    num_local_ffn_keys = []
    quant_suffix = (
        ".quant_weight"
        if config.use_offline_quant and config.moe_quant_type != "default"
        else ""
    )
    # Note(tangbinhan): Confirm whether the EP weight has been changed to weight_scale
    scale_suffix = (
        ".quant_scale"
        if config.use_offline_quant and config.moe_quant_type != "default"
        else ""
    )

    for i in range(config.moe_layer_start_index, config.num_layers):
        for j in range(
            config.num_experts_start_offset,
            config.num_experts_start_offset + config.num_experts_per_rank,
        ):
            ffn1_quant_key = (
                f"ernie.layers.{i}.mlp.experts.{j}.up_gate_proj.weight{quant_suffix}"
            )
            ffn2_quant_key = (
                f"ernie.layers.{i}.mlp.experts.{j}.down_proj.weight{quant_suffix}"
            )
            ffn1_scale_key = (
                f"ernie.layers.{i}.mlp.experts.{j}.up_gate_proj.weight{scale_suffix}"
            )
            ffn2_scale_key = (
                f"ernie.layers.{i}.mlp.experts.{j}.down_proj.weight{scale_suffix}"
            )
            num_local_ffn_keys.append(ffn1_quant_key)
            num_local_ffn_keys.append(ffn2_quant_key)
            num_local_ffn_keys.append(ffn1_scale_key)
            num_local_ffn_keys.append(ffn2_scale_key)

    for k in num_local_ffn_keys:
        if k in weight_list:
            filtered_map[k] = weight_list[k]

    state_dict = {}
    # Get all safetensor file paths that need to be opened
    safetensor_paths = set(filtered_map.values())

    # Open each safetensor file sequentially with progress bar
    for safetensor_path in tqdm(
        safetensor_paths, desc="Loading safetensor files", unit="file"
    ):
        with safe_open(
            os.path.join(model_path, safetensor_path), framework="np", device="cpu"
        ) as f:
            # Check if this file contains keys from filtered_map
            for k in filtered_map:
                if filtered_map[k] == safetensor_path and k in f.keys():
                    weight = f.get_tensor(k)
                    if not return_numpy:
                        weight = paddle.Tensor(weight, zero_copy=True)
                        weight = weight._copy_to(
                            paddle.framework._current_expected_place(), False
                        )
                    state_dict[k] = weight
    return state_dict


def safetensors_weights_iterator(
    safe_tensor_list: list[str],
):
    """
    safetensors_weights_iterator
    """
    for st_file in tqdm(
        safe_tensor_list,
        desc="Loading safetensors checkpoint shards",
    ):
        with safe_open(st_file, framework="np") as f:
            for name in f.keys():
                param = f.get_tensor(name)
                yield name, param


def fastsafetensors_weights_iterator(
    safetensor_list: list[str],
):
    """
    Return an iterator over tensors on GPU from a given safetensor_list.
    """
    world_size = dist.get_world_size()
    if world_size > 1:
        pg = dist.get_group()
        device = f"gpu:{pg.rank}" if paddle.is_compiled_with_cuda() else "cpu"
    else:
        pg = SingleGroup()
        device = f"gpu:{pg.rank()}" if paddle.is_compiled_with_cuda() else "cpu"

    safetensor_files_sub_lists = [
        safetensor_list[i : i + world_size]
        for i in range(0, len(safetensor_list), world_size)
    ]

    for st_file in tqdm(
        safetensor_files_sub_lists,
        desc="Loading fastsafetensors checkpoint shards",
    ):
        loader = SafeTensorsFileLoader(
            pg, device, nogds=True, debug_log=False, framework="paddle"
        )
        rank_file_map = {i: [f] for i, f in enumerate(st_file)}
        loader.add_filenames(rank_file_map)
        try:
            fb = loader.copy_files_to_device()
            try:
                keys = list(fb.key_to_rank_lidx.keys())
                for k in keys:
                    t = fb.get_tensor(k)
                    yield k, t
            finally:
                fb.close()
        finally:
            loader.close()


def load_pre_sharded_checkpoint(
    model_path: str, local_rank: int, use_fastsafetensor: bool = False
):
    """
    load_pre_sharded_checkpoint
    """
    state_dict = {}
    _, safetensor_files = get_all_safetensors(
        os.path.join(model_path, f"rank{local_rank}"),
        os.path.join(model_path, f"rank{local_rank}/model.safetensors.index.json"),
    )
    weights_iterator = safetensors_weights_iterator(safetensor_files)
    for name, weight in weights_iterator:
        state_dict[name] = weight
    return state_dict


def get_all_safetensors(model_path: str, index_json_path: str):
    """
    get_all_safetensors
    """
    safe_model_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(safe_model_path):
        safetensor_list = [safe_model_path]
        with safe_open(safe_model_path, framework="np", device="cpu") as f:
            key_name_list = f.keys()
        return key_name_list, safetensor_list
    else:
        with open(index_json_path, "r") as f:
            weight_map = json.load(f)["weight_map"]
        weight_files_in_index = set()
        for weight_name in weight_map:
            weight_files_in_index.add(os.path.join(model_path, weight_map[weight_name]))
        key_name_list = list(set(weight_map.keys()))
        safetensor_list = list(weight_files_in_index)
        safetensor_list.sort()
    return key_name_list, safetensor_list


def apply_quant_action(
    quant_map: Dict[str, partial],
    key: str,
    tensor: paddle.Tensor,
    state_dict: Dict[str, paddle.Tensor],
    quant_layer_instance_map: Dict[str, nn.Layer],
):
    """
    apply_quant_action
    """
    action = quant_map.pop(key)
    quant_weight_tensor, weight_quanter_tensor = action(
        key, tensor, quant_layer_instance_map
    )
    if quant_weight_tensor._is_initialized():
        quant_weight_key = key.replace("weight", "quant_weight")
        state_dict[quant_weight_key] = quant_weight_tensor
    if weight_quanter_tensor._is_initialized():
        weight_quanter_key = key.replace("weight", "weight_scale")
        state_dict[weight_quanter_key] = weight_quanter_tensor


def get_quant_layer_instance_map(cls: PretrainedModel, model_dict: Dict[str, nn.Layer]):
    """get_quant_layer_instance_map"""
    suffix_set = set(cls.quant_need_find_layer_list)
    quant_layer_map = {}

    remaining_suffixes = set(suffix_set)

    for key, layer in model_dict.items():
        for suffix in list(remaining_suffixes):
            if key.endswith(suffix):
                quant_layer_map[suffix] = layer
                remaining_suffixes.remove(suffix)
                break

        if not remaining_suffixes:
            break

    if not quant_layer_map:
        logger.error(
            "quant_map should not be empty. "
            "Pre-quantization is required, but _get_quantization_mappings is not implemented."
        )

    return quant_layer_map


def check_quantization_prerequisites(
    fd_config: FDConfig,
    cls: PretrainedModel,
    quant_filtered_map: Dict[str, partial],
    safetensor_keys: list[str],
    model_dict: Optional[Dict[str, nn.Layer]] = None,
) -> None:
    """check_quantization_prerequisites"""
    if fd_config.quant_config is not None and fd_config.model_config.pre_quant:
        if not hasattr(cls, "_get_quantization_mappings"):
            raise NotImplementedError(
                f"Class {cls.__name__} must implement method '_get_quantization_mappings'"
            )
        quant_map = cls._get_quantization_mappings(fd_config)
        if not quant_map:
            logger.error(
                f"quant_map should not be empty. \
            pre-quantization required, but _get_quantization_mappings is not implemented."
            )
        else:
            filtered_quant_map = cls._resolve_prefix_keys(
                quant_map.keys(), safetensor_keys
            )
            for k, v in filtered_quant_map.items():
                quant_filtered_map[v] = quant_map.pop(k)
            if not filtered_quant_map:
                logger.error(
                    "filtered_quant_map should not be empty. \
                The weights specified for quantization do not match the weights present in the model."
                )
        if not model_dict:
            logger.error(
                "Missing required argument 'model_dict' when calling load_tp_checkpoint_v1."
            )


def check_tensor_parallel_prerequisites(
    fd_config: FDConfig,
    cls: PretrainedModel,
    tensor_parallel_filtered_map: Dict[str, partial],
    safetensor_keys: list[str],
) -> None:
    """check_tensor_parallel_prerequisites"""
    if fd_config.parallel_config.tensor_parallel_degree > 1:
        tensor_parallel_map = cls._get_tensor_parallel_mappings(
            fd_config.model_config, is_split=True
        )
        if not tensor_parallel_map:
            logger.error(
                "filtered_quant_map should not be empty. \
                parallel splitting required, but _get_tensor_parallel_mappings is not implemented."
            )
        filtered_tp_keys = cls._resolve_prefix_keys(
            tensor_parallel_map.keys(), safetensor_keys
        )
        for k, v in filtered_tp_keys.items():
            tensor_parallel_filtered_map[v] = tensor_parallel_map.pop(k)
        if not tensor_parallel_filtered_map:
            logger.error(
                "tensor_parallel_filtered_map should not be empty. \
                The weights required for tensor parallel splitting are inconsistent with the model's weights."
            )


def load_tp_checkpoint_v1(
    model_path: str,
    cls: PretrainedModel,
    fd_config: FDConfig,
    model_dict: Optional[Dict[str, nn.Layer]] = None,
    use_fastsafetensor: bool = True,
):
    """
    This function currently supports GPU weight loading only.
    Loading NumPy tensors with safetensor is too slow, so use_fastsafetensor=True by default.
    """

    safetensor_keys, safetensor_files = get_all_safetensors(
        model_path, os.path.join(model_path, "model.safetensors.index.json")
    )

    if use_fastsafetensor:
        weights_iterator = fastsafetensors_weights_iterator(safetensor_files)
    else:
        weights_iterator = safetensors_weights_iterator(safetensor_files)

    tensor_parallel_filtered_map = {}
    check_tensor_parallel_prerequisites(
        fd_config,
        cls,
        tensor_parallel_filtered_map,
        safetensor_keys,
    )

    quant_filtered_map = {}
    check_quantization_prerequisites(
        fd_config, cls, quant_filtered_map, safetensor_keys, model_dict
    )
    quant_layer_instance_map = {}
    if quant_filtered_map:
        quant_layer_instance_map = get_quant_layer_instance_map(cls, model_dict)
    state_dict = {}
    for key, weight in weights_iterator:
        paddle.device.cuda.synchronize()
        if tensor_parallel_filtered_map and key in tensor_parallel_filtered_map:
            action = tensor_parallel_filtered_map.pop(key)
            tensor = action(weight).clone()
        else:
            tensor = weight.clone()
        if quant_filtered_map and key in quant_filtered_map:
            apply_quant_action(
                quant_filtered_map, key, tensor, state_dict, quant_layer_instance_map
            )
        else:
            state_dict[key] = tensor
        weight.value().get_tensor()._clear()
    return state_dict


def load_composite_checkpoint(
    model_path: str,
    cls: PretrainedModel,
    fd_config: FDConfig,
    model_dict: Optional[Dict[str, nn.Layer]] = None,
    return_numpy=True,
):
    """
    # This method supports loading checkpoints of three types:
    # 1. Expert Parallel (EP) checkpoint
    # 2. Tensor Parallel (TP) checkpoint
    # 3. Pre-sharded (pre-split) checkpoint
    """
    if fd_config.parallel_config.use_ep:
        state_dict = load_ep_checkpoint(
            model_path, fd_config.model_config, return_numpy=True
        )
    else:
        rank_dirs = [
            f
            for f in os.listdir(model_path)
            if f.startswith("rank") and os.path.isdir(os.path.join(model_path, f))
        ]
        if len(rank_dirs) > 1:
            if fd_config.parallel_config.tensor_parallel_degree != len(rank_dirs):
                raise ValueError(
                    f"Your model only supports loading with tp{len(rank_dirs)}"
                )
            state_dict = load_pre_sharded_checkpoint(
                model_path,
                fd_config.parallel_config.tensor_parallel_rank,
                use_fastsafetensor=False,
            )
        else:
            if not fd_config.load_config.load_weights_on == "cpu" and (
                current_platform.is_cuda() and current_platform.available()
            ):
                state_dict = load_tp_checkpoint_v1(
                    model_path, cls, fd_config, model_dict, use_fastsafetensor=True
                )
            else:
                state_dict = load_tp_checkpoint(
                    model_path, cls, fd_config.model_config, return_numpy=return_numpy
                )
    if not state_dict:
        raise ValueError(f"weight not found in state_dict !")
    return state_dict

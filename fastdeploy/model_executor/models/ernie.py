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

from __future__ import annotations

from functools import partial

import numpy as np
import paddle
from paddlenlp.transformers import PretrainedModel
from paddlenlp.utils.log import logger

from fastdeploy.config import LLMConfig, ModelConfig


def get_attr(layer, name):
    """
    get_attr
    """
    if getattr(layer, name, None) is not None:
        return getattr(layer, name, None)
    else:
        return get_attr(layer._layer, name)


class ErnieBotPretrainedModel(PretrainedModel):
    """
    ErnieBotPretrainedModel
    """

    config_class = LLMConfig

    def _init_weight(self, layer):
        """
        _init_weight
        """
        return None

    @classmethod
    def _get_tensor_parallel_mappings(cls, config: ModelConfig, is_split=True):
        """
        get_tensor_parallel_mappings
        """
        logger.info("erine bot inference model _get_tensor_parallel_mappings")

        from paddlenlp.transformers.conversion_utils import split_or_merge_func

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_attention_heads,
        )

        def gqa_qkv_split_func(
            weight,
            tensor_parallel_degree,
            tensor_parallel_rank,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
        ):

            def get_shape(tensor):
                return (tensor.get_shape()
                        if hasattr(tensor, "get_shape") else tensor.shape)

            def slice_tensor(tensor, start, end):
                shape = get_shape(tensor)
                if len(shape) == 1:
                    return tensor[start:end]
                else:
                    return tensor[..., start:end]

            q_end = num_attention_heads * head_dim
            k_end = q_end + num_key_value_heads * head_dim
            v_end = k_end + num_key_value_heads * head_dim

            q = slice_tensor(weight, 0, q_end)
            k = slice_tensor(weight, q_end, k_end)
            v = slice_tensor(weight, k_end, v_end)

            def split_tensor(tensor, degree):
                shape = get_shape(tensor)
                size = shape[-1]
                block_size = size // degree
                if hasattr(tensor, "get_shape"):
                    return [
                        slice_tensor(tensor, i * block_size,
                                     (i + 1) * block_size)
                        for i in range(degree)
                    ]
                else:
                    return np.split(tensor, degree, axis=-1)

            q_list = split_tensor(q, tensor_parallel_degree)
            k_list = split_tensor(k, tensor_parallel_degree)
            v_list = split_tensor(v, tensor_parallel_degree)

            if tensor_parallel_rank is None:
                return [
                    np.concatenate([q_i, k_i, v_i], axis=-1)
                    for q_i, k_i, v_i in zip(q_list, k_list, v_list)
                ]
            else:
                return np.concatenate(
                    [
                        q_list[tensor_parallel_rank],
                        k_list[tensor_parallel_rank],
                        v_list[tensor_parallel_rank],
                    ],
                    axis=-1,
                )

        def gqa_qkv_merge_func(weight_list, num_attention_heads,
                               num_key_value_heads, head_dim):
            tensor_parallel_degree = len(weight_list)
            num_attention_heads = num_attention_heads // tensor_parallel_degree
            num_key_value_heads = num_key_value_heads // tensor_parallel_degree

            is_paddle_tensor = not isinstance(weight_list[0], np.ndarray)

            def get_shape(tensor):
                return (tensor.get_shape()
                        if hasattr(tensor, "get_shape") else tensor.shape)

            def slice_tensor(tensor, start, end):
                if len(get_shape(tensor)) == 1:
                    return tensor[start:end]
                else:
                    return tensor[..., start:end]

            q_list, k_list, v_list = [], [], []

            for weight in weight_list:
                q_end = num_attention_heads * head_dim
                k_end = q_end + num_key_value_heads * head_dim
                v_end = k_end + num_key_value_heads * head_dim

                q = slice_tensor(weight, 0, q_end)
                k = slice_tensor(weight, q_end, k_end)
                v = slice_tensor(weight, k_end, v_end)

                q_list.append(q)
                k_list.append(k)
                v_list.append(v)

            merged = q_list + k_list + v_list

            if is_paddle_tensor:
                tensor = paddle.concat(merged, axis=-1)
                if tensor.place.is_gpu_place():
                    tensor = tensor._copy_to(paddle.CUDAPinnedPlace(), False)
                return tensor
            else:
                return np.concatenate(merged, axis=-1)

        if (config.num_key_value_heads is not None
                and config.num_key_value_heads != config.num_attention_heads):
            if is_split:
                qkv_fn = partial(
                    gqa_qkv_split_func,
                    tensor_parallel_degree=config.tensor_parallel_degree,
                    tensor_parallel_rank=config.tensor_parallel_rank,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=config.hidden_size // config.num_attention_heads,
                )
            else:
                qkv_fn = partial(
                    gqa_qkv_merge_func,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=config.hidden_size // config.num_attention_heads,
                )
        else:
            qkv_fn = partial(fn, is_column=True)

        def get_tensor_parallel_split_mappings(num_layers, moe_num_experts,
                                               moe_layer_start_index, is_mtp):
            final_actions = {}
            use_moe = moe_num_experts > 0
            if is_mtp:
                base_model_prefix = "ernie.mtp"
            else:
                base_model_prefix = "ernie"
            key = (f"{base_model_prefix}.embeddings.word_embeddings" if
                   not use_moe else f"{base_model_prefix}.embed_tokens.weight")
            base_actions = {
                "lm_head.weight": partial(fn, is_column=True),
                # "eh_proj.weight": partial(fn, is_column=True),
                key: partial(fn, is_column=False),
            }
            if use_moe and moe_layer_start_index > 0:
                base_actions[
                    f"{base_model_prefix}.layers.0.self_attn.qkv_proj.weight"] = qkv_fn
                base_actions[
                    f"{base_model_prefix}.layers.0.self_attn.o_proj.weight"] = partial(
                        fn, is_column=False)
                base_actions[
                    f"{base_model_prefix}.layers.0.mlp.up_gate_proj.weight"] = partial(
                        fn, is_column=True, is_naive_2fuse=True)
                base_actions[
                    f"{base_model_prefix}.layers.0.mlp.down_proj.weight"] = (
                        partial(fn, is_column=False))

                for expert_idx in range(moe_num_experts):
                    base_actions[
                        f"{base_model_prefix}.layers.{moe_layer_start_index}"
                        f".mlp.experts.{expert_idx}.up_gate_proj.weight"] = partial(
                            fn, is_column=True, is_naive_2fuse=True)
                    base_actions[
                        f"{base_model_prefix}.layers.{moe_layer_start_index}"
                        f".mlp.experts.{expert_idx}.down_proj.weight"] = partial(
                            fn, is_column=False)
            else:
                # (tangbinhan:todo) Splitting of non-MoE weights
                base_actions[
                    "decoder.layers.0.self_attn.qkv_proj.weight"] = partial(
                        fn, is_column=True)
                base_actions[
                    "decoder.layers.0.self_attn.qkv_proj.bias"] = partial(
                        fn, is_column=True)
                base_actions[
                    "decoder.layers.0.self_attn.out_proj.weight"] = partial(
                        fn, is_column=False)

                base_actions[
                    "decoder.layers.0.self_attn.out_proj.bias"] = partial(
                        fn, is_column=False)

                base_actions[
                    "decoder.layers.0.self_attn.linear1.weight"] = partial(
                        fn, is_column=True, is_naive_2fuse=True)
                base_actions[
                    "decoder.layers.0.self_attn.linear1.bias"] = partial(
                        fn, is_column=True, is_naive_2fuse=True)

                base_actions[
                    "decoder.layers.0.self_attn.linear2.weight"] = partial(
                        fn, is_column=False)
                base_actions[
                    "decoder.layers.0.self_attn.linear2.bias"] = partial(
                        fn, is_column=False)

            for key, action in base_actions.items():
                if (f"{base_model_prefix}.layers.0.mlp.up_gate_proj.weight"
                        in key
                        or f"{base_model_prefix}.layers.0.mlp.down_proj.weight"
                        in key):
                    for i in range(moe_layer_start_index):
                        final_actions[key.replace("layers.0.",
                                                  f"layers.{i}.")] = action
                elif f"layers.{moe_layer_start_index}.mlp.experts." in key:
                    for i in range(moe_layer_start_index, num_layers):
                        final_actions[key.replace(
                            f"layers.{moe_layer_start_index}.",
                            f"layers.{i}.")] = action
                elif f"{base_model_prefix}.layers.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("layers.0.",
                                                  f"layers.{i}.")] = action
                final_actions[key] = action
            return final_actions

        mappings = get_tensor_parallel_split_mappings(
            config.num_layers,
            config.moe_num_experts,
            config.moe_layer_start_index,
            config.is_mtp,
        )

        return mappings

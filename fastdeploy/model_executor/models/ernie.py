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

import logging
import os
from functools import partial
from typing import Dict, Union

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.distributed import fleet
from paddlenlp.transformers import PretrainedModel, register_base_model
from paddlenlp.utils.log import logger

from fastdeploy.config import LLMConfig, ModelConfig, WeightKeys
from fastdeploy.inference_args import GenerationPhase, InferenceArgs
from fastdeploy.model_executor.ops.gpu import (
    beam_search_softmax, draft_model_update, extract_text_token_output,
    get_padding_offset, get_token_penalty_multi_scores, mtp_save_first_token,
    mtp_save_first_token_dynamic, save_output, save_output_dynamic,
    set_stop_value_multi_ends, set_stop_value_multi_seqs,
    set_value_by_flags_and_idx, speculate_clear_accept_nums,
    speculate_get_output_padding_offset, speculate_get_padding_offset,
    speculate_get_seq_lens_output, speculate_get_token_penalty_multi_scores,
    speculate_rebuild_append_padding, speculate_save_output,
    speculate_save_output_dynamic, speculate_set_stop_value_multi_seqs,
    speculate_set_value_by_flags_and_idx, speculate_update_v3,
    speculate_verify, top_p_candidates, update_inputs, update_inputs_beam)
from fastdeploy.worker.model_runner import ForwardMeta

from ..layers.embeddings import VocabParallelEmbedding
from ..layers.lm_head import ParallelLMHead
from ..layers.normalization import RMSNorm
from ..layers.quantization import get_quantization_config
from .fused_transformer import FusedTransformer
from .model_base import ModelForCasualLM
from fastdeploy.model_executor.eplb.experts_manager import RedundantExpertManger

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


@register_base_model
class ErnieBotFusedModel(ErnieBotPretrainedModel):
    """
    ErnieBotFusedModel
    """

    def __init__(
        self,
        vocab_size=51200,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        ffn_hidden_size=3072,
        activation="gelu",
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
        dtype="float32",
        block_size=16,
        inv_compression_ratio=1.0,
        sequence_parallel=False,
        freeze_embedding=False,
        max_len=-1,
        use_rope=False,
        rope_theta=10000.0,
        rope_3d=False,
        weight_sharing=True,
        weight_sharing_add_bias=False,
        export_model_type="default",
        group_size=-1,
        model_path="",  # The path of Inference model.
        use_rmsnorm=False,
        msg_queue_id=None,
        use_fake_parameter=False,
        num_key_value_heads=-1,
        use_stop_seqs=False,
        cache_quant_dtype="default",
        has_zero_point=False,
        is_channel_wise=False,
        use_fast_ffn=False,
        speculate_method=None,
        speculate_max_draft_token_num=1,
        return_all_hidden_states=False,
        draft_type="None",
        start_layer_index=0,
        use_moe=False,
        moe_num_experts=None,
        moe_intermediate_size=None,
        moe_use_gate_correction_bias=False,
        moe_every2=False,
        moe_topk=8,
        moe_num_shared_experts=0,
        moe_layer_start_index=0,
        moe_use_ffn_shared_weight_and_bias=False,
        moe_group=False,
        moe_quant_type="default",
        use_ep=False,
        ep_just_for_test=True,
        generation_phase=GenerationPhase.PREFILL,
        use_micro_batch=False,
        weight_block_size=[-1, -1],
        scale_dir="None",
        output_via_mq=True,
        enable_redundant_experts: bool = False,
        redundant_experts_num: int = 0,
        max_batch_size: int = 128,
        use_offline_quant=False,
        sharing_model=None,
        llm_config=None,
    ):
        """
        Initializer for the ErnieBotFusedModel class.

        Args:
            vocab_size (int): The size of the vocabulary.
            hidden_size (int): The size of the hidden layers.
            num_layers (int): The number of hidden layers.
            num_attention_heads (int): The number of attention heads for each attention layer.
            ffn_hidden_size (int): The size of the intermediate feed-forward layer.
            activation (str): The activation function to use.
            hidden_dropout_prob (float): The dropout probability for hidden layers.
            max_position_embeddings (int): The maximum sequence length that this model can handle.
            type_vocab_size (int): The size of the token type vocabulary.
            initializer_range (float): The range of the initializer.
            dtype (str): The data type to use.
            block_size (int): The block size for processing.
            inv_compression_ratio (float): The inverse compression ratio.
            sequence_parallel (bool): Whether to use sequence parallel processing.
            freeze_embedding (bool): Whether to freeze the embedding layer.
            max_len (int): The maximum length of the input sequence.
            use_rope (bool): Whether to use RoPE (Rotary Position Embedding).
            weight_sharing (bool): Whether to share weights between layers.
            weight_sharing_add_bias (bool): Whether to add bias when sharing weights.
            export_model_type (str): The type of model to export.
            wint4_smooth (bool): Whether to use WINT4 smoothing.
            group_size (int): The size of the group.
            model_path (str): The path of the inference model.
            use_rmsnorm (bool): Whether to use RMS normalization.
            msg_queue_id (int, optional): The ID of the message queue.
            use_fake_parameter (bool): Whether to use fake parameters.
            num_key_value_heads (int): The number of key-value heads.
            use_stop_seqs (bool): Whether to use stop sequences.
            cache_quant_dtype (str): The data type for cached quantization.
            use_fast_ffn (bool): Whether to use a fast feed-forward network.
        """
        super(ErnieBotFusedModel, self).__init__(llm_config)
        self.msg_queue_id = msg_queue_id
        self.initializer_range = initializer_range
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_attention_heads = num_attention_heads
        self.ffn_hidden_size = ffn_hidden_size
        self.num_layers = num_layers
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.use_rope = use_rope
        self.max_len = max_len
        self.use_fast_ffn = use_fast_ffn
        self.use_ep = use_ep
        self.ep_just_for_test = ep_just_for_test
        self.generation_phase = generation_phase
        self.use_micro_batch = use_micro_batch

        self.output_via_mq = output_via_mq

        self.block_size = block_size
        self.inv_compression_ratio = inv_compression_ratio
        self.dtype = dtype
        self.use_stop_seqs = use_stop_seqs

        self.export_model_type = export_model_type
        self.group_size = group_size

        self.use_rmsnorm = use_rmsnorm
        self.num_key_value_heads = num_key_value_heads
        self.cache_quant_dtype = cache_quant_dtype
        self.use_moe = use_moe
        self.sharing_model = sharing_model
        if self.use_rmsnorm:
            self.norm_type = "rmsnorm"
        else:
            self.norm_type = "layernorm"

        if activation == "SwiGLU":
            activation = "swiglu"

        hcg = fleet.get_hybrid_communicate_group()
        mp_size = hcg.get_model_parallel_world_size()
        mp_rank = hcg.get_model_parallel_rank()

        self.mp_size = mp_size
        self.mp_rank = mp_rank

        self.speculate_method = speculate_method
        self.return_all_hidden_states = return_all_hidden_states
        if mp_size <= 1:
            sequence_parallel = False
            logging.warning(
                "If mp_size <= 1, sequence_parallel strategy will be turned off in GPTModelHybrid model."
            )

        self.inference_args = InferenceArgs(
            quant_type=export_model_type,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            mp_rank=mp_rank,
            mp_size=mp_size,
            model_path=model_path,
            use_fake_parameter=use_fake_parameter,
            max_position_embeddings=max_position_embeddings,
            has_zero_point=has_zero_point,
            is_channel_wise=is_channel_wise,
            speculate_method=speculate_method,
            speculate_max_draft_token_num=speculate_max_draft_token_num,
            use_moe=use_moe,
            moe_num_experts=moe_num_experts,
            weight_block_size=weight_block_size,
            moe_intermediate_size=moe_intermediate_size,
            moe_use_gate_correction_bias=moe_use_gate_correction_bias,
            moe_every2=moe_every2,
            moe_topk=moe_topk,
            moe_num_shared_experts=moe_num_shared_experts,
            moe_layer_start_index=moe_layer_start_index,
            moe_use_ffn_shared_weight_and_bias=
            moe_use_ffn_shared_weight_and_bias,
            moe_group=moe_group,
            moe_quant_type=moe_quant_type,
            use_ep=use_ep,
            generation_phase=generation_phase,
            use_micro_batch=use_micro_batch,
            start_layer_index=start_layer_index,
            scale_dir=scale_dir,
        )

        fmt_keys = WeightKeys(num_layers)
        is_mtp = draft_type in ["eagle", "mtp"]
        self.is_mtp = is_mtp
        base_model_prefix = "ernie.mtp" if is_mtp else "ernie"
        self.base_model_prefix = base_model_prefix

        llm_config.model_config.max_position_embeddings = max_position_embeddings
        llm_config.model_config.initializer_range = self.initializer_range
        llm_config.parallel_config.sequence_parallel = sequence_parallel
        llm_config.model_config.freeze_embedding = freeze_embedding
        llm_config.model_config.weight_sharing = weight_sharing
        llm_config.model_config.weight_sharing_add_bias = weight_sharing_add_bias
        llm_config.parallel_config.use_ep = use_ep
        llm_config.parallel_config.ep_size = 1 if use_ep else 1
        llm_config.model_config.rope_head_dim = hidden_size // num_attention_heads
        llm_config.model_config.prefix_name = "ernie.mtp" if is_mtp else "ernie"
        llm_config.model_config.use_rope = use_rope
        llm_config.parallel_config.column_cut = False
        llm_config.model_config.base_model_prefix = base_model_prefix
        llm_config.model_config.use_moe = use_moe
        if enable_redundant_experts and use_moe:
            self.redundant_table_manger = RedundantExpertManger(
                n_routed_experts=moe_num_experts,
                num_hidden_layers=num_layers,
                redundant_experts_num=redundant_experts_num,
                ep_size=mp_size,
            )
        else:
            self.redundant_table_manger = None
        if use_moe and moe_layer_start_index > 0:
            fmt_keys.norm_before_qkv_weight_keys = [
                f"ernie.layers.{i}.input_layernorm.weight"
                for i in range(num_layers)
            ]
            fmt_keys.norm_before_qkv_bias_keys = [
                None for i in range(num_layers)
            ]
            fmt_keys.qkv_linear_weight_keys = [
                f"ernie.layers.{i}.self_attn.qkv_proj.weight"
                for i in range(num_layers)
            ]
            fmt_keys.qkv_linear_bias_keys = [None for i in range(num_layers)]
            fmt_keys.out_linear_weight_keys = [
                f"ernie.layers.{i}.self_attn.o_proj.weight"
                for i in range(num_layers)
            ]
            fmt_keys.out_linear_bias_keys = [None for i in range(num_layers)]

            fmt_keys.ffn_layernorm_weight_keys = [
                f"ernie.layers.{i}.post_attention_layernorm.weight"
                for i in range(num_layers)
            ]
            fmt_keys.ffn_layernorm_bias_keys = [
                None for i in range(num_layers)
            ]
            fmt_keys.ffn1_weight_keys = [
                f"ernie.layers.{i}.mlp.up_gate_proj.weight"
                for i in range(num_layers)
            ]
            fmt_keys.ffn1_bias_keys = [None for i in range(num_layers)]
            fmt_keys.ffn2_weight_keys = [
                f"ernie.layers.{i}.mlp.down_proj.weight"
                for i in range(num_layers)
            ]
            fmt_keys.ffn2_bias_keys = [None for i in range(num_layers)]

            # MoE keys
            fmt_keys.moe_gate_weight_keys = "ernie.layers.{}.mlp.gate.weight"
            fmt_keys.moe_gate_correction_bias_keys = "ernie.layers.{}.mlp.moe_statics.e_score_correction_bias"
            fmt_keys.moe_ffn1_weight_keys = "ernie.layers.{}.mlp.experts.{}.up_gate_proj.weight"
            fmt_keys.moe_ffn2_weight_keys = "ernie.layers.{}.mlp.experts.{}.down_proj.weight"
            
            # only w4a8 use these keys
            # fmt_keys.moe_ffn1_weight_scale_keys = "ernie.layers.{}.mlp.experts.{}.up_gate_proj.weight_quanter"
            # fmt_keys.moe_ffn2_weight_scale_keys = "ernie.layers.{}.mlp.experts.{}.down_proj.weight_quanter"
            # fmt_keys.moe_ffn1_in_scale_keys = "ernie.layers.{}.mlp.experts.{}.up_gate_proj.activation_quanter"
            # fmt_keys.moe_ffn2_in_scale_keys = "ernie.layers.{}.mlp.experts.{}.down_proj.activation_quanter"

        else:
            fmt_keys.norm_before_qkv_weight_keys = [
                f"{base_model_prefix}.decoder.layers.{i}.norm1.weight"
                for i in range(num_layers)
            ]
            fmt_keys.norm_before_qkv_bias_keys = [
                f"{base_model_prefix}.decoder.layers.{i}.norm1.bias"
                for i in range(num_layers)
            ]
            fmt_keys.qkv_linear_weight_keys = [
                f"{base_model_prefix}.decoder.layers.{i}.self_attn.qkv_proj.weight"
                for i in range(num_layers)
            ]
            fmt_keys.qkv_linear_bias_keys = [
                f"{base_model_prefix}.decoder.layers.{i}.self_attn.qkv_proj.bias"
                for i in range(num_layers)
            ]
            fmt_keys.out_linear_weight_keys = [
                f"{base_model_prefix}.decoder.layers.{i}.self_attn.out_proj.weight"
                for i in range(num_layers)
            ]
            fmt_keys.out_linear_bias_keys = [
                f"{base_model_prefix}.decoder.layers.{i}.self_attn.out_proj.bias"
                for i in range(num_layers)
            ]

            fmt_keys.ffn_layernorm_weight_keys = [
                f"{base_model_prefix}.decoder.layers.{i}.norm2.weight"
                for i in range(num_layers)
            ]
            fmt_keys.ffn_layernorm_bias_keys = [
                f"{base_model_prefix}.decoder.layers.{i}.norm2.bias"
                for i in range(num_layers)
            ]
            fmt_keys.ffn1_weight_keys = [
                f"{base_model_prefix}.decoder.layers.{i}.linear1.weight"
                for i in range(num_layers)
            ]
            fmt_keys.ffn1_bias_keys = [
                f"{base_model_prefix}.decoder.layers.{i}.linear1.bias"
                for i in range(num_layers)
            ]
            fmt_keys.ffn2_weight_keys = [
                f"{base_model_prefix}.decoder.layers.{i}.linear2.weight"
                for i in range(num_layers)
            ]
            fmt_keys.ffn2_bias_keys = [
                f"{base_model_prefix}.decoder.layers.{i}.linear2.bias"
                for i in range(num_layers)
            ]
        if sharing_model is not None:
            self.embeddings = sharing_model.gpt.embeddings
        else:
            self.embeddings = VocabParallelEmbedding(
                llm_config=llm_config,
                num_embeddings=vocab_size,
                embedding_dim=hidden_size,
                params_dtype=paddle.get_default_dtype,
                prefix=(f"{base_model_prefix}.embeddings.word_embeddings"
                            if not use_moe else "ernie.embed_tokens"),
            )

        # get ring_id
        ring_id = -1
        try:
            model_parallel_group = hcg.get_model_parallel_group()
            ring_id = model_parallel_group.id
        except Exception:
            pass
        if mp_size == 1:
            ring_id = -1

        self.ffn1_concat = True
        self.use_smooth_quant = False
        self.fuse_ffn_act = False
        if ("float8" in self.inference_args.weight_dtype
                and self.inference_args.act_dtype
                == self.inference_args.weight_dtype):
            self.ffn1_concat = False
            self.fuse_ffn_act = True
        if (self.inference_args.weight_dtype == "int8"
                and self.inference_args.act_dtype
                == self.inference_args.weight_dtype):
            self.use_smooth_quant = True

        if self.inference_args.use_weight_only and self.inference_args.act_dtype in [
                "bfloat16",
                "float16",
                "float32",
        ]:
            quant_cls = get_quantization_config("weight_only")
            llm_config.quant_config = quant_cls.from_config({
                "weight_only_linear_arch":
                self.inference_args.weight_only_linear_arch,
                "algo":
                "weight_only_int8"
            })
        elif self.inference_args.weight_block_size[0] != -1:
            quant_cls = get_quantization_config("block_wise")
            llm_config.quant_config = quant_cls.from_config(
                {"weight_block_size": self.inference_args.weight_block_size})
        elif self.weight_dtype == "int4" and self.act_dtype in [
                "bfloat16",
                "float16",
                "float32",
        ]:
            quant_cls = get_quantization_config("weight_only")
            llm_config.quant_config = quant_cls.from_config({
                "weight_only_linear_arch":
                self.inference_args.weight_only_linear_arch,
                "algo":
                "weight_only_int4"
            })
        elif (self.inference_args.weight_dtype == "int4"
              and self.inference_args.act_dtype == "float8_e4m3fn"):  # W4Afp8
            quant_cls = get_quantization_config("w4afp8")
            llm_config.quant_config = quant_cls.from_config({
                "weight_scale_dict":
                self.inference_args.weight_scale_dict,
                "act_scale_dict":
                self.inference_args.act_scale_dict
            })
        elif self.inference_args.weight_dtype == "int8" and self.inference_args.act_dtype == self.weight_dtype:
            use_gemm_dequant = os.getenv("FLAGS_use_gemm_dequant")
            if use_gemm_dequant is not None:
                use_gemm_dequant = int(use_gemm_dequant) == 1
            else:
                use_gemm_dequant = False
            quant_cls = get_quantization_config("w8a8")
            llm_config.quant_config = quant_cls.from_config({
                "weight_scale_dict":
                self.inference_args.weight_scale_dict,
                "act_scale_dict":
                self.inference_args.act_scale_dict,
                "use_gemm_dequant":
                use_gemm_dequant
            })
        elif ("float8" in self.inference_args.weight_dtype
              and self.inference_args.act_dtype
              == self.inference_args.weight_dtype):
            quant_cls = get_quantization_config("wfp8afp8")
            llm_config.quant_config = quant_cls.from_config({
                "weight_scale_dict":
                self.inference_args.weight_scale_dict,
                "act_scale_dict":
                self.inference_args.act_scale_dict
            })

        else:
            llm_config.quant_config = None

        if self.inference_args.cachekv_dtype not in [
                "bfloat16", "float16", "float32"
        ]:
            quant_cls = get_quantization_config("kvcache")
            llm_config.kvcache_quant_config = quant_cls.from_config(
                {"cachekv_scale_dict": self.inference_args.cachekv_scale_dict})
        else:
            llm_config.kvcache_quant_config = None

        # we will move use_smooth_quant to quant_config later
        llm_config.model_config.speculate_method = self.speculate_method
        llm_config.model_config.use_smooth_quant = self.use_smooth_quant
        # we will remove later
        llm_config.model_config.weight_dtype = self.inference_args.weight_dtype
        # we will remove act_dtype later
        llm_config.model_config.act_dtype = self.inference_args.act_dtype
        llm_config.parallel_config.mp_size = mp_size
        llm_config.load_config.weight_keys = fmt_keys
        llm_config.quant_config.quant_round_type = self.inference_args.quant_round_type
        llm_config.quant_config.quant_max_bound = self.inference_args.quant_max_bound
        llm_config.quant_config.quant_min_bound = self.inference_args.quant_min_bound
        llm_config.load_config.act_scales = self.inference_args.act_scale_dict
        llm_config.load_config._post_init(llm_config.model_config)
        # cachekv
        if llm_config.kvcache_quant_config is not None:
            llm_config.kvcache_quant_config.cache_quant_type_str = \
                ("none" if self.inference_args.use_dynamic_cachekv_quant
                 else self.inference_args.cache_quant_type)
            llm_config.kvcache_quant_config.cachekv_dtype = self.inference_args.cachekv_dtype
            llm_config.kvcache_quant_config.has_zero_point = self.inference_args.has_zero_point
            llm_config.kvcache_quant_config.use_append_attn = self.inference_args.use_append_attn
            llm_config.kvcache_quant_config.is_channel_wise = self.inference_args.is_channel_wise
            llm_config.kvcache_quant_config.use_dynamic_cachekv_quant = self.inference_args.use_dynamic_cachekv_quant

        self.decoder = FusedTransformer(
            inference_args=self.inference_args,
            fmt_keys=fmt_keys,
            act_method=activation,
            rope_theta=rope_theta,
            rope_3d=rope_3d,
            ffn1_concat=self.ffn1_concat,
            use_smooth_quant=self.use_smooth_quant,
            fuse_ffn_act=self.fuse_ffn_act,
            ring_id=ring_id,
            return_all_hidden_states=self.return_all_hidden_states,
            base_model_prefix=base_model_prefix,
            draft_type=draft_type,
            llm_config=llm_config,
            redundant_table_manger=self.redundant_table_manger,
        )
        if sharing_model is not None:
            self.norm = sharing_model.gpt.norm
        else:
            self.norm = RMSNorm(
                llm_config,
                hidden_size=llm_config.model_config.hidden_size,
                eps=1e-5,
                prefix=f"{base_model_prefix}.norm",
            )

        if is_mtp:
            self.e_norm = RMSNorm(
                llm_config,
                hidden_size=llm_config.model_config.hidden_size,
                eps=1e-5,
                layer_name=f"{base_model_prefix}.e_norm",
            )
            self.h_norm = RMSNorm(
                llm_config,
                hidden_size=llm_config.model_config.hidden_size,
                eps=1e-5,
                layer_name=f"{base_model_prefix}.h_norm",
            )

            from paddle.distributed.fleet.meta_parallel import \
                ColumnParallelLinear

            self.eh_proj = ColumnParallelLinear(
                hidden_size * 2,
                hidden_size,
                has_bias=True,
                gather_output=True,
                fuse_matmul_bias=True,
            )

    def remove_padding(self, input_ids, seq_lens_this_time):
        """
        remove_padding
        """
        cum_offsets_now = paddle.cumsum(self.max_len - seq_lens_this_time)
        token_num = paddle.sum(seq_lens_this_time)
        (
            ids_remove_padding,
            cum_offsets,
            padding_offset,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = get_padding_offset(input_ids, cum_offsets_now, token_num,
                               seq_lens_this_time)
        return (
            ids_remove_padding,
            padding_offset,
            cum_offsets,
            cu_seqlens_q,
            cu_seqlens_k,
        )

    def speculate_remove_padding(self, input_ids, seq_lens_this_time,
                                 draft_tokens, seq_lens_encoder):
        """
        remove_padding
        """
        cum_offsets_now = paddle.cumsum(self.max_len - seq_lens_this_time)
        token_num = paddle.sum(seq_lens_this_time)
        (
            ids_remove_padding,
            cum_offsets,
            padding_offset,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = speculate_get_padding_offset(
            input_ids,
            draft_tokens,
            cum_offsets_now,
            token_num,
            seq_lens_this_time,
            seq_lens_encoder,
        )
        return (
            ids_remove_padding,
            padding_offset,
            cum_offsets,
            cu_seqlens_q,
            cu_seqlens_k,
        )

    def forward(
        self,
        input_ids,
        forward_meta: ForwardMeta,
        token_type_ids=None,
        image_features=None,
        attention_mask=None,  # for NPU
        rope_emb=None,
        caches=None,
        seq_lens_this_time=None,
        seq_lens_encoder=None,
        seq_lens_decoder=None,
        block_tables=None,
        beam_cache_offset=None,
        draft_tokens=None,
        output_padding_offset=None,
        step_idx=None,
        hidden_states=None,
    ):
        """
            Args:
            input_ids (Tensor, optional): Input sequence of shape `(batch_size, sequence_length)`. Defaults to None.
            attention_mask (Tensor, optional): Mask to avoid performing attention on padding tokens. Defaults to None.
            rope_emb (Tensor, optional): Rotary positional embeddings. Defaults to None.
            caches (List[Tensor], optional): List of cached decoder states. Defaults to None.
            seq_lens_this_time (Tensor, optional): Sequence lengths of this time step. Defaults to None.
            seq_lens_encoder (Tensor, optional): Sequence lengths of encoder. Defaults to None.
            seq_lens_decoder (Tensor, optional): Sequence lengths of decoder. Defaults to None.
            block_tables (Tensor, optional): Block table for compression. Defaults to None.
            beam_cache_offset (int, optional): Beam cache offset. Defaults to None.
            step_idx (int, optional): Step index. Defaults to None.

        Returns:
            Tensor: Output tensor of shape `(batch_size, sequence_length, hidden_size)`.
        """

        embedding_output = self.embeddings(
            ids_remove_padding=forward_meta.ids_remove_padding)
        if self.is_mtp:
            embedding_output = paddle.concat(
                [self.e_norm(embedding_output),
                 self.h_norm(hidden_states)],
                axis=-1)
            embedding_output = self.eh_proj(embedding_output)

        if isinstance(embedding_output, tuple):
            embedding_output = embedding_output[0]
        else:
            embedding_output = embedding_output

        if (self.inference_args.moe_config.use_moe
                and self.inference_args.moe_config.has_multimodality):
            token_type_ids = (forward_meta.ids_remove_padding ==
                              self.inference_args.moe_config.im_patch_id)
            image_mask = token_type_ids
            if image_mask.any():
                embedding_output[image_mask] = image_features.cast(
                    embedding_output.dtype)

        output = self.decoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            src=embedding_output,
            caches=caches,
            rotary_embs=forward_meta.rotary_embs,
            rotary_emb_dims=1,
            max_input_length=self.max_len,
            block_size=self.block_size,
            inv_compression_ratio=self.inv_compression_ratio,
            cum_offsets=forward_meta.cum_offsets,
            cu_seqlens_q=forward_meta.cu_seqlens_q,
            cu_seqlens_k=forward_meta.cu_seqlens_k,
            padding_offsets=forward_meta.padding_offset,
            block_tables=forward_meta.block_tables,
            seq_lens_this_time=forward_meta.seq_lens_this_time,
            seq_lens_encoder=forward_meta.seq_lens_encoder,
            seq_lens_decoder=forward_meta.seq_lens_decoder,
            attention_mask=attention_mask,  # for NPU
            beam_cache_offset=beam_cache_offset,
            draft_tokens=draft_tokens,
            output_padding_offset=output_padding_offset,
            return_all_hidden_states=self.return_all_hidden_states,
            forward_meta=forward_meta,
        )

        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output

        if (self.inference_args.moe_config.use_moe
                and self.inference_args.moe_config.has_multimodality):
            out = out.cast("float32")
            score_text = out

            mm_token_num_len = paddle.count_nonzero(token_type_ids).cast(
                "int32")

            if mm_token_num_len > 0:
                token_num = paddle.shape(forward_meta.ids_remove_padding)[0]
                token_type_ids = token_type_ids.reshape([-1])
                text_pos_shifted = token_type_ids[:token_num] == 0
                score_text = out[text_pos_shifted.reshape([-1])]

            max_seq_len, max_seq_len_index = paddle.topk(
                seq_lens_this_time.squeeze(-1), k=1)
            out = extract_text_token_output(
                max_seq_len,
                max_seq_len_index.cast("int32"),
                mm_token_num_len,
                seq_lens_this_time,
                forward_meta.cu_seqlens_q,
                score_text,
            )[0].cast(embedding_output.dtype)

        out = self.norm(out)

        if self.return_all_hidden_states:
            return out, forward_meta.cum_offsets
        else:
            return out


class ErnieForCausalLM(ModelForCasualLM):
    """
    ErnieForCausalLM
    """

    def __init__(self, llm_config):
        """
        Args:
            llm_config (LLMConfig): Configurations for the LLM model.

        Raises:
            ValueError: If the export_model_type is W8A8C8 or W8A8C16 and use_rmsnorm is True.
            ValueError: If use_rmsnorm is True and norm_type is not 'rmsnorm'.
            ValueError: If norm_type is not 'layernorm' or 'rmsnorm'.
            ValueError: If use_cache_kv_int8 is True and use_fake_parameter is True.
        """
        super(ErnieForCausalLM, self).__init__(llm_config)
        self.configs = llm_config
        self.ernie = ErnieBotFusedModel(
            vocab_size=self.configs.model_config.vocab_size,
            hidden_size=self.configs.model_config.hidden_size,
            max_len=self.configs.model_config.max_seq_len,
            block_size=self.configs.parallel_config.block_size,
            num_layers=self.configs.model_config.num_layers,
            num_attention_heads=self.configs.model_config.num_attention_heads,
            ffn_hidden_size=self.configs.model_config.ffn_hidden_size,
            activation="swiglu",
            hidden_dropout_prob=0,
            max_position_embeddings=self.configs.model_config.
            max_position_embeddings,
            type_vocab_size=1,
            dtype=self.configs.model_config.dtype,
            sequence_parallel=False,
            use_rope=True,
            rope_theta=self.configs.model_config.rope_theta,
            rope_3d=self.configs.model_config.rope_3d,
            weight_sharing=False,
            inv_compression_ratio=1.0 /
            self.configs.model_config.compression_ratio,
            export_model_type=self.configs.model_config.
            export_model_type,  # export model type.
            group_size=self.configs.model_config.group_size,
            model_path=self.configs.load_config.
            model_path,  # The path of Inference model.
            use_rmsnorm=self.configs.model_config.use_rmsnorm,
            msg_queue_id=self.configs.parallel_config.msg_queue_id,
            use_fake_parameter=self.configs.additional_config.
            use_fake_parameter,
            num_key_value_heads=self.configs.model_config.num_key_value_heads,
            use_stop_seqs=self.configs.model_config.use_stop_seqs,
            cache_quant_dtype=self.configs.tmp_config.cache_quant_dtype,
            has_zero_point=self.configs.tmp_config.has_zero_point,
            is_channel_wise=self.configs.tmp_config.is_channel_wise,
            use_fast_ffn=self.configs.model_config.use_fast_ffn,
            speculate_method=self.configs.speculative_config.speculate_method,
            speculate_max_draft_token_num=self.configs.speculative_config.
            speculate_max_draft_token_num,
            return_all_hidden_states=self.configs.model_config.
            return_all_hidden_states,
            draft_type=self.configs.speculative_config.draft_type,
            start_layer_index=self.configs.model_config.start_layer_index,
            use_moe=self.configs.moe_config.use_moe,
            moe_num_experts=self.configs.moe_config.num_experts,
            moe_intermediate_size=self.configs.moe_config.
            moe_intermediate_size,
            moe_use_gate_correction_bias=self.configs.moe_config.
            moe_use_gate_correction_bias,
            moe_every2=self.configs.moe_config.moe_every2,
            moe_topk=self.configs.moe_config.moe_topk,
            moe_num_shared_experts=self.configs.moe_config.
            moe_num_shared_experts,
            moe_layer_start_index=self.configs.moe_config.
            moe_layer_start_index,
            moe_use_ffn_shared_weight_and_bias=self.configs.moe_config.
            moe_use_ffn_shared_weight_and_bias,
            moe_group=self.configs.moe_config.moe_group,
            moe_quant_type=self.configs.moe_config.moe_quant_type,
            use_ep=self.configs.parallel_config.use_ep,
            ep_just_for_test=self.configs.additional_config.ep_just_for_test,
            generation_phase=self.configs.model_config.generation_phase,
            use_micro_batch=self.configs.parallel_config.use_micro_batch,
            weight_block_size=self.configs.tmp_config.weight_block_size,
            scale_dir=self.configs.load_config.scale_dir,
            output_via_mq=self.configs.model_config.output_via_mq,
            llm_config=self.configs,
        )

        self.msg_queue_id = self.ernie.msg_queue_id
        self.max_length = self.configs.decoding_config.max_dec_len
        self.min_length = self.configs.decoding_config.min_dec_len
        self.fake_server_p = self.configs.additional_config.fake_server_p
        self.decode_strategy = self.configs.decoding_config.decode_strategy
        self.speculate_max_candidate_len = self.configs.speculative_config.speculate_max_candidate_len
        self.speculate_verify_window = self.configs.speculative_config.speculate_verify_window
        self.use_moe = self.ernie.use_moe

        assert self.decode_strategy in [
            "greedy_search",
            "sampling",
            "beam_search",
            "speculate_decoding",
            "draft_model_sampling",
        ], f"`decode_strategy` must be one of 'greedy_search', 'sampling', \
            'speculate_decoding' or 'beam_search' but received {self.decode_strategy}."

        self.ori_vocab_size = self.configs.model_config.ori_vocab_size

        self.top_k = self.configs.moe_config.top_k
        self.bos_token_id = self.configs.decoding_config.bos_token_id
        self.pad_token_id = self.configs.decoding_config.pad_token_id
        self.num_return_sequences = self.configs.decoding_config.num_return_sequences
        self.weight_sharing = self.configs.model_config.weight_sharing
        self.weight_sharing_add_bias = self.configs.model_config.weight_sharing_add_bias

        self.export_model_type = self.configs.model_config.export_model_type
        self.group_size = self.configs.model_config.group_size
        self.weightonly_groupwise = True if self.group_size > 0 else False

        self.use_rmsnorm = self.configs.model_config.use_rmsnorm
        self.use_fake_parameter = self.configs.additional_config.use_fake_parameter
        self.cache_quant_dtype = self.configs.tmp_config.cache_quant_dtype
        if self.cache_quant_dtype == "default":
            self.cache_quant_dtype = paddle.get_default_dtype()
        self.use_fast_ffn = self.ernie.use_fast_ffn

        # for NPU
        self.hidden_size = self.configs.model_config.hidden_size
        self.num_attention_heads = self.configs.model_config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.rank = (paddle.distributed.fleet.get_hybrid_communicate_group().
                     get_model_parallel_rank())
        self.nranks = (paddle.distributed.fleet.get_hybrid_communicate_group().
                       get_model_parallel_world_size())
        self.root = 0
        self.ring_id = (paddle.distributed.fleet.get_hybrid_communicate_group(
        ).get_model_parallel_group().id)

        self.return_all_hidden_states = self.configs.model_config.return_all_hidden_states

        self.base_model_prefix = self.ernie.base_model_prefix

        if self.use_rmsnorm:
            self.norm_type = "rmsnorm"
            # rmsnorm use fp16/bf16 weight
            self.have_norm_bias = False
            self.is_norm_weight_type_fp32 = False
        else:
            # by default, use layernorm
            self.norm_type = "layernorm"
            # layernorm use fp32 weight
            self.have_norm_bias = True
            self.is_norm_weight_type_fp32 = True

        if self.weight_sharing:
            tie_word_embeddings = self.ernie.embeddings.word_embeddings.weight
        else:
            tie_word_embeddings = None

        if self.ernie.sharing_model is not None:
                self.lm_head = self.ernie.sharing_model.lm_head
        else:
            layer_prefix = None
            if self.use_moe:
                layer_prefix = "lm_head"
            else:
                layer_prefix = f"{self.base_model_prefix}"
            if self.use_moe:
                self.lm_head = ParallelLMHead(
                    llm_config=llm_config,
                    embedding_dim=self.hidden_size,
                    num_embeddings=self.ernie.vocab_size,
                    tie_word_embeddings=tie_word_embeddings,
                    prefix=layer_prefix,
                )
            else:
                self.lm_head = ParallelLMHead(
                    llm_config=llm_config,
                    embedding_dim=self.hidden_size,
                    num_embeddings=self.ernie.vocab_size,
                    tie_word_embeddings=tie_word_embeddings,
                    prefix=layer_prefix,
                )


    @classmethod
    def name(self):
        return "ErnieForCausalLM"

    @paddle.no_grad()
    def set_state_dict(self, state_dict: Dict[str, Union[np.ndarray,
                                                         paddle.Tensor]]):
        """
        Load model parameters from a given state dictionary.

        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
        try:
            self.ernie.embeddings.load_state_dict(state_dict)
            self.ernie.decoder.load_state_dict(state_dict)
            self.ernie.norm.load_state_dict(state_dict)
            self.lm_head.load_state_dict(state_dict)
            if self.ernie.is_mtp:
                self.ernie.e_norm.load_state_dict(state_dict)
                self.ernie.h_norm.load_state_dict(state_dict)
                self.ernie.eh_proj.weight.set_value(
                    paddle.to_tensor(
                        state_dict[f"{self.base_model_prefix}.eh_proj.weight"])
                )
                self.ernie.eh_proj.bias.set_value(
                    paddle.to_tensor(
                        state_dict[f"{self.base_model_prefix}.eh_proj.bias"]))
        except Exception:
            raise RuntimeError("set_state_dict error!!!")

    def get_output_padding_offset(self, seq_lens_this_time, seq_lens_encoder,
                                  seq_lens_decoder):
        """
        In the senerio of speculate decoding, the length of output token after rebuild_padding is no longer bsz.
        So we need to calculate the output_padding_offset after rebuild_padding.
        """
        seq_lens_output = speculate_get_seq_lens_output(
            seq_lens_this_time, seq_lens_encoder, seq_lens_decoder)
        out_token_num = paddle.sum(seq_lens_output)
        output_cum_offsets_tmp = paddle.cumsum(self.ernie.max_len -
                                               seq_lens_output)
        output_padding_offset, output_cum_offsets = speculate_get_output_padding_offset(
            output_cum_offsets_tmp, out_token_num, seq_lens_output,
            self.ernie.max_len)
        return output_padding_offset, output_cum_offsets

    def expand_inputs_for_generation(self,
                                     input_ids,
                                     expand_size,
                                     attention_mask=None,
                                     **model_kwargs):
        """
        Expand input IDs for generation.

        This method expands the input IDs by duplicating the first dimension and filling in the expanded indices.
        It also updates any other keyword arguments that may contain tensors with the same shape as `input_ids`.

        Args:
            input_ids (Tensor): Input IDs to be expanded.
            expand_size (int): Size of expansion.
            attention_mask (Tensor, optional): Attention mask to be updated. Defaults to None.
            **model_kwargs: Additional keyword arguments containing tensors with the same shape as `input_ids`.

        Returns:
            tuple: A tuple containing the expanded input IDs and the updated keyword arguments.
        """
        index = paddle.tile(
            paddle.arange(paddle.shape(input_ids)[0]).unsqueeze(-1),
            [1, expand_size],
        ).reshape([-1])

        input_ids = paddle.gather(input_ids, index)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = paddle.gather(
                attention_mask, index)

        if ("token_type_ids" in model_kwargs
                and model_kwargs["token_type_ids"] is not None):
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = paddle.gather(
                token_type_ids, index)

        if "position_ids" in model_kwargs and model_kwargs[
                "position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.gather(position_ids, index)

        if "seq_len" in model_kwargs and model_kwargs["seq_len"] is not None:
            seq_len = model_kwargs["seq_len"]
            model_kwargs["seq_len"] = paddle.gather(seq_len, index)

        if ("encoder_output" in model_kwargs
                and model_kwargs["encoder_output"] is not None):
            encoder_output = model_kwargs["encoder_output"]
            model_kwargs["encoder_output"] = paddle.gather(
                encoder_output, index)

        if "role_ids" in model_kwargs and model_kwargs["role_ids"] is not None:
            role_ids = model_kwargs["role_ids"]
            model_kwargs["role_ids"] = paddle.gather(role_ids, index)

        return input_ids, model_kwargs

    def prepare_inputs_for_generation(self, **kwargs):
        """
        Prepare input IDs for generation.

        This method prepares the input IDs for generation by expanding them based on the number of beams.

        Args:
            **kwargs: Keyword arguments containing tensors with the same shape as `input_ids`.

        Returns:
            dict: A dictionary containing the prepared input IDs and the updated keyword arguments.
        """
        # only last token for inputs_ids if cache is defined in kwargs
        input_ids = kwargs["input_ids"]
        image_features = kwargs.get("image_features", None)
        attention_mask = kwargs.get("attention_mask", None)
        block_tables = kwargs.get("block_tables", None)
        caches = kwargs.get("caches", None)
        beam_offset = kwargs.get("beam_offset", None)
        rope_emb = kwargs["rope_emb"]
        seq_lens_this_time = kwargs["seq_lens_this_time"]
        seq_lens_encoder = kwargs["seq_lens_encoder"]
        seq_lens_decoder = kwargs["seq_lens_decoder"]
        draft_tokens = kwargs.get("draft_tokens", None)
        output_padding_offset = kwargs.get("actual_output_padding_offset",
                                           None)
        hidden_states = kwargs.get("hidden_states", None)
        forward_meta = kwargs["forward_meta"]
        model_inputs = {
            "input_ids": input_ids,
            "image_features": image_features,
            "attention_mask": attention_mask,
            "rope_emb": rope_emb,
            "caches": caches,
            "seq_lens_this_time": seq_lens_this_time,
            "seq_lens_encoder": seq_lens_encoder,
            "seq_lens_decoder": seq_lens_decoder,
            "block_tables": block_tables,
            "beam_cache_offset": beam_offset,
            "draft_tokens": draft_tokens,
            "output_padding_offset": output_padding_offset,
            "hidden_states": hidden_states,
            "forward_meta": forward_meta,
        }
        return model_inputs

    def sampling(
        self,
        logits,
        **model_kwargs,
    ):
        """Sample from GPT using beam search and post process the generated sequence.

        Args:
            logits (Tensor): The id of the token indicating the end of a sentence.
            **model_kwargs: Other arguments for forward pass of GPT model.

        Returns:
            Tensor: The sampled tokens. The shape is [batch_size].
        """
        temperature = model_kwargs["temperature"]
        top_k = self.top_k
        top_p = model_kwargs["top_p"]
        eos_token_id = model_kwargs["eos_token_id"]
        penalty_score = model_kwargs["penalty_score"]
        frequency_score = model_kwargs["frequency_score"]
        presence_score = model_kwargs["presence_score"]

        def _post_process_(
            logits,
            top_k,
            top_p,
            penalty_score,
            frequency_score,
            presence_score,
            temperature,
            model_kwargs,
        ):
            """
            Post process the generated sequence.
            """
            step_idx = model_kwargs["step_idx"]

            set_value_by_flags_and_idx(
                model_kwargs["pre_ids"],
                model_kwargs["input_ids"],
                model_kwargs["seq_lens_this_time"],
                model_kwargs["seq_lens_encoder"],
                model_kwargs["seq_lens_decoder"],
                step_idx,
                model_kwargs["stop_flags"],
            )

            # pre-process distribution
            logits = get_token_penalty_multi_scores(
                model_kwargs["pre_ids"],
                logits,
                penalty_score,
                frequency_score,
                presence_score,
                temperature,
                model_kwargs["bad_tokens"],
                step_idx,
                model_kwargs["min_dec_len"],
                eos_token_id,
            )

            # sample
            probs = F.softmax(logits)
            _, next_tokens = paddle.tensor.top_p_sampling(
                probs, top_p, seed=-1)  # have random_seed
            """ !!! ep not need broadcast, here broadcast just for test !!! """
            if self.ernie.mp_size > 1 and (
                (not self.ernie.use_ep or self.ernie.ep_just_for_test) and
                (not self.fake_server_p)):
                paddle.distributed.broadcast(next_tokens, 0)

            paddle.assign(
                paddle.where(
                    model_kwargs["stop_flags"],
                    model_kwargs["step_idx"],
                    model_kwargs["step_idx"] + 1,
                ),
                model_kwargs["step_idx"],
            )
            length_cond = paddle.greater_equal(model_kwargs["step_idx"],
                                               model_kwargs["max_dec_len"])
            paddle.assign(
                paddle.logical_or(model_kwargs["stop_flags"], length_cond),
                model_kwargs["stop_flags"],
            )

            if self.ernie.use_stop_seqs:
                set_stop_value_multi_seqs(
                    next_tokens,
                    model_kwargs["pre_ids"],
                    step_idx,
                    model_kwargs["stop_flags"],
                    model_kwargs["seq_lens_this_time"],
                    model_kwargs["stop_seqs"],
                    model_kwargs["stop_seqs_len"],
                    eos_token_id,
                )
            else:
                set_stop_value_multi_ends(
                    next_tokens,
                    model_kwargs["stop_flags"],
                    model_kwargs["seq_lens_this_time"],
                    eos_token_id,
                    model_kwargs["next_tokens"],
                    False,
                )  # multi ends
            # update inputs
            with paddle.framework._no_check_dy2st_diff():
                update_inputs(
                    model_kwargs["stop_flags"],
                    model_kwargs["not_need_stop"],
                    model_kwargs["seq_lens_this_time"],
                    model_kwargs["seq_lens_encoder"],
                    model_kwargs["seq_lens_decoder"],
                    model_kwargs["input_ids"],
                    model_kwargs["stop_nums"],
                    next_tokens,
                    model_kwargs["is_block_step"],
                )
            if self.ernie.output_via_mq:
                if self.msg_queue_id is None:
                    save_output(
                        next_tokens,
                        model_kwargs["not_need_stop"],
                        self.ernie.mp_rank,
                        self.ernie.use_ep
                        and (not self.ernie.ep_just_for_test),
                    )
                else:
                    save_output_dynamic(
                        next_tokens,
                        model_kwargs["not_need_stop"],
                        self.ernie.mp_rank,
                        self.msg_queue_id,
                        self.ernie.use_ep
                        and (not self.ernie.ep_just_for_test),
                    )
            return next_tokens

        if ((not self.ernie.use_ep) or (self.ernie.ep_just_for_test)
                or (self.ernie.use_ep and model_kwargs["not_need_stop"])):
            # first decoder
            next_tokens = _post_process_(
                logits,
                top_k,
                top_p,
                penalty_score,
                frequency_score,
                presence_score,
                temperature,
                model_kwargs,
            )
        else:
            # fake ep
            fake_input = paddle.empty(
                shape=[0, self.ernie.inference_args.hidden_size],
                dtype=paddle.get_default_dtype(),
            )
            for i in range(
                    self.ernie.inference_args.moe_config.moe_layer_start_index,
                    self.ernie.inference_args.num_layers,
            ):
                self.ernie.decoder.moe_layers[i](fake_input)
            next_tokens = None

        return next_tokens

    def speculate_decoding(
        self,
        outputs,  # hidden_states
        **model_kwargs,
    ):
        """Sample from GPT using beam search and post process the generated sequence.

        Args:
            eos_token_id (int): The id of the token indicating the end of a sentence.
            top_p (float): If set to float < 1, only the tokens with probabilities greater than or equal to
                the threshold are kept for generation.
            penalty_score (dict): A dict containing penalty scores of different types.
            frequency_score (dict): A dict containing frequency score of each token.
            presence_score (dict): A dict containing presence score of each token.
            temperature (float, optional): The value used to module the logits. Defaults to None.
            min_tokens_to_keep (int, optional): Minimal number of tokens to keep for
                next step in decoding. Defaults to 1.
            **model_kwargs: Other arguments for forward pass of GPT model.

        Returns:
            Tensor: The sampled tokens. The shape is [batch_size].
        """
        temperature = model_kwargs["temperature"]
        top_p = model_kwargs["top_p"]
        eos_token_id = model_kwargs["eos_token_id"]
        penalty_score = model_kwargs["penalty_score"]
        frequency_score = model_kwargs["frequency_score"]
        presence_score = model_kwargs["presence_score"]

        def _post_process_(
            outputs,
            top_p,
            penalty_score,
            frequency_score,
            presence_score,
            temperature,
            model_kwargs,
        ):
            """
            Post process the generated sequence.
            """

            if self.return_all_hidden_states:
                all_hidden_states = outputs[0]
                cum_offsets = outputs[1]
                hidden_states = speculate_rebuild_append_padding(
                    all_hidden_states,
                    cum_offsets,
                    model_kwargs["seq_lens_encoder"],
                    model_kwargs["seq_lens_decoder"],
                    model_kwargs["actual_output_padding_offset"],
                    self.ernie.max_len,
                )
            else:
                hidden_states = outputs[0] if isinstance(outputs,
                                                         tuple) else outputs
            logits = self.lm_head(hidden_states)

            logits = paddle.cast(logits, paddle.float32)
            logits[:, self.ori_vocab_size:] = -float("inf")

            speculate_get_token_penalty_multi_scores(
                model_kwargs["pre_ids"],
                logits,
                penalty_score,
                frequency_score,
                presence_score,
                temperature,
                model_kwargs["bad_tokens"],
                model_kwargs["step_idx"],
                model_kwargs["min_dec_len"],
                eos_token_id,
                model_kwargs["seq_lens_this_time"],
                model_kwargs["actual_output_padding_offset"],
                model_kwargs["output_cum_offsets"],
                self.ernie.max_len,
            )

            # sample
            probs = F.softmax(logits)

            verify_scores, verify_tokens, actual_candidate_len = top_p_candidates(
                probs,
                top_p,
                model_kwargs["actual_output_padding_offset"],
                self.speculate_max_candidate_len,
                self.ernie.max_len,
            )

            speculate_verify(
                model_kwargs["accept_tokens"],
                model_kwargs["accept_num"],
                model_kwargs["step_idx"],
                model_kwargs["stop_flags"],
                model_kwargs["seq_lens_encoder"],
                model_kwargs["seq_lens_decoder"],
                model_kwargs[
                    "draft_tokens"],  # Both input and output, need to write the last 1 token accepted to position 0.
                model_kwargs["seq_lens_this_time"],
                verify_tokens,
                verify_scores,
                model_kwargs["max_dec_len"],
                eos_token_id,
                model_kwargs["is_block_step"],
                model_kwargs["output_cum_offsets"],
                actual_candidate_len,
                model_kwargs["actual_draft_token_num"],
                top_p,
                self.ernie.max_len,
                self.speculate_verify_window,
                True,  # enable_topp
            )

            # BroadCast
            if self.ernie.mp_size > 1:
                paddle.distributed.broadcast(model_kwargs["accept_tokens"], 0)
                paddle.distributed.broadcast(model_kwargs["accept_num"], 0)
                paddle.distributed.broadcast(model_kwargs["step_idx"], 0)
                paddle.distributed.broadcast(model_kwargs["stop_flags"], 0)

            if self.ernie.use_stop_seqs:
                speculate_set_stop_value_multi_seqs(
                    model_kwargs["accept_tokens"],
                    model_kwargs["accept_num"],
                    model_kwargs["pre_ids"],
                    model_kwargs["step_idx"],
                    model_kwargs["stop_flags"],
                    model_kwargs["seq_lens_this_time"],
                    model_kwargs["stop_seqs"],
                    model_kwargs["stop_seqs_len"],
                    eos_token_id,
                )

            # Update
            speculate_update_v3(
                model_kwargs["seq_lens_encoder"],
                model_kwargs["seq_lens_decoder"],
                model_kwargs["not_need_stop"],
                model_kwargs["draft_tokens"],
                model_kwargs["actual_draft_token_num"],
                model_kwargs["accept_tokens"],
                model_kwargs["accept_num"],
                model_kwargs["stop_flags"],
                model_kwargs["seq_lens_this_time"],
                model_kwargs["is_block_step"],
                model_kwargs["stop_nums"],
            )
            # Streaming output
            if not (self.ernie.speculate_method == "mtp" and
                    self.ernie.generation_phase == GenerationPhase.PREFILL):
                if self.msg_queue_id is None:
                    speculate_save_output(
                        model_kwargs["accept_tokens"],
                        model_kwargs["accept_num"],
                        model_kwargs["not_need_stop"],
                        self.rank,
                    )
                else:
                    speculate_save_output_dynamic(
                        model_kwargs["accept_tokens"],
                        model_kwargs["accept_num"],
                        model_kwargs["not_need_stop"],
                        self.rank,
                        self.msg_queue_id,
                    )

            # If seq_lens_decoder is 0 (means stop), accept_num should be set to 0
            speculate_clear_accept_nums(model_kwargs["accept_num"],
                                        model_kwargs["seq_lens_decoder"])

            # Update pre_ids through accept tokens
            speculate_set_value_by_flags_and_idx(
                model_kwargs["pre_ids"],
                model_kwargs["accept_tokens"],
                model_kwargs["accept_num"],
                model_kwargs["stop_flags"],
                model_kwargs["seq_lens_this_time"],
                model_kwargs["seq_lens_encoder"],
                model_kwargs["seq_lens_decoder"],
                model_kwargs["step_idx"],
            )

        output_padding_offset, output_cum_offsets = self.get_output_padding_offset(
            model_kwargs["seq_lens_this_time"],
            model_kwargs["seq_lens_encoder"],
            model_kwargs["seq_lens_decoder"],
        )
        model_kwargs["actual_output_padding_offset"] = output_padding_offset
        model_kwargs["output_cum_offsets"] = output_cum_offsets

        # first decoder
        _post_process_(
            outputs,
            top_p,
            penalty_score,
            frequency_score,
            presence_score,
            temperature,
            model_kwargs,
        )

        return outputs

    def beam_search(
        self,
        outputs,  # hidden_states
        **model_kwargs,
    ):
        """Sample from GPT using beam search and post process the generated sequence.

        Args:
            eos_token_id (int): The id of the token indicating the end of a sentence.
            penalty_score (dict): A dict containing penalty scores of different types.
            frequency_score (dict): A dict containing frequency score of each token.
            presence_score (dict): A dict containing presence score of each token.
            temperature (float, optional): The value used to module the logits. Defaults to None.
            **model_kwargs: Other arguments for forward pass of GPT model.

        Returns:
            Tensor: BeamHypotheses. The shape is [batch_size * beam_width, max_dec_len].
        """
        temperature = model_kwargs["temperature"]
        eos_token_id = model_kwargs["eos_token_id"]
        penalty_score = model_kwargs["penalty_score"]
        frequency_score = model_kwargs["frequency_score"]
        presence_score = model_kwargs["presence_score"]

        def _post_process_(
            outputs,
            penalty_score,
            frequency_score,
            presence_score,
            temperature,
            **model_kwargs,
        ):
            step_idx = model_kwargs["step_idx"]

            set_value_by_flags_and_idx(
                model_kwargs["pre_ids"],
                model_kwargs["input_ids"],
                model_kwargs["seq_lens_this_time"],
                model_kwargs["seq_lens_encoder"],
                model_kwargs["seq_lens_decoder"],
                step_idx,
                model_kwargs["stop_flags"],
            )
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = self.lm_head(logits)

            logits = paddle.cast(logits, paddle.float32)
            update_inputs_beam(
                model_kwargs["beam_width"].cpu(),
                model_kwargs["seq_lens_this_time"],
                model_kwargs["seq_lens_encoder"],
                model_kwargs["input_ids"],
                logits,
            )

            logits[:, self.ori_vocab_size:] = -float("inf")
            # pre-process distribution
            logits = get_token_penalty_multi_scores(
                model_kwargs["pre_ids"],
                logits,
                penalty_score,
                frequency_score,
                presence_score,
                temperature,
                model_kwargs["bad_tokens"],
                step_idx,
                model_kwargs["min_dec_len"],
                eos_token_id,
            )

            tmp_seq = paddle.where(
                model_kwargs["seq_lens_decoder"] == 0,
                model_kwargs["seq_lens_encoder"] - 1,
                model_kwargs["seq_lens_decoder"],
            )

            next_tokens, parent_ids = beam_search_softmax(
                logits=logits,
                seq_lens=tmp_seq.astype("int32"),
                stop_flags=model_kwargs["stop_flags"],
                end_ids=eos_token_id.astype("int32"),
                step_ids=step_idx.astype("int32"),
                max_dec_lens=model_kwargs["max_dec_len"].astype("int32"),
                block_tables=model_kwargs["block_tables"],
                cum_scores=model_kwargs["cum_score"],
                beam_cache_ids=model_kwargs["beam_cache_ids"],
                beam_hyps=model_kwargs["beam_hyps"],
                beam_hyps_score=model_kwargs["beam_hyps_score"],
                beam_finished=model_kwargs["beam_finished"],
                beam_width=model_kwargs["beam_width"],
                beam_group_num=model_kwargs["beam_group_num"],
                length_penalty=model_kwargs["beam_length_penalty"],
                diversity_penalty=model_kwargs["beam_diversity_penalty"],
                fuse_softmax=True,
                early_stop=False,
            )

            next_tokens = next_tokens.astype("int64")

            paddle.assign(
                paddle.where(
                    model_kwargs["beam_finished"],
                    model_kwargs["step_idx"],
                    model_kwargs["step_idx"] + 1,
                ),
                model_kwargs["step_idx"],
            )
            length_cond = paddle.greater_equal(model_kwargs["step_idx"],
                                               model_kwargs["max_dec_len"])
            paddle.assign(
                paddle.logical_or(model_kwargs["beam_finished"], length_cond),
                model_kwargs["beam_finished"],
            )

            set_stop_value_multi_ends(
                next_tokens,
                model_kwargs["beam_finished"],
                model_kwargs["seq_lens_this_time"],
                eos_token_id,
                model_kwargs["next_tokens"],
                True,
            )  # multi ends

            # update inputs
            update_inputs(
                model_kwargs["beam_finished"],
                model_kwargs["not_need_stop"],
                model_kwargs["seq_lens_this_time"],
                model_kwargs["seq_lens_encoder"],
                model_kwargs["seq_lens_decoder"],
                model_kwargs["input_ids"],
                model_kwargs["stop_nums"],
                next_tokens,
                model_kwargs["is_block_step"],
            )

        _post_process_(
            outputs,
            penalty_score,
            frequency_score,
            presence_score,
            temperature,
            **model_kwargs,
        )

        return model_kwargs["beam_hyps"]

    def draft_model_sampling(
        self,
        logits,
        **model_kwargs,
    ):
        """Sample from GPT using beam search and post process the generated sequence.

        Args:
            eos_token_id (int): The id of the token indicating the end of a sentence.
            top_p (float): If set to float < 1, only the tokens with probabilities greater than or equal to
                the threshold are kept for generation.
            **model_kwargs: Other arguments for forward pass of GPT model.

        Returns:
            Tensor: The sampled tokens. The shape is [batch_size].
        """
        top_p = model_kwargs["top_p"]
        eos_token_id = model_kwargs["eos_token_id"]

        def _post_process_(
            logits,
            top_p,
            model_kwargs,
        ):
            probs = F.softmax(logits)

            _, inter_next_tokens = paddle.tensor.top_p_sampling(probs,
                                                                top_p,
                                                                seed=-1)

            if self.ernie.mp_size > 1:
                paddle.distributed.broadcast(inter_next_tokens, 0)

            draft_model_update(
                inter_next_tokens,
                model_kwargs["draft_tokens"],
                model_kwargs["pre_ids"],
                model_kwargs["seq_lens_this_time"],
                model_kwargs["seq_lens_encoder"],
                model_kwargs["seq_lens_decoder"],
                model_kwargs["step_idx"],
                model_kwargs["output_cum_offsets"],
                model_kwargs["stop_flags"],
                model_kwargs["not_need_stop"],
                model_kwargs["max_dec_len"],
                eos_token_id,
                model_kwargs["base_model_draft_tokens"],
                self.ernie.max_len,
                model_kwargs["substep"],
            )
            if (self.ernie.speculate_method in ["mtp", "draft_model", "eagle"]
                    and self.ernie.generation_phase
                    == GenerationPhase.PREFILL):
                if self.msg_queue_id is None:
                    mtp_save_first_token(
                        model_kwargs["base_model_draft_tokens"],
                        model_kwargs["not_need_stop"],
                        self.ernie.mp_rank,
                        self.ernie.use_ep
                        and (not self.ernie.ep_just_for_test),
                    )
                else:
                    mtp_save_first_token_dynamic(
                        model_kwargs["base_model_draft_tokens"],
                        model_kwargs["not_need_stop"],
                        self.ernie.mp_rank,
                        self.msg_queue_id,
                        self.ernie.use_ep
                        and (not self.ernie.ep_just_for_test),
                    )
            return hidden_states

        output_padding_offset, output_cum_offsets = self.get_output_padding_offset(
            model_kwargs["seq_lens_this_time"],
            model_kwargs["seq_lens_encoder"],
            model_kwargs["seq_lens_decoder"],
        )
        model_kwargs["actual_output_padding_offset"] = output_padding_offset
        model_kwargs["output_cum_offsets"] = output_cum_offsets

        # first decoder
        hidden_states = _post_process_(logits, top_p, model_kwargs)

        return hidden_states

    def compute_logits(self, hidden_states):
        logits = self.lm_head(hidden_states)
        logits = paddle.cast(logits, paddle.float32)
        logits[:, self.ori_vocab_size:] = -float("inf")
        return logits

    def forward(self, **kwargs):
        model_inputs = self.prepare_inputs_for_generation(**kwargs)
        hidden_states = self.ernie(**model_inputs)
        return hidden_states

    def sample(
        self,
        logits,
        **sampler_kwargs,
    ):
        """
        Defines the forward pass of the model for generating text.

        Args:
            logits (Tensor): Logits tensor representing the probability distribution over the vocabulary.
            **sampler_kwargs: Additional keyword arguments for the sample.

        Returns:
            Tensor or list of Tensors: Generated tokens or decoded outputs.
        """
        sampler_kwargs["top_k"] = self.top_k
        num_return_sequences = self.num_return_sequences

        if self.decode_strategy == "sampling":
            if num_return_sequences > 1:
                sampler_kwargs[
                    "input_ids"], sampler_kwargs = self.expand_inputs_for_generation(
                        sampler_kwargs["input_ids"],
                        expand_size=num_return_sequences,
                        **sampler_kwargs)
            ret = self.sampling(
                logits,
                **sampler_kwargs,
            )
        elif self.decode_strategy == "draft_model_sampling":
            ret = self.draft_model_sampling(
                logits,
                **sampler_kwargs,
            )
        else:
            raise ValueError(
                f"Not support {self.decode_strategy} strategy yet!")
        return ret

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
# cipher_token=WjI1fQOvhN  # do not edit this line
from __future__ import annotations

import logging
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddlenlp.transformers import PretrainedModel
from paddlenlp.transformers import register_base_model
from paddlenlp.utils.log import logger

from ..layers.embeddings import Embeddings
from ..layers.lm_head import LMHead
from ..layers.lm_head import LMHeadAVX
from ..layers.lm_head import LMHeadNPU
from ..layers.normalization import Normalization
from .configuration import ModelConfig
from fastdeploy.inference_args import FMTKeys
from fastdeploy.inference_args import GenerationPhase
from fastdeploy.inference_args import InferenceArgs
from fastdeploy.platforms import current_platform

try:
    from paddlenlp.transformers.generation_utils import (
        ForcedBOSTokenLogitsProcessor,
        ForcedEOSTokenLogitsProcessor,
        HammingDiversityLogitsProcessor,
        LogitsProcessorList,
        RepetitionPenaltyLogitsProcessor,
    )
except ImportError:
    from paddlenlp.generation import (
        ForcedBOSTokenLogitsProcessor,
        ForcedEOSTokenLogitsProcessor,
        HammingDiversityLogitsProcessor,
        LogitsProcessorList,
        RepetitionPenaltyLogitsProcessor,
    )

if current_platform.is_cuda() and current_platform.available():
    try:
        from fastdeploy.model_executor.ops.gpu import (
            beam_search_softmax,
            draft_model_update,
            get_padding_offset,
            get_token_penalty_multi_scores,
            save_output,
            save_output_dynamic,
            mtp_save_first_token,
            mtp_save_first_token_dynamic,
            set_stop_value_multi_ends,
            set_stop_value_multi_seqs,
            set_value_by_flags_and_idx,
            speculate_clear_accept_nums,
            speculate_get_output_padding_offset,
            speculate_get_padding_offset,
            speculate_get_seq_lens_output,
            speculate_get_token_penalty_multi_scores,
            speculate_rebuild_append_padding,
            speculate_save_output_dynamic,
            speculate_save_output,
            speculate_set_stop_value_multi_seqs,
            speculate_set_value_by_flags_and_idx,
            speculate_update_v3,
            speculate_verify,
            top_p_candidates,
            update_inputs,
            update_inputs_beam,
            extract_text_token_output,
        )
    except Exception:
        raise ImportError(
            f"Verify environment consistency between compilation and FastDeploy installation. "
            f"And ensure the Paddle version supports FastDeploy's custom operators"
        )
elif paddle.is_compiled_with_xpu():
    from fastdeploy.model_executor.ops.gpu import (
        get_padding_offset,
        get_token_penalty_multi_scores,
        save_output,
        set_stop_value_multi_ends,
        set_value_by_flags_and_idx,
        update_inputs,
    )
elif paddle.is_compiled_with_custom_device("npu"):
    # custom ops for inference
    try:
        from fastdeploy.model_executor.ops.npu import (
            atb_broadcast,
            atb_top_p_sampling,
            get_token_penalty_multi_scores,
            mask_logits,
            remove_padding,
            save_output,
            save_output_dynamic,
            set_stop_value_multi_ends_v2 as set_stop_value_multi_ends,
            set_value_by_flags_and_idx_v2 as set_value_by_flags_and_idx,
            update_inputs,
        )
    except Exception:
        pass
else:  # CPU
    from fastdeploy.model_executor.ops.cpu import (
        get_padding_offset,
        get_token_penalty_multi_scores,
        save_output,
        save_output_dynamic,
        set_stop_value_multi_ends,
        set_value_by_flags_and_idx,
        simd_sort,
        update_inputs,
        xft_greedy_search,
    )

from .fused_avx_transformer import FusedAvxTransformer
from .fused_transformer import FusedTransformer


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

    config_class = ModelConfig

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
            multimodel_experts = getattr(config, "multimodel_experts", False)
            moe_num_experts = sum(
                moe_num_experts) if multimodel_experts else moe_num_experts
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
        wint4_smooth=False,
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
        use_avx512=False,
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
        embeddings_column_cut=False,
        erine_config=None,
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
            use_avx512 (bool): Whether to use AVX512 instructions.
        """
        super(ErnieBotFusedModel, self).__init__(erine_config)
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
        self.wint4_smooth = wint4_smooth
        self.group_size = group_size

        self.use_rmsnorm = use_rmsnorm
        self.num_key_value_heads = num_key_value_heads
        self.cache_quant_dtype = cache_quant_dtype
        self.use_moe = use_moe

        if self.use_rmsnorm:
            self.norm_type = "rmsnorm"
        else:
            self.norm_type = "layernorm"

        if self.norm_type == "rmsnorm":
            # rms_norm don't have norm_bias
            self.have_norm_bias = False
        elif self.norm_type == "layernorm":
            self.have_norm_bias = True

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

        if current_platform.is_npu() and (weight_sharing
                                          or weight_sharing_add_bias):
            logging.error(
                "weight_sharing and weight_sharing_add_bias is not supported to set True in NPU model."
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
            use_avx512=use_avx512,
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
            moe_use_ffn_shared_weight_and_bias=moe_use_ffn_shared_weight_and_bias,
            moe_group=moe_group,
            moe_quant_type=moe_quant_type,
            use_ep=use_ep,
            generation_phase=generation_phase,
            use_micro_batch=use_micro_batch,
            start_layer_index=start_layer_index,
            scale_dir=scale_dir,
        )

        fmt_keys = FMTKeys(num_layers)
        is_mtp = draft_type in ["eagle", "mtp"]
        self.is_mtp = is_mtp
        base_model_prefix = "ernie.mtp" if is_mtp else "ernie"
        self.base_model_prefix = base_model_prefix

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

        self.embeddings = Embeddings(
            layer_name=(f"{base_model_prefix}.embeddings.word_embeddings"
                        if not use_moe else "ernie.embed_tokens"),
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=self.initializer_range,
            sequence_parallel=sequence_parallel,
            freeze_embedding=freeze_embedding,
            weight_sharing=weight_sharing,
            weight_sharing_add_bias=weight_sharing_add_bias,
            use_rope=use_rope,
            rope_head_dim=hidden_size // num_attention_heads,
            prefix_name="ernie.mtp" if is_mtp else "ernie",
            use_ep=self.inference_args.use_ep,
            column_cut=embeddings_column_cut,
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
        if self.inference_args.use_avx512:
            self.decoder = FusedAvxTransformer(
                inference_args=self.inference_args,
                with_ln_bias=True,
                with_qkv_bias=True,
                with_out_linear_bias=True,
                with_ffn_ln_bias=True,
                with_gate_up_bias=True,
                with_ffn2_bias=True,
                activation="silu",
                norm_type=self.norm_type,
            )
        else:
            self.decoder = FusedTransformer(
                inference_args=self.inference_args,
                fmt_keys=fmt_keys,
                act_method=activation,
                rope_theta=rope_theta,
                rope_3d=rope_3d,
                norm_type=self.norm_type,
                ffn1_concat=self.ffn1_concat,
                use_smooth_quant=self.use_smooth_quant,
                fuse_ffn_act=self.fuse_ffn_act,
                ring_id=ring_id,
                have_norm_bias=self.have_norm_bias,
                return_all_hidden_states=self.return_all_hidden_states,
                base_model_prefix=base_model_prefix,
                draft_type=draft_type,
                max_len=max_len,
            )

            self.norm = Normalization(
                inference_args=self.inference_args,
                layer_name=f"{base_model_prefix}.decoder.norm",
                weight_key=(f"{self.base_model_prefix}.decoder.norm.weight"
                            if not self.use_moe else "ernie.norm.weight"),
                bias_key=(f"{self.base_model_prefix}.decoder.norm.bias" if
                          self.have_norm_bias and not self.is_mtp else None),
                norm_type=self.norm_type if not self.is_mtp else "rmsnorm",
                epsilon=1e-5,
            )

            if is_mtp:
                self.e_norm = Normalization(
                    inference_args=self.inference_args,
                    layer_name=f"{base_model_prefix}.e_norm",
                    weight_key=f"{base_model_prefix}.e_norm.weight",
                    bias_key=None,
                    norm_type="rmsnorm",
                    epsilon=1e-5,
                )
                self.h_norm = Normalization(
                    inference_args=self.inference_args,
                    layer_name=f"{base_model_prefix}.h_norm",
                    weight_key=f"{base_model_prefix}.h_norm.weight",
                    bias_key=None,
                    norm_type="rmsnorm",
                    epsilon=1e-5,
                )

                from paddle.distributed.fleet.meta_parallel import ColumnParallelLinear

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
        if current_platform.is_npu():
            remove_padding_ids = remove_padding(input_ids, seq_lens_this_time)
            embedding_output = self.embeddings(
                ids_remove_padding=remove_padding_ids)
        else:
            if self.speculate_method is not None:
                (
                    ids_remove_padding,
                    padding_offset,
                    cum_offsets,
                    cu_seqlens_q,
                    cu_seqlens_k,
                ) = self.speculate_remove_padding(input_ids,
                                                  seq_lens_this_time,
                                                  draft_tokens,
                                                  seq_lens_encoder)
            else:
                (
                    ids_remove_padding,
                    padding_offset,
                    cum_offsets,
                    cu_seqlens_q,
                    cu_seqlens_k,
                ) = self.remove_padding(input_ids, seq_lens_this_time)
            embedding_output = self.embeddings(
                ids_remove_padding=ids_remove_padding)
            if self.is_mtp:
                embedding_output = paddle.concat([
                    self.e_norm(embedding_output),
                    self.h_norm(hidden_states)
                ],
                                                 axis=-1)
                embedding_output = self.eh_proj(embedding_output)

        if isinstance(embedding_output, tuple):
            embedding_output = embedding_output[0]
        else:
            embedding_output = embedding_output

        if (self.inference_args.moe_config.use_moe
                and self.inference_args.moe_config.has_multimodality):
            token_type_ids = (ids_remove_padding ==
                              self.inference_args.moe_config.im_patch_id)
            image_mask = token_type_ids
            if image_mask.any():
                embedding_output[image_mask] = image_features.cast(
                    embedding_output.dtype)

        if self.inference_args.use_avx512:
            output = self.decoder(
                src=embedding_output,
                step_idx=step_idx,
                seq_lens_this_time=seq_lens_this_time,
                seq_lens_encoder=seq_lens_encoder,
                seq_lens_decoder=seq_lens_decoder,
            )
        else:
            output = self.decoder(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                src=embedding_output,
                caches=caches,
                rotary_embs=rope_emb,
                rotary_emb_dims=1,
                max_input_length=self.max_len,
                block_size=self.block_size,
                inv_compression_ratio=self.inv_compression_ratio,
                cum_offsets=None if current_platform.is_npu() else cum_offsets,
                cu_seqlens_q=None
                if current_platform.is_npu() else cu_seqlens_q,
                cu_seqlens_k=None
                if current_platform.is_npu() else cu_seqlens_k,
                padding_offsets=None
                if current_platform.is_npu() else padding_offset,
                block_tables=block_tables,
                seq_lens_this_time=seq_lens_this_time,
                seq_lens_encoder=seq_lens_encoder,
                seq_lens_decoder=seq_lens_decoder,
                attention_mask=attention_mask,  # for NPU
                beam_cache_offset=beam_cache_offset,
                draft_tokens=draft_tokens,
                output_padding_offset=output_padding_offset,
                return_all_hidden_states=self.return_all_hidden_states,
            )

        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output

        if (self.inference_args.moe_config.use_moe
                and self.inference_args.moe_config.has_multimodality):
            out = out.cast("float32")
            score_text = out
            score_image = None
            mm_token_num_len = paddle.count_nonzero(token_type_ids).cast(
                "int32")

            if mm_token_num_len > 0:
                token_num = paddle.shape(ids_remove_padding)[0]
                token_type_ids = token_type_ids.reshape([-1])
                image_mask_shifted = token_type_ids[:token_num] == 1
                text_pos_shifted = token_type_ids[:token_num] == 0
                score_text = out[text_pos_shifted.reshape([-1])]
                score_image = out[image_mask_shifted.reshape([-1])]
            max_seq_len, max_seq_len_index = paddle.topk(
                seq_lens_this_time.squeeze(-1), k=1)
            out = extract_text_token_output(
                max_seq_len,
                max_seq_len_index.cast("int32"),
                mm_token_num_len,
                seq_lens_this_time,
                cu_seqlens_q,
                score_text,
            )[0].cast(embedding_output.dtype)

        if not current_platform.is_npu():
            out = self.norm(out)

        if self.return_all_hidden_states:
            return out, cum_offsets
        else:
            return out


class ErnieBotForGeneration(nn.Layer):
    """
    ErnieBotForGeneration
    """

    def __init__(self, ernie, configs):
        """
        Args:
            ernie (ErnieBotFusedModel): ErnieBotFusedModel model used for generation.
            configs (dict): Configurations including parameters such as max_dec_len, min_dec_len, decode_strategy,
                ori_vocab_size, use_topp_sampling, use_top_k, top_k, inference, repetition_penalty, num_beams,
                num_beam_groups, length_penalty, early_stopping, bos_token_id, pad_token_id, decoder_start_token_id,
                forced_bos_token_id, forced_eos_token_id, num_return_sequences, diversity_rate, weight_sharing,
                weight_sharing_add_bias, export_model_type, group_size, use_rmsnorm,
                use_fake_parameter, cache_quant_dtype, hidden_size, num_attention_heads, rank, nranks, root, ring_id,
                beam_width, norm_type, have_norm_bias, is_norm_weight_type_fp32, cachekv_dtype, out_linear,
                use_fast_ffn.

        Raises:
            ValueError: If the export_model_type is W8A8C8 or W8A8C16 and use_rmsnorm is True.
            ValueError: If use_rmsnorm is True and norm_type is not 'rmsnorm'.
            ValueError: If norm_type is not 'layernorm' or 'rmsnorm'.
            ValueError: If use_cache_kv_int8 is True and use_fake_parameter is True.
        """
        super(ErnieBotForGeneration, self).__init__()
        self.ernie = ernie
        self.msg_queue_id = ernie.msg_queue_id
        # extra_parameters using for sharding stage3 to register extra_parameters
        self.extra_parameters = ([] if current_platform.is_npu() else [
            get_attr(self.ernie.embeddings.word_embeddings, "weight")
        ])
        self.configs = configs

        self.max_length = self.configs.get("max_dec_len", 20)
        self.min_length = self.configs.get("min_dec_len", 0)
        self.fake_server_p = self.configs.get("fake_server_p", False)
        self.decode_strategy = self.configs.get("decode_strategy", "sampling")
        self.speculate_max_candidate_len = self.configs.get(
            "speculate_max_candidate_len", 5)
        self.speculate_verify_window = self.configs.get(
            "speculate_verify_window", 2)
        self.use_moe = ernie.use_moe

        assert self.decode_strategy in [
            "greedy_search",
            "sampling",
            "beam_search",
            "speculate_decoding",
            "draft_model_sampling",
        ], f"`decode_strategy` must be one of 'greedy_search', 'sampling', \
            'speculate_decoding' or 'beam_search' but received {self.decode_strategy}."

        self.ori_vocab_size = self.configs["ori_vocab_size"]

        self.use_topp_sampling = self.configs.get("use_topp_sampling", True)
        self.use_top_k = self.configs.get("use_top_k", True)
        self.top_k = self.configs.get("top_k", 0)
        self.inference = self.configs.get("inference", True)
        self.repetition_penalty = self.configs.get("repetition_penalty", 1.0)
        self.length_penalty = self.configs.get("length_penalty", 0.0)
        self.early_stopping = self.configs.get("early_stopping", False)
        self.bos_token_id = self.configs.get("bos_token_id", None)
        # self.eos_token_id = self.configs.get('eos_token_id', None)
        self.pad_token_id = self.configs.get("pad_token_id", None)
        self.decoder_start_token_id = self.configs.get(
            "decoder_start_token_id", None)
        self.forced_bos_token_id = self.configs.get("forced_bos_token_id",
                                                    None)
        self.forced_eos_token_id = self.configs.get("forced_eos_token_id",
                                                    None)
        self.num_return_sequences = self.configs.get("num_return_sequences", 1)
        self.diversity_rate = self.configs.get("diversity_rate", 0.0)

        self.weight_sharing = self.configs.get("weight_sharing", False)
        self.weight_sharing_add_bias = self.configs.get(
            "weight_sharing_add_bias", False)

        self.export_model_type = self.configs.get("export_model_type",
                                                  "default")
        self.group_size = self.configs.get("group_size", -1)
        self.weightonly_groupwise = True if self.group_size > 0 else False

        self.use_rmsnorm = self.configs.get("use_rmsnorm", False)
        self.use_fake_parameter = self.configs.get("use_fake_parameter", False)
        self.cache_quant_dtype = self.configs.get("cache_quant_dtype",
                                                  "default")
        if self.cache_quant_dtype == "default":
            self.cache_quant_dtype = paddle.get_default_dtype()
        self.use_fast_ffn = self.ernie.use_fast_ffn

        # for NPU
        self.hidden_size = self.configs.get("hidden_size", 4096)
        self.num_attention_heads = self.configs.get("num_attention_heads", 32)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.rank = (paddle.distributed.fleet.get_hybrid_communicate_group().
                     get_model_parallel_rank())
        self.nranks = (paddle.distributed.fleet.get_hybrid_communicate_group().
                       get_model_parallel_world_size())
        self.root = 0
        self.ring_id = (paddle.distributed.fleet.get_hybrid_communicate_group(
        ).get_model_parallel_group().id)

        self.beam_width = self.configs.get("beam_width", 1)
        self.beam_group_num = self.configs.get("beam_group_num", 1)
        self.return_all_hidden_states = self.configs.get(
            "return_all_hidden_states", False)

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
        if self.ernie.inference_args.use_avx512:
            self.lm_head = LMHeadAVX(
                norm_layer_name=f"{self.base_model_prefix}.decoder.norm",
                linear_layer_name=f"{self.base_model_prefix}.output_linear.out_linear",
                input_dim=self.hidden_size,
                output_dim=self.ernie.vocab_size,
                have_norm_bias=self.have_norm_bias,
                have_ln_bias=True,
                alog="int8",
                hidden_size=self.hidden_size,
            )
        elif current_platform.is_npu():
            self.lm_head = LMHeadNPU(
                norm_layer_name=f"{self.base_model_prefix}.decoder.norm",
                linear_layer_name=f"{self.base_model_prefix}.output_linear.out_linear",
                input_dim=self.hidden_size,
                output_dim=self.ernie.vocab_size,
                epsilon=1e-5,
                norm_type=self.norm_type,
                have_norm_bias=self.have_norm_bias,
            )
        else:
            if self.weight_sharing:
                sharing_weight = self.ernie.embeddings.word_embeddings.weight
            else:
                sharing_weight = None
            if self.weight_sharing_add_bias:
                sharing_bias = self.ernie.embeddings.bias
            else:
                sharing_bias = None

            lmhead_name = ("server_nlg_mask_lm_trans_fc_"
                           if not self.ernie.is_mtp else
                           "mtp_server_nlg_mask_lm_trans_fc_")
            if self.use_moe:
                self.lm_head = LMHead(
                    layer_name=lmhead_name,
                    linear_weight_key="lm_head.weight",
                    linear_bias_key=None,
                    input_dim=self.hidden_size,
                    output_dim=self.ernie.vocab_size,
                    fused_linear=self.configs["fused_linear"],
                    sharing_weight=sharing_weight,
                    sharing_bias=sharing_bias,
                    use_ep=self.ernie.use_ep,
                )
            else:
                self.lm_head = LMHead(
                    layer_name=lmhead_name,
                    linear_weight_key=f"{self.base_model_prefix}.output_linear.out_linear.weight",
                    linear_bias_key=(
                        f"{self.base_model_prefix}.output_linear.out_linear.bias"
                        if self.have_norm_bias else None
                    ),
                    input_dim=self.hidden_size,
                    output_dim=self.ernie.vocab_size,
                    fused_linear=self.configs["fused_linear"],
                    sharing_weight=sharing_weight,
                    sharing_bias=sharing_bias,
                    use_ep=self.ernie.use_ep,
                )

    @paddle.no_grad()
    def set_state_dict(self, state_dict: dict[str,
                                              np.ndarray | paddle.Tensor]):
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
        except:
            raise RuntimeError("set_state_dict error!!!")

    def prepare_input_ids_for_generation(self,
                                         bos_token_id,
                                         encoder_output=None):
        """
        Prepare input ids for text generation.

        Args:
            bos_token_id (int): The beginning of sequence token id. This token will be used to initialize \
                the input ids.
            encoder_output (Tensor, optional): The output of the encoder. If provided, the batch size will be \
                inferred from its shape.Defaults to None.

        Returns:
            Tensor: A tensor of shape [batch_size, 1] filled with the bos_token_id,
                where batch_size is 1 if encoder_output is None, otherwise it is the batch size of encoder_output.

        Raises:
            ValueError: If bos_token_id is None and no encoder_output is provided.
        """
        batch_size = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no "
                             "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
        return paddle.ones([batch_size, 1], dtype="int64") * bos_token_id

    def prepare_attention_mask_for_generation(self, input_ids, pad_token_id,
                                              eos_token_id):
        """
        Prepare attention mask for sequence generation.

        Args:
            input_ids (Tensor): The input tensor of token ids, with shape [batch_size, sequence_length].
            pad_token_id (int, optional): The token id used for padding. If None, padding will not be considered.
            eos_token_id (int, optional): The token id representing the end of sentence.
                If provided, it will be checked whether the padding token id is the same as eos token id.

        Returns:
            Tensor: The attention mask tensor with shape [batch_size, 1, 1, sequence_length],
                where padded positions are masked with a large negative value (-1e4) to prevent attention to them.
        """
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(
            input_ids == pad_token_id).numpy().item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id))
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids == pad_token_id).astype(
                paddle.get_default_dtype()) * -1e4
        else:
            attention_mask = paddle.zeros_like(
                input_ids, dtype=paddle.get_default_dtype())
        return paddle.unsqueeze(attention_mask, axis=[1, 2])

    def update_scores_for_generation(self, scores, next_scores, length,
                                     unfinished_flag):
        """
        Update scores for generation process.

        Args:
            scores (Tensor): The initial scores of the tokens.
            next_scores (Tensor): The scores of the next tokens.
            length (Tensor): The length of the sequence corresponding to each score.
            unfinished_flag (Tensor): A boolean flag indicating whether the sequence is unfinished.

        Returns:
            Tensor: Updated scores for the generation process.
        """
        # update scores
        unfinished_scores = (scores * length + next_scores) / (length + 1)
        scores = paddle.where(unfinished_flag, unfinished_scores, scores)
        return scores

    def get_name_mappings_to_training(self):
        """Generate mapping between inference and training parameter names with MoE support."""

        # Extract configs with defaults
        configs = self.configs
        moe_layer_start_index = configs.get("moe_layer_start_index", 3)
        num_layers = configs.get("num_layers", 54)
        moe_use_gate_correction_bias = configs.get(
            "moe_use_gate_correction_bias", True)
        have_bias = configs.get("have_norm_bias", False)
        moe_num_experts = configs.get("moe_num_experts", 64)

        # Prepare placeholders
        place_holders = ["weight"] + (["bias"] if have_bias else [])

        # Initialize mapping dictionary
        infer_to_train = {}

        # Static mappings (non-layer specific)
        static_mappings = {
            "gpt.embeddings.word_embeddings.weight":
            "ernie.embed_tokens.weight",
            "gpt.norm.ln_weight": "ernie.norm.weight",
            "lm_head.out_linear.weight": "lm_head.weight"
        }
        infer_to_train.update(static_mappings)
        infer_base_name = "gpt.decoder"

        # Helper function to add layer mappings
        def _add_layer_mappings(layer_idx, is_moe_layer=False):
            # Handle special case for layer 0's input layernorm
            if layer_idx == 0:
                for ph in place_holders:
                    infer_key = f"{infer_base_name}.norm_before_qkv.ln_{ph}"
                    train_key = f"ernie.layers.{layer_idx}.input_layernorm.{ph}"
                    infer_to_train[infer_key] = train_key
            else:
                for ph in place_holders:
                    infer_key = f"{infer_base_name}.bias_residual_layernorm_layers.{layer_idx - 1}.ln_{ph}"
                    train_key = f"ernie.layers.{layer_idx}.input_layernorm.{ph}"
                    infer_to_train[infer_key] = train_key

            # Common attention mappings
            for ph in place_holders:
                infer_to_train[f"{infer_base_name}.qkv_linear_layers.{layer_idx}.qkv_{ph}"] = \
                    f"ernie.layers.{layer_idx}.self_attn.qkv_proj.{ph}"

                infer_to_train[f"{infer_base_name}.out_linear_layers.{layer_idx}.linear_{ph}"] = \
                    f"ernie.layers.{layer_idx}.self_attn.o_proj.{ph}"

            # Post-attention layernorm
            for ph in place_holders:
                infer_to_train[f"{infer_base_name}.ffn_layernorm_layers.{layer_idx}.ln_{ph}"] = \
                    f"ernie.layers.{layer_idx}.post_attention_layernorm.{ph}"

            if not is_moe_layer:
                # Dense FFN mappings
                for ph in place_holders:
                    infer_to_train[f"{infer_base_name}.ffn1_layers.{layer_idx}.ffn1_{ph}"] = \
                        f"ernie.layers.{layer_idx}.mlp.up_gate_proj.{ph}"

                    infer_to_train[f"{infer_base_name}.ffn2_layers.{layer_idx}.linear_{ph}"] = \
                        f"ernie.layers.{layer_idx}.mlp.down_proj.{ph}"
            else:
                # MoE specific mappings
                infer_to_train[f"{infer_base_name}.moe_layers.{layer_idx}.gate_weight"] = \
                    f"ernie.layers.{layer_idx}.mlp.gate.weight"

                if moe_use_gate_correction_bias:
                    infer_to_train[f"{infer_base_name}.moe_layers.{layer_idx}.gate_correction_bias"] = \
                        f"ernie.layers.{layer_idx}.mlp.moe_statics.e_score_correction_bias"

                # MoE experts mappings
                for expert_idx in range(moe_num_experts):
                    for ph in place_holders:
                        # FFN1 (up_gate_proj)
                        ffn1_key = f"{infer_base_name}.moe_layers.{layer_idx}.moe_ffn1_weight"
                        if ffn1_key not in infer_to_train:
                            infer_to_train[ffn1_key] = []
                        infer_to_train[ffn1_key].append(
                            f"ernie.layers.{layer_idx}.mlp.experts.{expert_idx}.up_gate_proj.{ph}"
                        )

                        # FFN2 (down_proj)
                        ffn2_key = f"{infer_base_name}.moe_layers.{layer_idx}.moe_ffn2_weight"
                        if ffn2_key not in infer_to_train:
                            infer_to_train[ffn2_key] = []
                        infer_to_train[ffn2_key].append(
                            f"ernie.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.{ph}"
                        )

        # Process non-MoE layers
        for layer_idx in range(moe_layer_start_index):
            _add_layer_mappings(layer_idx, is_moe_layer=False)

        # Process MoE layers
        for layer_idx in range(moe_layer_start_index, num_layers):
            _add_layer_mappings(layer_idx, is_moe_layer=True)

        return infer_to_train

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

    def get_logits_processor(
        self,
        min_length=None,
        max_length=None,
        eos_token_id=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        num_beams=1,
        num_beam_groups=1,
        diversity_rate=0.0,
        repetition_penalty=None,
    ):
        """
            Gets the list of logits processors to be applied before the argmax in beam search.
        By default, this includes:
        1. Hamming Diversity Logits Processor (if diversity_rate > 0)
        2. Repetition Penalty Logits Processor (if repetition_penalty != 1.0)
        3. Forced BOS Token Logits Processor (if forced_bos_token_id is not None)
        4. Forced EOS Token Logits Processor (if forced_eos_token_id is not None)

        Args:
            min_length (int, optional): Minimum length of the generated sequences (default: None).
            max_length (int, optional): Maximum length of the generated sequences (default: None).
            eos_token_id (int, optional): End-of-sequence token id (default: None).
            forced_bos_token_id (int, optional): Beginning-of-sequence token id (default: None).
            forced_eos_token_id (int, optional): End-of-sequence token id (default: None).
            num_beams (int, optional): Number of beams for beam search (default: 1).
            num_beam_groups (int, optional): Number of groups for dynamic beam search (default: 1).
            diversity_rate (float, optional): Hamming distance diversity rate (default: 0.0).
            repetition_penalty (float, optional): Repetition penalty (default: None).

        Returns:
            LogitsProcessorList: The list of logits processors to be applied before the argmax in beam search.
        """
        processors = LogitsProcessorList()

        if num_beam_groups > 1 and diversity_rate > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_rate=diversity_rate,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups,
                ))
        if repetition_penalty is not None and repetition_penalty != 1.0:
            processors.append(
                RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if forced_bos_token_id is not None:
            processors.append(
                ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
        if forced_eos_token_id is not None:
            processors.append(
                ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
        # TODO
        # Add more pre_processing for distribution

        return processors

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
        if self.ernie.inference_args.use_avx512:
            input_ids = kwargs["input_ids"]
            seq_lens_encoder = kwargs["seq_lens_encoder"]
            seq_lens_decoder = kwargs["seq_lens_decoder"]
            seq_lens_this_time = kwargs["seq_lens_this_time"]
            step_idx = kwargs["step_idx"]
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": None,
                "rope_emb": None,
                "caches": None,
                "seq_lens_this_time": seq_lens_this_time,
                "seq_lens_encoder": seq_lens_encoder,
                "seq_lens_decoder": seq_lens_decoder,
                "block_tables": None,
                "beam_cache_offset": None,
                "step_idx": step_idx,
            }
        else:
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
            }
        return model_inputs

    def sample(
        self,
        eos_token_id,
        top_k,
        top_p,
        penalty_score,
        frequency_score,
        presence_score,
        temperature=None,
        min_tokens_to_keep=1,
        **model_kwargs,
    ):
        """Sample from GPT using beam search and post process the generated sequence.

        Args:
            eos_token_id (int): The id of the token indicating the end of a sentence.
            top_k (int): Number of highest probability vocabulary tokens to keep for top-k-filtering.
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

        def _forward_(**args):
            """
            Forward pass of GPT model.
            """
            model_inputs = self.prepare_inputs_for_generation(**args)
            return self.ernie(**model_inputs)

        def TopPProcess(probs: paddle.Tensor, top_p: float,
                        min_tokens_to_keep: int):
            """
            Filter a distribution of integer ids according to the top_p value.

            Args:
                probs(paddle.Tensor): Probabilities of the tokens.
                top_p(float): Keep only top_p tokens with highest probability.
                min_tokens_to_keep(int): Minimal number of tokens to keep for next step in decoding.
            Returns:
                paddle.Tensor: Filtered distribution.
            """
            sorted_indices, sorted_probs = simd_sort(probs)
            cumulative_probs = paddle.cumsum(sorted_probs, axis=-1)
            # Remove tokens with cumulative probs above the top_p, But keep at
            # least min_tokens_to_keep tokens
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Set 'min_tokens_to_keep - 1' because the first token is kept
                sorted_indices_to_remove[:, :min_tokens_to_keep - 1] = 0
            sorted_indices_to_remove = paddle.cast(sorted_indices_to_remove,
                                                   dtype="int64")
            sorted_indices_to_remove[:,
                                     1:] = sorted_indices_to_remove[:, :
                                                                    -1].clone(
                                                                    )
            sorted_indices_to_remove[:, 0] = 0
            # Scatter sorted tensors to original indexing
            sorted_indices = (sorted_indices + paddle.arange(
                probs.shape[0], dtype="int64").unsqueeze(-1) * probs.shape[-1])
            condition = paddle.scatter(
                sorted_indices_to_remove.flatten(),
                sorted_indices.flatten(),
                sorted_indices_to_remove.flatten(),
            )
            condition = paddle.cast(condition, "bool").reshape(probs.shape)
            probs = paddle.where(condition, paddle.full_like(probs, 0.0),
                                 probs)
            return probs

        def _post_process_(
            outputs,
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

            hidden_states = outputs[0] if isinstance(outputs,
                                                     tuple) else outputs

            if current_platform.is_npu():
                # NPU uses custom op lm_head, mask_logits for logits calculation
                logits = self.lm_head(hidden_states)
                mask_logits(logits, vocab_size=self.ori_vocab_size)
                logits = paddle.cast(logits, paddle.float32)

                # NPU diffs in input parametes of get_token_penalty_multi_scores op.
                logits = get_token_penalty_multi_scores(
                    model_kwargs["pre_ids"],
                    logits,
                    penalty_score,
                    frequency_score,
                    presence_score,
                    step_idx,
                    model_kwargs["min_dec_len"],
                    eos_token_id,
                )
                logits = logits / temperature
            else:
                logits = self.lm_head(hidden_states)

                logits = paddle.cast(logits, paddle.float32)
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

            # sample
            probs = F.softmax(logits)
            if current_platform.is_npu():
                # NPU: use custom op atb_top_p_sampling
                _, next_tokens = atb_top_p_sampling(probs,
                                                    top_p,
                                                    random_seed=-1)
            elif self.ernie.inference_args.use_avx512:
                # topp sampling
                if top_p > 0.0 and top_p < 1.0:
                    min_tokens_to_keep = 1
                    probs = TopPProcess(probs, top_p, min_tokens_to_keep)
                    next_tokens = paddle.multinomial(probs)
                else:
                    # greedy_search
                    next_tokens = xft_greedy_search(probs)
            else:
                _, next_tokens = paddle.tensor.top_p_sampling(
                    probs, top_p, seed=-1)  # have random_seed
            """ !!! ep not need broadcast, here broadcast just for test !!! """
            if self.ernie.mp_size > 1 and (
                (not self.ernie.use_ep or self.ernie.ep_just_for_test) and
                (not self.fake_server_p)):
                if current_platform.is_npu():
                    # NPU: use custom op atb_broadcast
                    atb_broadcast(
                        next_tokens,
                        rank=self.rank,
                        nranks=self.nranks,
                        root=self.root,
                        ring_id=self.ring_id,
                    )
                else:
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
                if current_platform.is_npu():
                    set_stop_value_multi_ends(
                        next_tokens,
                        model_kwargs["stop_flags"],
                        model_kwargs["seq_lens_this_time"],
                        eos_token_id,
                        model_kwargs["next_tokens"],
                    )  # multi ends
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
            # encoder
            outputs = _forward_(**model_kwargs)  # [bs, 1, dim_embed]
            # first decoder
            next_tokens = _post_process_(
                outputs,
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
        eos_token_id,
        top_p,
        penalty_score,
        frequency_score,
        presence_score,
        temperature=None,
        min_tokens_to_keep=1,
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

        def _forward_(**args):
            """
            Forward pass of GPT model.
            """
            model_inputs = self.prepare_inputs_for_generation(**args)
            return self.ernie(**model_inputs)

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

        # encoder
        outputs = _forward_(**model_kwargs)  # [bs, 1, dim_embed]
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
        eos_token_id,
        penalty_score,
        frequency_score,
        presence_score,
        temperature=None,
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

        def _forward_(**args):
            model_inputs = self.prepare_inputs_for_generation(**args)
            return self.ernie(**model_inputs)

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

        # encoder
        outputs = _forward_(**model_kwargs)  # [bs, 1, dim_embed]

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
        eos_token_id,
        top_p,
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

        def _forward_(**args):
            model_inputs = self.prepare_inputs_for_generation(**args)
            return self.ernie(**model_inputs)

        def _post_process_(
            outputs,
            top_p,
            model_kwargs,
        ):

            hidden_states = outputs[0] if isinstance(outputs,
                                                     tuple) else outputs

            logits = self.lm_head(hidden_states)

            logits = paddle.cast(logits, paddle.float32)
            logits[:, self.ori_vocab_size:] = -float("inf")

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

        outputs = _forward_(**model_kwargs)  # [bs, 1, dim_embed]
        # first decoder
        hidden_states = _post_process_(outputs, top_p, model_kwargs)

        return hidden_states

    def forward(
        self,
        input_ids=None,  # update
        image_features=None,
        stop_seqs=None,
        stop_seqs_len=None,
        temperature=None,
        top_p=None,
        eos_token_id=None,
        penalty_score=None,
        frequency_score=None,
        presence_score=None,
        next_tokens=None,
        is_block_step=None,
        seq_lens_this_time=None,  # update
        seq_lens_encoder=None,  # update
        seq_lens_decoder=None,  # update
        step_idx=None,  # update
        stop_flags=None,  # update
        pre_ids=None,  # update
        rope_emb=None,
        min_dec_len=None,
        max_dec_len=None,
        stop_nums=None,
        bad_tokens=None,
        not_need_stop=None,
        block_tables=None,  # [args.bs, max_num_blocks]
        caches=[],
        attention_mask=None,  # for NPU
        beam_offset=None,
        beam_cache_ids=None,
        cum_score=None,
        beam_hyps=None,
        beam_hyps_score=None,
        beam_finished=None,
        beam_width=None,
        beam_group_num=None,
        beam_length_penalty=None,
        beam_diversity_penalty=None,
        draft_tokens=None,
        accept_tokens=None,
        accept_num=None,
        actual_draft_token_num=None,
        **model_kwargs,
    ):
        """
        Defines the forward pass of the model for generating text.

        Args:
            input_ids (Tensor, optional): The input token ids to the model.
            stop_seqs (Tensor, optional): Sequence ids indicating where to stop decoding.
            stop_seqs_len (Tensor, optional): Lengths of the stop sequences.
            temperature (float, optional): Temperature for sampling from the output distribution.
            top_p (float, optional): Probability threshold for top-p sampling.
            eos_token_id (int, optional): End-of-sequence token id.
            penalty_score (Tensor, optional): Penalty scores for certain tokens.
            frequency_score (Tensor, optional): Frequency scores for certain tokens.
            presence_score (Tensor, optional): Presence scores for certain tokens.
            next_tokens (Tensor, optional): Tokens generated in the previous step.
            is_block_step (bool, optional): Indicates if this is a blocking step.
            seq_lens_this_time (Tensor, optional): Sequence lengths for this step.
            seq_lens_encoder (Tensor, optional): Sequence lengths of the encoder.
            seq_lens_decoder (Tensor, optional): Sequence lengths of the decoder.
            step_idx (int, optional): Index of the current decoding step.
            stop_flags (Tensor, optional): Flags indicating whether decoding should stop.
            pre_ids (Tensor, optional): Previous ids used for decoding.
            rope_emb (Tensor, optional): Embeddings for ROPE.
            min_dec_len (int, optional): Minimum decoding length.
            max_dec_len (int, optional): Maximum decoding length.
            stop_nums (int, optional): Number of stop conditions.
            bad_tokens (Tensor, optional): Tokens that should not be generated.
            not_need_stop (bool, optional): Indicates if stopping conditions should be ignored.
            block_tables (Tensor, optional): Block tables for controlling decoding.
            caches (list, optional): Decoder caches from previous steps.
            attention_mask (Tensor, optional): Attention mask for the input ids.
            beam_offset (int, optional): Beam search offset.
            beam_cache_ids (Tensor, optional): Beam search cache ids.
            cum_score (Tensor, optional): Cumulative scores for beam search.
            beam_hyps (list, optional): Beam search hypotheses.
            beam_hyps_score (Tensor, optional): Scores for beam search hypotheses.
            beam_finished (bool, optional): Indicates if decoding has finished for some beams.
            beam_width (int): The beam width of beam search.
            beam_group_num (int): The num of groups in beam search.
            beam_length_penalty (int): The length penalty for beam search.
            beam_diversity_penalty (float): The diversity penaly for group beam search
            **model_kwargs: Additional keyword arguments for the model.

        Returns:
            Tensor or list of Tensors: Generated tokens or decoded outputs.
        """
        temperature = temperature
        top_k = self.top_k
        top_p = top_p
        bos_token_id = self.bos_token_id
        eos_token_id = eos_token_id
        pad_token_id = self.pad_token_id
        decoder_start_token_id = self.decoder_start_token_id
        forced_bos_token_id = self.forced_bos_token_id
        forced_eos_token_id = self.forced_eos_token_id
        num_return_sequences = self.num_return_sequences

        bos_token_id = (bos_token_id if bos_token_id is not None else getattr(
            self.ernie, "bos_token_id", None))
        pad_token_id = (pad_token_id if pad_token_id is not None else getattr(
            self.ernie, "pad_token_id", None))
        forced_bos_token_id = (forced_bos_token_id
                               if forced_bos_token_id is not None else getattr(
                                   self.ernie, "forced_bos_token_id", None))
        forced_eos_token_id = (forced_eos_token_id
                               if forced_eos_token_id is not None else getattr(
                                   self.ernie, "forced_eos_token_id", None))
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else
            getattr(self.ernie, "decoder_start_token_id", None))
        model_kwargs["input_ids"] = input_ids
        model_kwargs["image_features"] = image_features
        model_kwargs["attention_mask"] = attention_mask
        model_kwargs["seq_lens_this_time"] = seq_lens_this_time
        model_kwargs["seq_lens_encoder"] = seq_lens_encoder
        model_kwargs["seq_lens_decoder"] = seq_lens_decoder
        model_kwargs["step_idx"] = step_idx
        model_kwargs["stop_flags"] = stop_flags
        model_kwargs["pre_ids"] = pre_ids
        model_kwargs["min_dec_len"] = min_dec_len
        model_kwargs["max_dec_len"] = max_dec_len
        model_kwargs["rope_emb"] = rope_emb
        model_kwargs["stop_nums"] = stop_nums
        model_kwargs["bad_tokens"] = bad_tokens
        model_kwargs["not_need_stop"] = not_need_stop
        model_kwargs["block_tables"] = block_tables
        model_kwargs["next_tokens"] = next_tokens
        model_kwargs["is_block_step"] = is_block_step
        model_kwargs["stop_seqs"] = stop_seqs
        model_kwargs["stop_seqs_len"] = stop_seqs_len
        model_kwargs["caches"] = caches
        model_kwargs["beam_offset"] = beam_offset
        model_kwargs["beam_cache_ids"] = beam_cache_ids
        model_kwargs["cum_score"] = cum_score
        model_kwargs["beam_hyps"] = beam_hyps
        model_kwargs["beam_hyps_score"] = beam_hyps_score
        model_kwargs["beam_finished"] = beam_finished
        model_kwargs["beam_width"] = beam_width
        model_kwargs["beam_group_num"] = beam_group_num
        model_kwargs["beam_length_penalty"] = beam_length_penalty
        model_kwargs["beam_diversity_penalty"] = beam_diversity_penalty
        # speculate decoding related parameters
        model_kwargs["draft_tokens"] = draft_tokens
        model_kwargs["accept_tokens"] = accept_tokens
        model_kwargs["accept_num"] = accept_num
        model_kwargs["actual_draft_token_num"] = actual_draft_token_num

        if self.decode_strategy == "sampling":
            if num_return_sequences > 1:
                input_ids, model_kwargs = self.expand_inputs_for_generation(
                    input_ids,
                    expand_size=num_return_sequences,
                    **model_kwargs)
            ret = self.sample(
                eos_token_id,
                top_k,
                top_p,
                penalty_score,
                frequency_score,
                presence_score,
                temperature,
                **model_kwargs,
            )
        elif self.decode_strategy == "beam_search":
            ret = self.beam_search(
                eos_token_id,
                penalty_score,
                frequency_score,
                presence_score,
                temperature,
                **model_kwargs,
            )
            return ret
        elif self.decode_strategy == "speculate_decoding":
            ret = self.speculate_decoding(
                eos_token_id,
                top_p,
                penalty_score,
                frequency_score,
                presence_score,
                temperature,
                **model_kwargs,
            )
        elif self.decode_strategy == "draft_model_sampling":
            ret = self.draft_model_sampling(
                eos_token_id,
                top_p,
                **model_kwargs,
            )
        else:
            raise ValueError(
                f"Not support {self.decode_strategy} strategy yet!")
        return ret

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

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from paddlenlp.transformers import PretrainedModel, register_base_model
from paddlenlp.utils.log import logger
from paddle.incubate.nn.functional import blha_get_max_len

from fastdeploy.config import LLMConfig, ModelConfig, WeightKeys
from fastdeploy.inference_args import GenerationPhase, InferenceArgs

from ..layers.embeddings import VocabParallelEmbedding
from ..layers.lm_head import LMHead
from ..layers.normalization import RMSNorm
from ..layers.activation import SiluAndMul
from ..layers.attention.base import Attention
from ..layers.linear import (MergedColumnParallelLinear,
                             QKVParallelLinear, RowParallelLinear)
from ..layers.normalization import LayerNorm, RMSNorm
from .model_base import ModelForCasualLM


try:
    from paddlenlp.transformers.generation_utils import (
        ForcedBOSTokenLogitsProcessor, ForcedEOSTokenLogitsProcessor,
        HammingDiversityLogitsProcessor, LogitsProcessorList,
        RepetitionPenaltyLogitsProcessor)
except ImportError:
    from paddlenlp.generation import (ForcedBOSTokenLogitsProcessor,
                                      ForcedEOSTokenLogitsProcessor,
                                      HammingDiversityLogitsProcessor,
                                      LogitsProcessorList,
                                      RepetitionPenaltyLogitsProcessor)

from fastdeploy.model_executor.ops.gpu import (
    get_block_shape_and_split_kv_block, rebuild_padding,
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

from ..layers.quantization import get_quantization_config
from .fused_transformer import FusedTransformer

class Qwen2MLP(nn.Layer):
    """
    """
    def __init__(
        self,
        llm_config: LLMConfig,
        inference_args: InferenceArgs,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.nranks = llm_config.parallel_config.mp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            llm_config=llm_config,
            prefix=f"{prefix}.linear1",
            with_bias=False,
            activation=llm_config.model_config.hidden_act,
            use_fast_ffn=True,
        )
        
        self.down_proj = RowParallelLinear(
            llm_config=llm_config,
            prefix=f"{prefix}.down_proj",
            input_size=(llm_config.model_config.ffn_hidden_size //
                        self.nranks),
            output_size=llm_config.model_config.hidden_size,
            with_bias=False,
        )
        
        self.act_fn = SiluAndMul(
            inference_args=inference_args,
            bias=getattr(self.gate_up_proj, "linear_bias", None),
            act_method=llm_config.model_config.hidden_act,
        )

    def load_state_dict(self, state_dict):
        """
        """
        self.gate_up_proj.load_state_dict(state_dict)
        self.down_proj.load_state_dict(state_dict)

    def forward(self, x):
        """
        """
        gate_up_out = self.gate_up_proj(x)
        act_out = self.act_fn(gate_up_out)
        down_out = self.down_proj(act_out)
        return down_out

class Qwen2Attention(nn.Layer):
    """
    """
    def __init__(self,
                 llm_config: LLMConfig,
                 inference_args: InferenceArgs,
                 generation_phase: GenerationPhase = GenerationPhase.DECODER,
                 prefix: str = "") -> None:
        super().__init__()
        self.nranks = inference_args.mp_size
        self.num_heads = inference_args.num_attention_heads // self.nranks

        self.qkv_proj = QKVParallelLinear(
                llm_config=llm_config,
                prefix=f"{prefix}.qkv_proj",
                with_bias=True
            )

        self.o_proj = RowParallelLinear(
                llm_config=llm_config,
                prefix=f"{prefix}.o_proj",
                input_size=self.num_heads *
                (llm_config.model_config.hidden_size //
                 llm_config.model_config.num_attention_heads),
                output_size=llm_config.model_config.hidden_size,
            )

        self.attn = Attention(
                inference_args=inference_args,
                prefix=prefix,
                rope_theta=llm_config.model_config.rope_theta,
                out_scale=-1,
                qkv_scale=None,
                use_neox_rotary_style=True,
                qkv_bias=getattr(self.qkv_proj, "qkv_bias", None),
                linear_shift=getattr(self.o_proj, "linear_shift", None),
                linear_smooth=getattr(self.o_proj, "linear_smooth", None),
            )

    def load_state_dict(self, state_dict):
        """
        """
        self.qkv_proj.load_state_dict(state_dict)
        self.o_proj.load_state_dict(state_dict)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        input_ids: paddle.Tensor,
        rotary_embs: paddle.Tensor,
        rotary_emb_dims: paddle.Tensor,
        key_cache: paddle.Tensor,
        value_cache: paddle.Tensor,
        **attn_args
    ):
        """
        """
        qkv_out = self.qkv_proj(hidden_states)

        atten_out = self.attn(
            qkv=qkv_out,
            input_ids=input_ids,
            rotary_embs=rotary_embs,
            rotary_emb_dims=rotary_emb_dims,
            key_cache=key_cache,
            value_cache=value_cache,
            pre_key_cache=None,
            pre_value_cache=None,
            pre_caches_length=0,
            attn_mask=None,
            kv_signal_data=None,
            **attn_args,
        )

        output = self.o_proj(atten_out)
        return output


class Qwen2DecoderLayer(nn.Layer):
    """
    """
    def __init__(
        self,
        llm_config: LLMConfig,
        inference_args: InferenceArgs,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = Qwen2Attention(
            llm_config=llm_config,
            inference_args=inference_args,
            prefix=f"{prefix}.self_attn",
        )

        self.mlp = Qwen2MLP(
            llm_config=llm_config,
            inference_args=inference_args,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(
            llm_config,
            hidden_size=llm_config.model_config.hidden_size,
            eps=1e-6,
            prefix=f"{prefix}.input_layernorm",
        )

        self.post_attention_layernorm = RMSNorm(
            llm_config,
            hidden_size=llm_config.model_config.hidden_size,
            eps=1e-6,
            prefix=f"{prefix}.post_attention_layernorm",
        )

    def load_state_dict(self, state_dict):
        """
        """
        self.self_attn.load_state_dict(state_dict)
        self.mlp.load_state_dict(state_dict)
        self.input_layernorm.load_state_dict(state_dict)
        self.post_attention_layernorm.load_state_dict(state_dict)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor,
        input_ids: paddle.Tensor,
        rotary_embs: paddle.Tensor,
        rotary_emb_dims: paddle.Tensor,
        key_cache: paddle.Tensor,
        value_cache: paddle.Tensor,
        **attn_args
    ):
        """
        """
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            input_ids=input_ids,
            rotary_embs=rotary_embs,
            rotary_emb_dims=rotary_emb_dims,
            key_cache=key_cache,
            value_cache=value_cache,
            **attn_args
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2Model(nn.Layer):
    """
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
        super().__init__()
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

        is_mtp = draft_type in ["eagle", "mtp"]
        self.is_mtp = is_mtp

        llm_config.model_config.max_position_embeddings = max_position_embeddings
        llm_config.model_config.initializer_range = self.initializer_range
        llm_config.parallel_config.sequence_parallel = sequence_parallel
        llm_config.model_config.freeze_embedding = freeze_embedding
        llm_config.model_config.weight_sharing = weight_sharing
        llm_config.model_config.weight_sharing_add_bias = weight_sharing_add_bias
        llm_config.parallel_config.use_ep = use_ep
        llm_config.parallel_config.ep_size = 1 if use_ep else 1
        llm_config.model_config.rope_head_dim = hidden_size // num_attention_heads
        llm_config.model_config.prefix_name = "qwen2.mtp" if is_mtp else "qwen2"
        llm_config.model_config.use_rope = use_rope
        llm_config.parallel_config.column_cut = False
        llm_config.model_config.use_moe = use_moe

        self.llm_config = llm_config

        self.embeddings = VocabParallelEmbedding(
            llm_config=llm_config,
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=(f"{llm_config.model_config.prefix_name}.embeddings.word_embeddings"
                        if not use_moe else f"{llm_config.model_config.prefix_name}.embed_tokens"),
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
        elif self.inference_args.weight_dtype == "int4" and self.inference_args.act_dtype in [
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
        llm_config.quant_config.quant_round_type = self.inference_args.quant_round_type
        llm_config.quant_config.quant_max_bound = self.inference_args.quant_max_bound
        llm_config.quant_config.quant_min_bound = self.inference_args.quant_min_bound
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

        self.embeddings = VocabParallelEmbedding(
            llm_config=llm_config,
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=(f"{llm_config.model_config.prefix_name}.embed_tokens"),
        )

        self.layers = [
            Qwen2DecoderLayer(
                llm_config=llm_config,
                inference_args=self.inference_args,
                prefix=f"{llm_config.model_config.prefix_name}.layers.{i}"
            )
            for i in range(num_layers)
        ]

        self.last_layernorm = LayerNorm(
            llm_config,
            prefix="",
            hidden_size=llm_config.model_config.hidden_size,
            eps=1e-6
        )

        self.norm = RMSNorm(
            llm_config,
            hidden_size=llm_config.model_config.hidden_size,
            eps=1e-5,
            prefix=f"{llm_config.model_config.prefix_name}.norm",
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
            padding_offsets,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = get_padding_offset(input_ids, cum_offsets_now, token_num,
                               seq_lens_this_time)
        return (
            ids_remove_padding,
            padding_offsets,
            cum_offsets,
            cu_seqlens_q,
            cu_seqlens_k,
        )

    def load_state_dict(self, state_dict: dict[str,
                                              np.ndarray | paddle.Tensor]):
        """
        Load model parameters from a given state dictionary.

        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
        self.embeddings.load_state_dict(state_dict)
        self.norm.load_state_dict(state_dict)
        for i in range(self.num_layers):
            self.layers[i].load_state_dict(state_dict)


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
        mtp_hidden_states=None,
    ):
        """
        """
        kwargs = {}
        kwargs["input_ids"] = input_ids
        kwargs["token_type_ids"] = token_type_ids
        kwargs["image_features"] = image_features
        kwargs["attention_mask"] = attention_mask
        kwargs["rotary_embs"] = rope_emb
        kwargs["rotary_emb_dims"] = 1
        kwargs["caches"] = caches
        kwargs["seq_lens_this_time"] = seq_lens_this_time
        kwargs["seq_lens_encoder"] = seq_lens_encoder
        kwargs["seq_lens_decoder"] = seq_lens_decoder
        kwargs["block_tables"] = block_tables
        kwargs["beam_cache_offset"] = beam_cache_offset
        kwargs["draft_tokens"] = draft_tokens
        kwargs["output_padding_offset"] = output_padding_offset
        kwargs["step_idx"] = step_idx
        kwargs["max_input_length"] = self.max_len

        (
            ids_remove_padding,
            padding_offsets,
            cum_offsets,
            cu_seqlens_q,
            cu_seqlens_k,
        ) = self.remove_padding(input_ids, seq_lens_this_time)

        kwargs["padding_offsets"] = padding_offsets
        kwargs["cum_offsets"] = cum_offsets
        kwargs["cu_seqlens_q"] = cu_seqlens_q
        kwargs["cu_seqlens_k"] = cu_seqlens_k

        embedding_output = self.embeddings(
            ids_remove_padding=ids_remove_padding)

        if isinstance(embedding_output, tuple):
            kwargs["hidden_states"] = embedding_output[0]
        else:
            kwargs["hidden_states"] = embedding_output

        kwargs["encoder_block_shape_q"] = 64
        kwargs["decoder_block_shape_q"] = 16
        kwargs["max_partition_size"] = 32768
        kwargs["encoder_max_partition_size"] = 32768

        (
            kwargs["encoder_batch_ids"],
            kwargs["encoder_tile_ids_per_batch"],
            kwargs["encoder_num_blocks"],
            kwargs["kv_batch_ids"],
            kwargs["kv_tile_ids_per_batch"],
            kwargs["kv_num_blocks"],
            kwargs["decoder_batch_ids"],
            kwargs["decoder_tile_ids_per_batch"],
            kwargs["decoder_num_blocks"],
            kwargs["max_len_kv"],
            set_max_lengths,
        ) = get_block_shape_and_split_kv_block(
            kwargs.get("seq_lens_encoder", None),
            kwargs.get("seq_lens_decoder", None),
            kwargs.get("seq_lens_this_time", None),
            kwargs.get("cum_offsets", None),
            kwargs.get("encoder_block_shape_q", 64),
            kwargs.get("decoder_block_shape_q", 16),
            self.llm_config.model_config.num_attention_heads // self.llm_config.model_config.num_key_value_heads,
            kwargs.get("block_size", 64),
            self.inference_args.speculate_max_draft_token_num + 1,
        )
        kwargs["set_max_lengths"] = set_max_lengths
        kwargs["residual"] = None

        for i in range(self.num_layers):
            kwargs["key_cache"] = kwargs["caches"][2 * i]
            kwargs["value_cache"] = kwargs["caches"][2 * i + 1]

            kwargs["hidden_states"], kwargs["residual"] = self.layers[i](**kwargs)

        kwargs["hidden_states"], _ = self.last_layernorm(kwargs["hidden_states"], kwargs["residual"])

        kwargs["hidden_states"] = rebuild_padding(
            kwargs.get("hidden_states", None),
            kwargs.get("cum_offsets", None),
            kwargs.get("seq_lens_this_time", None),
            kwargs.get("seq_lens_decoder", None),
            kwargs.get("seq_lens_encoder", None),
            kwargs.get("output_padding_offset", None),
            kwargs.get("max_input_length", -1),
        )

        if isinstance(kwargs["hidden_states"][0], tuple):
            out = kwargs["hidden_states"][0]
        else:
            out = kwargs["hidden_states"]

        out = self.norm(out)

        return out


class Qwen2ForCausalLM(ModelForCasualLM):
    """
    Qwen2ForCausalLM
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
        super(Qwen2ForCausalLM, self).__init__(llm_config)
        self.configs = llm_config
        self.qwen2 = Qwen2Model(
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
            moe_use_ffn_shared_weight_and_bias=self.configs.moe_config.moe_use_ffn_shared_weight_and_bias,
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

        self.msg_queue_id = self.qwen2.msg_queue_id
        self.max_length = self.configs.decoding_config.max_dec_len
        self.min_length = self.configs.decoding_config.min_dec_len
        self.fake_server_p = self.configs.additional_config.fake_server_p
        self.decode_strategy = self.configs.decoding_config.decode_strategy
        self.speculate_max_candidate_len = self.configs.speculative_config.speculate_max_candidate_len
        self.speculate_verify_window = self.configs.speculative_config.speculate_verify_window
        self.use_moe = self.qwen2.use_moe

        self.ori_vocab_size = self.configs.model_config.ori_vocab_size

        self.use_top_k = self.configs.moe_config.use_top_k
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
        self.use_fast_ffn = self.qwen2.use_fast_ffn

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

        lmhead_name = ("server_nlg_mask_lm_trans_fc_" if not self.qwen2.is_mtp
                       else "mtp_server_nlg_mask_lm_trans_fc_")
        self.lm_head = LMHead(
            layer_name=lmhead_name,
            linear_weight_key="lm_head.weight",
            linear_bias_key=None,
            input_dim=self.hidden_size,
            output_dim=self.qwen2.vocab_size,
            fused_linear=self.configs.model_config.fused_linear,
        )

    @classmethod
    def name(self):
        """
        """
        return "Qwen2ForCausalLM"

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        """
        Load model parameters from a given state dictionary.

        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
        self.qwen2.load_state_dict(state_dict)
        self.lm_head.load_state_dict(state_dict)

    def get_output_padding_offset(self, seq_lens_this_time, seq_lens_encoder,
                                  seq_lens_decoder):
        """
        In the senerio of speculate decoding, the length of output token after rebuild_padding is no longer bsz.
        So we need to calculate the output_padding_offset after rebuild_padding.
        """
        seq_lens_output = speculate_get_seq_lens_output(
            seq_lens_this_time, seq_lens_encoder, seq_lens_decoder)
        out_token_num = paddle.sum(seq_lens_output)
        output_cum_offsets_tmp = paddle.cumsum(self.qwen2.max_len -
                                               seq_lens_output)
        output_padding_offset, output_cum_offsets = speculate_get_output_padding_offset(
            output_cum_offsets_tmp, out_token_num, seq_lens_output,
            self.qwen2.max_len)
        return output_padding_offset, output_cum_offsets

    def expand_inputs_for_generation(self,
                                     input_ids,
                                     expand_size,
                                     attention_mask=None,
                                     **kwargs):
        """
        Expand input IDs for generation.

        This method expands the input IDs by duplicating the first dimension and filling in the expanded indices.
        It also updates any other keyword arguments that may contain tensors with the same shape as `input_ids`.

        Args:
            input_ids (Tensor): Input IDs to be expanded.
            expand_size (int): Size of expansion.
            attention_mask (Tensor, optional): Attention mask to be updated. Defaults to None.
            **kwargs: Additional keyword arguments containing tensors with the same shape as `input_ids`.

        Returns:
            tuple: A tuple containing the expanded input IDs and the updated keyword arguments.
        """
        index = paddle.tile(
            paddle.arange(paddle.shape(input_ids)[0]).unsqueeze(-1),
            [1, expand_size],
        ).reshape([-1])

        input_ids = paddle.gather(input_ids, index)

        if attention_mask is not None:
            kwargs["attention_mask"] = paddle.gather(
                attention_mask, index)

        if ("token_type_ids" in kwargs
                and kwargs["token_type_ids"] is not None):
            token_type_ids = kwargs["token_type_ids"]
            kwargs["token_type_ids"] = paddle.gather(
                token_type_ids, index)

        if "position_ids" in kwargs and kwargs[
                "position_ids"] is not None:
            position_ids = kwargs["position_ids"]
            kwargs["position_ids"] = paddle.gather(position_ids, index)

        if "seq_len" in kwargs and kwargs["seq_len"] is not None:
            seq_len = kwargs["seq_len"]
            kwargs["seq_len"] = paddle.gather(seq_len, index)

        if ("encoder_output" in kwargs
                and kwargs["encoder_output"] is not None):
            encoder_output = kwargs["encoder_output"]
            kwargs["encoder_output"] = paddle.gather(
                encoder_output, index)

        if "role_ids" in kwargs and kwargs["role_ids"] is not None:
            role_ids = kwargs["role_ids"]
            kwargs["role_ids"] = paddle.gather(role_ids, index)

        return input_ids, kwargs

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
        # hidden_states = kwargs.get("hidden_states", None)
        # forward_meta = kwargs["forward_meta"]
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
            "mtp_hidden_states": None,
            # "forward_meta": forward_meta,
        }
        return model_inputs

    def sampling(
        self,
        logits,
        **kwargs,
    ):
        """Sample from GPT using beam search and post process the generated sequence.

        Args:
            logits (Tensor): The id of the token indicating the end of a sentence.
            **kwargs: Other arguments for forward pass of GPT model.

        Returns:
            Tensor: The sampled tokens. The shape is [batch_size].
        """
        temperature = kwargs["temperature"]
        top_k = self.top_k
        top_p = kwargs["top_p"]
        eos_token_id = kwargs["eos_token_id"]
        penalty_score = kwargs["penalty_score"]
        frequency_score = kwargs["frequency_score"]
        presence_score = kwargs["presence_score"]

        def _post_process_(
            logits,
            top_k,
            top_p,
            penalty_score,
            frequency_score,
            presence_score,
            temperature,
            kwargs,
        ):
            """
            Post process the generated sequence.
            """
            step_idx = kwargs["step_idx"]

            set_value_by_flags_and_idx(
                kwargs["pre_ids"],
                kwargs["input_ids"],
                kwargs["seq_lens_this_time"],
                kwargs["seq_lens_encoder"],
                kwargs["seq_lens_decoder"],
                step_idx,
                kwargs["stop_flags"],
            )

            # pre-process distribution
            logits = get_token_penalty_multi_scores(
                kwargs["pre_ids"],
                logits,
                penalty_score,
                frequency_score,
                presence_score,
                temperature,
                kwargs["bad_tokens"],
                step_idx,
                kwargs["min_dec_len"],
                eos_token_id,
            )

            # sample
            probs = F.softmax(logits)
            _, next_tokens = paddle.tensor.top_p_sampling(
                probs, top_p, seed=-1)  # have random_seed
            """ !!! ep not need broadcast, here broadcast just for test !!! """
            if self.qwen2.mp_size > 1 and (
                (not self.qwen2.use_ep or self.qwen2.ep_just_for_test) and
                (not self.fake_server_p)):
                paddle.distributed.broadcast(next_tokens, 0)

            paddle.assign(
                paddle.where(
                    kwargs["stop_flags"],
                    kwargs["step_idx"],
                    kwargs["step_idx"] + 1,
                ),
                kwargs["step_idx"],
            )
            length_cond = paddle.greater_equal(kwargs["step_idx"],
                                               kwargs["max_dec_len"])
            paddle.assign(
                paddle.logical_or(kwargs["stop_flags"], length_cond),
                kwargs["stop_flags"],
            )

            if self.qwen2.use_stop_seqs:
                set_stop_value_multi_seqs(
                    next_tokens,
                    kwargs["pre_ids"],
                    step_idx,
                    kwargs["stop_flags"],
                    kwargs["seq_lens_this_time"],
                    kwargs["stop_seqs"],
                    kwargs["stop_seqs_len"],
                    eos_token_id,
                )
            else:
                set_stop_value_multi_ends(
                    next_tokens,
                    kwargs["stop_flags"],
                    kwargs["seq_lens_this_time"],
                    eos_token_id,
                    kwargs["next_tokens"],
                    False,
                )  # multi ends
            # update inputs
            with paddle.framework._no_check_dy2st_diff():
                update_inputs(
                    kwargs["stop_flags"],
                    kwargs["not_need_stop"],
                    kwargs["seq_lens_this_time"],
                    kwargs["seq_lens_encoder"],
                    kwargs["seq_lens_decoder"],
                    kwargs["input_ids"],
                    kwargs["stop_nums"],
                    next_tokens,
                    kwargs["is_block_step"],
                )
            if self.qwen2.output_via_mq:
                if self.msg_queue_id is None:
                    save_output(
                        next_tokens,
                        kwargs["not_need_stop"],
                        self.qwen2.mp_rank,
                        self.qwen2.use_ep
                        and (not self.qwen2.ep_just_for_test),
                    )
                else:
                    save_output_dynamic(
                        next_tokens,
                        kwargs["not_need_stop"],
                        self.qwen2.mp_rank,
                        self.msg_queue_id,
                        self.qwen2.use_ep
                        and (not self.qwen2.ep_just_for_test),
                    )
            return next_tokens

        if ((not self.qwen2.use_ep) or (self.qwen2.ep_just_for_test)
                or (self.qwen2.use_ep and kwargs["not_need_stop"])):
            # first decoder
            next_tokens = _post_process_(
                logits,
                top_k,
                top_p,
                penalty_score,
                frequency_score,
                presence_score,
                temperature,
                kwargs,
            )
        else:
            # fake ep
            fake_input = paddle.empty(
                shape=[0, self.qwen2.inference_args.hidden_size],
                dtype=paddle.get_default_dtype(),
            )
            for i in range(
                    self.qwen2.inference_args.moe_config.moe_layer_start_index,
                    self.qwen2.inference_args.num_layers,
            ):
                self.qwen2.decoder.moe_layers[i](fake_input)
            next_tokens = None

        return next_tokens

    def speculate_decoding(
        self,
        outputs,  # hidden_states
        **kwargs,
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
            **kwargs: Other arguments for forward pass of GPT model.

        Returns:
            Tensor: The sampled tokens. The shape is [batch_size].
        """
        temperature = kwargs["temperature"]
        top_p = kwargs["top_p"]
        eos_token_id = kwargs["eos_token_id"]
        penalty_score = kwargs["penalty_score"]
        frequency_score = kwargs["frequency_score"]
        presence_score = kwargs["presence_score"]

        def _post_process_(
            outputs,
            top_p,
            penalty_score,
            frequency_score,
            presence_score,
            temperature,
            kwargs,
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
                    kwargs["seq_lens_encoder"],
                    kwargs["seq_lens_decoder"],
                    kwargs["actual_output_padding_offset"],
                    self.qwen2.max_len,
                )
            else:
                hidden_states = outputs[0] if isinstance(outputs,
                                                         tuple) else outputs
            logits = self.lm_head(hidden_states)

            logits = paddle.cast(logits, paddle.float32)
            logits[:, self.ori_vocab_size:] = -float("inf")

            speculate_get_token_penalty_multi_scores(
                kwargs["pre_ids"],
                logits,
                penalty_score,
                frequency_score,
                presence_score,
                temperature,
                kwargs["bad_tokens"],
                kwargs["step_idx"],
                kwargs["min_dec_len"],
                eos_token_id,
                kwargs["seq_lens_this_time"],
                kwargs["actual_output_padding_offset"],
                kwargs["output_cum_offsets"],
                self.qwen2.max_len,
            )

            # sample
            probs = F.softmax(logits)

            verify_scores, verify_tokens, actual_candidate_len = top_p_candidates(
                probs,
                top_p,
                kwargs["actual_output_padding_offset"],
                self.speculate_max_candidate_len,
                self.qwen2.max_len,
            )

            speculate_verify(
                kwargs["accept_tokens"],
                kwargs["accept_num"],
                kwargs["step_idx"],
                kwargs["stop_flags"],
                kwargs["seq_lens_encoder"],
                kwargs["seq_lens_decoder"],
                kwargs[
                    "draft_tokens"],  # Both input and output, need to write the last 1 token accepted to position 0.
                kwargs["seq_lens_this_time"],
                verify_tokens,
                verify_scores,
                kwargs["max_dec_len"],
                eos_token_id,
                kwargs["is_block_step"],
                kwargs["output_cum_offsets"],
                actual_candidate_len,
                kwargs["actual_draft_token_num"],
                top_p,
                self.qwen2.max_len,
                self.speculate_verify_window,
                True,  # enable_topp
            )

            # BroadCast
            if self.qwen2.mp_size > 1:
                paddle.distributed.broadcast(kwargs["accept_tokens"], 0)
                paddle.distributed.broadcast(kwargs["accept_num"], 0)
                paddle.distributed.broadcast(kwargs["step_idx"], 0)
                paddle.distributed.broadcast(kwargs["stop_flags"], 0)

            if self.qwen2.use_stop_seqs:
                speculate_set_stop_value_multi_seqs(
                    kwargs["accept_tokens"],
                    kwargs["accept_num"],
                    kwargs["pre_ids"],
                    kwargs["step_idx"],
                    kwargs["stop_flags"],
                    kwargs["seq_lens_this_time"],
                    kwargs["stop_seqs"],
                    kwargs["stop_seqs_len"],
                    eos_token_id,
                )

            # Update
            speculate_update_v3(
                kwargs["seq_lens_encoder"],
                kwargs["seq_lens_decoder"],
                kwargs["not_need_stop"],
                kwargs["draft_tokens"],
                kwargs["actual_draft_token_num"],
                kwargs["accept_tokens"],
                kwargs["accept_num"],
                kwargs["stop_flags"],
                kwargs["seq_lens_this_time"],
                kwargs["is_block_step"],
                kwargs["stop_nums"],
            )
            # Streaming output
            if not (self.qwen2.speculate_method == "mtp" and
                    self.qwen2.generation_phase == GenerationPhase.PREFILL):
                if self.msg_queue_id is None:
                    speculate_save_output(
                        kwargs["accept_tokens"],
                        kwargs["accept_num"],
                        kwargs["not_need_stop"],
                        self.rank,
                    )
                else:
                    speculate_save_output_dynamic(
                        kwargs["accept_tokens"],
                        kwargs["accept_num"],
                        kwargs["not_need_stop"],
                        self.rank,
                        self.msg_queue_id,
                    )

            # If seq_lens_decoder is 0 (means stop), accept_num should be set to 0
            speculate_clear_accept_nums(kwargs["accept_num"],
                                        kwargs["seq_lens_decoder"])

            # Update pre_ids through accept tokens
            speculate_set_value_by_flags_and_idx(
                kwargs["pre_ids"],
                kwargs["accept_tokens"],
                kwargs["accept_num"],
                kwargs["stop_flags"],
                kwargs["seq_lens_this_time"],
                kwargs["seq_lens_encoder"],
                kwargs["seq_lens_decoder"],
                kwargs["step_idx"],
            )

        output_padding_offset, output_cum_offsets = self.get_output_padding_offset(
            kwargs["seq_lens_this_time"],
            kwargs["seq_lens_encoder"],
            kwargs["seq_lens_decoder"],
        )
        kwargs["actual_output_padding_offset"] = output_padding_offset
        kwargs["output_cum_offsets"] = output_cum_offsets

        # first decoder
        _post_process_(
            outputs,
            top_p,
            penalty_score,
            frequency_score,
            presence_score,
            temperature,
            kwargs,
        )

        return outputs

    def beam_search(
        self,
        outputs,  # hidden_states
        **kwargs,
    ):
        """Sample from GPT using beam search and post process the generated sequence.

        Args:
            eos_token_id (int): The id of the token indicating the end of a sentence.
            penalty_score (dict): A dict containing penalty scores of different types.
            frequency_score (dict): A dict containing frequency score of each token.
            presence_score (dict): A dict containing presence score of each token.
            temperature (float, optional): The value used to module the logits. Defaults to None.
            **kwargs: Other arguments for forward pass of GPT model.

        Returns:
            Tensor: BeamHypotheses. The shape is [batch_size * beam_width, max_dec_len].
        """
        temperature = kwargs["temperature"]
        eos_token_id = kwargs["eos_token_id"]
        penalty_score = kwargs["penalty_score"]
        frequency_score = kwargs["frequency_score"]
        presence_score = kwargs["presence_score"]

        def _post_process_(
            outputs,
            penalty_score,
            frequency_score,
            presence_score,
            temperature,
            **kwargs,
        ):
            step_idx = kwargs["step_idx"]

            set_value_by_flags_and_idx(
                kwargs["pre_ids"],
                kwargs["input_ids"],
                kwargs["seq_lens_this_time"],
                kwargs["seq_lens_encoder"],
                kwargs["seq_lens_decoder"],
                step_idx,
                kwargs["stop_flags"],
            )
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = self.lm_head(logits)

            logits = paddle.cast(logits, paddle.float32)
            update_inputs_beam(
                kwargs["beam_width"].cpu(),
                kwargs["seq_lens_this_time"],
                kwargs["seq_lens_encoder"],
                kwargs["input_ids"],
                logits,
            )

            logits[:, self.ori_vocab_size:] = -float("inf")
            # pre-process distribution
            logits = get_token_penalty_multi_scores(
                kwargs["pre_ids"],
                logits,
                penalty_score,
                frequency_score,
                presence_score,
                temperature,
                kwargs["bad_tokens"],
                step_idx,
                kwargs["min_dec_len"],
                eos_token_id,
            )

            tmp_seq = paddle.where(
                kwargs["seq_lens_decoder"] == 0,
                kwargs["seq_lens_encoder"] - 1,
                kwargs["seq_lens_decoder"],
            )

            next_tokens, parent_ids = beam_search_softmax(
                logits=logits,
                seq_lens=tmp_seq.astype("int32"),
                stop_flags=kwargs["stop_flags"],
                end_ids=eos_token_id.astype("int32"),
                step_ids=step_idx.astype("int32"),
                max_dec_lens=kwargs["max_dec_len"].astype("int32"),
                block_tables=kwargs["block_tables"],
                cum_scores=kwargs["cum_score"],
                beam_cache_ids=kwargs["beam_cache_ids"],
                beam_hyps=kwargs["beam_hyps"],
                beam_hyps_score=kwargs["beam_hyps_score"],
                beam_finished=kwargs["beam_finished"],
                beam_width=kwargs["beam_width"],
                beam_group_num=kwargs["beam_group_num"],
                length_penalty=kwargs["beam_length_penalty"],
                diversity_penalty=kwargs["beam_diversity_penalty"],
                fuse_softmax=True,
                early_stop=False,
            )

            next_tokens = next_tokens.astype("int64")

            paddle.assign(
                paddle.where(
                    kwargs["beam_finished"],
                    kwargs["step_idx"],
                    kwargs["step_idx"] + 1,
                ),
                kwargs["step_idx"],
            )
            length_cond = paddle.greater_equal(kwargs["step_idx"],
                                               kwargs["max_dec_len"])
            paddle.assign(
                paddle.logical_or(kwargs["beam_finished"], length_cond),
                kwargs["beam_finished"],
            )

            set_stop_value_multi_ends(
                next_tokens,
                kwargs["beam_finished"],
                kwargs["seq_lens_this_time"],
                eos_token_id,
                kwargs["next_tokens"],
                True,
            )  # multi ends

            # update inputs
            update_inputs(
                kwargs["beam_finished"],
                kwargs["not_need_stop"],
                kwargs["seq_lens_this_time"],
                kwargs["seq_lens_encoder"],
                kwargs["seq_lens_decoder"],
                kwargs["input_ids"],
                kwargs["stop_nums"],
                next_tokens,
                kwargs["is_block_step"],
            )

        _post_process_(
            outputs,
            penalty_score,
            frequency_score,
            presence_score,
            temperature,
            **kwargs,
        )

        return kwargs["beam_hyps"]

    def draft_model_sampling(
        self,
        logits,
        **kwargs,
    ):
        """Sample from GPT using beam search and post process the generated sequence.

        Args:
            eos_token_id (int): The id of the token indicating the end of a sentence.
            top_p (float): If set to float < 1, only the tokens with probabilities greater than or equal to
                the threshold are kept for generation.
            **kwargs: Other arguments for forward pass of GPT model.

        Returns:
            Tensor: The sampled tokens. The shape is [batch_size].
        """
        top_p = kwargs["top_p"]
        eos_token_id = kwargs["eos_token_id"]

        def _post_process_(
            logits,
            top_p,
            kwargs,
        ):
            probs = F.softmax(logits)

            _, inter_next_tokens = paddle.tensor.top_p_sampling(probs,
                                                                top_p,
                                                                seed=-1)

            if self.qwen2.mp_size > 1:
                paddle.distributed.broadcast(inter_next_tokens, 0)

            draft_model_update(
                inter_next_tokens,
                kwargs["draft_tokens"],
                kwargs["pre_ids"],
                kwargs["seq_lens_this_time"],
                kwargs["seq_lens_encoder"],
                kwargs["seq_lens_decoder"],
                kwargs["step_idx"],
                kwargs["output_cum_offsets"],
                kwargs["stop_flags"],
                kwargs["not_need_stop"],
                kwargs["max_dec_len"],
                eos_token_id,
                kwargs["base_model_draft_tokens"],
                self.qwen2.max_len,
                kwargs["substep"],
            )
            if (self.qwen2.speculate_method in ["mtp", "draft_model", "eagle"]
                    and self.qwen2.generation_phase
                    == GenerationPhase.PREFILL):
                if self.msg_queue_id is None:
                    mtp_save_first_token(
                        kwargs["base_model_draft_tokens"],
                        kwargs["not_need_stop"],
                        self.qwen2.mp_rank,
                        self.qwen2.use_ep
                        and (not self.qwen2.ep_just_for_test),
                    )
                else:
                    mtp_save_first_token_dynamic(
                        kwargs["base_model_draft_tokens"],
                        kwargs["not_need_stop"],
                        self.qwen2.mp_rank,
                        self.msg_queue_id,
                        self.qwen2.use_ep
                        and (not self.qwen2.ep_just_for_test),
                    )
            return hidden_states

        output_padding_offset, output_cum_offsets = self.get_output_padding_offset(
            kwargs["seq_lens_this_time"],
            kwargs["seq_lens_encoder"],
            kwargs["seq_lens_decoder"],
        )
        kwargs["actual_output_padding_offset"] = output_padding_offset
        kwargs["output_cum_offsets"] = output_cum_offsets

        # first decoder
        hidden_states = _post_process_(logits, top_p, kwargs)

        return hidden_states

    def compute_logits(self, hidden_states):
        """
        """
        logits = self.lm_head(hidden_states)
        logits = paddle.cast(logits, paddle.float32)
        logits[:, self.ori_vocab_size:] = -float("inf")
        return logits

    def forward(self, **kwargs):
        """
        """
        model_inputs = self.prepare_inputs_for_generation(**kwargs)
        hidden_states = self.qwen2(**model_inputs)
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

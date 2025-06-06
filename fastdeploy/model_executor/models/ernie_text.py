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

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import paddle
from paddle import nn

from fastdeploy.config import LLMConfig
from fastdeploy.model_executor.layers.activation import SiluAndMul
from fastdeploy.model_executor.layers.attention import Attention
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.linear import (
    MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear)
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import LayerNorm, RMSNorm
from fastdeploy.worker.model_runner import ForwardMeta

from .model_base import ModelForCasualLM


class Ernie45TMLP(nn.Layer):
    """
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.nranks = llm_config.parallel_config.mp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            llm_config=llm_config,
            prefix=f"{prefix}.up_gate_proj",
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
            llm_config=llm_config,
            bias=None,
            act_method=llm_config.model_config.hidden_act,
        )

    def load_state_dict(self, state_dict):
        self.gate_up_proj.load_state_dict(state_dict)
        self.down_proj.load_state_dict(state_dict)

    def forward(self, hidden_states: paddle.Tensor):
        gate_up_out = self.gate_up_proj(hidden_states)
        act_out = self.act_fn(gate_up_out)
        down_out = self.down_proj(act_out)
        return down_out


class Ernie45TAttention(nn.Layer):

    def __init__(self, llm_config: LLMConfig, layer_id: int,
                 prefix: str) -> None:
        super().__init__()

        nranks = llm_config.parallel_config.mp_size

        self.qkv_proj = QKVParallelLinear(
            llm_config=llm_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            llm_config=llm_config,
            prefix=f"{prefix}.o_proj",
            input_size=(llm_config.model_config.hidden_size // nranks),
            output_size=llm_config.model_config.hidden_size,
        )

        self.attn = Attention(
            llm_config=llm_config,
            layer_id=layer_id,
            prefix=prefix,
        )

    def load_state_dict(self, state_dict):
        self.qkv_proj.load_state_dict(state_dict)
        self.o_proj.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
    ):
        qkv_out = self.qkv_proj(hidden_states)

        attn_out = self.attn(
            qkv=qkv_out,
            forward_meta=forward_meta,
        )

        output = self.o_proj(attn_out)

        return output


class Ernie45TDecoderLayer(nn.Layer):

    def __init__(
        self,
        llm_config: LLMConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_id = int(prefix.split(sep='.')[-1])

        self.self_attn = Ernie45TAttention(
            llm_config=llm_config,
            layer_id=layer_id,
            prefix=f"{prefix}.self_attn",
        )

        if (llm_config.moe_config.num_experts is not None
                and layer_id >= llm_config.moe_config.moe_layer_start_index):
            self.mlp = FusedMoE(
                llm_config=llm_config,
                moe_intermediate_size=llm_config.moe_config.
                moe_intermediate_size,
                num_experts=llm_config.moe_config.num_experts,
                top_k=llm_config.moe_config.top_k,
                moe_use_gate_correction_bias=llm_config.moe_config.
                moe_use_gate_correction_bias,
                moe_quant_type=llm_config.moe_config.moe_quant_type,
                layer_idx=layer_id,
                gate_weight_key=f"{prefix}.mlp.gate.weight",
                gate_correction_bias_key=
                f"{prefix}.mlp.moe_statics.e_score_correction_bias",
                ffn1_expert_weight_key=
                f"{prefix}.mlp.experts.{{}}.up_gate_proj.weight",
                ffn2_expert_weight_key=
                f"{prefix}.mlp.experts.{{}}.down_proj.weight",
            )
        else:
            self.mlp = Ernie45TMLP(
                llm_config=llm_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(
            llm_config,
            hidden_size=llm_config.model_config.hidden_size,
            eps=1e-5,
            prefix=f"{prefix}.input_layernorm",
        )

        self.post_attention_layernorm = RMSNorm(
            llm_config,
            hidden_size=llm_config.model_config.hidden_size,
            eps=1e-5,
            prefix=f"{prefix}.post_attention_layernorm",
        )

    def load_state_dict(self, state_dict):
        self.self_attn.load_state_dict(state_dict)
        self.mlp.load_state_dict(state_dict)
        self.input_layernorm.load_state_dict(state_dict)
        self.post_attention_layernorm.load_state_dict(state_dict)

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor = None,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        print("[debug6]\n", hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            forward_meta=forward_meta,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        print("[debug5]\n", hidden_states)

        hidden_states = self.mlp(hidden_states)

        print("[debug5.5]\n", hidden_states)

        return hidden_states, residual


class Ernie45TModel(nn.Layer):

    def __init__(
        self,
        llm_config: LLMConfig = None,
    ):
        """
        Initializer for the ErnieBotFusedModel class.

        Args:

        """
        super().__init__()

        self.num_layers = llm_config.model_config.num_layers

        self.embeddings = VocabParallelEmbedding(
            llm_config=llm_config,
            num_embeddings=llm_config.model_config.vocab_size,
            embedding_dim=llm_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=(f"{llm_config.model_config.prefix_name}.embed_tokens"),
        )

        self.hidden_layers = [
            Ernie45TDecoderLayer(
                llm_config=llm_config,
                prefix=f"{llm_config.model_config.prefix_name}.layers.{i}")
            for i in range(self.num_layers)
        ]

        self.last_layernorm = LayerNorm(
            llm_config,
            prefix="",
            hidden_size=llm_config.model_config.hidden_size,
            eps=1e-5)

        self.norm = RMSNorm(
            llm_config,
            hidden_size=llm_config.model_config.hidden_size,
            eps=1e-5,
            prefix=f"{llm_config.model_config.prefix_name}.norm",
        )

    def load_state_dict(self, state_dict):
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
            self.hidden_layers[i].load_state_dict(state_dict)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        """
        """

        hidden_states = self.embeddings(ids_remove_padding=ids_remove_padding)

        print("[debug4]\n", hidden_states)

        residual = None
        for i in range(self.num_layers):
            hidden_states, residual = self.hidden_layers[i](forward_meta,
                                                            hidden_states,
                                                            residual)

        hidden_states, _ = self.last_layernorm(hidden_states, residual)

        out = self.norm(hidden_states)

        return out


class ErnieForCausalLM(ModelForCasualLM):
    """
    ErnieForCausalLM
    """

    def __init__(self, llm_config: LLMConfig):
        """
        Args:
            llm_config (LLMConfig): Configurations for the LLM model.
        """
        super(ErnieForCausalLM, self).__init__(llm_config)

        self.model = Ernie45TModel(llm_config=llm_config)

        self.ori_vocab_size = llm_config.model_config.ori_vocab_size

        self.lm_head = ParallelLMHead(
            llm_config=llm_config,
            embedding_dim=llm_config.model_config.hidden_size,
            num_embeddings=llm_config.model_config.vocab_size,
            prefix="lm_head",
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
        self.model.load_state_dict(state_dict)
        self.lm_head.load_state_dict(state_dict)

    def compute_logits(self, hidden_states: paddle.Tensor):
        logits = self.lm_head(hidden_states)
        logits = paddle.cast(logits, paddle.float32)
        logits[:, self.ori_vocab_size:] = -float("inf")

        return logits

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        hidden_states = self.model(ids_remove_padding, forward_meta)

        return hidden_states

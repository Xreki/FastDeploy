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
from paddlenlp.utils.log import logger

from fastdeploy.config import FDConfig
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

    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.nranks = fd_config.parallel_config.mp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.up_gate_proj",
            with_bias=False,
            activation=fd_config.model_config.hidden_act,
            use_fast_ffn=True,
        )

        self.down_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.down_proj",
            input_size=(fd_config.model_config.ffn_hidden_size //
                        self.nranks),
            output_size=fd_config.model_config.hidden_size,
            with_bias=False,
        )

        self.act_fn = SiluAndMul(
            fd_config=fd_config,
            bias=None,
            act_method=fd_config.model_config.hidden_act,
        )

    def load_state_dict(self, state_dict):
        self.gate_up_proj.load_state_dict(state_dict)
        self.down_proj.load_state_dict(state_dict)

    def forward(self, hidden_states: paddle.Tensor):
        gate_up_out = self.gate_up_proj(hidden_states)
        act_out = self.act_fn(gate_up_out)
        down_out = self.down_proj(act_out)
        return down_out


class Ernie45TMoE(nn.Layer):

    def __init__(self, pd_config: PDConfig, layer_id: int,
                 prefix: str) -> None:
        super().__init__()

        self.fused_moe = FusedMoE(
            pd_config=pd_config,
            moe_intermediate_size=pd_config.moe_config.moe_intermediate_size,
            num_experts=pd_config.moe_config.num_experts,
            top_k=pd_config.moe_config.top_k,
            moe_use_gate_correction_bias=pd_config.moe_config.
            moe_use_gate_correction_bias,
            moe_quant_type=pd_config.moe_config.moe_quant_type,
            layer_idx=layer_id,
            gate_weight_key=f"{prefix}.gate.weight",
            gate_correction_bias_key=
            f"{prefix}.moe_statics.e_score_correction_bias",
            ffn1_expert_weight_key=f"{prefix}.experts.{{}}.up_gate_proj.weight",
            ffn2_expert_weight_key=f"{prefix}.experts.{{}}.down_proj.weight",
        )

    def load_state_dict(self, state_dict):
        self.fused_moe.load_state_dict(state_dict)

    def forward(self, hidden_states: paddle.Tensor):
        out = self.fused_moe(hidden_states)
        return out


class Ernie45TAttention(nn.Layer):

    def __init__(self, fd_config: FDConfig, layer_id: int,
                 prefix: str) -> None:
        super().__init__()

        nranks = fd_config.parallel_config.mp_size

        self.qkv_proj = QKVParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=(fd_config.model_config.hidden_size // nranks),
            output_size=fd_config.model_config.hidden_size,
        )

        self.attn = Attention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=prefix,
            use_neox_rotary_style=False,
        )

    def load_state_dict(self, state_dict):
        self.qkv_proj.load_state_dict(state_dict)
        self.o_proj.load_state_dict(state_dict)
        self.attn.load_state_dict(state_dict)

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
        fd_config: FDConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_id = int(prefix.split(sep='.')[-1])

        self.self_attn = Ernie45TAttention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.self_attn",
        )

        if (pd_config.moe_config.num_experts is not None
                and layer_id >= pd_config.moe_config.moe_layer_start_index):
            self.mlp = Ernie45TMoE(
                pd_config=pd_config,
                layer_id=layer_id,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Ernie45TMLP(
                fd_config=fd_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=1e-5,
            prefix=f"{prefix}.input_layernorm",
        )

        self.post_attention_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
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

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            forward_meta=forward_meta,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Ernie45TModel(nn.Layer):

    def __init__(
        self,
        fd_config: FDConfig = None,
    ):
        """
        Initializer for the Ernie45TModel class.

        Args:

        """
        super().__init__()

        self.num_layers = fd_config.model_config.num_layers
        fd_config.model_config.prefix_name = "ernie"

        self.embeddings = VocabParallelEmbedding(
            fd_config=fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=(f"{fd_config.model_config.prefix_name}.embed_tokens"),
        )

        self.hidden_layers = [
            Ernie45TDecoderLayer(
                fd_config=fd_config,
                prefix=f"{fd_config.model_config.prefix_name}.layers.{i}")
            for i in range(self.num_layers)
        ]

        self.last_layernorm = LayerNorm(
            fd_config,
            prefix="",
            hidden_size=fd_config.model_config.hidden_size,
            eps=1e-5)

        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=1e-5,
            prefix=f"{fd_config.model_config.prefix_name}.norm",
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
            logger.info(f"Start load layer {i}")
            self.hidden_layers[i].load_state_dict(state_dict)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        hidden_states = self.embeddings(ids_remove_padding=ids_remove_padding)

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

    def __init__(self, fd_config: FDConfig):
        """
        Args:
            fd_config (FDConfig): Configurations for the LLM model.
        """
        super(ErnieForCausalLM, self).__init__(fd_config)

        self.model = Ernie45TModel(fd_config=fd_config)

        self.ori_vocab_size = fd_config.model_config.ori_vocab_size

        self.lm_head = ParallelLMHead(
            fd_config=fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
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

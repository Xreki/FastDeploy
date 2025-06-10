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

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import paddle
from paddle import nn
from paddlenlp.utils.log import logger

from fastdeploy.config import LLMConfig
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import LayerNorm, RMSNorm
from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.model_executor.models.ernie_text import (Ernie45TAttention,
                                                         Ernie45TMLP)
from fastdeploy.model_executor.ops.gpu import (extract_text_token_output,
                                               text_image_gather_scatter,
                                               text_image_index_out)
from fastdeploy.worker.model_runner import ForwardMeta

from ..model_base import ModelForCasualLM


class Ernie45TVLMLP(Ernie45TMLP):
    pass


class Ernie45TVLAttention(Ernie45TAttention):
    pass


@dataclass
class VLMoEMeta:
    image_input: Optional[paddle.Tensor] = None
    text_input: Optional[paddle.Tensor] = None
    text_index: Optional[paddle.Tensor] = None
    image_index: Optional[paddle.Tensor] = None


class Ernie45TVLMoE(nn.Layer):

    def __init__(self, llm_config: LLMConfig, layer_id: int,
                 prefix: str) -> None:
        super().__init__()

        # TODO: Fix 传参
        self.fused_moe_text = FusedMoE(
            llm_config=llm_config,
            moe_intermediate_size=llm_config.moe_config.
            moe_intermediate_size[0],
            num_experts=llm_config.moe_config.num_experts[0],
            top_k=llm_config.moe_config.top_k,
            moe_use_gate_correction_bias=llm_config.moe_config.
            moe_use_gate_correction_bias,
            moe_quant_type=llm_config.moe_config.moe_quant_type,
            layer_idx=layer_id,
            moe_tag="Text",
            gate_weight_key=f"{prefix}.gate.weight",
            gate_correction_bias_key=
            f"{prefix}.moe_statics.e_score_correction_bias",
            ffn1_expert_weight_key=f"{prefix}.experts.{{}}.up_gate_proj.weight",
            ffn2_expert_weight_key=f"{prefix}.experts.{{}}.down_proj.weight",
        )
        self.fused_moe_text.load_gate_state_dict = self.load_gate_state_dict_text

        self.fused_moe_image = FusedMoE(
            llm_config=llm_config,
            moe_intermediate_size=llm_config.moe_config.
            moe_intermediate_size[1],
            num_experts=llm_config.moe_config.num_experts[1],
            top_k=llm_config.moe_config.top_k,
            moe_use_gate_correction_bias=llm_config.moe_config.
            moe_use_gate_correction_bias,
            moe_quant_type=llm_config.moe_config.moe_quant_type,
            layer_idx=layer_id,
            moe_tag="Image",
            gate_weight_key=f"{prefix}.gate.weight_1",
            gate_correction_bias_key=
            f"{prefix}.moe_statics.e_score_correction_bias",
            ffn1_expert_weight_key=f"{prefix}.experts.{{}}.up_gate_proj.weight",
            ffn2_expert_weight_key=f"{prefix}.experts.{{}}.down_proj.weight",
        )
        self.fused_moe_image.load_gate_state_dict = self.load_gate_state_dict_image

    def load_gate_state_dict_text(self, state_dict):
        """
        load_gate_state_dict function for text
        """
        # gate_correction_bias
        if self.fused_moe_text.moe_use_gate_correction_bias:
            gate_correction_bias_tensor = get_tensor(
                state_dict[self.fused_moe_text.gate_correction_bias_key])

            self.fused_moe_text.gate_correction_bias = self.fused_moe_text.create_parameter(
                shape=[1, self.fused_moe_text.num_experts],
                dtype="float32",
            )

            self.fused_moe_text.gate_correction_bias.set_value(
                gate_correction_bias_tensor[0].unsqueeze(0))
        else:
            self.fused_moe_text.gate_correction_bias = None

        up_gate_proj_weight = []
        down_proj_weight = []
        for j in range(0, self.fused_moe_text.num_experts):
            up_gate_proj_weight.append(
                get_tensor(
                    state_dict.pop(
                        self.fused_moe_text.ffn1_expert_weight_key.format(j))))
            down_proj_weight.append(
                get_tensor(
                    state_dict.pop(
                        self.fused_moe_text.ffn2_expert_weight_key.format(j))))
        return up_gate_proj_weight, down_proj_weight

    def load_gate_state_dict_image(self, state_dict):
        """
        load_gate_state_dict function for image
        """
        # gate_correction_bias
        if self.fused_moe_image.moe_use_gate_correction_bias:
            gate_correction_bias_tensor = get_tensor(
                state_dict[self.fused_moe_image.gate_correction_bias_key])

            self.fused_moe_image.gate_correction_bias = self.fused_moe_image.create_parameter(
                shape=[1, self.fused_moe_image.num_experts],
                dtype="float32",
            )

            self.fused_moe_image.gate_correction_bias.set_value(
                gate_correction_bias_tensor[1].unsqueeze(0))
        else:
            self.fused_moe_image.gate_correction_bias = None

        up_gate_proj_weight = []
        down_proj_weight = []
        for j in range(
                self.fused_moe_text.num_experts,
                self.fused_moe_text.num_experts +
                self.fused_moe_image.num_experts):
            up_gate_proj_weight.append(
                get_tensor(
                    state_dict.pop(
                        self.fused_moe_image.ffn1_expert_weight_key.format(
                            j))))
            down_proj_weight.append(
                get_tensor(
                    state_dict.pop(
                        self.fused_moe_image.ffn2_expert_weight_key.format(
                            j))))
        return up_gate_proj_weight, down_proj_weight

    def load_state_dict(self, state_dict):
        self.fused_moe_text.load_state_dict(state_dict)
        self.fused_moe_image.load_state_dict(state_dict)
        state_dict.pop(self.fused_moe_text.gate_correction_bias_key)

    def forward(self, hidden_states: paddle.Tensor, vl_moe_meta: VLMoEMeta):
        image_input = vl_moe_meta.get("image_input", None)
        if image_input is not None:
            token_type_ids = vl_moe_meta.get("token_type_ids", None)
            text_input = vl_moe_meta.get("text_input", None)
            text_index = vl_moe_meta.get("text_index", None)
            image_index = vl_moe_meta.get("image_index", None)
            text_image_gather_scatter(hidden_states, text_input, image_input,
                                      token_type_ids, text_index, image_index,
                                      True)
            text_out = self.text_moe_layer(text_input)
            image_out = self.image_moe_layer(image_input)
            text_image_gather_scatter(hidden_states, text_out, image_out,
                                      token_type_ids, text_index, image_index,
                                      False)
        else:
            hidden_states = self.text_moe_layer(hidden_states)
        return hidden_states


class Ernie45TVLDecoderLayer(nn.Layer):

    def __init__(
        self,
        llm_config: LLMConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_id = int(prefix.split(sep='.')[-1])

        self.self_attn = Ernie45TVLAttention(
            llm_config=llm_config,
            layer_id=layer_id,
            prefix=f"{prefix}.self_attn",
        )

        if (llm_config.moe_config.num_experts is not None
                and layer_id >= llm_config.moe_config.moe_layer_start_index):
            self.mlp = Ernie45TVLMoE(
                llm_config=llm_config,
                layer_id=layer_id,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Ernie45TVLMLP(
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
        vl_moe_meta: VLMoEMeta = None,
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

        if isinstance(self.mlp, Ernie45TVLMoE):
            hidden_states = self.mlp(hidden_states, vl_moe_meta)
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Ernie45TVLModel(nn.Layer):

    def __init__(
        self,
        llm_config: LLMConfig = None,
    ):
        """
        Initializer for the Ernie45TVLModel class.

        Args:

        """
        super().__init__()

        self.num_layers = llm_config.model_config.num_layers
        self.im_patch_id = llm_config.moe_config.im_patch_id
        self._dtype = llm_config.model_config.dtype
        llm_config.model_config.prefix_name = "ernie"

        self.embeddings = VocabParallelEmbedding(
            llm_config=llm_config,
            num_embeddings=llm_config.model_config.vocab_size,
            embedding_dim=llm_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=(f"{llm_config.model_config.prefix_name}.embed_tokens"),
        )

        self.hidden_layers = [
            Ernie45TVLDecoderLayer(
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
            logger.info(f"Start load layer {i}")
            self.hidden_layers[i].load_state_dict(state_dict)

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        text_input = None
        image_input = None
        text_index = None
        image_index = None
        image_token_num = 0

        hidden_states = self.embeddings(ids_remove_padding=ids_remove_padding)

        # -----------------------
        image_mask = ids_remove_padding == self.im_patch_id
        token_type_ids = image_mask.cast("int32")
        token_num = hidden_states.shape[0]
        image_token_num = paddle.count_nonzero(token_type_ids).cast("int32")
        text_token_num = ((token_num - image_token_num) if
                          (token_num - image_token_num) > 0 else 1)
        if image_mask.any():
            hidden_states[image_mask] = forward_meta.image_features.cast(
                self._dtype)
            text_input = paddle.full(
                shape=[text_token_num, hidden_states.shape[1]],
                fill_value=1,
                dtype=self._dtype)
            image_input = paddle.full(
                shape=[image_token_num, hidden_states.shape[1]],
                fill_value=1,
                dtype=self._dtype)
            text_index = paddle.zeros_like(token_type_ids)
            image_index = paddle.zeros_like(token_type_ids)
            text_image_index_out(token_type_ids, text_index, image_index)

        vl_moe_meta = VLMoEMeta(
            text_input=text_input,
            image_input=image_input,
            text_index=text_index,
            image_index=image_index,
        )
        # -----------------------

        residual = None
        for i in range(self.num_layers):
            hidden_states, residual = self.hidden_layers[i](
                forward_meta,
                hidden_states,
                residual,
                vl_moe_meta,
            )

        hidden_states, _ = self.last_layernorm(hidden_states, residual)

        # -----------------------
        hidden_states = hidden_states.cast("float32")
        score_text = hidden_states

        if image_input is not None:
            token_type_ids = token_type_ids.reshape([-1])
            text_pos_shifted = token_type_ids[:token_num] == 0
            score_text = hidden_states[text_pos_shifted.reshape([-1])]
        max_seq_len, max_seq_len_index = paddle.topk(
            forward_meta.seq_lens_this_time.squeeze(-1), k=1)
        hidden_states = extract_text_token_output(
            max_seq_len,
            max_seq_len_index.cast("int32"),
            image_token_num,
            forward_meta.seq_lens_this_time,
            forward_meta.cu_seqlens_q,
            score_text,
        )[0].cast(self._dtype)
        # -----------------------

        out = self.norm(hidden_states)

        return out


class ErnieMoEVLForCausalLM(ModelForCasualLM):
    """
    ErnieMoEVLForCausalLM
    """

    def __init__(self, llm_config: LLMConfig):
        """
        Args:
            llm_config (LLMConfig): Configurations for the LLM model.
        """
        super(ErnieMoEVLForCausalLM, self).__init__(llm_config)

        self.model = Ernie45TVLModel(llm_config=llm_config)

        self.ori_vocab_size = llm_config.model_config.ori_vocab_size

        self.lm_head = ParallelLMHead(
            llm_config=llm_config,
            embedding_dim=llm_config.model_config.hidden_size,
            num_embeddings=llm_config.model_config.vocab_size,
            prefix="lm_head",
        )

    @classmethod
    def name(self):
        return "ErnieMoEVLForCausalLM"

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

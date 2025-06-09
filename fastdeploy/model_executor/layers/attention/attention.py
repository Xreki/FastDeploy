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

from typing import Optional

import paddle
from paddle import nn
from paddlenlp.utils.log import logger

from fastdeploy.model_executor.layers.utils import get_tensor
from fastdeploy.worker.model_runner import ForwardMeta


class Attention(nn.Layer):
    """
    The AttentionLayer.
    """

    def __init__(
        self,
        llm_config,
        layer_id: int,
        logit_cap: float = 0.0,
        v_head_dim: int = -1,
        rope_type: str = "",
        qkv_bias: Optional[paddle.Tensor] = None,
        qkv_scale: Optional[paddle.Tensor] = None,
        prefix: str = "",
        out_scale: float = -1.,
        linear_shift=None,
        linear_smooth=None,
        use_neox_rotary_style=False,
        cache_k_scale_key=None,
        cache_v_scale_key=None,
    ) -> None:
        """
        Initializes `LMLayer` with the given parameters.

        Args:
            llm_config (dict): The config of LM model.
            layer_id (int): The id of current layer.
            logit_cap (float, optional): The cap for logits. Defaults to 0.0.
            v_head_dim (int, optional): The head dim of value. Defaults to -1.
            rope_type (str, optional): The type of RoPE. Defaults to "".
            qkv_bias (Optional[paddle.Tensor], optional): The bias of QKV. Defaults to None.
            qkv_scale (Optional[paddle.Tensor], optional): The scale of QKV. Defaults to None.
            prefix (str, optional): The name of current layer. Defaults to "".
            linear_shift (Optional[paddle.Tensor], optional): The shift of linear. Defaults to None.
            linear_smooth (Optional[paddle.Tensor], optional): The smooth of linear. Defaults to None.

        Raises:
            ValueError: If the `v_head_dim` is less than 0.
        """
        super().__init__()
        self.num_heads = llm_config.model_config.num_attention_heads // llm_config.parallel_config.mp_size
        self.head_dim = llm_config.model_config.hidden_size // llm_config.model_config.num_attention_heads
        self.kv_num_heads = llm_config.model_config.num_key_value_heads // llm_config.parallel_config.mp_size
        self.layer_id = layer_id
        self.logit_cap = logit_cap
        self.v_head_dim = v_head_dim if v_head_dim > 0 else self.head_dim
        self.rope_type = rope_type
        self.qk_head_dim = self.head_dim
        # not use
        self.tp_q_head_num = self.num_heads
        self.tp_k_head_num = self.num_heads
        self.tp_v_head_num = self.num_heads
        # not use
        self.scaling = 1.0 / (self.head_dim**0.5)
        self.linear_shift = linear_shift
        self.linear_smooth = linear_smooth
        self.qkv_bias = qkv_bias
        self.qkv_scale = qkv_scale
        self._dtype = self._helper.get_default_dtype()
        self.out_scale = out_scale
        self.use_neox_rotary_style = use_neox_rotary_style
        if llm_config.kv_cache_config.kvcache_quant_config is not None:
            self.kvcache_quant_method = llm_config.kv_cache_config.kvcache_quant_config.get_quant_method(
                self)
            self.kvcache_quant_method.create_weights(self)
        if llm_config.quant_config is not None:
            self.quant_max_bound = llm_config.quant_config.quant_max_bound
            self.quant_min_bound = llm_config.quant_config.quant_min_bound
        self.cache_k_scale_key = cache_k_scale_key
        self.cache_v_scale_key = cache_v_scale_key

    def load_state_dict(self, state_dict):
        """
        Attention only have quant related scales not other parameters.
        """
        if self.cache_k_scale_key is not None:
            self.cache_quant_type_str = "cache_int8"
            logger.info(
                f"Attention is running in {self.cache_quant_type_str} mode")
            max_bound = 127
            add_scale_attrs = [
                "cache_k_scale", "cache_k_out_scale", "cache_v_scale",
                "cache_v_out_scale"
            ]
            weight_keys = [self.cache_k_scale_key, self.cache_v_scale_key]

            for i in range(2):
                max_value = get_tensor(state_dict.pop(weight_keys[i]))
                max_value = max_value.cast(paddle.get_default_dtype())
                max_value.reshape_([-1])
                quant_scale_tensor = max_bound / max_value
                dequant_scale_tensor = max_value / max_bound

                # quant_scale and dequant_scale
                for j, scale_tensor in enumerate(
                    [quant_scale_tensor, dequant_scale_tensor]):
                    tmp_name = add_scale_attrs[2 * i + j]
                    setattr(
                        self, tmp_name,
                        self.create_parameter(
                            shape=scale_tensor.shape,
                            dtype=scale_tensor.dtype,
                        ))
                    getattr(self, tmp_name).set_value(scale_tensor)

    def forward(
        self,
        q: paddle.Tensor = None,
        k: paddle.Tensor = None,
        v: paddle.Tensor = None,
        qkv: paddle.Tensor = None,
        forward_meta: ForwardMeta = None,
    ):
        """
        The forward function of attention layer.
        args:
            q: the query tensor
            k: the key tensor
            v: the value tensor
            forward_meta: the forward meta data
        """
        return forward_meta.attn_backend.forward(
            q,
            k,
            v,
            qkv,
            self,
            forward_meta,
        )

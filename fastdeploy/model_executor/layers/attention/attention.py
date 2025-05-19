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

# cipher_token=WjI1fQOvhN  # do not edit this line

import os

from fastdeploy.model_executor.model_runner import ForwardMeta

import paddle
from paddle import nn

class Attention(nn.Layer):
    """
    The AttentionLayer.
    """

    def __init__(self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        layer_id: int,
        logit_cap: float = 0.0,
        v_head_dim: int = -1,
        rope_type: str = "") -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.layer_id = layer_id
        self.logit_cap = logit_cap
        self.v_head_dim = v_head_dim if v_head_dim > 0 else head_dim
        self.rope_type = rope_type
        self.qk_head_dim = head_dim
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_heads
        self.tp_v_head_num = num_heads
        self.k_scale = 1.0
        self.v_scale = 1.0
        self.scaling = 1.0 / (head_dim ** 0.5)

    def forward(
        self,
        q,
        k,
        v,
        forward_batch: ForwardMeta,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        return forward_batch.attn_backend.forward(
            q,
            k,
            v,
            self,
            forward_batch,
            save_kv_cache,
            **kwargs,
        )
        
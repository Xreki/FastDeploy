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
from abc import abstractmethod
from typing import Optional

import paddle

import fastdeploy
import fastdeploy.model_executor.ops.gpu.deep_gemm as deep_gemm

from ..utils import per_block_cast_to_fp8
from .quant_base import QuantConfigBase, QuantMethodBase


class WFP8AFP8Config(QuantConfigBase):

    def __init__(self, weight_block_size: list = [-1, -1]) -> None:
        super().__init__()
        self.weight_block_size = weight_block_size

    def get_name(self) -> str:
        return "wfp8afp8"

    @classmethod
    def from_config(cls, config: dict) -> "WFP8AFP8Config":
        weight_block_size = config["weight_block_size"]
        return cls(weight_block_size)

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        return WFP8AFP8LinearMethod(self)


class WFP8AFP8LinearMethod(QuantMethodBase):

    def __init__(
        self,
        quant_config: WFP8AFP8Config,
    ) -> None:
        super().__init__()
        self.quant_config = quant_config

    @abstractmethod
    def create_weights(self, layer):
        layer.linear_weight_scale = self.create_parameter(
            shape=[
                (layer.embed_dim + 127) // 128,
                (layer.num_heads * layer.head_dim + 127) // 128,
            ],
            attr=paddle.ParamAttr(name=layer.layer_name +
                                  ".weight_block_scale"),
            dtype="float32",
            is_bias=False,
        )

    def process_weights_after_loading(self, layer, weights) -> None:
        weight_tensor = weights.transpose([1, 0])
        quanted_weight_tensor, weight_block_scale_tensor = (
            per_block_cast_to_fp8(weight_tensor))
        layer.linear_weight.copy_(quanted_weight_tensor, False)
        layer.linear_weight_scale.set_value(weight_block_scale_tensor)

    @abstractmethod
    def apply(self, layer, x):
        x, x_scale_tensor = fastdeploy.model_executor.ops.gpu.per_token_quant_padding(
            x, self.quant_config.weight_block_size[0])
        linear_out = paddle.empty(
            (x.shape[0], layer.llm_config.model_config.hidden_size),
            dtype=paddle.bfloat16)
        deep_gemm.gemm_fp8_fp8_bf16_nt(
            (x, x_scale_tensor),
            (layer.linear_weight, layer.linear_weight_scale),
            linear_out,
        )
        return linear_out

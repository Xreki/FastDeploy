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

from ..moe import FusedMoE
from . import get_quantization_config
from .quant_base import QuantConfigBase, QuantMethodBase


class MixQuantConfig(QuantConfigBase):
    """
    Quantization config for layers that has different quantization methods.
    """

    def __init__(
        self,
        dense_quant_type: str,
        moe_quant_type: str,
    ) -> None:
        super().__init__()
        self.dense_quant_type = dense_quant_type
        self.moe_quant_type = moe_quant_type
        self.quant_max_bound = 0
        self.quant_min_bound = 0
        self.quant_round_type = 0

    def name(self) -> str:
        return "mix_quant"

    @classmethod
    def from_config(cls, config: dict) -> "MixQuantConfig":
        return cls(config['dense_quant_type'], config['moe_quant_type'])

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        if isinstance(layer, FusedMoE):
            return get_quantization_config(self.moe_quant_type).from_config(
                {}).get_quant_method(layer)
        else:
            return get_quantization_config(self.dense_quant_type).from_config(
                {}).get_quant_method(layer)

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
quantization module
"""
from typing import Dict, List, Type

from fastdeploy.platforms import current_platform

from .quant_base import QuantConfigBase
from enum import Enum


class QuantzationMethods(str, Enum):
    """QuantzationMethods"""
    WINT2 = "wint2"
    WINT4 = "wint4"
    WINT8 = "wint8"
    WEIGHT_ONLY = "weight_only"
    BLOCK_WISE = "block_wise"
    W4AFP8 = "w4afp8"
    W8A8 = "w8a8"
    W4A8 = "w4a8"
    WFP8AFP8 = "wfp8afp8"
    KVCACHE = "kvcache"
    MIX_QUANT = "mix_quant"


def get_quantization_config(quantization: str) -> Type[QuantConfigBase]:
    """
    Get the quantization config class by the quantization name.
    """
    try:
        quantization_enum = QuantzationMethods(quantization)
    except ValueError:
        raise ValueError(f"Invalid quantization method: {quantization}")

    from .block_wise import BlockWiseConfig
    from .kv_cache import KvCacheQuantConfig
    from .mix_quant import MixQuantConfig
    from .w4a8 import W4A8Config
    from .w4afp8 import W4AFP8Config
    from .w8a8 import W8A8Config
    from .weight_only import WeightOnlyConfig, WINT4Config, WINT8Config
    from .wfp8afp8 import WFP8AFP8Config
    from .wint2 import WINT2Config

    QUANT = QuantzationMethods
    method_to_config: Dict[str, Type[QuantConfigBase]] = {
        QUANT.WINT2: WINT2Config,
        QUANT.WINT4: WINT4Config,
        QUANT.WINT8: WINT8Config,
        QUANT.WEIGHT_ONLY: WeightOnlyConfig,
        QUANT.BLOCK_WISE: BlockWiseConfig,
        QUANT.W4AFP8: W4AFP8Config,
        QUANT.W8A8: W8A8Config,
        QUANT.W4A8: W4A8Config,
        QUANT.WFP8AFP8: WFP8AFP8Config,
        QUANT.KVCACHE: KvCacheQuantConfig,
        QUANT.MIX_QUANT: MixQuantConfig,
    }

    if not current_platform.is_xpu():
        from .block_wise import BlockWiseConfig

        method_to_config[QUANT.BLOCK_WISE] = BlockWiseConfig

    return method_to_config[quantization_enum]

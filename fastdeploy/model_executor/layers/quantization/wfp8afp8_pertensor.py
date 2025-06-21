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

from ..utils import get_tensor
from .quant_base import QuantConfigBase, QuantMethodBase


class WFP8AFP8PerTensorConfig(QuantConfigBase):
    """
    Quantization config for weight and activation with FP8.
    """

    def __init__(self) -> None:
        """
        Nothing else to do!
        """
        super().__init__()

    def name(self) -> str:
        """
        Nothing else to do!
        """
        return "wfp8afp8_pertensor"

    @classmethod
    def from_config(cls, config: dict) -> "WFP8AFP8PerTensorConfig":
        """
        Nothing else to do!
        """
        return cls()

    def get_quant_method(self, layer) -> Optional[QuantMethodBase]:
        """
        return method according to this config!
        """
        return WFP8AFP8PerTensorLinearMethod(self)


class WFP8AFP8PerTensorLinearMethod(QuantMethodBase):
    """
    Weight and activation quantization method for linear layer with per tensor FP8
    """

    def __init__(
        self,
        quant_config: WFP8AFP8PerTensorConfig,
    ) -> None:
        """
        Nothing special to do!
        """
        super().__init__()
        self.quant_config = quant_config
        self.quant_max_bound = 448
        self.quant_min_bound = -448
        self.quant_round_type = 1
        self.weight_dtype = "float8_e4m3fn"

    def create_weights(self, layer):
        """
        Nothing to do!
        """
        pass

    def process_loaded_weights(self, layer, weights, state_dict) -> None:
        """
        Read fp8 weight, act scale, weight scale
        """
        weights = weights.transpose([1, 0]).contiguous()
        layer.linear_weight.copy_(weights.view("float8_e4m3fn"), False)

        tmp_name = f"{layer.prefix}.act_scale"
        act_scale = get_tensor(state_dict.pop(tmp_name)).cast("float32")
        tmp_name = f"{layer.prefix}.quant_scale"
        weight_scale = get_tensor(state_dict.pop(tmp_name)).cast("float32")

        self.act_scale = act_scale.item()
        self.total_scale = (act_scale * weight_scale).item()

    def apply(self, layer, x):
        """
        compute!
        """
        from fastdeploy.model_executor.ops.gpu import \
            cutlass_fp8_fp8_half_gemm_fused

        from ..utils import create_hadamard_matrix_map

        hadamard_matrix = create_hadamard_matrix_map[x.shape[-1]]
        new_x = paddle.matmul(x.cast("float32"), hadamard_matrix)
        fp8_x = new_x / self.act_scale
        fp8_x = fp8_x.astype("float8_e4m3fn")

        linear_out = cutlass_fp8_fp8_half_gemm_fused(
            fp8_x,
            layer.linear_weight,
            transpose_x=False,
            transpose_y=True,
            bias=None,
            scale=self.total_scale,
            output_dtype="bfloat16",
            activation_type="identity")
        return linear_out

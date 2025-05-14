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


class RTNLinearMethod(QuantMethodBase):
    def __init__(
        self,
        weight_bits: int,
        act_bits: int,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.act_bits = act_bits

    @abstractmethod
    def create_weights(self, layer, *weight_args, **extra_weight_attrs):
        weight_only_scale_name = layer.layer_name + ".weight_only_scale"
        layer.linear_weight_scale = layer.create_parameter(
            shape=[layer.embed_dim],
            attr=paddle.ParamAttr(name=weight_only_scale_name),
            dtype=layer._dtype,
            is_bias=False,
        )

    def process_weights_after_loading(self, layer, weights) -> None:
        return

    @abstractmethod
    def apply(self, layer, *args, **kwargs):
        if self.weight_bits == 8 and self.act_bits is None:
            linear_out = weight_only_linear(
                x,
                weight=layer.linear_weight,
                weight_scale=layer.linear_weight_scale,
                weight_dtype=layer.weight_dtype,
                arch=layer.inference_args.weight_only_linear_arch,
            )
        else:
            raise ValueError(
                f"Linear is not implemented for W[{self.weight_dtype}]A[{self.act_dtype}] yet."
            )
        return linear_out


class RTNGPULinearMethod(RTNLinearMethod):
    def __init__(
        self,
        weight_bits: int,
        act_bits: int,
    ) -> None:
        super().__init__(weight_bits, act_bits)

    def process_weights_after_loading(self, layer, weight) -> None:
        if self.weight_bits == 8 and self.act_dtype is None:  # WINT8
            quanted_weight_tensor, weight_scale_tensor = weight_quantize(
                weight,
                algo="weight_only_int8",
                arch=self.inference_args.weight_only_linear_arch,
            )

            self.linear_weight.set_value(quanted_weight_tensor)
            self.linear_weight_scale.set_value(
                weight_scale_tensor.astype(paddle.get_default_dtype())
            )
        else:
            raise ValueError(
                f"GPULinear is not implemented for W[{self.weight_bits}]A[{self.act_bits}] yet."
            )


class RTNXPULinearMethod(RTNLinearMethod):
    def __init__(
        self,
        weight_bits: int,
        act_bits: int,
    ) -> None:
        super().__init__(weight_bits, act_bits)

    def process_weights_after_loading(self, layer, weight) -> None:
        if self.weight_bits == 8 and self.act_dtype is None:  # WINT8
            quanted_weight_tensor, weight_scale_tensor = xpu_quant_weight(
                weight.cpu().numpy()
            )
            self.linear_weight.set_value(quanted_weight_tensor)
            self.linear_weight_scale.set_value(
                weight_scale_tensor.astype(paddle.get_default_dtype())
            )
        else:
            raise ValueError(
                f"XPULinear is not implemented for W[{self.weight_bits}]A[{self.act_bits}] yet."
            )


class RTNLinearConfig(QuantConfigBase):
    def __init__(
        self,
        weight_bits: int,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits

    @abstractmethod
    def get_name(self) -> str:
        return "RTN"

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "RTNConfigBase":
        return cls(config["weight_bits"])

    @abstractmethod
    def get_quant_method(self, layer) -> Optional[QuantizeMethodBase]:
        return RTNLinearMethod(self)

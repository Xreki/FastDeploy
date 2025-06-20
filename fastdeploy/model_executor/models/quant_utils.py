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

from fastdeploy.config import FDConfig
from typing import Optional
from paddle import nn
from paddleformers.utils.log import logger
from typing import Dict
import paddle
from enum import Enum
from fastdeploy.model_executor.layers.quantization import QuantzationMethods as QUANT


class PrePostQuantMethod(Enum):
    """PrePostQuantMethod"""

    QKV = "qkv"
    OUT_LINEAR = "out_linear"
    FFN1 = "ffn1"
    FFN2 = "ffn2"
    MOE_FFN1 = "moe_ffn1"
    MOE_FFN2 = "moe_ffn2"


quant_mapping = {}


def register_quant(method_key: PrePostQuantMethod):
    """register_quant"""

    def decorator(func):
        """decorator"""
        if method_key in quant_mapping:
            raise ValueError(f"Duplicate registration for {method_key}")
        quant_mapping[method_key] = func
        return func

    return decorator


def get_quant_func(pre_post_quant_key: str):
    """
    get_quant_func
    """
    if isinstance(pre_post_quant_key, str):
        try:
            pre_post_quant_key = PrePostQuantMethod(pre_post_quant_key)
        except ValueError:
            raise KeyError(
                f"Invalid quant pre_post_quant_key string: {pre_post_quant_key}"
            )
    elif not isinstance(pre_post_quant_key, PrePostQuantMethod):
        raise TypeError(
            f"Expected str or PrePostQuantMethod, got {type(pre_post_quant_key)}"
        )

    if pre_post_quant_key not in quant_mapping:
        raise KeyError(f"No quant function registered for: {pre_post_quant_key}")

    return quant_mapping[pre_post_quant_key]


def is_quant_type(
    fd_config: FDConfig, target: str, quant_type: str = "dense_quant_type"
) -> bool:
    """is_quant_type"""
    qname = fd_config.quant_config.name()
    quant_type = getattr(fd_config.quant_config, quant_type, None)
    return qname == target or (qname == QUANT.MIX_QUANT and quant_type == target)


@register_quant(PrePostQuantMethod.QKV)
def qkv_pre_post_quantization_func(
    before: bool = True,
    qkv: Optional[paddle.Tensor] = None,
    qkv_scale: Optional[paddle.Tensor] = None,
    fd_config: Optional[FDConfig] = None,
):
    """qkv_pre_post_quantization_func"""
    if before:
        if fd_config.model_config.num_key_value_heads <= 0:
            qkv = qkv.reshape(
                [
                    fd_config.model_config.hidden_size,
                    (
                        fd_config.model_config.num_attention_heads
                        // fd_config.parallel_config.tensor_parallel_degree
                    ),
                    3,
                    fd_config.model_config.head_dim,
                ]
            ).transpose([2, 1, 3, 0])
        else:
            qkv = (
                qkv.reshape(
                    [
                        fd_config.model_config.hidden_size,
                        (
                            fd_config.model_config.num_attention_heads
                            // fd_config.parallel_config.tensor_parallel_degree
                            + 2
                            * (
                                fd_config.model_config.num_key_value_heads
                                // fd_config.parallel_config.tensor_parallel_degree
                            )
                        ),
                        fd_config.model_config.head_dim,
                    ]
                )
                .transpose([1, 2, 0])
                .reshape([-1, fd_config.model_config.hidden_size])
            )
        if is_quant_type(fd_config, QUANT.WINT8) or is_quant_type(
            fd_config, QUANT.WINT4
        ):
            qkv = qkv.reshape([-1, fd_config.model_config.hidden_size]).transpose(
                [1, 0]
            )
        return qkv, None
    else:
        if is_quant_type(fd_config, QUANT.WINT8) or is_quant_type(
            fd_config, QUANT.WINT4
        ):
            qkv = qkv.reshape([-1, fd_config.model_config.hidden_size])
        elif qkv_scale is not None and is_quant_type(fd_config, QUANT.W4AFP8):
            qkv_scale = paddle.view(qkv_scale, paddle.get_default_dtype())
        return qkv, qkv_scale


def common_befor_func(
    weight: paddle.Tensor,
    fd_config: Optional[FDConfig] = None,
):
    """common_befor_func"""
    if (
        is_quant_type(fd_config, QUANT.W8A8)
        or is_quant_type(fd_config, QUANT.BLOCK_WISE)
        or is_quant_type(fd_config, QUANT.WFP8AFP8)
    ):
        weight = weight.transpose([1, 0])
    return weight


def command_after_func(
    weight: paddle.Tensor,
    weight_scale: Optional[paddle.Tensor] = None,
    fd_config: Optional[FDConfig] = None,
):
    """command_after_func"""
    if weight_scale is not None and is_quant_type(fd_config, QUANT.BLOCK_WISE):
        weight_scale = paddle.view(weight_scale, paddle.get_default_dtype())
    return weight, weight_scale


@register_quant(PrePostQuantMethod.OUT_LINEAR)
def outlinear_pre_post_quantization_func(
    before: bool = True,
    out_linear_weight: Optional[paddle.Tensor] = None,
    out_linear_scale: Optional[paddle.Tensor] = None,
    fd_config: Optional[FDConfig] = None,
):
    """outlinear_pre_post_quantization_func"""
    if before:
        out_linear_weight = common_befor_func(out_linear_weight, fd_config)
    else:
        out_linear_weight, out_linear_scale = command_after_func(
            out_linear_weight, out_linear_scale, fd_config
        )
    return out_linear_weight, out_linear_scale


@register_quant(PrePostQuantMethod.FFN1)
def ffn1_pre_post_quantization_func(
    before: bool = True,
    ffn1_weight: Optional[paddle.Tensor] = None,
    ffn1_weight_scale: Optional[paddle.Tensor] = None,
    fd_config: Optional[FDConfig] = None,
):
    """ffn1_pre_post_quantization_func"""
    if before:
        if (
            fd_config.moe_config.num_experts > 0
            and fd_config.moe_config.moe_use_ffn_shared_weight_and_bias
        ):
            # not fast ffn:
            ffn1_weight = paddle.concat(
                [ffn1_weight[:, ::2], ffn1_weight[:, 1::2]], axis=1
            )
        ffn1_weight = common_befor_func(ffn1_weight, fd_config)
    else:
        ffn1_weight, ffn1_weight_scale = command_after_func(
            ffn1_weight, ffn1_weight_scale, fd_config
        )
    return ffn1_weight, ffn1_weight_scale


@register_quant(PrePostQuantMethod.FFN2)
def ffn2_pre_post_quantization_func(
    before: bool = True,
    ffn2_weight: Optional[paddle.Tensor] = None,
    ffn2_weight_scale: Optional[paddle.Tensor] = None,
    fd_config: Optional[FDConfig] = None,
):
    """ffn2_pre_post_quantization_func"""
    if before:
        ffn2_weight = common_befor_func(ffn2_weight, fd_config)
    else:
        ffn2_weight, ffn2_weight_scale = command_after_func(
            ffn2_weight, ffn2_weight_scale, fd_config
        )
    return ffn2_weight, ffn2_weight_scale


@register_quant(PrePostQuantMethod.MOE_FFN1)
def moe_ffn1_pre_post_quantization_func(
    before: bool = True,
    moe_ffn1_weight: Optional[paddle.Tensor] = None,
    moe_ffn1_weight_scale: Optional[paddle.Tensor] = None,
    fd_config: Optional[FDConfig] = None,
):
    """moe_ffn1_pre_post_quantization_func"""
    if before:
        if is_quant_type(fd_config, QUANT.BLOCK_WISE, "moe_quant_type"):
            moe_ffn1_weight = moe_ffn1_weight.transpose([1, 0])
        elif is_quant_type(fd_config, QUANT.W4A8, "moe_quant_type"):
            moe_ffn1_weight = moe_ffn1_weight.cast("int8")
    else:
        if is_quant_type(fd_config, QUANT.W4A8, "moe_quant_type"):
            moe_ffn1_weight = moe_ffn1_weight.reshape(
                [-1, fd_config.model_config.hidden_size // 2]
            )
        elif is_quant_type(fd_config, QUANT.WINT4, "moe_quant_type"):
            moe_ffn1_weight = moe_ffn1_weight.reshape(
                [
                    fd_config.model_config.hidden_size,
                    (
                        fd_config.moe_config.moe_intermediate_size
                        // fd_config.parallel_config.tensor_parallel_degree
                    ),
                ]
            )
        elif is_quant_type(fd_config, QUANT.WINT8, "moe_quant_type"):
            moe_ffn1_weight = moe_ffn1_weight.reshape(
                [
                    fd_config.model_config.hidden_size,
                    (
                        fd_config.moe_config.moe_intermediate_size
                        // fd_config.parallel_config.tensor_parallel_degree
                    )
                    * 2,
                ]
            )
    return moe_ffn1_weight, moe_ffn1_weight_scale


@register_quant(PrePostQuantMethod.MOE_FFN2)
def moe_ffn2_pre_post_quantization_func(
    before: bool = True,
    moe_ffn2_weight: Optional[paddle.Tensor] = None,
    moe_ffn2_weight_scale: Optional[paddle.Tensor] = None,
    fd_config: Optional[FDConfig] = None,
):
    """moe_ffn2_pre_post_quantization_func"""
    if before:
        if is_quant_type(fd_config, QUANT.BLOCK_WISE, "moe_quant_type"):
            moe_ffn2_weight = moe_ffn2_weight.transpose([1, 0])
        elif is_quant_type(fd_config, QUANT.W4A8, "moe_quant_type"):
            moe_ffn2_weight = moe_ffn2_weight.cast("int8")
    else:
        if is_quant_type(fd_config, QUANT.W4A8, "moe_quant_type"):
            moe_ffn2_weight = moe_ffn2_weight.reshape(
                [fd_config.model_config.hidden_size, -1]
            )
        elif is_quant_type(fd_config, QUANT.WINT4, "moe_quant_type"):

            moe_ffn2_weight = moe_ffn2_weight.reshape(
                [
                    (
                        fd_config.moe_config.moe_intermediate_size
                        // fd_config.parallel_config.tensor_parallel_degree
                    ),
                    fd_config.model_config.hidden_size // 2,
                ]
            )
        elif is_quant_type(fd_config, QUANT.WINT8, "moe_quant_type"):
            moe_ffn2_weight = moe_ffn2_weight.reshape(
                [
                    (
                        fd_config.moe_config.moe_intermediate_size
                        // fd_config.parallel_config.tensor_parallel_degree
                    ),
                    fd_config.model_config.hidden_size,
                ]
            )

    return moe_ffn2_weight, moe_ffn2_weight_scale


def quantization_func(
    fd_config: FDConfig,
):
    """quantization_func"""

    def fn(
        key: str,
        tensor: paddle.Tensor,
        quant_layer_instance_map: Dict[str, nn.Layer],
        quant_fn_key: str = "",
        quant_layer_key: str = "",
    ):
        """fn"""
        PrePostQuantFn = get_quant_func(quant_fn_key)
        if PrePostQuantFn is not None:
            tensor, _ = PrePostQuantFn(True, tensor, None, fd_config)
        quant_layer = quant_layer_instance_map[quant_layer_key]
        quant_method = fd_config.quant_config.get_quant_method(quant_layer)
        if quant_method is None:
            raise ValueError(f"quant_method should not be None.")
        try:
            (quanted_weight_tensor, weight_scale_tensor) = (
                quant_method.apply_weight_quantization(tensor)
            )
        except Exception as e:
            raise ValueError(
                f"Expected apply_weight_quantization is missing from {quant_method}"
            )

        if PrePostQuantFn is not None:
            quanted_weight_tensor, weight_scale_tensor = PrePostQuantFn(
                False, quanted_weight_tensor, weight_scale_tensor, fd_config
            )
        return quanted_weight_tensor, weight_scale_tensor

    return fn

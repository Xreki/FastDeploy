"""
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import nn
from paddle.framework import LayerHelper, in_dynamic_or_pir_mode


def fused_act_bias_wrapper(
    x,
    bias=None,
    dequant_scales=None,
    shift=None,
    smooth=None,
    act_method="gelu",
    compute_dtype="default",
    quant_scale=-1,
    quant_round_type=0,
    quant_max_bound=0,
    quant_min_bound=0,
):
    """
    Fused activation with bias and optional dequantization.

    Args:
        x (Tensor): The input tensor.
        bias (Tensor, optional): The bias tensor. Default: None.
        dequant_scales (Tensor, optional): The dequantization scale tensor. Default: None.
        shift (Tensor, optional): The shift tensor. Default: None.
        smooth (Tensor, optional): The smooth tensor. Default: None.
        act_method (str, optional): The activation method. Default: "gelu".
        compute_dtype (str, optional): The data type for computation. Default: "default".
        quant_scale (float, optional): The quantization scale. Default: -1.
        quant_round_type (int, optional): The rounding type for quantization. Default: 0.
        quant_max_bound (float, optional): The maximum bound for quantization. Default: 0.
        quant_min_bound (float, optional): The minimum bound for quantization. Default: 0.

    Returns:
        Tensor: The output tensor after fused activation with bias and optional dequantization.

    """
    if in_dynamic_or_pir_mode():
        return paddle._C_ops.fused_bias_act(
            x,
            bias,
            dequant_scales,
            shift,
            smooth,
            act_method,
            compute_dtype,
            quant_scale,
            quant_round_type,
            quant_max_bound,
            quant_min_bound,
        )
    helper = LayerHelper("fused_bias_act")
    if x.dtype == "int32":
        if compute_dtype == "bf16":
            dtype = "uint16"
        elif compute_dtype == "fp16":
            dtype = "float16"
        elif compute_dtype == "fp32":
            dtype = "float32"
        out = helper.create_variable_for_type_inference(dtype=dtype)
    else:
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

    inputs = {}
    inputs["x"] = x
    if bias is not None:
        inputs["bias"] = bias
    if dequant_scales is not None:
        inputs["dequant_scales"] = dequant_scales

    if shift is not None:
        inputs["shift"] = shift

    if smooth is not None:
        inputs["smooth"] = smooth

    attrs = {
        "act_method": act_method,
        "compute_dtype": compute_dtype,
        "quant_scale": quant_scale,
        "quant_round_type": quant_round_type,
        "quant_max_bound": quant_max_bound,
        "quant_min_bound": quant_min_bound,
    }

    helper.append_op(
        type="fused_bias_act",
        inputs=inputs,
        outputs={"out": out},
        attrs=attrs,
    )
    return out


class Activation(nn.Layer):
    """
    Activation Layer
    """

    def __init__(
        self,
        inference_args,
        bias=None,
        act_method="gelu",
        dequant_scales=None,
        shift=None,
        smooth=None,
        quant_scale=-1,
    ):
        """
        Initialize the activation layer with optional parameters for quantization, bias,
        activation method, and more.

        Args:
            inference_args (Any): Arguments related to inference, including quantization
                settings.
            bias (Optional[Tensor]): Optional bias term to be added to the output.
            act_method (str, optional): Activation method to be applied.
                Defaults to "gelu".
            dequant_scales (Optional[List[float]]): Dequantization scales, used in
                quantization scenarios.
            shift (Optional[float]): Shift factor, used in quantization scenarios.
            smooth (Optional[float]): Smoothing factor, used for specific activation
                functions.
            quant_scale (float, optional): Quantization scale, used in quantization
                scenarios. Defaults to -1, indicating no quantization.

        Raises:
            ValueError: If the default data type is not supported (only float32, float16,
                and bfloat16 are supported).
        """
        super().__init__()

        self.bias = bias
        self.act_method = act_method
        self.dequant_scales = dequant_scales
        self.shift = shift
        self.smooth = smooth
        self.quant_scale = quant_scale
        self.quant_round_type = inference_args.quant_round_type
        self.quant_max_bound = inference_args.quant_max_bound
        self.quant_min_bound = inference_args.quant_min_bound

        self._dtype = self._helper.get_default_dtype()
        if self._dtype == "bfloat16":
            self._fuse_kernel_compute_dtype = "bf16"
        elif self._dtype == "float16":
            self._fuse_kernel_compute_dtype = "fp16"
        elif self._dtype == "float32":
            self._fuse_kernel_compute_dtype = "fp32"
        else:
            raise ValueError(f"Just support float32, float16 and \
                    bfloat16 as default dtype, but received {self._dtype}")

        # fp8 is not support smooth quantization
        if "float8" in inference_args.act_dtype:
            self.dequant_scales = None
            self.shift = None
            self.smooth = None

    def forward(self, x):
        """
        Forward propagation of the custom activation layer.

        Args:
            x (Tensor): Input tensor to the activation layer.

        Returns:
            Tensor: Output tensor.
        """
        return fused_act_bias_wrapper(
            x,
            bias=self.bias,
            act_method=self.act_method,
            compute_dtype=self._fuse_kernel_compute_dtype,
            dequant_scales=self.dequant_scales,
            shift=self.shift,
            smooth=self.smooth,
            quant_scale=self.quant_scale,
            quant_round_type=self.quant_round_type,
            quant_max_bound=self.quant_max_bound,
            quant_min_bound=self.quant_min_bound,
        )

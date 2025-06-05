"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# cipher_token=WjI1fQOvhN  # do not edit this line
import os

import fastdeploy
from paddlenlp.utils.log import logger

import paddle
from paddle import nn
from paddle.nn.quant import weight_only_linear, weight_quantize

from fastdeploy.platforms.utils import (
    convert_to_npu_dequant_scale,
    xpu_quant_weight,
)

import fastdeploy.model_executor.ops.gpu.deep_gemm as deep_gemm
from .utils import per_block_cast_to_fp8, _set_var_distributed, get_tensor
from fastdeploy.platforms import current_platform


class Linear(nn.Layer):
    """
    Linear Layer
    """

    def __init__(
        self,
        inference_args,
        layer_name,
        weight_key,
        bias_key=None,
        dim_feedforward=None,
        skip_quant=False,
        use_smooth_quant=True,
        shift_key=None,
        smooth_key=None,
    ):
        """
        Initialize a linear layer with additional parameters for inference and quantization.

        Args:
            inference_args (dict or object): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
            layer_name (str): Unique name of the layer, used for naming internal attributes,
                you can give it any name you like.
            weight_key (str): Key name of weight in the pdparams state dict.
            bias_key (str): Key name of bias in the pdparams state dict. Defaults to None, means no bias.
            dim_feedforward (int, optional): Size of intermediate layer. Defaults to None.
            skip_quant (bool, optional): Whether to skip quantization for this layer.
                Defaults to False.
            use_smooth_quant (bool, optional): Whether to use smooth quantization for this
                layer. Smooth quantization introduces additional parameters to improve
                quantization accuracy. Defaults to True.
            shift_key (str): Key name of linear_shift in the pdparams state dict.
            smooth_key (str): Key name of smooth_weight in the pdparams state dict.

        """
        super().__init__()
        self.inference_args = inference_args
        self.with_bias = bias_key is not None
        self.skip_quant = skip_quant
        self.use_smooth_quant = use_smooth_quant
        self.weight_dtype = inference_args.weight_dtype
        self.act_dtype = inference_args.act_dtype
        self.nranks = inference_args.mp_size
        self.embed_dim = inference_args.hidden_size
        self.head_dim = inference_args.head_dim
        self.num_heads = inference_args.num_attention_heads // self.nranks
        self.dim_feedforward = (
            inference_args.dim_feedforward
            if dim_feedforward is None
            else dim_feedforward
        ) // self.nranks

        self.weight_key = weight_key
        self.bias_key = bias_key
        self.shift_key = shift_key
        self.smooth_key = smooth_key

        self.layer_name = layer_name
        self.weight_name = self.layer_name + ".weight"
        self.bias_name = self.layer_name + ".bias"
        self.weight_only_scale_name = self.layer_name + ".weight_only_scale"
        self.out_scale_name = self.layer_name + ".out_scale"
        if self.use_smooth_quant:
            self.shift_name = self.layer_name + ".shift_bias"
            self.smooth_name = self.layer_name + ".smooth_weight"
        self._dtype = self._helper.get_default_dtype()

        self.use_gemm_dequant = os.getenv("FLAGS_use_gemm_dequant")
        if self.use_gemm_dequant is not None:
            self.use_gemm_dequant = int(self.use_gemm_dequant) == 1
        else:
            self.use_gemm_dequant = False
        self.use_offline_quant = inference_args.use_offline_quant
        if inference_args.use_weight_only:
            self.init_weight_only_scale()
        if self.inference_args.weight_block_size[0] != -1:
            logger.debug("linear use_fp8_blockwise")
            self.init_weight_block_scale()
        if (
            inference_args.weight_dtype == "int8" and inference_args.act_dtype == "int8"
        ) or (
            "float8" in inference_args.weight_dtype
            and "float8" in inference_args.act_dtype
        ):
            self.set_ptq_scale()  # init and load scale
        self.init_weight()

    def init_weight_block_scale(self):
        """init_weight_block_scale for fp8"""
        self.linear_weight_scale = self.create_parameter(
            shape=[
                (self.embed_dim + 127) // 128,
                (self.num_heads * self.head_dim + 127) // 128,
            ],
            attr=paddle.ParamAttr(name=self.layer_name + ".weight_block_scale"),
            dtype="float32",
            is_bias=False,
        )

    def is_y_transposed(self):
        """
        Returns whether the y tensor should be transposed for inference.
        Args:
            None.

        Returns:
            bool, whether the y tensor should be transposed for inference.
        """
        if current_platform.is_dcu():
            return False
        elif current_platform.is_npu():
            return True
        else:  # GPU
            if self.weight_dtype == "int4":
                return True
            if self.weight_dtype == "int8":
                return True
            if "float8" in self.weight_dtype:
                return True
            # bf16/fp16/fp32 y is not transposed
            return False

    def init_weight_shape(self, trans=False):
        """
        Initialize the weight shape for the first feedforward network layer.

        Args:
            trans (bool, optional): Whether to transpose the weight shape.
                Defaults to False. If True, the shape will be reversed.

        Returns:
            None.
        """
        self.linear_weight_shape = [
            self.num_heads * self.head_dim,
            self.embed_dim,
        ]
        if trans:
            self.linear_weight_shape.reverse()
        if self.use_smooth_quant:
            self.linear_shift_shape = [self.num_heads * self.head_dim]
            self.linear_smooth_shape = [self.num_heads * self.head_dim]
        if self.weight_dtype == "int4":
            self.linear_weight_shape[0] //= 2

    def init_weight(self):
        """
        Initialize the weights and biases.
        """
        self.init_weight_shape(self.is_y_transposed())

        self.linear_weight = self.create_parameter(
            shape=self.linear_weight_shape,
            attr=paddle.ParamAttr(name=self.weight_name),
            dtype=self.get_weight_create_dtype(),
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        self.linear_bias = None
        if self.with_bias:
            self.linear_bias = self.create_parameter(
                shape=[self.embed_dim],
                attr=paddle.ParamAttr(name=self.bias_name),
                dtype=self._dtype,
                is_bias=True,
            )

        if self.nranks > 0:
            # row parallel
            _set_var_distributed(self.linear_weight, split_axis=0)

        # smooth quant
        self.linear_shift = None
        self.linear_smooth = None
        if self.use_smooth_quant:
            self.linear_shift = self.create_parameter(
                shape=self.linear_shift_shape,
                attr=paddle.ParamAttr(name=self.shift_name),
                dtype=self._dtype,
                is_bias=False,
            )
            self.linear_smooth = self.create_parameter(
                shape=self.linear_smooth_shape,
                attr=paddle.ParamAttr(name=self.smooth_name),
                dtype=self._dtype,
                is_bias=False,
            )

    def get_weight_create_dtype(self):
        """
        Get the data type for creating weights based on quantization settings.

        Args:
            self (object): The instance of the class where this method is defined.

        Returns:
            str: The data type for creating weights. It depends on the quantization settings:
                - If `self.skip_quant` is True, returns the original data type `self._dtype`.
                - If `self.weight_dtype` is "int4", returns "int8" to ensure compatibility or optimization.
                - Otherwise, returns the specified weight data type `self.weight_dtype`.
        """
        if self.skip_quant:
            return self._dtype
        if self.weight_dtype == "int4":
            return "int8"
        # TODO(wangzhe24) create_parameter not support FP8
        if "float8" in self.weight_dtype:
            return self._dtype
        return self.weight_dtype

    def init_weight_only_scale(self):
        """
        Initialize the weight scale.
        """
        self.linear_weight_scale = self.create_parameter(
            shape=[self.embed_dim],
            attr=paddle.ParamAttr(name=self.weight_only_scale_name),
            dtype=self._dtype,
            is_bias=False,
        )

    def set_ptq_scale(self):
        """
        Set the post-training quantization (PTQ) scale for the layer.

        This method fetches weight and input activation scales from the inference arguments,
        and computes the output scale for the layer.
        It also handles skipping quantization for missing scales.

        Args:
            None (Method operates on the instance's attributes and arguments.)

        Returns:
            None (Modifies the instance's attributes.)

        Raises:
            None
        """
        if self.inference_args.weight_block_size[0] != -1:
            return

        weight_scale = self.inference_args.weight_scale_dict.get(
            self.layer_name + ".weight_quanter"
        )
        in_scale = self.inference_args.act_scale_dict.get(
            self.layer_name + ".activation_quanter"
        )

        if weight_scale is None or in_scale is None:
            logger.debug(f"{self.layer_name} skip quant")
            self.skip_quant = True
            return

        if "float8" in self.weight_dtype:
            max_range = 448.0
            self.scalar_scale_name = self.layer_name + ".scalar_weight_quanter"
            self.scalar_scale = self.create_parameter(
                shape=([1]),
                attr=paddle.ParamAttr(name=self.scalar_scale_name),
                dtype="float32",
            )
            self.scalar_scale.set_value(
                paddle.to_tensor([1.0 / (max_range * in_scale)], dtype="float32")
            )
            linear_out_scale = paddle.to_tensor(weight_scale / max_range).astype(
                "float32"
            )
        else:
            max_range = 127.0
            linear_out_scale = paddle.to_tensor(
                weight_scale / (max_range * max_range * in_scale)
            ).astype("float32")
        self.linear_out_scale = self.create_parameter(
            shape=[self.embed_dim],
            attr=paddle.ParamAttr(name=self.out_scale_name),
            dtype="float32",
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        self.linear_out_scale.set_value(convert_to_npu_dequant_scale(linear_out_scale))

    def load_offline_quant_state_dict(self, quant_weight, quant_scale=None):
        """
        Load offline the checkpoint state dictionary into the layer.
        """
        if quant_scale is None:
            if "float8" in self.weight_dtype:
                self.linear_weight.copy_(quant_weight, False)
            else:
                self.linear_weight.set_value(quant_weight)
        else:
            if self.inference_args.weight_block_size[0] != -1:
                self.linear_weight.copy_(quant_weight.view(paddle.float8_e4m3fn), False)
            else:
                self.linear_weight.set_value(quant_weight)
            self.linear_weight_scale.set_value(quant_scale)

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        # weight
        if self.use_offline_quant:
            self.load_offline_quant_state_dict(
                quant_weight=get_tensor(
                    state_dict.pop(self.weight_key + ".quant_weight")
                ),
                quant_scale=get_tensor(
                    state_dict.pop(self.weight_key + ".quant_scale")
                ),
            )
        else:
            weight_tensor = get_tensor(state_dict.pop(self.weight_key))
            if self.skip_quant:
                weight_tensor = weight_tensor.cast(self._dtype)
            else:
                if self.inference_args.weight_block_size[0] != -1:
                    weight_tensor = weight_tensor.transpose([1, 0])
                    quanted_weight_tensor, weight_block_scale_tensor = (
                        per_block_cast_to_fp8(weight_tensor)
                    )
                    self.linear_weight.copy_(quanted_weight_tensor, False)
                    self.linear_weight_scale.set_value(weight_block_scale_tensor)
                elif self.weight_dtype == "int8" and self.act_dtype in [
                    "bfloat16",
                    "float16",
                    "float32",
                ]:  # WINT8
                    if paddle.is_compiled_with_cuda():
                        quanted_weight_tensor, weight_scale_tensor = weight_quantize(
                            weight_tensor,
                            algo="weight_only_int8",
                            arch=self.inference_args.weight_only_linear_arch,
                        )
                    elif paddle.is_compiled_with_xpu():
                        quanted_weight_tensor, weight_scale_tensor = xpu_quant_weight(
                            weight_tensor.cpu().numpy()
                        )
                    else:
                        raise ValueError("Not supported platform.")
                    self.linear_weight.set_value(quanted_weight_tensor)
                    self.linear_weight_scale.set_value(
                        weight_scale_tensor.astype(paddle.get_default_dtype())
                    )
                elif self.weight_dtype == "int4" and self.act_dtype in [
                    "bfloat16",
                    "float16",
                    "float32",
                ]:  # WINT4
                    quanted_weight_tensor, weight_scale_tensor = weight_quantize(
                        weight_tensor.cpu(),
                        algo="weight_only_int4",
                        arch=self.inference_args.weight_only_linear_arch,
                    )
                    self.linear_weight.set_value(quanted_weight_tensor)
                    self.linear_weight_scale.set_value(weight_scale_tensor)
                elif (
                    self.weight_dtype == "int4" and self.act_dtype == "float8_e4m3fn"
                ):  # W4Afp8
                    quanted_weight_tensor, weight_scale_tensor = (
                        fastdeploy.model_executor.ops.gpu.scaled_gemm_f8_i4_f16_weight_quantize(
                            paddle.cast(weight_tensor, "float32").cpu(),
                            groupsize=-1,
                            scale_dtype="float16",
                        )
                    )
                    weight_scale_tensor = paddle.view(weight_scale_tensor, self._dtype)
                    self.linear_weight.set_value(quanted_weight_tensor)
                    self.linear_weight_scale.set_value(weight_scale_tensor)
                else:  # bf16/fp16/fp32, A8W8, FP8
                    if self.is_y_transposed():
                        weight_tensor = weight_tensor.transpose([1, 0])
                    weight_tensor = paddle.cast(weight_tensor, self.weight_dtype)
                    if (
                        "float8" in self.weight_dtype
                    ):  # TODO(wangzhe24) FP8 cannot use set_value now
                        self.linear_weight.copy_(weight_tensor, False)
                    else:
                        self.linear_weight.set_value(weight_tensor)

        # bias
        if self.with_bias:
            bias_tensor = paddle.to_tensor(get_tensor(state_dict.pop(self.bias_key)))
            self.linear_bias.set_value(bias_tensor)

        # smooth quant
        if self.use_smooth_quant:
            if self.shift_key in state_dict:
                shift_tensor = get_tensor(state_dict.pop(self.shift_key)).astype(
                    paddle.get_default_dtype()
                )
            else:
                shift_tensor = paddle.zeros(
                    shape=[
                        (
                            self.inference_args.num_attention_heads
                            // self.inference_args.mp_size
                        )
                        * (
                            self.inference_args.hidden_size
                            // self.inference_args.num_attention_heads
                        )
                    ],
                    dtype=paddle.get_default_dtype(),
                )
            self.linear_shift.set_value(shift_tensor)
            if self.smooth_key in state_dict:
                smooth_tensor = get_tensor(state_dict.pop(self.smooth_key)).astype(
                    paddle.get_default_dtype()
                )
            else:
                smooth_tensor = paddle.ones(
                    shape=[
                        (
                            self.inference_args.num_attention_heads
                            // self.inference_args.mp_size
                        )
                        * (
                            self.inference_args.hidden_size
                            // self.inference_args.num_attention_heads
                        )
                    ],
                    dtype=paddle.get_default_dtype(),
                )
            self.linear_smooth.set_value(smooth_tensor)

    def forward(self, x):
        """
        Forward function for FFN1Split.

        Args:
            x (Tensor): Input tensor to the FFN1Split layer.

        Returns:
            Tensor: Output tensor.

        Raises:
            NotImplementedError: If the weight dtype is not float8 or act dtype is not equal to weight dtype.
        """
        if self.skip_quant:
            linear_out = paddle.matmul(x, self.linear_weight, False, True)
            return linear_out
        if self.inference_args.weight_block_size[0] != -1:
            x, x_scale_tensor = fastdeploy.model_executor.ops.gpu.per_token_quant_padding(
                x, self.inference_args.weight_block_size[0]
            )
            linear_out = paddle.empty(
                (x.shape[0], self.inference_args.hidden_size), dtype=paddle.bfloat16
            )
            deep_gemm.gemm_fp8_fp8_bf16_nt(
                (x, x_scale_tensor),
                (self.linear_weight, self.linear_weight_scale),
                linear_out,
            )
        elif self.inference_args.use_weight_only and self.act_dtype in [
            "bfloat16",
            "float16",
            "float32",
        ]:
            linear_out = weight_only_linear(
                x,
                weight=self.linear_weight,
                weight_scale=self.linear_weight_scale,
                weight_dtype=self.weight_dtype,
                arch=self.inference_args.weight_only_linear_arch,
            )
        elif self.weight_dtype == "int8" and self.act_dtype == self.weight_dtype:
            if self.use_gemm_dequant:
                linear_out = fastdeploy.model_executor.ops.gpu.gemm_dequant(
                    x, self.linear_weight, self.linear_out_scale, self._dtype
                )
            else:
                linear_out = paddle.matmul(x, self.linear_weight, False, True)
                linear_out = fastdeploy.model_executor.ops.gpu.dequant_int8(
                    linear_out, self.linear_out_scale, self._dtype
                )
        elif self.weight_dtype == "int4" and self.act_dtype == "float8_e4m3fn":
            linear_out = fastdeploy.model_executor.ops.gpu.scaled_gemm_f8_i4_f16(
                x,
                self.linear_weight,
                self.linear_weight_scale,
                zero_points=None,
                bias=None,
                out_scale=self.inference_args.weight_scale_dict.get(
                    self.layer_name + ".weight_quanter"
                )
                / (
                    self.inference_args.act_scale_dict.get(
                        self.layer_name + ".activation_quanter"
                    )
                    * 448
                    * 448
                ),
                groupsize=0,
                out_dtype=self._dtype,
            )
        elif "float8" in self.weight_dtype and self.act_dtype == self.weight_dtype:
            linear_out = fastdeploy.model_executor.ops.gpu.per_channel_fp8_fp8_half_gemm_fused(
                x,
                self.linear_weight,
                bias=None,
                scalar_scale=self.scalar_scale,
                channel_scale=self.linear_out_scale,
                transpose_x=False,
                transpose_y=True,
                output_dtype=self._dtype,
            )
        elif (
            self.weight_dtype in ["bfloat16", "float16", "float32"]
            and self.act_dtype == self.weight_dtype
        ):
            linear_out = paddle.matmul(x, self.linear_weight)
        else:
            raise ValueError(
                f"Linear is not implemented for W[{self.weight_dtype}]A[{self.act_dtype}] yet."
            )
        return linear_out


class FFN2(Linear):
    """
    FFN2 is a Linear layer with different weight dimensions.
    """

    def __init__(
        self,
        inference_args,
        layer_name,
        weight_key,
        bias_key=None,
        dim_feedforward=None,
        skip_quant=False,
        use_smooth_quant=True,
        shift_key=None,
        smooth_key=None,
    ):
        """
        Initialize a linear layer with additional parameters for inference and quantization.

        Args:
            inference_args (dict or object): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
            layer_name (str): Name of the layer, used for naming internal attributes.
            with_bias (bool, optional): Whether to include a bias term in the layer.
                Defaults to True.
            dim_feedforward (int, optional): Size of intermediate layer. Defaults to None.
            skip_quant (bool, optional): Whether to skip quantization for this layer.
                Defaults to False.
            use_smooth_quant (bool, optional): Whether to use smooth quantization for this
                layer. Smooth quantization introduces additional parameters to improve
                quantization accuracy. Defaults to True.
        """
        super(FFN2, self).__init__(
            inference_args,
            layer_name,
            weight_key,
            bias_key,
            dim_feedforward,
            skip_quant,
            use_smooth_quant,
            shift_key,
            smooth_key,
        )

    def init_weight_block_scale(self):
        """init_weight_block_scale for fp8"""
        self.linear_weight_scale = self.create_parameter(
            shape=[(self.embed_dim + 127) // 128, (self.dim_feedforward + 127) // 128],
            attr=paddle.ParamAttr(name=self.layer_name + ".weight_block_scale"),
            dtype="float32",
            is_bias=False,
        )

    def init_weight_shape(self, trans=False):
        """
        Initialize the weight shape for the first feedforward network layer.

        Args:
            trans (bool, optional): Whether to transpose the weight shape.
                Defaults to False. If True, the shape will be reversed.

        Returns:
            None.
        """
        self.linear_weight_shape = [self.dim_feedforward, self.embed_dim]
        if trans:
            self.linear_weight_shape.reverse()
        if self.use_smooth_quant:
            self.linear_shift_shape = [self.dim_feedforward]
            self.linear_smooth_shape = [self.dim_feedforward]
        if self.weight_dtype == "int4":
            self.linear_weight_shape[0] //= 2

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        # weight
        weight_tensor = get_tensor(state_dict.pop(self.weight_key))
        if self.skip_quant:
            if self.is_y_transposed():
                weight_tensor = weight_tensor.transpose([1, 0])
            weight_tensor = weight_tensor.cast(self._dtype)
            self.linear_weight.set_value(weight_tensor)
        else:
            if self.inference_args.weight_block_size[0] != -1:
                weight_tensor = weight_tensor.transpose([1, 0])
                quanted_weight_tensor, weight_block_scale_tensor = (
                    per_block_cast_to_fp8(weight_tensor)
                )
                self.linear_weight.copy_(quanted_weight_tensor, False)
                self.linear_weight_scale.set_value(weight_block_scale_tensor)
            elif self.weight_dtype == "int8" and self.act_dtype in [
                "bfloat16",
                "float16",
                "float32",
            ]:  # WINT8
                if paddle.is_compiled_with_cuda():
                    quanted_weight_tensor, weight_scale_tensor = weight_quantize(
                        weight_tensor,
                        algo="weight_only_int8",
                        arch=self.inference_args.weight_only_linear_arch,
                    )
                elif paddle.is_compiled_with_xpu():
                    quanted_weight_tensor, weight_scale_tensor = xpu_quant_weight(
                        weight_tensor.cpu().numpy()
                    )
                else:
                    raise ValueError("Not supported platform.")
                self.linear_weight.set_value(quanted_weight_tensor)
                self.linear_weight_scale.set_value(
                    weight_scale_tensor.astype(paddle.get_default_dtype())
                )
            elif self.weight_dtype == "int4" and self.act_dtype in [
                "bfloat16",
                "float16",
                "float32",
            ]:  # WINT4
                quanted_weight_tensor, weight_scale_tensor = weight_quantize(
                    weight_tensor.cpu(),
                    algo="weight_only_int4",
                    arch=self.inference_args.weight_only_linear_arch,
                )
                self.linear_weight.set_value(quanted_weight_tensor)
                self.linear_weight_scale.set_value(weight_scale_tensor)
            elif (
                self.weight_dtype == "int4" and self.act_dtype == "float8_e4m3fn"
            ):  # W4Afp8
                quanted_weight_tensor, weight_scale_tensor = (
                    fastdeploy.model_executor.ops.gpu.scaled_gemm_f8_i4_f16_weight_quantize(
                        paddle.cast(weight_tensor, "float32").cpu(),
                        groupsize=-1,
                        scale_dtype="float16",
                    )
                )
                weight_scale_tensor = paddle.view(weight_scale_tensor, self._dtype)
                self.linear_weight.set_value(quanted_weight_tensor)
                self.linear_weight_scale.set_value(weight_scale_tensor)
            else:  # bf16/fp16/fp32, A8W8, FP8
                if self.is_y_transposed():
                    weight_tensor = weight_tensor.transpose([1, 0])
                weight_tensor = paddle.cast(weight_tensor, self.weight_dtype)
                if (
                    "float8" in self.weight_dtype
                ):  # TODO(wangzhe24) FP8 cannot use set_value now
                    self.linear_weight.copy_(weight_tensor, False)
                else:
                    self.linear_weight.set_value(weight_tensor)

        # bias
        if self.with_bias:
            bias_tensor = get_tensor(state_dict.pop(self.bias_key))
            self.linear_bias.set_value(bias_tensor)

        # smooth quant
        if self.use_smooth_quant:
            if self.shift_key in state_dict:
                shift_tensor = get_tensor(state_dict.pop(self.shift_key)).astype(
                    paddle.get_default_dtype()
                )
            else:
                shift_tensor = paddle.zeros(
                    shape=self.linear_shift_shape,
                    dtype=paddle.get_default_dtype(),
                )
            self.linear_shift.set_value(shift_tensor)
            if self.smooth_key in state_dict:
                smooth_tensor = get_tensor(state_dict.pop(self.smooth_key)).astype(
                    paddle.get_default_dtype()
                )
            else:
                smooth_tensor = paddle.ones(
                    shape=self.linear_smooth_shape,
                    dtype=paddle.get_default_dtype(),
                )
            self.linear_smooth.set_value(smooth_tensor)

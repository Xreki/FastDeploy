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
import fastdeploy
import numpy as np
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


class FFN1(nn.Layer):
    """
    FFN1 Layer
    """

    def __init__(
        self,
        inference_args,
        layer_name,
        weight_key,
        bias_key=None,
        dim_feedforward=None,
        skip_quant=False,
        activation="gelu",
        use_fast_ffn=False,
    ):
        """
        Initializes internal parameters based on the provided `inference_args` and
        layer configuration, including weight and bias names, quantization settings,
        and the type of activation function.

        Args:
            inference_args (InferenceArgs): Configuration object containing parameters
                for inference, such as data types and layer sizes.
            layer_name (str): Unique name of the layer, you can give it any name you like.
            weight_key (str): Key name of weight in the pdparams state dict.
            bias_key (str): Key name of bias in the pdparams state dict. Defaults to None, means no bias.
            dim_feedforward (int, optional): Size of intermediate layer. Defaults to None.
            skip_quant (bool, optional): Whether to skip quantization for this layer.
                Defaults to False.
            activation (str, optional): Activation function to use. Defaults to "gelu".
            use_fast_ffn (bool, optional): Whether to use a faster FFN implementation.
                Defaults to False.
        """
        super().__init__()
        self.inference_args = inference_args
        self.with_bias = bias_key is not None
        self.skip_quant = skip_quant
        self.activation = activation
        self.use_fast_ffn = use_fast_ffn
        self.weight_dtype = inference_args.weight_dtype
        self.act_dtype = inference_args.act_dtype
        self.nranks = inference_args.mp_size
        self.embed_dim = inference_args.hidden_size
        self.dim_feedforward = (
            inference_args.dim_feedforward
            if dim_feedforward is None
            else dim_feedforward
        ) // self.nranks

        self.weight_key = weight_key
        self.bias_key = bias_key

        self.layer_name = layer_name
        self.weight_name = self.layer_name + ".weight"
        self.bias_name = self.layer_name + ".bias"
        self.weight_only_scale_name = self.layer_name + ".weight_only_scale"
        self.out_scale_name = self.layer_name + ".out_scale"
        self.use_offline_quant = inference_args.use_offline_quant

        self._dtype = self._helper.get_default_dtype()

        if inference_args.use_weight_only:
            self.init_weight_only_scale()
        if self.inference_args.weight_block_size[0] != -1:
            self.init_weight_block_scale()
        if inference_args.weight_dtype == "int8" and inference_args.act_dtype == "int8":
            self.set_ptq_scale()  # init and load scale
        self.init_weight()

    def init_weight_block_scale(self):
        """
        Initialize the weight scale shape for fp8 gemm.
        """
        if self.activation.endswith("glu"):
            n = self.dim_feedforward * 2
        else:
            n = self.dim_feedforward
        k = self.embed_dim
        self.ffn1_weight_scale = self.create_parameter(
            shape=[(n + 127) // 128, (k + 127) // 128],
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
        self.ffn1_weight_shape = (
            [self.embed_dim, self.dim_feedforward * 2]
            if self.activation.endswith("glu")
            else [self.embed_dim, self.dim_feedforward]
        )
        if trans:
            self.ffn1_weight_shape.reverse()
        if self.weight_dtype == "int4":
            self.ffn1_weight_shape[0] //= 2

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

    def init_weight(self):
        """
        Initialize the weights and biases.
        """
        self.init_weight_shape(self.is_y_transposed())

        self.ffn1_weight = self.create_parameter(
            shape=self.ffn1_weight_shape,
            attr=paddle.ParamAttr(name=self.weight_name),
            dtype=self.get_weight_create_dtype(),
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        self.ffn1_bias = None
        if self.with_bias:
            self.ffn1_bias = self.create_parameter(
                shape=(
                    [self.dim_feedforward * 2]
                    if self.activation.endswith("glu")
                    else [self.dim_feedforward]
                ),
                attr=paddle.ParamAttr(name=self.bias_name),
                dtype=self._dtype,
                is_bias=True,
            )
        if self.nranks > 0:
            # column parallel
            _set_var_distributed(self.ffn1_weight, split_axis=1)
            _set_var_distributed(self.ffn1_bias, split_axis=0)

    def init_weight_only_scale(self):
        """
        Initialize the weight scale.
        """
        self.ffn1_weight_scale = self.create_parameter(
            shape=(
                [self.dim_feedforward * 2]
                if self.activation.endswith("glu")
                else [self.dim_feedforward]
            ),
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

        self.ffn1_out_scale = self.create_parameter(
            shape=(
                [self.dim_feedforward * 2]
                if self.activation.endswith("glu")
                else [self.dim_feedforward]
            ),
            attr=paddle.ParamAttr(name=self.out_scale_name),
            dtype="float32",
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        weight_scale_numpy = weight_scale / (127.0 * 127.0 * in_scale)
        converted_weight_scale = np.zeros(
            list(weight_scale_numpy.shape), dtype=weight_scale_numpy.dtype
        )
        out_dim = converted_weight_scale.shape[-1]
        if not self.use_fast_ffn:
            converted_weight_scale[: out_dim // 2] = weight_scale_numpy[::2]
            converted_weight_scale[out_dim // 2 :] = weight_scale_numpy[1::2]
        else:
            converted_weight_scale[:] = weight_scale_numpy[:]
        self.ffn1_out_scale.set_value(
            convert_to_npu_dequant_scale(converted_weight_scale).astype("float32")
        )

    def load_offline_quant_state_dict(self, quant_weight, quant_scale=None):
        """
        Load offline the checkpoint state dictionary into the layer.
        """
        if quant_scale is None:
            if "float8" in self.weight_dtype:
                self.ffn1_weight.copy_(quant_weight, False)
            else:
                self.ffn1_weight.set_value(quant_weight)
        else:
            if self.inference_args.weight_block_size[0] != -1:
                self.ffn1_weight.copy_(quant_weight.view(paddle.float8_e4m3fn), False)
            else:
                self.ffn1_weight.set_value(quant_weight)
            self.ffn1_weight_scale.set_value(quant_scale)

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
            converted_weight_tensor = paddle.zeros(
                shape=list(weight_tensor.shape), dtype=weight_tensor.dtype
            )
            if not self.use_fast_ffn:
                converted_weight_tensor = paddle.concat(
                    [weight_tensor[:, ::2], weight_tensor[:, 1::2]], axis=1
                )
            else:
                converted_weight_tensor = weight_tensor

            # set weight
            if self.skip_quant:
                self.ffn1_weight.set_value(converted_weight_tensor)
            else:
                if self.inference_args.weight_block_size[0] != -1:
                    converted_weight_tensor = converted_weight_tensor.transpose([1, 0])
                    quanted_weight_tensor, weight_block_scale_tensor = (
                        per_block_cast_to_fp8(converted_weight_tensor)
                    )
                    self.ffn1_weight.copy_(quanted_weight_tensor, False)
                    self.ffn1_weight_scale.set_value(weight_block_scale_tensor)
                elif self.weight_dtype == "int8" and self.act_dtype in [
                    "bfloat16",
                    "float16",
                    "float32",
                ]:  # WINT8
                    if paddle.is_compiled_with_cuda():
                        quanted_weight_tensor, weight_scale_tensor = weight_quantize(
                            converted_weight_tensor,
                            algo="weight_only_int8",
                            arch=self.inference_args.weight_only_linear_arch,
                        )
                    elif paddle.is_compiled_with_xpu():
                        quanted_weight_tensor, weight_scale_tensor = xpu_quant_weight(
                            converted_weight_tensor.cpu().numpy()
                        )
                    self.ffn1_weight.set_value(quanted_weight_tensor)
                    self.ffn1_weight_scale.set_value(
                        weight_scale_tensor.astype(paddle.get_default_dtype())
                    )
                elif self.weight_dtype == "int4" and self.act_dtype in [
                    "bfloat16",
                    "float16",
                    "float32",
                ]:  # WINT4
                    quanted_weight_tensor, weight_scale_tensor = weight_quantize(
                        converted_weight_tensor.cpu(),
                        algo="weight_only_int4",
                        arch=self.inference_args.weight_only_linear_arch,
                    )
                    self.ffn1_weight.set_value(quanted_weight_tensor)
                    self.ffn1_weight_scale.set_value(weight_scale_tensor)
                elif self.weight_dtype == "int4" and self.act_dtype == "float8_e4m3fn":
                    quanted_weight_tensor, weight_scale_tensor = (
                        fastdeploy.model_executor.ops.gpu.scaled_gemm_f8_i4_f16_weight_quantize(
                            paddle.cast(converted_weight_tensor, "float32").cpu(),
                            groupsize=-1,
                            scale_dtype="float16",
                        )
                    )
                    weight_scale_tensor = paddle.view(weight_scale_tensor, self._dtype)
                    self.ffn1_weight.set_value(quanted_weight_tensor)
                    self.ffn1_weight_scale.set_value(weight_scale_tensor)
                else:  # bf16/fp16/fp32, A8W8, FP8
                    if self.is_y_transposed():
                        converted_weight_tensor = converted_weight_tensor.transpose(
                            [1, 0]
                        )
                    converted_weight_tensor = paddle.cast(
                        converted_weight_tensor, self.weight_dtype
                    )
                    if (
                        "float8" in self.weight_dtype
                    ):  # TODO(wangzhe24) FP8 cannot use set_value now
                        self.ffn1_weight.copy_(converted_weight_tensor, False)
                    else:
                        self.ffn1_weight.set_value(converted_weight_tensor)
        # bias
        if self.with_bias:
            bias_tensor = get_tensor(state_dict.pop(self.bias_key)).astype(
                paddle.get_default_dtype()
            )
            converted_bias_tensor = paddle.zeros(
                shape=list(bias_tensor.shape), dtype=bias_tensor.dtype
            )
            if not self.use_fast_ffn:
                converted_bias_tensor = paddle.concat(
                    [bias_tensor[::2], bias_tensor[1::2]], axis=0
                )
            else:
                converted_bias_tensor = bias_tensor
            self.ffn1_bias.set_value(converted_bias_tensor)

    def forward(self, x):
        """
         The forward pass computes the output of the FFN1 layer based on the input tensor `x`.

        Args:
            x (Tensor): Input tensor to the FFN1 layer.

        Returns:
            Tensor: Output tensor of the FFN1 layer after processing.

        Raises:
            ValueError: If the combination of weight dtype and activation dtype is not supported.
        """
        if self.skip_quant:
            ffn1_out = paddle.matmul(x, self.ffn1_weight)
            return ffn1_out
        if self.inference_args.weight_block_size[0] != -1:
            x, x_scale_tensor = fastdeploy.model_executor.ops.gpu.per_token_quant_padding(
                x, self.inference_args.weight_block_size[0]
            )
            ffn1_out = paddle.empty(
                (x.shape[0], self.ffn1_weight_shape[0]), dtype=paddle.bfloat16
            )
            deep_gemm.gemm_fp8_fp8_bf16_nt(
                (x, x_scale_tensor),
                (self.ffn1_weight, self.ffn1_weight_scale),
                ffn1_out,
            )
        elif self.inference_args.use_weight_only and self.act_dtype in [
            "bfloat16",
            "float16",
            "float32",
        ]:
            ffn1_out = weight_only_linear(
                x,
                weight=self.ffn1_weight,
                weight_scale=self.ffn1_weight_scale,
                weight_dtype=self.weight_dtype,
                arch=self.inference_args.weight_only_linear_arch,
            )
        elif self.weight_dtype == "int8" and self.act_dtype == self.weight_dtype:
            ffn1_out = paddle.matmul(x, self.ffn1_weight, False, True)
        elif self.weight_dtype == "int4" and self.act_dtype == "float8_e4m3fn":
            ffn1_out = fastdeploy.model_executor.ops.gpu.scaled_gemm_f8_i4_f16(
                x,
                self.ffn1_weight,
                self.ffn1_weight_scale,
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
                groupsize=-1,
                out_dtype=self._dtype,
            )
        elif "float8" in self.weight_dtype and self.act_dtype == self.weight_dtype:
            ffn1_out = fastdeploy.model_executor.ops.gpu.cutlass_fp8_fp8_half_gemm_fused(
                x,
                self.ffn1_weight,
                bias=None,
                transpose_x=False,
                transpose_y=True,
                scale=self.inference_args.weight_scale_dict.get(
                    self.layer_name + ".weight_quanter"
                )
                / (
                    self.inference_args.act_scale_dict.get(
                        self.layer_name + ".activation_quanter"
                    )
                    * 448
                    * 448
                ),
                output_dtype=self._dtype,
                activation_type="identity",
            )
        elif (
            self.weight_dtype in ["bfloat16", "float16", "float32"]
            and self.act_dtype == self.weight_dtype
        ):
            ffn1_out = paddle.matmul(x, self.ffn1_weight)
        else:
            raise ValueError(
                f"FFN1 is not implemented for W[{self.weight_dtype}]A[{self.act_dtype}] yet."
            )
        return ffn1_out


class FFN1Split(nn.Layer):
    """
    FFN1Split is a layer that performs the first linear of the input tensor using two matrices: gate and up.
    """

    def __init__(
        self,
        inference_args,
        gate_layer_name,
        up_layer_name,
        with_bias=True,
        skip_quant=False,
        activation="swiglu",
    ):
        """
        Initialize the FFN1 layer with configuration arguments and naming conventions.

        Args:
            inference_args (InferenceArgs): Configuration arguments for inference, including data types and sizes.
            gate_layer_name (str): Name of the gate layer, used for naming conventions and accessing weights.
            up_layer_name (str): Name of the up layer, also used for naming conventions and accessing weights.
            with_bias (bool, optional): Whether to include bias terms in the layer weights. Defaults to True.
            skip_quant (bool, optional): Whether to skip quantization for this layer. Defaults to False.
            activation (str, optional): Activation function to use. Defaults to "swiglu".
        """
        super().__init__()
        self.inference_args = inference_args
        self.with_bias = with_bias
        self.skip_quant = skip_quant
        self.activation = activation
        self.weight_dtype = inference_args.weight_dtype
        self.act_dtype = inference_args.act_dtype
        self.nranks = inference_args.mp_size
        self.embed_dim = inference_args.hidden_size
        self.dim_feedforward = inference_args.ffn_hidden_size

        self.gate_layer_name = gate_layer_name
        self.gate_weight_layer_name = self.gate_layer_name + ".weight"
        self.gate_bias_layer_name = self.gate_layer_name + ".bias"

        self.up_layer_name = up_layer_name
        self.up_weight_layer_name = self.up_layer_name + ".weight"
        self.up_bias_layer_name = self.up_layer_name + ".bias"

        self._dtype = self._helper.get_default_dtype()

        # trick method to get in scale
        self.ffn1_layer_name = self.gate_layer_name.replace("gate", "linear1")
        # trick method to get out scale
        self.ffn2_layer_name = self.gate_layer_name.replace("gate", "linear2")

        if inference_args.use_weight_only:
            self.init_weight_only_scale()
        if self.weight_dtype == "int8" and self.act_dtype == "int8":
            self.set_ptq_scale()
        self.init_weight()

    def init_weight_shape(self, trans=False):
        """
        Initialize the weight shape for the first feedforward network layer.

        Args:
            trans (bool, optional): Whether to transpose the weight shape.
                Defaults to False. If True, the shape will be reversed.

        Returns:
            None.
        """
        self.gate_weight_shape = [self.embed_dim, self.dim_feedforward]
        self.up_weight_shape = [self.embed_dim, self.dim_feedforward]
        if self.inference_args.use_weight_only and self.weight_dtype == "int4":
            self.gate_weight_shape[0] //= 2
            self.up_weight_shape[0] //= 2
        if trans:
            self.gate_weight_shape.reverse()
            self.up_weight_shape.reverse()

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
        else:
            if self.weight_dtype == "int4":
                return True
            if self.weight_dtype == "int8":
                return True
            if "float8" in self.weight_dtype:
                return True
            # bf16/fp16/fp32 y is not transposed
            return False

    def init_weight(self):
        """
        Initialize the weights and biases.
        """
        self.init_weight_shape(self.is_y_transposed())

        self.gate_weight = self.create_parameter(
            shape=self.gate_weight_shape,
            attr=paddle.ParamAttr(name=self.gate_weight_layer_name),
            dtype=self.get_weight_create_dtype(),
            is_bias=False,
        )
        self.up_weight = self.create_parameter(
            shape=self.up_weight_shape,
            attr=paddle.ParamAttr(name=self.up_weight_layer_name),
            dtype=self.get_weight_create_dtype(),
            is_bias=False,
        )

        self.gate_bias = None
        self.up_bias = None
        if self.with_bias:
            self.gate_bias = self.create_parameter(
                shape=[self.dim_feedforward],
                attr=paddle.ParamAttr(name=self.gate_bias_layer_name),
                dtype=self._dtype,
                is_bias=True,
            )
            self.up_bias = self.create_parameter(
                shape=[self.dim_feedforward],
                attr=paddle.ParamAttr(name=self.up_bias_layer_name),
                dtype=self._dtype,
                is_bias=True,
            )
        if self.nranks > 0:
            # column parallel
            _set_var_distributed(self.gate_weight, split_axis=1)
            _set_var_distributed(self.up_weight, split_axis=1)
            _set_var_distributed(self.gate_bias, split_axis=0)
            _set_var_distributed(self.up_bias, split_axis=0)

    def init_weight_only_scale(self):
        """
        Initialize the weight scale.
        """
        raise NotImplementedError("FFN1Split only support fp8 now")

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
        raise NotImplementedError("FFN1Split only support fp8 now")

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        gate_weight_tensor = get_tensor(
            state_dict.pop(self.gate_weight_layer_name)
        ).cast(self.weight_dtype)
        up_weight_tensor = get_tensor(state_dict.pop(self.up_weight_layer_name)).cast(
            self.weight_dtype
        )
        if self.is_y_transposed():
            gate_weight_tensor = gate_weight_tensor.transpose([1, 0])
            up_weight_tensor = up_weight_tensor.transpose([1, 0])
        # TODO(wangzhe24) set_value not support FP8
        self.gate_weight.copy_(gate_weight_tensor, False)
        self.up_weight.copy_(up_weight_tensor, False)
        if self.with_bias:
            self.gate_bias.set_value(
                get_tensor(state_dict.pop(self.gate_bias_layer_name))
            )
            self.up_bias.set_value(get_tensor(state_dict.pop(self.up_bias_layer_name)))

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
        if "float8" in self.weight_dtype and self.act_dtype == self.weight_dtype:
            ffn1_out = fastdeploy.model_executor.ops.gpu.cutlass_fp8_fp8_fp8_dual_gemm_fused(
                x,
                self.gate_weight,
                self.up_weight,
                transpose_x=False,
                transpose_y=True,
                bias0=self.gate_bias,
                bias1=self.up_bias,
                scale0=self.inference_args.weight_scale_dict.get(
                    self.gate_layer_name + ".weight_quanter"
                )
                / (
                    self.inference_args.act_scale_dict.get(
                        self.ffn1_layer_name + ".activation_quanter"
                    )
                    * 448
                    * 448
                ),
                scale1=self.inference_args.weight_scale_dict.get(
                    self.up_layer_name + ".weight_quanter"
                )
                / (
                    self.inference_args.act_scale_dict.get(
                        self.ffn1_layer_name + ".activation_quanter"
                    )
                    * 448
                    * 448
                ),
                scale_out=self.inference_args.act_scale_dict.get(
                    self.ffn2_layer_name + ".activation_quanter"
                )
                * 448,
                activation_type=self.activation,
            )
        else:
            raise NotImplementedError("FFN1Split only support fp8 now")
        return ffn1_out

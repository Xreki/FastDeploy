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

import paddle
from paddle import nn

# cipher_token=WjI1fQOvhN  # do not edit this line
import fastdeploy
from fastdeploy.platforms import current_platform

from .utils import _set_var_distributed, get_tensor


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
            state_dict.pop(self.gate_weight_layer_name)).cast(
                self.weight_dtype)
        up_weight_tensor = get_tensor(state_dict.pop(
            self.up_weight_layer_name)).cast(self.weight_dtype)
        if self.is_y_transposed():
            gate_weight_tensor = gate_weight_tensor.transpose([1, 0])
            up_weight_tensor = up_weight_tensor.transpose([1, 0])
        # TODO(wangzhe24) set_value not support FP8
        self.gate_weight.copy_(gate_weight_tensor, False)
        self.up_weight.copy_(up_weight_tensor, False)
        if self.with_bias:
            self.gate_bias.set_value(
                get_tensor(state_dict.pop(self.gate_bias_layer_name)))
            self.up_bias.set_value(
                get_tensor(state_dict.pop(self.up_bias_layer_name)))

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
                    self.gate_layer_name + ".weight_quanter") /
                (self.inference_args.act_scale_dict.get(
                    self.ffn1_layer_name + ".activation_quanter") * 448 * 448),
                scale1=self.inference_args.weight_scale_dict.get(
                    self.up_layer_name + ".weight_quanter") /
                (self.inference_args.act_scale_dict.get(
                    self.ffn1_layer_name + ".activation_quanter") * 448 * 448),
                scale_out=self.inference_args.act_scale_dict.get(
                    self.ffn2_layer_name + ".activation_quanter") * 448,
                activation_type=self.activation,
            )
        else:
            raise NotImplementedError("FFN1Split only support fp8 now")
        return ffn1_out

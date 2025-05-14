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
import paddle
from paddle import nn
from paddle.incubate.nn.functional import fused_layer_norm, fused_rms_norm
from .utils import get_tensor


class Normalization(nn.Layer):
    """
    Normalization layer.
    """

    def __init__(
        self,
        inference_args,
        layer_name,
        weight_key=None,
        bias_key=None,
        epsilon=1e-5,
        norm_type="layernorm",
        linear_bias=None,
        quant_scale=None,
    ):
        """
        Initializes the normalization layer.

        Args:
            inference_args (object): Contains layer-specific configurations such as hidden size.
            layer_name (str): Unique name of the layer for identification.
            weight_key (str): Key name of weight in the pdparams state dict. Defaults to None, means no weight.
            bias_key (str): Key name of bias in the pdparams state dict. Defaults to None, means no bias.
            epsilon (float, optional): Small value added to the variance to avoid division by zero. Defaults to 1e-5.
            norm_type (str, optional): Type of normalization to use, supports 'layernorm' or 'rmsnorm'.
                Defaults to 'layernorm'.
            linear_bias (float, optional): Initial bias value for the linear layer (if used). Defaults to None.
            quant_scale (float, optional): Quantization scale factor. Defaults to None for no quantization.

        Raises:
            NotImplementedError: If the specified norm_type is not supported.
        """
        super().__init__()
        self.inference_args = inference_args
        self.layer_name = layer_name
        self.with_weight = weight_key is not None
        self.with_bias = bias_key is not None
        self.epsilon = epsilon
        self.norm_type = norm_type

        if self.norm_type == "layernorm":
            self.norm_func = fused_layer_norm
        elif self.norm_type == "rmsnorm":
            self.norm_func = fused_rms_norm
        else:
            raise NotImplementedError("Only support norm type of [layernorm, rmsnorm]")

        self.linear_bias = linear_bias
        self.quant_scale = quant_scale

        self.weight_key = weight_key
        self.bias_key = bias_key

        self.embed_dim = inference_args.hidden_size
        self.weight_name = self.layer_name + ".weight"
        self.bias_name = self.layer_name + ".bias"
        self._dtype = self._helper.get_default_dtype()

        self._norm_weight_dtype = (
            "float32" if self.norm_type == "layernorm" else self._dtype
        )
        self.init_weight()

    def init_weight(self):
        """
        Initialize the weights and biases.
        """

        self.ln_weight = None
        if self.with_weight:
            self.ln_weight = self.create_parameter(
                attr=paddle.ParamAttr(name=self.weight_name),
                shape=[self.embed_dim],
                default_initializer=nn.initializer.Constant(value=1.0),
                dtype=self._norm_weight_dtype,
            )
        self.ln_bias = None
        if self.with_bias:
            self.ln_bias = self.create_parameter(
                attr=paddle.ParamAttr(name=self.bias_name),
                shape=[self.embed_dim],
                is_bias=True,
                dtype=self._norm_weight_dtype,
            )

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """

        # weight
        weight_tensor = paddle.cast(
            get_tensor(state_dict.pop(self.weight_key)), self._norm_weight_dtype
        )
        self.ln_weight.set_value(weight_tensor)

        # bias
        if self.with_bias:
            bias_tensor = paddle.cast(
                get_tensor(state_dict.pop(self.bias_key)), self._norm_weight_dtype
            )
            self.ln_bias.set_value(bias_tensor)

    def forward(self, x, residual_input=None):
        """
        Defines the forward computation of the layer.

        Args:
            x (paddle.Tensor): Input tensor to be normalized.
            residual_input (paddle.Tensor, optional): Residual input tensor for residual connection.
                Defaults to None. If provided, the normalization layer will also return the residual
                output for further computation.

        Returns:
            paddle.Tensor or tuple of paddle.Tensor:
                - If `residual_input` is None, returns the normalized output tensor.
                - If `residual_input` is provided, returns a tuple of (normalized_output, residual_output).
                  The `residual_output` is the result of applying the normalization and possibly other
                  operations (like linear transformation) on the `residual_input`.
        """
        norm_out = self.norm_func(
            x,
            norm_weight=self.ln_weight,
            norm_bias=self.ln_bias,
            epsilon=self.epsilon,
            begin_norm_axis=1,
            bias=self.linear_bias,
            residual=residual_input,
            quant_scale=-1 if self.quant_scale is None else self.quant_scale,
            quant_round_type=self.inference_args.quant_round_type,
            quant_max_bound=self.inference_args.quant_max_bound,
            quant_min_bound=self.inference_args.quant_min_bound,
        )
        if residual_input is not None:
            return norm_out[0], norm_out[1]
        else:
            return norm_out[0]

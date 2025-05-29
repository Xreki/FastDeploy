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
from paddle.nn.quant import weight_quantize

import fastdeploy

from .utils import (_set_var_distributed, divide, get_tensor,
                    per_block_cast_to_fp8)


class LinearBase(nn.Layer):
    """
    LinearBase Layer
    """

    def __init__(
        self,
        llm_config,
        layer_name: str = "",
        input_size: int = None,
        output_size: int = None,
        weight_key=None,
        bias_key=None,
        skip_quant=False,
    ):
        """
        Initializes a linear layer and provides additional parameters required for inference and quantization.

        Args:
            llm_config (LLMConfig): Inference-related parameters containing attributes such as
                weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
            layer_name (str): Unique name of the layer, used to name internal attributes.
                Can be arbitrarily named.
            input_size (int, optional): Number of input features. Defaults to None.
            output_size (int, optional): Number of output features. Defaults to None.
            weight_key (Any, optional): Key for weights. Defaults to None.
            bias_key (Any, optional): Key for biases. Defaults to None.
            skip_quant (bool, optional): Whether to skip quantization. Defaults to False.

        Raises:
            NotImplementedError: Raised if the current platform is not a CUDA platform.
        """
        super().__init__()
        if current_platform.is_cuda():
            self.forward = self.forward_cuda
        else:
            raise NotImplementedError

        self.llm_config = llm_config
        self.skip_quant = skip_quant
        self.use_smooth_quant = llm_config.model_config.use_smooth_quant
        self.weight_dtype = llm_config.model_config.weight_dtype
        self.act_dtype = llm_config.model_config.act_dtype
        self.input_size = input_size
        self.output_size = output_size
        self.weight_key = weight_key
        self.bias_key = bias_key
        self.with_bias = True if self.bias_key is not None else False

        self.shift_key = f"{layer_name}.shift_bias"
        self.smooth_key = f"{layer_name}.smooth_weight"

        self.layer_name = layer_name
        self.weight_name = self.layer_name + ".weight"
        self.bias_name = self.layer_name + ".bias"
        self.out_scale_name = self.layer_name + ".out_scale"
        if self.use_smooth_quant:
            self.shift_name = self.layer_name + ".shift_bias"
            self.smooth_name = self.layer_name + ".smooth_weight"
        self._dtype = self._helper.get_default_dtype()

        if llm_config.quant_config:
            self.quant_method = llm_config.quant_config.get_quant_method(self)

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
            self.input_size,
            self.output_size,
        ]
        if trans:
            self.linear_weight_shape.reverse()
        if self.use_smooth_quant:
            self.linear_shift_shape = [self.output_size]
            self.linear_smooth_shape = [self.output_size]
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
                shape=[self.output_size],
                attr=paddle.ParamAttr(name=self.bias_name),
                dtype=self._dtype,
                is_bias=True,
            )

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

        if "float8" in self.weight_dtype:
            return "float8_e4m3fn"
        return self.weight_dtype

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        # weight
        assert self.weight_key is not None, 'weight_key should not be None.'
        weight_tensor = get_tensor(state_dict.pop(self.weight_key))

        if self.llm_config.quant_config:
            self.quant_method.process_loaded_weights(self, weight_tensor)
        else:
            self.linear_weight.set_value(weight_tensor)

        # bias
        if self.with_bias:
            bias_tensor = paddle.to_tensor(
                get_tensor(state_dict.pop(self.bias_key)))
            self.linear_bias.set_value(bias_tensor)

        # smooth quant
        if self.use_smooth_quant:
            if self.shift_key in state_dict:
                shift_tensor = get_tensor(state_dict.pop(
                    self.shift_key)).astype(paddle.get_default_dtype())
            else:
                shift_tensor = paddle.zeros(
                    shape=self.linear_shift_shape,
                    dtype=paddle.get_default_dtype(),
                )
            self.linear_shift.set_value(shift_tensor)
            if self.smooth_key in state_dict:
                smooth_tensor = get_tensor(state_dict.pop(
                    self.smooth_key)).astype(paddle.get_default_dtype())
            else:
                smooth_tensor = paddle.ones(
                    shape=[self.linear_smooth_shape],
                    dtype=paddle.get_default_dtype(),
                )
            self.linear_smooth.set_value(smooth_tensor)

    def forward_cuda(self, x):
        """
        Forward function for ColumnParallelLinear.

        Args:
            x (Tensor): Input tensor to the ColumnParallelLinear layer.

        Returns:
            Tensor: Output tensor.

        Raises:
            NotImplementedError: If the weight dtype is not float8 or act dtype is not equal to weight dtype.
        """
        if self.llm_config.quant_config:
            linear_out = self.quant_method.apply(self, x)
        else:
            linear_out = paddle.matmul(x, self.linear_weight)

        return linear_out

    def forward(self, x):
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """
    ReplicatedLinear Layer
    """

    def __init__(
        self,
        llm_config,
        layer_name: str = "",
        input_size: int = None,
        output_size: int = None,
        weight_key=None,
        bias_key=None,
        skip_quant=False,
    ):
        """
        Initialize a linear layer with additional parameters for inference and quantization.

        Args:
            llm_config (LLMConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
            layer_name (str): Unique name of the layer, used for naming internal attributes,
                you can give it any name you like.
            layer_index (int): The index of the linear layer in the model

        """
        super().__init__(llm_config=llm_config,
                         layer_name=layer_name,
                         input_size=input_size,
                         output_size=output_size,
                         weight_key=weight_key,
                         bias_key=bias_key,
                         skip_quant=skip_quant)
        self.nranks = llm_config.parallel_config.mp_size
        self.input_size = input_size
        self.init_weight()
        self.quant_method.create_weights(self)

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
                shape=[self.output_size],
                attr=paddle.ParamAttr(name=self.bias_name),
                dtype=self._dtype,
                is_bias=True,
            )

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


class ColumnParallelLinear(LinearBase):
    """
    ColumnParallelLinear Layer
    """

    def __init__(
        self,
        llm_config,
        layer_name: str = "",
        input_size: int = None,
        output_size: int = None,
        weight_key=None,
        bias_key=None,
        skip_quant=False,
    ):
        """
        Initialize a linear layer with additional parameters for inference and quantization.

        Args:
            llm_config (LLMConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
            layer_name (str): Unique name of the layer, used for naming internal attributes,
                you can give it any name you like.
            layer_index (int): The index of the linear layer in the model

        """
        super().__init__(llm_config=llm_config,
                         layer_name=layer_name,
                         input_size=input_size,
                         output_size=output_size,
                         weight_key=weight_key,
                         bias_key=bias_key,
                         skip_quant=skip_quant)
        self.nranks = llm_config.parallel_config.mp_size
        self.input_size = input_size
        self.output_size = divide(output_size, self.nranks)
        self.init_weight()

        self.quant_method.create_weights(self)

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
        if self.nranks > 0:
            # col parallel
            _set_var_distributed(self.linear_weight, split_axis=-1)

        self.linear_bias = None
        if self.with_bias:
            self.linear_bias = self.create_parameter(
                shape=[self.output_size],
                attr=paddle.ParamAttr(name=self.bias_name),
                dtype=self._dtype,
                is_bias=True,
            )
            if self.nranks > 0:
                # col parallel
                _set_var_distributed(self.linear_bias, split_axis=-1)

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


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    MergedColumnParallelLinear Layer.
    """

    def __init__(
        self,
        llm_config,
        layer_name,
        weight_key,
        bias_key=None,
        activation="gelu",
        use_fast_ffn=False,
        skip_quant=False,
    ):
        """Packed linear layers with column parallelism.

        Initialize the fused ffn1 Linear layer with given parameters.

        Args:
            llm_config (LLMConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.

            layer_name (str): Unique name of the layer, used for naming weights and biases.
            weight_key (str): Key name of weight in the pdparams state dict.
            bias_key (str): Key name of bias in the pdparams state dict. Defaults to None, means no bias.
            with_bias (bool, optional): Whether to include bias term. Defaults to True.
            activation (str, optional): Activation function to use. Defaults to "gelu".
            use_fast_ffn (bool, optional): Whether to use a faster FFN implementation.
                Defaults to False.
            skip_quant (bool, optional): Whether to skip quantization steps. Defaults to False.
        """
        self.use_fast_ffn = use_fast_ffn
        self.activation = activation
        self.embed_dim = llm_config.model_config.hidden_size
        self.dim_feedforward = llm_config.model_config.ffn_hidden_size
        self.nranks = llm_config.parallel_config.mp_size
        self.dim_feedforward_per_rank = divide(self.dim_feedforward,
                                               self.nranks)
        input_size = self.embed_dim
        output_size = self.dim_feedforward * 2 if self.activation.endswith(
            "glu") else self.dim_feedforward
        super().__init__(llm_config=llm_config,
                         layer_name=layer_name,
                         input_size=input_size,
                         output_size=output_size,
                         weight_key=weight_key,
                         bias_key=bias_key,
                         skip_quant=skip_quant)

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        # weight
        assert self.weight_key is not None, 'weight_key should not be None.'
        if self.weight_key in state_dict.keys():
            weight_tensor = get_tensor(state_dict.pop(self.weight_key))
        else:
            gate_weight_key = self.weight_key.replace("linear1", "gate_proj")
            up_weight_key = self.weight_key.replace("linear1", "up_proj")
            gate_tensor = get_tensor(state_dict.pop(gate_weight_key))
            up_tensor = get_tensor(state_dict.pop(up_weight_key))
            weight_tensor = paddle.concat([gate_tensor, up_tensor], axis=-1)

        if not self.use_fast_ffn:
            converted_weight_tensor = paddle.concat(
                [weight_tensor[:, ::2], weight_tensor[:, 1::2]], axis=1)
        else:
            converted_weight_tensor = weight_tensor

        state_dict[self.weight_key] = converted_weight_tensor

        super().load_state_dict(state_dict)


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKVParallelLinear Layer.
    """

    def __init__(self, llm_config, layer_name, weight_key, bias_key=None):
        """
        Initialize the QKV Linear layer with given parameters.

        Args:
            llm_config (LLMConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.

            layer_name (str): Unique name of the layer, used for naming weights and biases.
            weight_key (str): Key name of weight in the pdparams state dict.
            bias_key (str): Key name of bias in the pdparams state dict. Defaults to None, means no bias.
            with_bias (bool, optional): Whether to include bias term. Defaults to True.
            skip_quant (bool, optional): Whether to skip quantization steps. Defaults to False.
        """
        self.num_heads = llm_config.model_config.num_attention_heads
        self.kv_num_heads = llm_config.model_config.num_key_value_heads
        self.embed_dim = llm_config.model_config.hidden_size
        self.head_dim = llm_config.model_config.head_dim
        self.nranks = llm_config.parallel_config.mp_size
        self.num_heads_per_rank = divide(self.num_heads, self.nranks)
        self.kv_num_heads_per_rank = divide(self.kv_num_heads, self.nranks)
        input_size = self.embed_dim
        output_size = (self.num_heads + 2 * self.kv_num_heads) * self.head_dim
        super().__init__(llm_config=llm_config,
                         layer_name=layer_name,
                         input_size=input_size,
                         output_size=output_size,
                         weight_key=weight_key,
                         bias_key=bias_key)

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        # weight
        assert self.weight_key is not None, 'weight_key should not be None.'
        # qkv fused in disk
        if self.weight_key in state_dict.keys():
            weight_tensor = get_tensor(state_dict.pop(self.weight_key))
        else:
            q_weight_key = self.weight_key.replace("qkv_proj", "q_proj")
            k_weight_key = self.weight_key.replace("qkv_proj", "k_proj")
            v_weight_key = self.weight_key.replace("qkv_proj", "v_proj")
            q_tensor = get_tensor(state_dict.pop(q_weight_key))
            k_tensor = get_tensor(state_dict.pop(k_weight_key))
            v_tensor = get_tensor(state_dict.pop(v_weight_key))
            weight_tensor = paddle.concat([q_tensor, k_tensor, v_tensor],
                                          axis=-1)

        if self.llm_config.quant_config:
            self.quant_method.process_loaded_weights(self, weight_tensor)
        else:
            self.linear_weight.set_value(weight_tensor)

        # bias
        if self.with_bias:
            bias_tensor = paddle.to_tensor(
                get_tensor(state_dict.pop(self.bias_key)))
            self.linear_bias.set_value(bias_tensor)

        # smooth quant
        if self.use_smooth_quant:
            if self.shift_key in state_dict:
                shift_tensor = get_tensor(state_dict.pop(
                    self.shift_key)).astype(paddle.get_default_dtype())
            else:
                shift_tensor = paddle.zeros(
                    shape=self.linear_shift_shape,
                    dtype=paddle.get_default_dtype(),
                )
            self.linear_shift.set_value(shift_tensor)
            if self.smooth_key in state_dict:
                smooth_tensor = get_tensor(state_dict.pop(
                    self.smooth_key)).astype(paddle.get_default_dtype())
            else:
                smooth_tensor = paddle.ones(
                    shape=[self.linear_smooth_shape],
                    dtype=paddle.get_default_dtype(),
                )
            self.linear_smooth.set_value(smooth_tensor)


class RowParallelLinear(nn.Layer):
    """
    RowParallelLinear Layer
    """

    def __init__(
        self,
        llm_config,
        layer_name="",
        layer_index=0,
    ):
        """
        Initialize a linear layer with additional parameters for inference and quantization.

        Args:
            llm_config (LLMConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.
            layer_name (str): Unique name of the layer, used for naming internal attributes,
                you can give it any name you like.
            layer_index (int): The index of the linear layer in the model

        """
        super().__init__()
        self.llm_config = llm_config
        self.skip_quant = False
        self.use_smooth_quant = llm_config.model_config.use_smooth_quant
        self.weight_dtype = llm_config.model_config.weight_dtype
        self.act_dtype = llm_config.model_config.act_dtype
        self.nranks = llm_config.parallel_config.mp_size
        self.embed_dim = llm_config.model_config.hidden_size
        self.head_dim = llm_config.model_config.hidden_size // llm_config.model_config.num_attention_heads
        self.num_heads = llm_config.model_config.num_attention_heads // self.nranks
        self.dim_feedforward = llm_config.model_config.ffn_hidden_size // self.nranks

        self.weight_key = llm_config.load_config.weight_keys.out_linear_weight_keys[
            layer_index]
        self.bias_key = llm_config.load_config.weight_keys.out_linear_bias_keys[
            layer_index]
        self.with_bias = True if self.bias_key is not None else False

        self.shift_key = f"{layer_name}.shift_bias"
        self.smooth_key = f"{layer_name}.smooth_weight"

        self.layer_name = layer_name
        self.weight_name = self.layer_name + ".weight"
        self.bias_name = self.layer_name + ".bias"
        self.weight_only_scale_name = self.layer_name + ".weight_only_scale"
        self.out_scale_name = self.layer_name + ".out_scale"
        if self.use_smooth_quant:
            self.shift_name = self.layer_name + ".shift_bias"
            self.smooth_name = self.layer_name + ".smooth_weight"
        self._dtype = self._helper.get_default_dtype()

        if llm_config.quant_config:
            self.quant_method = llm_config.quant_config.get_quant_method(self)
            self.quant_method.create_weights(self)

        self.init_weight()

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

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        # weight
        weight_tensor = get_tensor(state_dict.pop(self.weight_key))

        if self.llm_config.quant_config:
            self.quant_method.process_loaded_weights(self, weight_tensor)
        else:
            self.linear_weight.set_value(weight_tensor)

        # bias
        if self.with_bias:
            bias_tensor = paddle.to_tensor(
                get_tensor(state_dict.pop(self.bias_key)))
            self.linear_bias.set_value(bias_tensor)

        # smooth quant
        if self.use_smooth_quant:
            if self.shift_key in state_dict:
                shift_tensor = get_tensor(state_dict.pop(
                    self.shift_key)).astype(paddle.get_default_dtype())
            else:
                shift_tensor = paddle.zeros(
                    shape=[(self.inference_args.num_attention_heads //
                            self.inference_args.mp_size) *
                           (self.inference_args.hidden_size //
                            self.inference_args.num_attention_heads)],
                    dtype=paddle.get_default_dtype(),
                )
            self.linear_shift.set_value(shift_tensor)
            if self.smooth_key in state_dict:
                smooth_tensor = get_tensor(state_dict.pop(
                    self.smooth_key)).astype(paddle.get_default_dtype())
            else:
                smooth_tensor = paddle.ones(
                    shape=[(self.inference_args.num_attention_heads //
                            self.inference_args.mp_size) *
                           (self.inference_args.hidden_size //
                            self.inference_args.num_attention_heads)],
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
        if self.llm_config.quant_config:
            linear_out = self.quant_method.apply(self, x)
        else:
            linear_out = paddle.matmul(x, self.linear_weight)

        return linear_out


class FFN2(RowParallelLinear):
    """
    FFN2 is a Linear layer with different weight dimensions.
    """

    def __init__(
        self,
        inference_args,
        layer_name,
        weight_key,
        bias_key=None,
        skip_quant=False,
        use_smooth_quant=True,
        shift_key=None,
        smooth_key=None,
        llm_config=None,
        layer_index=0,
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
            skip_quant (bool, optional): Whether to skip quantization for this layer.
                Defaults to False.
            use_smooth_quant (bool, optional): Whether to use smooth quantization for this
                layer. Smooth quantization introduces additional parameters to improve
                quantization accuracy. Defaults to True.
        """

        super(FFN2, self).__init__(
            llm_config,
            layer_name=layer_name,
            layer_index=layer_index,
        )
        self.inference_args = inference_args
        self.weight_key = weight_key
        self.bias_key = bias_key
        self.skip_quant = skip_quant
        self.use_smooth_quant = use_smooth_quant
        self.shift_key = shift_key
        self.smooth_key = smooth_key
        self.with_bias = True if self.bias_key is not None else False

    def init_weight_block_scale(self):
        """init_weight_block_scale for fp8"""
        self.linear_weight_scale = self.create_parameter(
            shape=[(self.embed_dim + 127) // 128,
                   (self.dim_feedforward + 127) // 128],
            attr=paddle.ParamAttr(name=self.layer_name +
                                  ".weight_block_scale"),
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
                    per_block_cast_to_fp8(weight_tensor))
                self.linear_weight.copy_(quanted_weight_tensor, False)
                self.linear_weight_scale.set_value(weight_block_scale_tensor)
            elif self.weight_dtype == "int8" and self.act_dtype in [
                    "bfloat16",
                    "float16",
                    "float32",
            ]:  # WINT8
                quanted_weight_tensor, weight_scale_tensor = weight_quantize(
                    weight_tensor,
                    algo="weight_only_int8",
                    arch=self.inference_args.weight_only_linear_arch,
                )
                self.linear_weight.set_value(quanted_weight_tensor)
                self.linear_weight_scale.set_value(
                    weight_scale_tensor.astype(paddle.get_default_dtype()))
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
            elif (self.weight_dtype == "int4"
                  and self.act_dtype == "float8_e4m3fn"):  # W4Afp8
                quanted_weight_tensor, weight_scale_tensor = (
                    fastdeploy.model_executor.ops.gpu.
                    scaled_gemm_f8_i4_f16_weight_quantize(
                        paddle.cast(weight_tensor, "float32").cpu(),
                        groupsize=-1,
                        scale_dtype="float16",
                    ))
                weight_scale_tensor = paddle.view(weight_scale_tensor,
                                                  self._dtype)
                self.linear_weight.set_value(quanted_weight_tensor)
                self.linear_weight_scale.set_value(weight_scale_tensor)
            else:  # bf16/fp16/fp32, A8W8, FP8
                if self.is_y_transposed():
                    weight_tensor = weight_tensor.transpose([1, 0])
                weight_tensor = paddle.cast(weight_tensor, self.weight_dtype)
                if ("float8" in self.weight_dtype
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
                shift_tensor = get_tensor(state_dict.pop(
                    self.shift_key)).astype(paddle.get_default_dtype())
            else:
                shift_tensor = paddle.zeros(
                    shape=self.linear_shift_shape,
                    dtype=paddle.get_default_dtype(),
                )
            self.linear_shift.set_value(shift_tensor)
            if self.smooth_key in state_dict:
                smooth_tensor = get_tensor(state_dict.pop(
                    self.smooth_key)).astype(paddle.get_default_dtype())
            else:
                smooth_tensor = paddle.ones(
                    shape=self.linear_smooth_shape,
                    dtype=paddle.get_default_dtype(),
                )
            self.linear_smooth.set_value(smooth_tensor)

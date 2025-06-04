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
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed import fleet
from .utils import get_tensor

try:
    from fastdeploy.model_executor.ops.npu import lm_head
except ImportError:
    pass

try:
    from fastdeploy.model_executor.ops.cpu import avx_weight_only
except ImportError:
    pass


def parallel_matmul(lm_output, logit_weights, parallel_output):
    """
    Performs parallel matrix multiplication for large-scale language models.

    Args:
        lm_output (Tensor): The output tensor from the language model layers,
            which will be multiplied with the logit weights.
        logit_weights (Tensor): The weights used in the matrix multiplication,
            typically the weights of the output layer.
        parallel_output (bool): A flag indicating whether to return the parallel
            outputs or concatenate them. If True, returns the outputs from the
            parallel computation directly. If False, concatenates the outputs
            across the model parallel group before returning.

    Returns:
        Tensor: The result of the matrix multiplication. If `parallel_output` is True,
            returns the parallel outputs. If `parallel_output` is False and
            model parallel world size is greater than 1, returns the concatenated
            outputs across the model parallel group. Otherwise, returns the direct
            matrix multiplication result.
    """
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_model_parallel_group()
    world_size = hcg.get_model_parallel_world_size()
    # rank = hcg.get_model_parallel_rank()

    if world_size > 1:
        input_parallel = paddle.distributed.collective._c_identity(
            lm_output, group=model_parallel_group
        )

        logits = paddle.matmul(input_parallel, logit_weights, transpose_y=True)

        if parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(
            logits, group=model_parallel_group
        )
    else:
        logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
        return logits


def parallel_linear(lm_output, logit_weights, parallel_output, bias):
    """
    Perform parallel linear transformation on lm_output.

    Args:
        lm_output (Tensor): The input tensor to the linear transformation, typically the output of a language model.
        logit_weights (Tensor): The weights of the linear layer, with shape [vocab_size, hidden_size].
        parallel_output (bool): Whether to return parallel outputs or concatenated outputs.
        bias (Tensor): The bias term to be added to the linear transformation.

    Returns:
        Tensor: The transformed tensor. If parallel_output is True, returns parallel tensors.
                Otherwise, returns concatenated tensors across the model parallel group.
    """
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_model_parallel_group()
    world_size = hcg.get_model_parallel_world_size()
    # rank = hcg.get_model_parallel_rank()
    bias += 0.0

    if world_size > 1:
        input_parallel = paddle.distributed.collective._c_identity(
            lm_output, group=model_parallel_group
        )
        bias_parallel = paddle.distributed.collective._c_identity(
            bias, group=model_parallel_group
        )

        logits = paddle.matmul(input_parallel, logit_weights, transpose_y=True)
        logits += bias_parallel

        if parallel_output:
            return logits

        return paddle.distributed.collective._c_concat(
            logits, group=model_parallel_group
        )
    else:
        logits = paddle.matmul(lm_output, logit_weights, transpose_y=True)
        logits += bias

        return logits


class FusedRMSNorm(nn.Layer):
    """
    FusedRMSNorm is a layer that applies RMSNorm normalization to the input tensor.
    """

    def __init__(self, hidden_size, epsilon=1e-5):
        """
        Initialize the FusedRMSNorm layer.

        Args:
            hidden_size (int): The size of the last hidden layer in the model.
                This is the number of features in the output of the Transformer layer.
            epsilon (float, optional): A small scalar to avoid division by zero in
                optimization. Defaults to 1e-5.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = epsilon

    def forward(self, hidden_states):
        """
        Applies the RMSNorm layer to the input.

        Args:
            hidden_states (Tensor): The input tensor of shape [batch_size, sequence_length, hidden_dim].
                It is the output from the last transformer encoder layer.

        Returns:
            Tensor or Tuple(Tensor, Tensor):
                If `output_attentions` is set in the config, a tuple will be returned containing the RMSNorm output and
                the attention weights. Otherwise, just the RMSNorm output will be returned.
                The RMSNorm output has shape [batch_size, sequence_length, hidden_dim].
        """

        result = paddle.incubate.nn.functional.fused_rms_norm(
            hidden_states,
            self.weight,
            None,
            self.variance_epsilon,
            begin_norm_axis=1,
        )
        if isinstance(result, tuple):
            return result[0]
        return result


class LMHead(nn.Layer):
    """
    LMHead is a layer that performs linear transformation on the input tensor.
    """

    def __init__(
        self,
        layer_name,
        linear_weight_key,
        linear_bias_key,
        input_dim=None,
        output_dim=None,
        fused_linear=False,
        activation=None,
        sequence_parallel=False,
        sharing_weight=None,
        sharing_bias=None,
        column_cut=True,
        use_ep=False,
    ):
        """
        Initialize the LMHead module.

        Args:
            layer_name (str): Unique name of the layer, you can give it any name you like.
            weight_key (str): Key name of weight in the pdparams state dict.
            linear_weight_key (str): Key name of linear weight in the pdparams state dict.
            linear_bias_key (str): Key name of linear bias in the pdparams state dict.
            input_dim (int, optional): The input dimension of the normalization layer and linear layer,
                defaults to None.
            output_dim (int, optional): The output dimension of the linear layer, defaults to None.
            fused_linear (bool, optional): Whether to use fused linear layer for matmul and bias addition,
                defaults to False.
            activation (str, optional): The activation function to apply after the linear layer,
                None if no activation, defaults to None.
            sequence_parallel (bool, optional): Whether to enable sequence parallelism, currently not used,
                defaults to False.
            sharing_weight (bool, optional): Whether to share weights across model parallel ranks, defaults to None.
            sharing_bias (bool, optional): Whether to share biases across model parallel ranks, defaults to None.
            column_cut (bool, optional): The weight distributed on your gpu cards is divided by row or column.
                Defaults to True means divide by column.
                When vocab_size can not be divided by world_size but hidden_size can,
                we can consider split embedding weight by row.
        """
        super(LMHead, self).__init__()
        self.linear_weight_key = linear_weight_key
        self.linear_bias_key = linear_bias_key
        self.use_ep = use_ep
        self.column_cut = column_cut

        hcg = fleet.get_hybrid_communicate_group()
        mp_rank = hcg.get_model_parallel_rank()
        ColumnParallelLinear = fleet.meta_parallel.ColumnParallelLinear
        RowParallelLinear = fleet.meta_parallel.RowParallelLinear

        self.sharing_weight = sharing_weight
        self.sharing_bias = sharing_bias

        if self.sharing_weight is None:
            if self.use_ep:
                self.weight = self.create_parameter(
                    shape=[input_dim, output_dim],
                    attr=None,
                    dtype=paddle.get_default_dtype(),
                    is_bias=False,
                )
            else:
                if self.column_cut:
                    need_gather = True
                    self.out_linear = ColumnParallelLinear(
                        input_dim,
                        output_dim,
                        mp_group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                        weight_attr=None,
                        has_bias=True if self.linear_bias_key is not None else False,
                        gather_output=need_gather,
                        fuse_matmul_bias=fused_linear,  # False diff更小
                    )
                else:
                    self.out_linear = RowParallelLinear(
                        input_dim,
                        output_dim,
                        mp_group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                        weight_attr=None,
                        has_bias=True if self.linear_bias_key is not None else False,
                        input_is_parallel=False,
                        fuse_matmul_bias=fused_linear,  # False diff更小
                    )

                self.out_linear.weight.name = layer_name + str(mp_rank) + ".w_0"
                if self.linear_bias_key is not None:
                    self.out_linear.bias.name = layer_name + str(mp_rank) + ".b_0"

        self.activation = activation
        if self.activation is not None:
            if activation == "SwiGLU":
                self.act = paddle.nn.Silu()
            else:
                self.act = getattr(F, activation)

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """

        if self.sharing_weight is None:
            if self.use_ep:
                self.weight.set_value(
                    get_tensor(state_dict.pop(self.linear_weight_key)).astype(
                        paddle.get_default_dtype()
                    )
                )
            else:
                self.out_linear.weight.set_value(
                    get_tensor(state_dict.pop(self.linear_weight_key)).astype(
                        paddle.get_default_dtype()
                    )
                )

                if self.linear_bias_key is not None:
                    bias = get_tensor(state_dict.pop(self.linear_bias_key)).astype(
                            paddle.get_default_dtype())
                    self.out_linear.bias.set_value(bias)

    def forward(self, input):
        """
        Defines the forward computation of the layer.

        Args:
            input (Tensor): The input tensor to the layer.

        Returns:
            Tensor: The output tensor after processing through the layer.
        """
        logits = input
        if self.sharing_weight is not None and self.sharing_bias is not None:
            logits = parallel_linear(
                logits, self.sharing_weight, False, self.sharing_bias
            )
        elif self.sharing_weight is not None:
            logits = parallel_matmul(logits, self.sharing_weight, False)
        else:
            if self.use_ep:
                logits = paddle.matmul(logits, self.weight)
            else:
                logits = self.out_linear(logits)
            if self.activation is not None:
                if self.activation == "SwiGLU":
                    logits = self.act(logits)
                elif self.activation == "gelu":
                    logits = F.gelu(logits, approximate=True)
                else:
                    logits = self.act(logits)
        return logits


class LMHeadNPU(nn.Layer):
    """
    LMHeadNPU is a layer that performs linear transformation on the input tensor.
    """

    def __init__(
        self,
        norm_layer_name,
        linear_layer_name,
        input_dim=None,
        output_dim=None,
        activation=None,
        epsilon=1e-5,
        trans_weight=True,
        norm_type="layernorm",
        have_norm_bias=True,
        name="",
    ):
        """
        Initialize LMHeadNPU class.

        Args:
            norm_layer_name (str): The name of the normalization layer.
            linear_layer_name (str): The name of the linear layer.
            input_dim (int, optional): The dimension of the input tensor. Defaults to None.
            output_dim (int, optional): The dimension of the output tensor. Defaults to None.
            activation (str, optional): The activation function to use.
                Supported options are any PaddlePaddle activation function or "SwiGLU".
                Defaults to None.
            epsilon (float, optional): A value added to the denominator for numerical stability.
                Defaults to 1e-5.
            trans_weight (bool, optional): Whether to transpose the weight matrix of the linear layer.
                Defaults to True.
            norm_type (str, optional): The type of normalization to use.
                Defaults to "layernorm".
            have_norm_bias (bool, optional): Whether to include bias in the normalization layer.
                Defaults to True.
            name (str, optional): The name of the layer. Defaults to "".
        """
        super(LMHeadNPU, self).__init__()
        self.norm_layer_name = norm_layer_name
        self.norm_weight_layer_name = self.norm_layer_name + ".weight"
        self.norm_bias_layer_name = self.norm_layer_name + ".bias"

        self.rank = (
            paddle.distributed.fleet.get_hybrid_communicate_group().get_model_parallel_rank()
        )
        self.nranks = (
            paddle.distributed.fleet.get_hybrid_communicate_group().get_model_parallel_world_size()
        )
        self.root = 0
        self.ring_id = (
            paddle.distributed.fleet.get_hybrid_communicate_group()
            .get_model_parallel_group()
            .id
        )
        self.trans_weight = trans_weight
        self.epsilon = epsilon
        self.have_norm_bias = have_norm_bias
        self.norm_weight = self.create_parameter(
            shape=[input_dim],
            attr=None,
            dtype=self._helper.get_default_dtype(),
            is_bias=False,
        )
        if self.have_norm_bias:
            self.norm_bias = self.create_parameter(
                shape=[input_dim],
                attr=None,
                dtype=self._helper.get_default_dtype(),
                is_bias=True,
            )
        else:
            self.norm_bias = None

        self.linear_layer_name = linear_layer_name
        self.linear_weight_layer_name = self.linear_layer_name + ".weight"
        self.linear_bias_layer_name = self.linear_layer_name + ".bias"
        self.linear_weight = self.create_parameter(
            shape=(
                [output_dim // self.nranks, input_dim]
                if trans_weight
                else [input_dim, output_dim // self.nranks]
            ),
            attr=None,
            dtype=self._helper.get_default_dtype(),
            is_bias=False,
        )
        if self.have_norm_bias:
            self.linear_bias = self.create_parameter(
                shape=[output_dim // self.nranks],
                attr=None,
                dtype=self._helper.get_default_dtype(),
                is_bias=True,
            )
        else:
            self.linear_bias = None

        self.name = name
        self.linear_weight.name = self.name + str(self.rank) + ".w_0"
        if self.have_norm_bias:
            self.linear_bias.name = self.name + str(self.rank) + ".b_0"
        self.activation = activation
        if self.activation is not None:
            if activation == "SwiGLU":
                self.act = paddle.nn.Silu()
            else:
                self.act = getattr(F, activation)
        self.norm_type = norm_type

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """

        self.norm_weight.set_value(
            get_tensor(state_dict.pop(self.norm_weight_layer_name))
        )
        self.linear_weight.set_value(
            get_tensor(state_dict.pop(self.linear_weight_layer_name))
        )
        if self.have_norm_bias:
            self.norm_bias.set_value(
                get_tensor(state_dict.pop(self.norm_bias_layer_name))
            )
            self.linear_bias.set_value(
                get_tensor(state_dict.pop(self.linear_bias_layer_name))
            )

    def forward(self, input):
        """
        Defines the forward computation of the layer.

        Args:
            input (Tensor): The input tensor to the layer.

        Returns:
            Tensor: The output tensor after processing through the layer.
        """

        return lm_head(
            input,
            self.norm_weight,
            self.norm_bias,
            self.linear_weight,
            self.linear_bias,
            epsilon=self.epsilon,
            trans_weight=self.trans_weight,
            act="none" if self.activation is None else self.activation,
            norm_type=self.norm_type,
            rank=self.rank,
            nranks=self.nranks,
            root=self.root,
            ring_id=self.ring_id,
        )


class LMHeadAVX(nn.Layer):
    """
    LMHeadAVX is a layer that performs linear transformation on the input tensor.
    """

    def __init__(
        self,
        norm_layer_name,
        linear_layer_name,
        norm_type="layernorm",
        input_dim=None,
        output_dim=None,
        have_norm_bias=True,
        have_ln_bias=True,
        hidden_size=None,
        alog="int8",
    ):
        """
        Initialize the LMHeadAVX class.

        Args:
            norm_layer_name (str): The name of the normalization layer.
            linear_layer_name (str): The name of the linear layer.
            norm_type (str, optional): The type of normalization, defaults to 'layernorm'.
                Options include 'layernorm' and 'rmsnorm'.
            input_dim (int, optional): The input dimension of the linear layer.
            output_dim (int, optional): The output dimension of the linear layer.
            have_norm_bias (bool, optional): Whether the normalization layer has a bias term, defaults to True.
            have_ln_bias (bool, optional): Whether the linear layer has a bias , defaults to True.
            hidden_size(int, optional): The hidden size of the model. Defaults to None.
            alog (str, optional): The activation log quantization type, defaults to 'int8'.
        """
        super(LMHeadAVX, self).__init__()
        self.norm_layer_name = norm_layer_name
        self.norm_weight_layer_name = self.norm_layer_name + ".weight"
        self.norm_bias_layer_name = self.norm_layer_name + ".bias"
        self.norm_type = norm_type
        self.hidden_size = hidden_size
        if self.norm_type == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size, epsilon=1e-5)
        elif self.norm_type == "rmsnorm":
            self.norm = FusedRMSNorm(self.hidden_size, epsilon=1e-5)
        else:
            raise NotImplementedError(f"Unsupported norm type: {self.norm_type}")
        self.linear_layer_name = linear_layer_name
        self.linear_weight_layer_name = self.linear_layer_name + ".weight"
        self.linear_bias_layer_name = self.linear_layer_name + ".bias"
        self.alog = alog
        self.have_ln_bias = have_ln_bias
        self.have_norm_bias = have_norm_bias
        self.linear_weight = self.create_parameter(
            shape=[input_dim, output_dim],
            attr=paddle.ParamAttr(name=self.linear_weight_layer_name),
            dtype="float32",
            is_bias=False,
        )
        if self.have_ln_bias:
            self.linear_bias = self.create_parameter(
                shape=[output_dim],
                attr=paddle.ParamAttr(name=self.linear_bias_layer_name),
                dtype="float32",
                is_bias=True,
            )
        else:
            self.linear_bias = None

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        self.norm.weight.set_value(state_dict.pop(self.norm_weight_layer_name))
        self.linear_weight.set_value(state_dict.pop(self.linear_weight_layer_name))
        if self.have_norm_bias:
            self.norm.bias.set_value(state_dict.pop(self.norm_bias_layer_name))
        if self.have_ln_bias:
            self.linear_bias.set_value(state_dict.pop(self.linear_bias_layer_name))

    def forward(self, input):
        """
        Defines the forward computation of the layer.

        Args:
            input (Tensor): The input tensor to the layer.

        Returns:
            Tensor: The output tensor after processing through the layer.
        """

        logits = self.norm(input)
        logits = avx_weight_only(
            logits,
            self.linear_weight,
            self.linear_bias,
            alog=self.alog,
            trans=False,
        )
        return logits

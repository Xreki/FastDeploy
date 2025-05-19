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
from .utils import get_tensor

try:
    from fastdeploy.model_executor.ops.cpu import xft_llama_all_layer
except ImportError:
    pass


class AvxFusedTransformer(nn.Layer):
    """
    AvxFusedTransformer Layer
    """

    def __init__(
        self,
        inference_args,
        with_ln_bias,
        with_qkv_bias,
        with_out_linear_bias,
        with_ffn_ln_bias,
        with_gate_up_bias,
        with_ffn2_bias,
        activation,
        norm_type,
    ):
        """
        AvxFusedTransformer layer for Transformer-based models optimized with Intel AVX-512/AMX instruction sets.

        Args:
            inference_args (object): infrence arguments, such as number of attention heads, hidden size, etc.
            with_ln_bias (bool): Whether to add bias to the layer normalization layers.
            with_qkv_bias (bool): Whether to add bias to the query, key, and value projection layers.
            with_out_linear_bias (bool): Whether to add bias to the output linear layer.
            with_ffn_ln_bias (bool): Whether to add bias to the layer normalization in the feedforward network.
            with_gate_up_bias (bool): Whether to add bias to the gate up layer (if applicable).
            with_ffn2_bias (bool): Whether to add bias to the second feedforward network layer (if applicable).
            activation (callable): The activation function to use in the feedforward network.
            norm_type (str): The type of normalization to use.
        """
        super().__init__()
        self.inference_args = inference_args
        # config
        self.kv_num_heads = inference_args.num_key_value_heads
        self._dtype = self._helper.get_default_dtype()
        self._compute_type = self.inference_args.act_dtype
        self._with_ln_bias = with_ln_bias
        self._with_qkv_bias = with_qkv_bias
        self._with_out_linear_bias = with_out_linear_bias
        self._with_ffn_ln_bias = with_ffn_ln_bias
        self._with_gate_up_bias = with_gate_up_bias
        self._with_ffn2_bias = with_ffn2_bias
        self.head_dim = (
            self.inference_args.hidden_size // self.inference_args.num_attention_heads
        )
        self.activation = activation
        self.norm_type = norm_type
        self.init_weight()

    def _add_parameter(self, param):
        """
        Add a parameter to the model if it does not already exist.
        If the parameter is None, no operation is performed.

        Args:
            param (Optional[Parameter]): The parameter to be added, can be None. Defaults to None.
                The Parameter type, containing information such as the parameter name and weights.

        Returns:
            None.

        Raises:
            AssertionError: If the parameter's name already exists in the model's parameters.
        """
        if param is None:
            return
        assert param.name not in self._parameters
        self._parameters[param.name] = param

    def init_weight(self):
        """
        Initialize the weights and biases.
        """
        # 权重
        # ln
        self.ln_weights, self.ln_biases = [], []
        # qkv
        self.qkv_weights, self.qkv_biases = [], []
        # out_linear
        self.linear_weights, self.linear_biases = [], []
        # ffn ln
        self.ffn_ln_weights, self.ffn_ln_biases = [], []
        # ffn
        self.gate_weights, self.gate_biases = [], []
        self.up_weights, self.up_biases = [], []
        self.ffn2_weights, self.ffn2_biases = [], []

        for i in range(self.inference_args.num_layers):
            ln_weight = self.create_parameter(
                attr=paddle.ParamAttr(name=f"gpt.decoder.layers.{i}.norm1.weight"),
                shape=[self.inference_args.hidden_size],
                dtype=self._dtype,
            )
            ln_bias = None
            if self._with_ln_bias:
                ln_bias = self.create_parameter(
                    attr=paddle.ParamAttr(name=f"gpt.decoder.layers.{i}.norm1.bias"),
                    shape=[self.inference_args.hidden_size],
                    is_bias=True,
                    dtype=self._dtype,
                )

            qkv_weight = self.create_parameter(
                shape=[
                    self.inference_args.hidden_size,
                    (
                        self.inference_args.num_attention_heads
                        + 2 * self.inference_args.num_key_value_heads
                    )
                    * self.head_dim,
                ],
                attr=paddle.ParamAttr(
                    name=f"gpt.decoder.layers.{i}.self_attn.qkv_proj.weight"
                ),
                dtype=self._dtype,
                is_bias=False,
            )

            qkv_bias = None
            if self._with_qkv_bias:
                qkv_bias = self.create_parameter(
                    shape=[
                        (
                            self.inference_args.num_attention_heads
                            + 2 * self.inference_args.num_key_value_heads
                        )
                        * self.head_dim
                    ],
                    attr=paddle.ParamAttr(
                        name=f"gpt.decoder.layers.{i}.self_attn.qkv_proj.bias"
                    ),
                    dtype=self._dtype,
                    is_bias=True,
                )

            linear_weight = self.create_parameter(
                shape=[
                    self.inference_args.num_attention_heads * self.head_dim,
                    self.inference_args.hidden_size,
                ],
                attr=paddle.ParamAttr(
                    name=f"gpt.decoder.layers.{i}.self_attn.out_proj.weight"
                ),
                dtype=self._dtype,
                is_bias=False,
            )
            linear_bias = None
            if self._with_out_linear_bias:
                linear_bias = self.create_parameter(
                    shape=[self.inference_args.hidden_size],
                    attr=paddle.ParamAttr(
                        name=f"gpt.decoder.layers.{i}.self_attn.out_proj.bias"
                    ),
                    dtype=self._dtype,
                    is_bias=True,
                )

            ffn_ln_weight = self.create_parameter(
                shape=[self.inference_args.hidden_size],
                attr=paddle.ParamAttr(name=f"gpt.decoder.layers.{i}.norm2.weight"),
                is_bias=False,
                dtype=self._dtype,
            )

            ffn_ln_bias = None
            if self._with_ffn_ln_bias:
                ffn_ln_bias = self.create_parameter(
                    shape=[self.inference_args.hidden_size],
                    attr=paddle.ParamAttr(name=f"gpt.decoder.layers.{i}.norm2.bias"),
                    is_bias=True,
                    dtype=self._dtype,
                )

            gate_weight = self.create_parameter(
                shape=[
                    self.inference_args.hidden_size,
                    self.inference_args.ffn_hidden_size,
                ],
                attr=paddle.ParamAttr(name=f"gpt.decoder.layers.{i}.gate.weight"),
                dtype=self._dtype,
                is_bias=False,
            )
            up_weight = self.create_parameter(
                shape=[
                    self.inference_args.hidden_size,
                    self.inference_args.ffn_hidden_size,
                ],
                attr=paddle.ParamAttr(name=f"gpt.decoder.layers.{i}.up.weight"),
                dtype=self._dtype,
                is_bias=False,
            )

            gate_bias = None
            up_bias = None
            if self._with_gate_up_bias:
                gate_bias = self.create_parameter(
                    shape=[self.inference_args.ffn_hidden_size],
                    attr=paddle.ParamAttr(name=f"gpt.decoder.layers.{i}.gate.bias"),
                    dtype=self._dtype,
                    is_bias=True,
                )
                up_bias = self.create_parameter(
                    shape=[self.inference_args.ffn_hidden_size],
                    attr=paddle.ParamAttr(name=f"gpt.decoder.layers.{i}.up.bias"),
                    dtype=self._dtype,
                    is_bias=True,
                )

            ffn2_weight = self.create_parameter(
                shape=[
                    self.inference_args.ffn_hidden_size,
                    self.inference_args.hidden_size,
                ],
                attr=paddle.ParamAttr(name=f"gpt.decoder.layers.{i}.linear2.weight"),
                dtype=self._dtype,
                is_bias=False,
            )

            ffn2_bias = None
            if self._with_ffn2_bias:
                ffn2_bias = self.create_parameter(
                    shape=[self.inference_args.hidden_size],
                    attr=paddle.ParamAttr(name=f"gpt.decoder.layers.{i}.linear2.bias"),
                    dtype=self._dtype,
                    is_bias=True,
                )

            self.ln_weights.append(ln_weight)
            self.ln_biases.append(ln_bias)
            self.qkv_weights.append(qkv_weight)
            self.qkv_biases.append(qkv_bias)
            self.linear_weights.append(linear_weight)
            self.linear_biases.append(linear_bias)
            self.ffn_ln_weights.append(ffn_ln_weight)
            self.ffn_ln_biases.append(ffn_ln_bias)
            self.gate_weights.append(gate_weight)
            self.gate_biases.append(gate_bias)
            self.up_weights.append(up_weight)
            self.up_biases.append(up_bias)
            self.ffn2_weights.append(ffn2_weight)
            self.ffn2_biases.append(ffn2_bias)

            self._add_parameter(ln_weight)
            self._add_parameter(ln_bias)
            self._add_parameter(qkv_weight)
            self._add_parameter(qkv_bias)
            self._add_parameter(linear_weight)
            self._add_parameter(linear_bias)
            self._add_parameter(ffn_ln_weight)
            self._add_parameter(ffn_ln_bias)
            self._add_parameter(gate_weight)
            self._add_parameter(gate_bias)
            self._add_parameter(up_weight)
            self._add_parameter(up_bias)
            self._add_parameter(ffn2_weight)
            self._add_parameter(ffn2_bias)

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        for i in range(self.inference_args.num_layers):
            # ln
            ln_weight_tensor = paddle.cast(
                get_tensor(state_dict.pop(f"gpt.decoder.layers.{i}.norm1.weight")),
                self._dtype,
            )
            self.ln_weights[i].set_value(ln_weight_tensor)
            if self._with_ln_bias:
                ln_bias_tensor = paddle.cast(
                    get_tensor(state_dict.pop(f"gpt.decoder.layers.{i}.norm1.bias")),
                    self._dtype,
                )
                self.ln_biases[i].set_value(ln_bias_tensor)
            # qkv
            # weight
            if self.inference_args.num_key_value_heads <= 0:
                qkv_proj_weight = (
                    get_tensor(
                        state_dict.pop(
                            f"gpt.decoder.layers.{i}.self_attn.qkv_proj.weight"
                        )
                    )
                    .reshape(
                        [
                            self.inference_args.hidden_size,
                            self.inference_args.num_attention_heads
                            // self.inference_args.mp_size,
                            3,
                            self.inference_args.hidden_size
                            // self.inference_args.num_attention_heads,
                        ]
                    )
                    .transpose([2, 1, 3, 0])
                )
            else:
                qkv_proj_weight = get_tensor(
                    state_dict.pop(f"gpt.decoder.layers.{i}.self_attn.qkv_proj.weight")
                ).reshape(
                    [
                        self.inference_args.hidden_size,
                        self.inference_args.num_attention_heads
                        // self.inference_args.mp_size
                        + 2
                        * self.inference_args.num_key_value_heads
                        // self.inference_args.mp_size,
                        self.inference_args.hidden_size
                        // self.inference_args.num_attention_heads,
                    ]
                )
                single_qkv_proj_weights = paddle.split(
                    qkv_proj_weight,
                    self.inference_args.num_key_value_heads
                    // self.inference_args.mp_size,
                    axis=1,
                )
                q_weights, k_weights, v_weights = [], [], []
                for single_qkv_proj_weight in single_qkv_proj_weights:
                    q_weight, k_weight, v_weight = paddle.split(
                        single_qkv_proj_weight,
                        [
                            self.inference_args.num_attention_heads
                            // self.inference_args.num_key_value_heads,
                            1,
                            1,
                        ],
                        axis=1,
                    )
                    q_weights.append(q_weight)
                    k_weights.append(k_weight)
                    v_weights.append(v_weight)
                q_weight = paddle.concat(q_weights, axis=1)
                k_weight = paddle.concat(k_weights, axis=1)
                v_weight = paddle.concat(v_weights, axis=1)
                qkv_proj_weight = paddle.concat([q_weight, k_weight, v_weight], axis=1)
            qkv_proj_weight = qkv_proj_weight.reshape(
                [self.inference_args.hidden_size, -1]
            )
            self.qkv_weights[i].set_value(qkv_proj_weight)
            if self._with_qkv_bias:
                if self.inference_args.num_key_value_heads <= 0:
                    qkv_bias = (
                        get_tensor(
                            state_dict.pop(
                                f"gpt.decoder.layers.{i}.self_attn.qkv_proj.bias"
                            )
                        )
                        .reshape(
                            [
                                self.inference_args.num_attention_heads
                                // self.inference_args.mp_size,
                                3,
                                self.inference_args.hidden_size
                                // self.inference_args.num_attention_heads,
                            ]
                        )
                        .transpose([1, 0, 2])
                    )
                else:
                    # GQA
                    qkv_bias = get_tensor(
                        state_dict.pop(
                            f"gpt.decoder.layers.{i}.self_attn.qkv_proj.bias"
                        )
                    ).reshape(
                        [
                            self.inference_args.num_attention_heads
                            // self.inference_args.mp_size
                            + 2
                            * self.inference_args.num_key_value_heads
                            // self.inference_args.mp_size,
                            self.inference_args.hidden_size
                            // self.inference_args.num_attention_heads,
                        ]
                    )
                    single_qkv_biases = paddle.split(
                        qkv_bias,
                        self.inference_args.num_key_value_heads
                        // self.inference_args.mp_size,
                        axis=0,
                    )
                    q_biases, k_biases, v_biases = [], [], []
                    for single_qkv_bias in single_qkv_biases:
                        q_bias, k_bias, v_bias = paddle.split(
                            single_qkv_bias,
                            [
                                self.inference_args.num_attention_heads
                                // self.inference_args.num_key_value_heads,
                                1,
                                1,
                            ],
                            axis=0,
                        )
                        q_biases.append(q_bias)
                        k_biases.append(k_bias)
                        v_biases.append(v_bias)
                    q_bias = paddle.concat(q_biases, axis=0)
                    k_bias = paddle.concat(k_biases, axis=0)
                    v_bias = paddle.concat(v_biases, axis=0)
                    qkv_bias = paddle.concat([q_bias, k_bias, v_bias], axis=0)
                qkv_bias = qkv_bias.reshape([-1])
                self.qkv_biases[i].set_value(qkv_bias)
            # out_linear
            linear_weight_tensor = paddle.cast(
                get_tensor(
                    state_dict.pop(f"gpt.decoder.layers.{i}.self_attn.out_proj.weight")
                ),
                self._dtype,
            )
            self.linear_weights[i].set_value(linear_weight_tensor)
            if self._with_out_linear_bias:
                linear_biase_tensor = paddle.cast(
                    get_tensor(
                        state_dict.pop(
                            f"gpt.decoder.layers.{i}.self_attn.out_proj.bias"
                        )
                    ),
                    self._dtype,
                )
                self.linear_biases[i].set_value(linear_biase_tensor)
            # ffnln
            ffn_ln_weight_tensor = paddle.cast(
                get_tensor(state_dict.pop(f"gpt.decoder.layers.{i}.norm2.weight")),
                self._dtype,
            )
            self.ffn_ln_weights[i].set_value(ffn_ln_weight_tensor)
            if self._with_ffn_ln_bias:
                ffn_ln_biase_tensor = paddle.cast(
                    get_tensor(state_dict.pop(f"gpt.decoder.layers.{i}.norm2.bias")),
                    self._dtype,
                )
                self.ffn_ln_biases[i].set_value(ffn_ln_biase_tensor)
            # gate and up
            ffn1_weight_tensor = paddle.to_tensor(
                state_dict[f"gpt.decoder.layers.{i}.linear1.weight"]
            )
            converted_ffn1_weight_tensor = paddle.zeros(
                shape=list(ffn1_weight_tensor.shape),
                dtype=ffn1_weight_tensor.dtype,
            )
            out_dim = converted_ffn1_weight_tensor.shape[-1]
            converted_ffn1_weight_tensor[:, : out_dim // 2] = ffn1_weight_tensor[
                :, 0::2
            ]
            converted_ffn1_weight_tensor[:, out_dim // 2 :] = ffn1_weight_tensor[
                :, 1::2
            ]
            gate_up_list = paddle.split(
                converted_ffn1_weight_tensor, num_or_sections=2, axis=-1
            )
            self.gate_weights[i].set_value(gate_up_list[0])
            self.up_weights[i].set_value(gate_up_list[1])
            if self._with_gate_up_bias:
                ffn1_bias_tensor = paddle.to_tensor(
                    state_dict[f"gpt.decoder.layers.{i}.linear1.bias"]
                )
                converted_ffn1_bias_tensor = paddle.zeros(
                    shape=list(ffn1_bias_tensor.shape),
                    dtype=ffn1_weight_tensor.dtype,
                )
                converted_ffn1_bias_tensor[: out_dim // 2] = ffn1_bias_tensor[0::2]
                converted_ffn1_bias_tensor[out_dim // 2 :] = ffn1_bias_tensor[1::2]
                gate_up_bias_list = paddle.split(
                    converted_ffn1_bias_tensor, num_or_sections=2, axis=-1
                )
                self.gate_biases[i].set_value(gate_up_bias_list[0])
                self.up_biases[i].set_value(gate_up_bias_list[1])
            # ffn2
            ffn2_weight_tensor = paddle.cast(
                get_tensor(state_dict.pop(f"gpt.decoder.layers.{i}.linear2.weight")),
                self._dtype,
            )
            self.ffn2_weights[i].set_value(ffn2_weight_tensor)
            if self._with_ln_bias:
                ffn2_biase_tensor = paddle.cast(
                    get_tensor(state_dict.pop(f"gpt.decoder.layers.{i}.linear2.bias")),
                    self._dtype,
                )
                self.ffn2_biases[i].set_value(ffn2_biase_tensor)

    def forward(
        self,
        src,
        step_idx=None,
        seq_lens_this_time=None,
        seq_lens_encoder=None,
        seq_lens_decoder=None,
        **kwargs,
    ):
        """
        Defines the forward pass of the Transformer layer.

        Args:
            src (Tensor): The input sequence data of shape [batch_size, sequence_length, hidden_size].
            step_idx (int, optional): The current step index for incremental decoding.
            seq_lens_this_time (Tensor, optional): The sequence lengths of the current batch.
            seq_lens_encoder (Tensor, optional): The sequence lengths of the encoder output.
            seq_lens_decoder (Tensor, optional): The sequence lengths of the decoder output.
            **kwargs: Additional keyword arguments for future compatibility and extensions.

        Returns:
            Tensor: The output of the Transformer layer at the last position of the sequence.
                    Shape: [batch_size, hidden_size]
        """
        if int(seq_lens_encoder[0]) != 0:
            past_seq_len = paddle.zeros_like(seq_lens_encoder, dtype="int64")
        else:
            past_seq_len = paddle.cast(paddle.clone(seq_lens_decoder), "int64")
        bs = seq_lens_this_time.shape[0]
        src = src.reshape([bs, -1, self.inference_args.hidden_size])
        xft_out = xft_llama_all_layer(
            src,  # input
            self.ln_weights,  # ln1Gamma
            self.ln_biases,  # ln1Beta
            self.qkv_weights,  # qkvWeight
            self.qkv_biases,  # qkvBias
            self.linear_weights,  # attnOutWeight
            self.linear_biases,  # attnOutBias
            self.ffn_ln_weights,  # ln2Gamma
            self.ffn_ln_biases,  # ln2Beta
            self.gate_weights,  # gateWeight
            self.gate_biases,  # gateBias
            self.up_weights,  # upWeight
            self.up_biases,  # upbias
            self.ffn2_weights,  # downWeight
            self.ffn2_biases,  # upbias
            past_seq_len,  # pastSeqLen
            seq_lens_this_time,  # currentSeqLen
            step_idx,  # step
            self.inference_args.hidden_size,  # hiddensize
            self.inference_args.num_layers,  # totalLayer
            self._compute_type,  # computeType
            self.activation,  # activation
            self.norm_type,  # normType
            self.head_dim,  # attHeadDim
            self.inference_args.num_attention_heads,  # attHeadNum
            self.inference_args.num_key_value_heads,  # kvHeadNum
            self.inference_args.max_position_embeddings,  # maxPositions
            self.inference_args.max_position_embeddings,  # maxPosEmbed
            self.inference_args.ffn_hidden_size,  # intermediateSize
        )
        return xft_out[:, -1, :]

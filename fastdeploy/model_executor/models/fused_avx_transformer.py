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
from paddle import nn

from ..layers.transformer import AvxFusedTransformer


class FusedAvxTransformer(nn.Layer):
    """
    FusedAvxTransformer Layer
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
        Initialize the fused transformer model.
        Only supports Intel AVX512/AMX instruction sets.

        Args:
            inference_args (dict or similar): Configuration arguments for the transformer during inference.
            with_ln_bias (bool): Whether to include bias in layer normalization layers.
            with_qkv_bias (bool): Whether to include bias in the query/key/value projections.
            with_out_linear_bias (bool): Whether to include bias in the output linear layer.
            with_ffn_ln_bias (bool): Whether to include bias in the feed-forward network's layer normalization.
            with_gate_up_bias (bool): Whether to include bias in the gate upscaling mechanism.
            with_ffn2_bias (bool): Whether to include bias in the second feed-forward network (if applicable).
            activation (str or function): Activation function used in the transformer layers.
            norm_type (str): Normalization type used in the transformer layers.
        """
        super().__init__()
        self.inference_args = inference_args
        self.transformer = AvxFusedTransformer(
            inference_args,
            with_ln_bias,
            with_qkv_bias,
            with_out_linear_bias,
            with_ffn_ln_bias,
            with_gate_up_bias,
            with_ffn2_bias,
            activation,
            norm_type,
        )

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        self.transformer.load_state_dict(state_dict)

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
        Defines the forward pass of the model.

        This method encapsulates the forward pass logic of the model, which typically involves passing the input through
        the transformer layer.

        Args:
            src (Tensor): The input tensor to the transformer.
            step_idx (int, optional): The step index for autoregressive decoding.
            seq_lens_this_time (Tensor, optional): The sequence lengths for the current decoding step.
            seq_lens_encoder (Tensor, optional): The sequence lengths of the encoder outputs.
            seq_lens_decoder (Tensor, optional): The sequence lengths of the decoder inputs.
            **kwargs: Additional keyword arguments that will be passed to the transformer layer.

        Returns:
            Tensor: The output of the transformer layer.
        """
        return self.transformer(
            src,
            step_idx=step_idx,
            seq_lens_this_time=seq_lens_this_time,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            **kwargs,
        )

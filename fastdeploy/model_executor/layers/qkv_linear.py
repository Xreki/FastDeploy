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
from fastdeploy.model_executor.layers.linear import ColumnParallelLinear
from fastdeploy.model_executor.layers.utils import divide


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

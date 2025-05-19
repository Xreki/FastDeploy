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
from paddle.distributed import fleet

from .utils import get_tensor
from fastdeploy.platforms import current_platform

try:
    from fastdeploy.model_executor.ops.npu import word_embedding_parallel
except ImportError:
    pass


class Embeddings(nn.Layer):
    """
    Embeddings Layer
    """

    def __init__(
        self,
        layer_name,
        vocab_size,
        hidden_size=768,
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
        sequence_parallel=False,
        freeze_embedding=False,
        weight_sharing=True,
        weight_sharing_add_bias=False,
        use_rope=True,
        rope_head_dim=None,
        column_cut=False,
        prefix_name="",
        use_ep=False,
    ):
        """
        Initialize the embedding layer for the model.

        Args:
            layer_name (str): Name of the layer.
            vocab_size (int): Vocabulary size of the embedding layer.
            hidden_size (int, optional): Hidden size of the embedding vectors. Defaults to 768.
            hidden_dropout_prob (float, optional): Dropout probability for the embedding vectors.
                Defaults to 0.1.
            max_position_embeddings (int, optional): Maximum number of positional embeddings.
                Defaults to 512.
            type_vocab_size (int, optional): Type vocabulary size. Not used in this snippet.
                Defaults to 16.
            initializer_range (float, optional): Standard deviation of the normal initializer.
                Defaults to 0.02.
            sequence_parallel (bool, optional): Whether to enable sequence parallelism.
                Defaults to False.
            freeze_embedding (bool, optional): Whether to freeze the embedding layer during training.
                Defaults to False.
            weight_sharing (bool, optional): Whether to share weights with another layer.
                Defaults to True.
            weight_sharing_add_bias (bool, optional): Whether to add bias when weight sharing is enabled.
                Defaults to False.
            use_rope (bool, optional): Whether to use RoPE (Rotary Position Embedding).
                Defaults to True.
            rope_head_dim (int, optional): Head dimension for RoPE (if used). Defaults to None.
            column_cut (bool, optional): The embedding weight distributed on your gpu cards is divided by row or column.
                Defaults to False means divide by row.
                When vocab_size can not be divided by world_size but hidden_size can,
                we can consider split embedding weight by column.
        """
        super().__init__()
        hcg = fleet.get_hybrid_communicate_group()
        self.mp_rank = hcg.get_model_parallel_rank()
        self.column_cut = column_cut
        self.world_size = hcg.get_model_parallel_world_size()
        self.ring_id = hcg.get_model_parallel_group().id  # for NPU
        self._word_emb_name = (
            prefix_name + "word_embedding_expanded_" + str(self.mp_rank) + ".w_0"
        )
        self._pos_emb_name = prefix_name + "pos_embedding_0.w_0"
        self.use_rope = use_rope
        self.rope_head_dim = rope_head_dim
        self.use_ep = use_ep

        self.sequence_parallel = sequence_parallel
        if current_platform.is_npu():
            # npu call custom op to calculate parallel word_embedding
            self.word_embeddings = self.create_parameter(
                shape=[vocab_size, hidden_size // self.world_size],
                attr=None,
                dtype=self._helper.get_default_dtype(),
                is_bias=False,
            )
        else:
            # gpu
            if use_ep:
                self.word_embeddings = nn.Embedding(
                    vocab_size,
                    hidden_size,
                )
            else:
                if not self.column_cut:
                    self.word_embeddings = fleet.meta_parallel.VocabParallelEmbedding(
                        vocab_size,
                        hidden_size,
                        mp_group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                        weight_attr=paddle.ParamAttr(
                            name=self._word_emb_name,
                            initializer=nn.initializer.Normal(
                                mean=0.0, std=initializer_range
                            ),
                        ),
                    )
                else:
                    # column cut embedding
                    self.word_embeddings = nn.Embedding(
                        vocab_size,
                        hidden_size // self.world_size,
                    )
                    self.word_embeddings.weight.is_distributed = True
                    self.word_embeddings.weight.split_axis = 1

        if not self.use_rope:
            self.position_embeddings = nn.Embedding(
                max_position_embeddings,
                hidden_size,
                weight_attr=paddle.ParamAttr(
                    name=self._pos_emb_name,
                    initializer=nn.initializer.Normal(mean=0.0, std=initializer_range),
                ),
            )

        self.layer_name = layer_name

        if weight_sharing and weight_sharing_add_bias:
            if self.world_size > 1:
                bias_name = "server_nlg_mask_lm_out_fc_" + str(self.mp_rank) + ".b_0"
            else:
                bias_name = "server_nlg_mask_lm_out_fc.b_0"
            mask_lm_out_bias_attr = paddle.ParamAttr(
                name=bias_name,
                initializer=paddle.nn.initializer.Constant(value=0.0),
            )
            assert vocab_size % self.world_size == 0
            if use_ep:
                self.bias = self.create_parameter(
                    shape=[vocab_size],
                    dtype=paddle.get_default_dtype(),
                    attr=mask_lm_out_bias_attr,
                    is_bias=True,
                )
            else:
                self.bias = self.create_parameter(
                    shape=[vocab_size // self.world_size],
                    dtype=paddle.get_default_dtype(),
                    attr=mask_lm_out_bias_attr,
                    is_bias=True,
                )
                self.bias.is_distributed = True

        if freeze_embedding:
            self.word_embeddings.weight.learning_rate = 0.0
            if not self.use_rope:
                self.position_embeddings.weight.learning_rate = 0.0

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.rope_head_dim_shape_tensor = paddle.ones(
            (self.rope_head_dim), dtype="int8"
        )

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        if current_platform.is_npu():
            self.word_embeddings.set_value(
                get_tensor(state_dict.pop(self.layer_name + ".weight"))
            )
        else:
            self.word_embeddings.weight.set_value(
                get_tensor(state_dict.pop(self.layer_name + ".weight")).astype(
                    paddle.get_default_dtype()
                )
            )

    def forward(self, ids_remove_padding=None):
        """
        Defines the forward computation of the layer.

        Args:
            ids_remove_padding (Tensor, optional): Tensor of token IDs, with padding removed.
                If None, no input is provided.

        Returns:
            Tensor: Embedded tensor representation of the input IDs.
        """
        if current_platform.is_npu():
            # npu
            input_embedings = word_embedding_parallel(
                ids_remove_padding,
                self.word_embeddings,
                parallel_type="ColumnParallel",
                rank=self.mp_rank,
                nranks=self.world_size,
                root=0,
                ring_id=self.ring_id,
            )
        else:
            # gpu
            if self.use_ep:
                input_embedings = self.word_embeddings(ids_remove_padding)
            else:
                if self.column_cut:
                    input_embedings = self.word_embeddings(ids_remove_padding)
                    inputs_embeds_temp = []
                    paddle.distributed.all_gather(
                        inputs_embeds_temp,
                        input_embedings,
                        group=fleet.get_hybrid_communicate_group().get_model_parallel_group(),
                        sync_op=True,
                    )
                    input_embedings = paddle.concat(inputs_embeds_temp, -1)
                else:
                    input_embedings = self.word_embeddings(ids_remove_padding)

        return input_embedings

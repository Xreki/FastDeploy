"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.incubate.nn.functional import blha_get_max_len
import fastdeploy
from paddlenlp.utils.log import logger

try:
    from paddle.base.core import EventHandle
    from paddle.distributed.communication import deep_ep
except ImportError:
    logger.warning("import EventHandle and deep_ep Failed!")


class MicroBatch:
    """
    MicroBatch
    """

    def __init__(self, batch_size):
        """
        Initialize the MicroBatch.
        """
        self.batch_size = batch_size
        self.token_num = 0
        self.start_batch_id = 0
        self.end_batch_id = 0
        self.start_token_id = 0
        self.end_token_id = 0
        self.compute_event = deep_ep.utils.EventOverlap(EventHandle())
        self.comunication_event = deep_ep.utils.EventOverlap(EventHandle())
        self.args = {}

    def get_block_shape_and_split_kv_block(
        self,
        encoder_block_shape_q,
        decoder_block_shape_q,
        encoder_max_partition_size,
        max_partition_size,
        group_size,
        total_draft_token_num,
    ):
        """
        Split the block of attention computing for current micro-batch.

        Args:
            encoder_block_shape_q (int): The block shape of append attention in prefill stage.
            decoder_block_shape_q (int): The block shape of append attention in decode stage.
            encoder_max_partition_size (int): The kv tokens size of per chunk in prefill stage.
            max_partition_size (int): The kv tokens size of per chunk in decode stage.
            group_size (int): Group size in gqa attention.
            total_draft_token_num (int): Max token_num of queries in speculate mode.
        """
        self.args["encoder_block_shape_q"] = encoder_block_shape_q
        self.args["decoder_block_shape_q"] = decoder_block_shape_q
        self.args["encoder_max_partition_size"] = encoder_max_partition_size
        self.args["max_partition_size"] = max_partition_size
        (
            self.args["encoder_batch_ids"],
            self.args["encoder_tile_ids_per_batch"],
            self.args["encoder_num_blocks"],
            self.args["kv_batch_ids"],
            self.args["kv_tile_ids_per_batch"],
            self.args["kv_num_blocks"],
            self.args["decoder_batch_ids"],
            self.args["decoder_tile_ids_per_batch"],
            self.args["decoder_num_blocks"],
            self.args["max_len_kv"],
            self.args["set_max_lengths"],
        ) = fastdeploy.model_executor.ops.gpu.get_block_shape_and_split_kv_block(
            self.args["seq_lens_encoder"],
            self.args["seq_lens_decoder"],
            self.args["seq_lens_this_time"],
            self.args["cum_offsets"],
            encoder_block_shape_q,
            decoder_block_shape_q,
            group_size,
            self.args["block_size"],
            total_draft_token_num,
        )

    def blha_get_max_len(self):
        """
        Get the max length for current micro-batch.
        """
        (self.args["max_enc_len_this_time"], self.args["max_dec_len_this_time"]) = (
            blha_get_max_len(
                self.args["seq_lens_encoder"],
                self.args["seq_lens_decoder"],
                self.args["cum_offsets"],
            )
        )


class MicroBatchControl:
    """
    MicroBatchControl
    """

    def __init__(self):
        """
        Initialize the MicroBatch.
        """
        self.micro_batches = []
        self.micro_batch_num = 0
        self.total_batch_size = 0

    def split_micro_batch(self, micro_batch_num, **kwargs):
        """
        Split the batch to multiple micro-batch.

        Args:
            micro_batch_num (int): The num of micro-batch.
        """
        self.micro_batches = []
        cum_offsets = kwargs.get("cum_offsets", None)
        total_batch_size = cum_offsets.shape[0]
        self.total_batch_size = total_batch_size
        micro_bsz = (total_batch_size + micro_batch_num - 1) // micro_batch_num
        token_idx = 0
        for i in range(micro_batch_num):
            start_bidx = i * micro_bsz
            end_bidx = min(start_bidx + micro_bsz, total_batch_size)
            cur_batch_size = end_bidx - start_bidx
            mb = MicroBatch(cur_batch_size)
            mb.start_batch_id = start_bidx
            mb.end_batch_id = end_bidx
            mb.args["cum_offsets"] = (
                cum_offsets[start_bidx:end_bidx] - cum_offsets[start_bidx]
            )
            mb.args["block_tables"] = kwargs.get("block_tables", None)[
                start_bidx:end_bidx
            ]
            mb.args["seq_lens_encoder"] = kwargs.get("seq_lens_encoder", None)[
                start_bidx:end_bidx
            ]
            mb.args["seq_lens_decoder"] = kwargs.get("seq_lens_decoder", None)[
                start_bidx:end_bidx
            ]
            mb.args["seq_lens_this_time"] = kwargs.get("seq_lens_this_time", None)[
                start_bidx:end_bidx
            ]
            token_num = mb.args["seq_lens_this_time"].sum().item()
            mb.token_num = token_num
            mb.args["padding_offsets"] = (
                kwargs.get("padding_offsets", None)[token_idx: token_idx + token_num]
                - kwargs.get("padding_offsets", None)[token_idx]
            )
            mb.args["max_input_length"] = kwargs.get("max_input_length", -1)
            mb.args["block_size"] = kwargs.get("block_size", 64)
            mb.start_token_id = token_idx
            token_idx += token_num
            mb.end_token_id = token_idx
            self.micro_batches.append(mb)
        self.micro_batch_num = micro_batch_num

    def get_block_shape_and_split_kv_block(
        self,
        encoder_block_shape_q,
        decoder_block_shape_q,
        encoder_max_partition_size,
        max_partition_size,
        group_size,
        total_draft_token_num,
    ):
        """
        Split the block of attention computing for all micro-batch.

        Args:
            encoder_block_shape_q (int): The block shape of append attention in prefill stage.
            decoder_block_shape_q (int): The block shape of append attention in decode stage.
            encoder_max_partition_size (int): The kv tokens size of per chunk in prefill stage.
            max_partition_size (int): The kv tokens size of per chunk in decode stage.
            group_size (int): Group size in gqa attention.
            total_draft_token_num (int): Max token_num of queries in speculate mode.
        """
        for mb in self.micro_batches:
            mb.get_block_shape_and_split_kv_block(
                encoder_block_shape_q,
                decoder_block_shape_q,
                encoder_max_partition_size,
                max_partition_size,
                group_size,
                total_draft_token_num,
            )

    def blha_get_max_len(self):
        """
        Get the max length for all micro-batch.
        """
        for mb in self.micro_batches:
            mb.blha_get_max_len()

    def wait_attn(self, micro_batch_id):
        """
        Wait attention computing is finished.

        Args:
            micro_batch_id (int): The index of micro-batch.
        """
        # wait in dispatch
        pass

    def wait_moe(self, micro_batch_id):
        """
        Wait moe computing is finished.

        Args:
            micro_batch_id (int): The index of micro-batch.
        """
        # wait in combine
        pass

    def wait_dispatch(self, micro_batch_id):
        """
        Wait diapatch communication is finished.

        Args:
            micro_batch_id (int): The index of micro-batch.
        """
        self.micro_batches[micro_batch_id].comunication_event.current_stream_wait()

    def wait_combine(self, micro_batch_id):
        """
        Wait combine communication is finished.

        Args:
            micro_batch_id (int): The index of micro-batch.
        """
        self.micro_batches[micro_batch_id].comunication_event.current_stream_wait()

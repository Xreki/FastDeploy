
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


from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Optional
from typing import List, Optional, Tuple, Union
from typing import TYPE_CHECKING

import abc
import paddle
import numpy as np
import logging

if TYPE_CHECKING:
    from fastdeploy.model_executor.layers.attention import AttentionBackend, Attention

logger = logging.getLogger(__name__)

class ForwardMode(IntEnum):
    """
    Forward mode used during attention.
    """

    # for prefill and extend
    EXTEND = auto()
    # for generation
    DECODE = auto()

    def is_prefill(self):
        """Whether it's a prefill forward"""
        return self == ForwardMode.EXTEND

    def is_decode(self):
        """Whether it's a decode forward"""
        return self == ForwardMode.DECODE

class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(
        self,
        size: int,
        max_context_len: int
    ):

        self.size = size
        self.max_context_len = max_context_len
        self.req_to_token = paddle.zeros(
            (size, max_context_len), dtype=paddle.int32
        )
        self.free_slots = list(range(size))

    def write(self, indices, values):
        """Write data into request buffer"""
        self.req_to_token[indices] = values

    def available_size(self):
        """Get number of slots left"""
        return len(self.free_slots)

    def alloc(self, need_size: int) -> List[int]:
        """Allocate `need_size` slots"""
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        """Free slot"""
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self):
        """Clear all slots"""
        self.free_slots = list(range(self.size))

class KVCache(abc.ABC):
    """Abstract base class representing a key value cache"""
    @abc.abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Return cached keys and values given layer id.
        Args:
        layer_id: int
        Returns:
            tuple: (keys, values)
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: 'Attention',
        loc: paddle.Tensor,
        cache_k: paddle.Tensor,
        cache_v: paddle.Tensor,
    ) -> None:
        """
        Set cached keys and values given layer id.
        Args:
        layer: Attention
        loc: paddle.Tensor
        cache_k: paddle.Tensor
        cache_v: paddle.Tensor
        """
        raise NotImplementedError()


    @abc.abstractmethod
    def transfer(self, indices, flat_data):
        """Transfer kv_data between devices"""
        raise NotImplementedError()

    @abc.abstractmethod
    def transfer_per_layer(self, indices, flat_data, layer_id):
        """Not used yet"""
        raise NotImplementedError()

    def register_layer_transfer_counter(self, layer_transfer_counter):
        """Not used yet"""
        self.layer_transfer_counter = layer_transfer_counter

class MHATokenToKVPool(KVCache):

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: paddle.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        if dtype in (paddle.int8, paddle.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = paddle.uint8
        else:
            self.store_dtype = dtype

        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self._create_buffers()


        k_size, v_size = self.get_kv_size_bytes()
        GB = 1024 * 1024 * 1024
        logger.info(
            f"KV Cache is allocated. #tokens: {size}, K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB"
        )

    def _create_buffers(self):
        # [size, head_num, head_dim] for each layer
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.k_buffer = [
            paddle.zeros(
                (self.size + self.page_size, self.head_num, self.head_dim),
                dtype=self.store_dtype,
            )
            for _ in range(self.layer_num)
        ]
        self.v_buffer = [
            paddle.zeros(
                (self.size + self.page_size, self.head_num, self.head_dim),
                dtype=self.store_dtype,
            )
            for _ in range(self.layer_num)
        ]

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += np.prod(k_cache.shape) * 4
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += np.prod(v_cache.shape) * 4
        return k_size_bytes, v_size_bytes


    def transfer(self, indices, flat_data):
        # transfer prepared data from host to device
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        k_data, v_data = flat_data[0], flat_data[1]
        for i in range(self.layer_num):
            self.k_buffer[i][indices] = k_data[i]
            self.v_buffer[i][indices] = v_data[i]

    def transfer_per_layer(self, indices, flat_data, layer_id):
        # transfer prepared data for a specific layer from host to device
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        k_data, v_data = flat_data[0], flat_data[1]
        self.k_buffer[layer_id][indices] = k_data
        self.v_buffer[layer_id][indices] = v_data

    def get_key_buffer(self, layer_id: int):

        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id].view(self.dtype)
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):

        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id].view(self.dtype)
        return self.v_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: 'Attention',
        loc: paddle.Tensor,
        cache_k: paddle.Tensor,
        cache_v: paddle.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ):
        layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        self.k_buffer[layer_id][loc] = cache_k
        self.v_buffer[layer_id][loc] = cache_v

@dataclass
class ForwardMeta():
    """
    ForwardMeta is used to store the global meta information of the forward.
    """
    # The forward mode
    forward_mode: ForwardMode
    # The batch size
    batch_size: int
    # The input ids
    input_ids: paddle.Tensor
    # The indices of requests in the req_to_token_pool
    req_pool_indices: paddle.Tensor
    # The sequence length
    seq_lens: paddle.Tensor
    # The indices of output tokens in the token_to_kv_pool
    out_cache_loc: paddle.Tensor

    # The sum of all sequence lengths
    seq_lens_sum: int

    # Optional seq_lens on cpu
    seq_lens_cpu: Optional[paddle.Tensor] = None

    # For extend
    extend_num_tokens: Optional[int] = None
    extend_seq_lens: Optional[paddle.Tensor] = None
    extend_prefix_lens: Optional[paddle.Tensor] = None
    extend_start_loc: Optional[paddle.Tensor] = None
    extend_prefix_lens_cpu: Optional[List[int]] = None
    extend_seq_lens_cpu: Optional[List[int]] = None

    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool: KVCache = None
    attn_backend: 'AttentionBackend' = None

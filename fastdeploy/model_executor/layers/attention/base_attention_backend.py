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

# Copyright 2023-2024 SGLang Team
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

from __future__ import annotations

from abc import ABC, abstractmethod
import paddle

from fastdeploy.worker.model_runner import ForwardMeta, ForwardMode


class AttentionBackend(ABC):
    """The base class of attention backends"""

    @abstractmethod
    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize the forward metadata."""
        raise NotImplementedError()

    def forward(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        layer: paddle.nn.Layer,
        forward_meta: ForwardMeta,
        **kwargs,
    ):
        """
        Run a forward.
        args:
            q: The query tensor.
            k: The key tensor.
            v: The value tensor.
            layer: The layer that will be used for the forward.
            forward_meta: The forward metadata.
        """
        if forward_meta.forward_mode.is_mixed():
            return self.forward_mixed(
                q,
                k,
                v,
                layer,
                forward_meta,
                **kwargs,
            )
        elif forward_meta.forward_mode.is_decode():
            return self.forward_decode(
                q,
                k,
                v,
                layer,
                forward_meta,
                **kwargs,
            )
        else:
            return self.forward_extend(
                q,
                k,
                v,
                layer,
                forward_meta,
                **kwargs,
            )

    def forward_mixed(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        layer: paddle.nn.Layer,
        forward_meta: ForwardMeta,
    ):
        """Run a forward for mix."""
        raise NotImplementedError()

    def forward_decode(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        layer: paddle.nn.Layer,
        forward_meta: ForwardMeta,
    ):
        """Run a forward for decode."""
        raise NotImplementedError()

    def forward_extend(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        layer: paddle.nn.Layer,
        forward_meta: ForwardMeta,
    ):
        """Run a forward for extend."""
        raise NotImplementedError()

"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import time
from dataclasses import dataclass, asdict, fields
from typing import TYPE_CHECKING, Optional, Union, Any
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.utils import data_processor_logger

from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.utils import data_processor_logger


@dataclass
class Request:
    def __init__(
        self,
        request_id: str,
        prompt: Optional[Union[str, list[str]]],
        prompt_token_ids: Optional[list[int]],
        prompt_token_ids_len: Optional[int],
        messages: Optional[list[list[dict[str, Any]]]],
        history: Optional[list[list[str]]],
        system: Optional[Union[str, list[str]]],
        sampling_params: SamplingParams,
        eos_token_ids: Optional[list[int]],
        arrival_time: float,
        preprocess_start_time: Optional[float] = None,
        preprocess_end_time: Optional[float] = None,
        multi_modal_inputs: Optional[dict] = None,
        raw_request: bool = True
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_token_ids_len = prompt_token_ids_len
        self.messages = messages
        self.system = system
        self.sampling_params = sampling_params
        self.history = history
        self.eos_token_ids = eos_token_ids

        self.arrival_time = arrival_time
        self.preprocess_start_time = preprocess_start_time
        self.preprocess_end_time = preprocess_end_time
        self.raw_request = raw_request

        # Multi-modal related
        # TODO
        self.multi_modal_inputs = multi_modal_inputs

    @classmethod
    def from_dict(cls, d: dict):
        data_processor_logger.debug(f"{d}")
        sampling_params = SamplingParams.from_dict(d)
        return cls(
            request_id=d["request_id"],
            prompt=d.get("prompt"),
            prompt_token_ids=d.get("prompt_token_ids"),
            prompt_token_ids_len=d.get("prompt_token_ids_len"),
            messages=d.get("messages"),
            system=d.get("system"),
            history=d.get("history"),
            sampling_params=sampling_params,
            eos_token_ids=d.get("eos_token_ids"),
            arrival_time=d.get("arrival_time", time.time()),
            preprocess_start_time=d.get("preprocess_start_time"),
            preprocess_end_time=d.get("preprocess_end_time"),
            multi_modal_inputs=d.get("multi_modal_inputs"),
            raw_request=d.get("raw_request", True)
        )

    def to_dict(self) -> dict:
        """convert Request into a serializable dict """
        data = {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "prompt_token_ids": self.prompt_token_ids,
            "prompt_token_ids_len": self.prompt_token_ids_len,
            "messages": self.messages,
            "system": self.system,
            "history": self.history,
            "eos_token_ids": self.eos_token_ids,
            "arrival_time": self.arrival_time,
            "preprocess_start_time": self.preprocess_start_time,
            "preprocess_end_time": self.preprocess_end_time,
            "multi_modal_inputs": self.multi_modal_inputs,
            "raw_request": self.raw_request
        }
        data.update(asdict(self.sampling_params))
        return data

    def get(self, key: str, default_value=None):
        if hasattr(self, key):
            return getattr(self, key)
        elif hasattr(self.sampling_params, key):
            return getattr(self.sampling_params, key)
        else:
            return default_value

    def set(self, key, value):
        if hasattr(self.sampling_params, key):
            setattr(self.sampling_params, key, value)
        else:
            setattr(self, key, value)

    def __repr__(self) -> str:
        return (f"Request(request_id={self.request_id}, "
                f"prompt={self.prompt!r}, "
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"sampling_params={self.sampling_params})")


@dataclass
class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index: The index of the output in the request.
        text: The generated output text.
        token_ids: The token IDs of the generated output text.
    """

    index: int
    token_ids: list[int]
    text: Optional[str] = None
    reasoning_content: Optional[str] = None

    @classmethod
    def from_dict(cls, req_dict: dict[str, Any]) -> 'CompletionOutput':
        """Create instance from dict arguments"""
        return cls(**{
            field.name: req_dict[field.name] if field.name in req_dict else field.default
            for field in fields(cls)
        })

    def __repr__(self) -> str:
        return (f"CompletionOutput(index={self.index}, "
                f"text={self.text!r}, "
                f"token_ids={self.token_ids}, "
                f"reasoning_content={self.reasoning_content!r}")


@dataclass
class RequestMetrics:
    """Metrics associated with a request.

    Attributes:
        arrival_time: The time when the request arrived.
        inference_start_time: The time when the inference started.
        first_token_time: The time when the first token was generated.
        time_in_queue: The time the request spent in the queue.
        model_forward_time: The time spent in the model forward pass when this
                            request was in the batch.
        model_execute_time: The time spent in the model execute function. This
                            will include model forward, block/sync across
                            workers, cpu-gpu sync time and sampling time.

    """
    arrival_time: float
    inference_start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    time_in_queue: Optional[float] = None
    preprocess_cost_time: Optional[float] = None
    model_forward_time: Optional[float] = None
    model_execute_time: Optional[float] = None

    @classmethod
    def from_dict(cls, req_dict: dict[str, Any]) -> 'RequestMetrics':
        """Create instance from dict arguments"""
        return cls(**{
            field.name: req_dict[field.name] if field.name in req_dict else field.default
            for field in fields(cls)
        })


class RequestOutput:
    """The output data of a completion request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
                For encoder/decoder models, this is the
                decoder input prompt.
        prompt_token_ids: The token IDs of the prompt.
                          For encoder/decoder models, this is the
                          decoder input prompt token ids.
        prompt_logprobs: The log probabilities to return per prompt token.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
        metrics: Metrics associated with the request.
        lora_request: The LoRA request that was used to generate the output.
        encoder_prompt: The encoder prompt string of the request.
                        None if decoder-only.
        encoder_prompt_token_ids: The token IDs of the encoder prompt.
                                  None if decoder-only.
        num_cached_tokens: The number of tokens with prefix cache hit.
    """

    def __init__(
        self,
        request_id: str,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[list[int]] = None,
        outputs: CompletionOutput = None,
        finished: bool = False,
        metrics: Optional[RequestMetrics] = None,
        num_cached_tokens: Optional[int] = 0,
        error_code: Optional[int] = 200,
        error_msg: Optional[str] = None,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs
        self.finished = finished
        self.metrics = metrics
        self.num_cached_tokens = num_cached_tokens
        self.error_code = error_code
        self.error_msg = error_msg

    def add(self, next_output: "RequestOutput") -> None:
        """Merge RequestOutput into this one"""

        self.prompt = next_output.prompt
        self.prompt_token_ids = next_output.prompt_token_ids
        self.finished |= next_output.finished
        self.outputs.index = next_output.outputs.index
        self.outputs.token_ids.extend(next_output.outputs.token_ids)
        if next_output.metrics.arrival_time is not None and self.metrics.inference_start_time is not None:
            self.metrics.model_forward_time = next_output.metrics.arrival_time - \
                self.metrics.inference_start_time
        if next_output.metrics.arrival_time is not None and self.metrics.arrival_time is not None:
            self.metrics.model_execute_time = next_output.metrics.arrival_time - \
                self.metrics.arrival_time

    def __repr__(self) -> str:
        return (f"RequestOutput(request_id={self.request_id}, "
                f"prompt={self.prompt!r}, "
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"outputs={self.outputs}, "
                f"metrics={self.metrics}, "
                f"num_cached_tokens={self.num_cached_tokens})")

    @classmethod
    def from_dict(cls, d: dict):
        """Create instance from dict arguments"""
        completion_output = CompletionOutput.from_dict(d.pop("outputs"))
        metrics = RequestMetrics.from_dict(d.pop("metrics"))
        return RequestOutput(**d, outputs=completion_output, metrics=metrics)

    def to_dict(self):
        """convert RequestOutput into a serializable dict """
        if self.prompt_token_ids is None:
            self.prompt_token_ids = []

        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "prompt_token_ids": self.prompt_token_ids,
            "outputs": None if self.outputs is None else asdict(self.outputs),
            "finished": self.finished,
            "metrics": None if self.metrics is None else asdict(self.metrics),
            "num_cached_tokens": self.num_cached_tokens,
            "error_code": self.error_code,
            "error_msg": self.error_msg,
        }

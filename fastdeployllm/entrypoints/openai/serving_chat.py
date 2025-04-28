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

import asyncio
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Callable, Optional, Union, List
import uuid

from fastapi import Request
from pydantic import BaseModel
from fastdeployllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionResponseStreamChoice,
    ChatMessage,
    UsageInfo,
    ChatCompletionResponse,
    ErrorResponse,
)

from fastdeployllm.utils import api_server_logger

from fastdeployllm.engine.request import RequestOutput


async def async_wrapper(sync_gen):
    loop = asyncio.get_event_loop()
    while True:
        try:
            # 在独立线程中执行同步生成器
            item = await loop.run_in_executor(None, next, sync_gen)
            yield item
            if item.get("finished", False):
                break
        except StopIteration:
            api_server_logger.info("Sync generator has been fully traversed.")
            break


class OpenAIServingChat:
    """
    OpenAI-style chat completions serving
    """

    def __init__(self, engine_client):
        self.engine_client = engine_client

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest
    ):
        """
        Create a new chat completion using the specified parameters.
        """
        request_id = f"chatcmpl-{uuid.uuid4()}"
        api_server_logger.info(f"create chat completion request: {request_id}")

        try:
            current_req_dict = request.to_dict_for_infer()
            generator = self.engine_client.generate(
                current_req_dict,
                request.stream
            )
            async_gen = async_wrapper(generator)
        except ValueError as e:
            return ErrorResponse(code=400, message=str(e))

        if request.stream:
            return self.chat_completion_stream_generator(
                request, async_gen, request_id, request.model)
        else:
            try:
                return await self.chat_completion_full_generator(
                    request, async_gen, request_id, request.model)
            except ValueError as e:
                return ErrorResponse(code=400, message=str(e))

    def _create_streaming_error_response(self, message: str) -> str:
        error_response = ErrorResponse(
            code=400,
            message=message,
        )
        return error_response.model_dump_json()

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncGenerator,
        request_id: str,
        model_name: str
    ):
        """
        Streaming chat completion generator.
        """
        created_time = int(time.time())
        chunk_object_type: str = "chat.completion.chunk"
        first_iteration = True
        num_choices = 1
        previous_num_tokens = [0] * num_choices
        num_prompt_tokens = 0

        stream_options = request.stream_options
        if stream_options is None:
            include_usage = False
            include_continuous_usage = False
        else:
            include_usage = stream_options.include_usage
            include_continuous_usage = stream_options.continuous_usage_stats
        api_server_logger.info(f"include usage: {include_usage}, include cont usage: {include_continuous_usage}")

        try:
            async for res in result_generator:
                if first_iteration:
                    num_prompt_tokens = len(res["prompt_token_ids"])
                    num_cached_tokens = res["num_cached_tokens"]
                    for i in range(num_choices):
                        choice = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(role="assistant", content="")
                        )
                        if request.metadata is not None and request.metadata.get("training", False):
                            choice.delta.token_ids = list(res["prompt_token_ids"])
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice],
                            model=model_name
                        )
                        if include_continuous_usage:
                            chunk.usage = UsageInfo(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                total_tokens=num_prompt_tokens
                            )
                        yield f"data: {chunk.model_dump_json(exclude_unset=True)} \n\n"
                    first_iteration = False
                api_server_logger.debug(f"The chat completion stream chunk {res}")
                output = res["outputs"]
                delta_text = output["text"]

                previous_num_tokens[0] += len(output["token_ids"])
                delta_message = DeltaMessage(content=delta_text, reasoning_content=output["reasoning_content"])

                choice = ChatCompletionResponseStreamChoice(
                    index=output["index"],
                    delta=delta_message
                )
                if res["finished"]:
                    if request.max_tokens is None or output["index"] + 1 != request.max_tokens:
                        choice.finish_reason = "stop"
                    else:
                        choice.finish_reason = "length"

                if request.metadata is not None and request.metadata.get("training", False) and delta_text != "":
                    choice.delta.token_ids = output["token_ids"]
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[choice],
                    model=model_name
                )
                if include_continuous_usage:
                    chunk.usage = UsageInfo(
                        prompt_tokens=num_prompt_tokens,
                        completion_tokens=previous_num_tokens[0],
                        total_tokens=num_prompt_tokens + previous_num_tokens[0]
                    )
                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

            if include_usage:
                completion_tokens = sum(previous_num_tokens)
                usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens
                )
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=usage
                )
                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

        except Exception as e:
            error_data = self._create_streaming_error_response(str(e))
            yield f"data: {error_data}\n\n"
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncGenerator,
        request_id: str,
        model_name: str
    ):
        """
        Full chat completion generator.
        """
        created_time = int(time.time())
        final_res = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return ErrorResponse(code=499, message="Client disconnected")

        if not final_res:
            return ErrorResponse(code=500, message="No response generated")

        choices = []
        output = final_res["outputs"]
        message = ChatMessage(
            role="assistant",
            content=output["text"],
            reasoning_content=output["reasoning_content"]
        )

        choice = ChatCompletionResponseChoice(
            index=output["index"],
            message=message,
            finish_reason=None
        )
        if request.max_tokens is None or output["index"] + 1 != request.max_tokens:

            choice.finish_reason = "stop"
        else:
            choice.finish_reason = "length"
        choices.append(choice)

        num_prompt_tokens = len(final_res["prompt_token_ids"])
        num_generated_tokens = len(output["token_ids"])
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens
        )

        return ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage
        )

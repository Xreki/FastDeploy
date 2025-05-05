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

import asyncio
import aiozmq
import json
from aiozmq import zmq
from asyncio import FIRST_COMPLETED, AbstractEventLoop, Task
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from typing import Optional, Union, cast, TypeVar, List
import uuid
from fastapi import Request

from fastdeploy.entrypoints.openai.protocol import ErrorResponse, CompletionRequest, CompletionResponse, CompletionStreamResponse, CompletionResponseStreamChoice, CompletionResponseChoice,UsageInfo
from fastdeploy.utils import api_server_logger
from fastdeploy.engine.request import RequestOutput


class OpenAIServingCompletion:
    def __init__(self, engine_client):
        self.engine_client = engine_client

    async def create_completion(self, request: CompletionRequest):
        """
        Create a completion for the given prompt.
        """
        created_time = int(time.time())
        request_id = f"cmpl-{uuid.uuid4()}"
        api_server_logger.info(f"initialize request {request_id}")
        request_prompt_ids = None
        request_prompts = None
        try:
            if isinstance(request.prompt, str):
                request_prompts = [request.prompt]
            elif isinstance(request.prompt, list) and all(isinstance(item,  int) for item in request.prompt):
                request_prompt_ids = [request.prompt]
            elif isinstance(request.prompt, list) and all(isinstance(item, str) for item in request.prompt):
                request_prompts = request.prompt
            elif isinstance(request.prompt, list):
                for item in request.prompt:
                    if isinstance(item, list) and all(isinstance(x, int) for x in item):
                        continue
                    else:
                        raise ValueError("Prompt must be a string, a list of strings or a list of integers.")
                request_prompt_ids = request.prompt
            else:
                raise ValueError("Prompt must be a string, a list of strings or a list of integers.")
        except Exception as e:
            return ErrorResponse(message=str(e), code=400)

        if request_prompt_ids is not None:
            request_prompts = request_prompt_ids
        num_choices = len(request_prompts)

        api_server_logger.info(f"start inference for request {num_choices}")

        try:
            for idx, prompt in enumerate(request_prompts):
                request_id_idx = f"{request_id}-{idx}"
                current_req_dict = request.to_dict_for_infer(request_id_idx, prompt)
                try:
                    current_req_dict["arrival_time"] = time.time()
                    self.engine_client.format_and_add_data(current_req_dict)
                except Exception as e:
                    return ErrorResponse(message=str(e), code=400) 

            if request.stream:
                return self.completion_stream_generator(
                    request=request,
                    num_choices = num_choices,
                    request_id=request_id,
                    created_time=created_time,
                    model_name=request.model
                )
            else:
                try:
                    return await self.completion_full_generator(
                        request=request,
                        num_choices=num_choices,
                        request_id=request_id,
                        created_time=created_time,
                        model_name=request.model
                    )
                except ValueError as e:
                    return ErrorResponse(code=400, message=str(e))

        except ValueError as e:
            return ErrorResponse(message=str(e), code=400)


    async def completion_full_generator(
        self,
        request: CompletionRequest,
        num_choices: int,
        request_id: str,
        created_time: int,
        model_name: str,
    ):
        """
        Process the full completion request with multiple choices.
        """
        dealer = None
        try:
            request_ids = [f"{request_id}-{i}" for i in range(num_choices)]
            # create dealer
            dealer = await aiozmq.create_zmq_stream(
                zmq.DEALER,
                connect="ipc:///dev/shm/router.ipc"
            )

            for rid in request_ids:
                dealer.write([b"", rid.encode("utf-8")])

            valid_results = [dict()] * num_choices
            while num_choices > 0:
                raw_data = await dealer.read()
                data = json.loads(raw_data[-1].decode("utf-8"))
                rid = int(data["request_id"].split("-")[-1])

                self.engine_client.data_processor.process_response_dict(
                    data, stream=False
                )
                if data.get("finished", False):
                    valid_results[rid] = data
                    num_choices -= 1

            return self.request_output_to_completion_response(
                final_res_batch=valid_results,
                request=request,
                request_id=request_id,
                created_time=created_time,
                model_name=model_name
            )
        except Exception as e:
            api_server_logger.error(
                f"Error in completion_full_generator: {e}", exc_info=True
            )
            raise
        finally:
            if dealer is not None:
                dealer.close()


    async def completion_stream_generator(
        self,
        request: CompletionRequest,
        num_choices: int,
        request_id: str,
        created_time: int,
        model_name: str
    ):
        """
        Process the stream completion request.
        """
        try:
            dealer = await aiozmq.create_zmq_stream(
                zmq.DEALER,
                connect="ipc:///dev/shm/router.ipc"
            )

            for i in range(num_choices):
                req_id = f"{request_id}-{i}"
                dealer.write([b"", req_id.encode('utf-8')])  # 发送多路请求
            output_tokens = [0] * num_choices
            inference_start_time = [0] * num_choices
            while num_choices > 0:
                raw_data = await dealer.read()
                res = json.loads(raw_data[-1].decode('utf-8'))
                idx = int(res["request_id"].split("-")[-1])
                self.engine_client.data_processor.process_response_dict(res, stream=True)
                if res['metrics']['first_token_time'] is not None:
                    arrival_time = res['metrics']['first_token_time']
                    inference_start_time[idx] = res['metrics']['inference_start_time']
                else:
                    arrival_time = res['metrics']['arrival_time'] - inference_start_time[idx]
                # api_server_logger.info(f"{arrival_time}")

                output = res["outputs"]
                chunk = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[CompletionResponseStreamChoice(
                        index=idx,
                        text=output["text"],
                        reasoning_content=output["reasoning_content"],
                        arrival_time=arrival_time
                    )]
                )

                output_tokens[idx] += 1

                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

                if res["finished"]:
                    num_choices -= 1
                    if getattr(request, "stream_options", None) and request.stream_options.include_usage:
                        usage_chunk = CompletionStreamResponse(
                            id=request_id,
                            created=created_time,
                            model=model_name,
                            choices=[],
                            usage=UsageInfo(
                                prompt_tokens=len(res.get("prompt_token_ids", [])),
                                completion_tokens=output_tokens[idx]
                            )
                        )
                        yield f"data: {usage_chunk.model_dump_json(exclude_unset=True)}\n\n"


        except Exception as e:
            yield f"data: {ErrorResponse(message=str(e), code=400).model_dump_json(exclude_unset=True)}\n\n"
        finally:
            if dealer is not None:
                dealer.close()
            yield "data: [DONE]\n\n"


    def request_output_to_completion_response(
        self,
        final_res_batch: List[RequestOutput],
        request: CompletionRequest,
        request_id: str,
        created_time: int,
        model_name: str,
    ) -> CompletionResponse:
        choices: List[CompletionResponseChoice] = []
        num_prompt_tokens = 0
        num_generated_tokens = 0

        for final_res in final_res_batch:
            prompt_token_ids = final_res["prompt_token_ids"]
            assert prompt_token_ids is not None
            prompt_text = final_res["prompt"]

            output = final_res["outputs"]
            if request.echo:
                assert prompt_text is not None
                if request.max_tokens == 0:
                    token_ids = prompt_token_ids
                    output_text = prompt_text
                else:
                    token_ids = [*prompt_token_ids, *output["token_ids"]]
                    output_text = prompt_text + output["text"]
            else:
                token_ids = output["token_ids"]
                output_text = output["text"]

            choice_data = CompletionResponseChoice(
                index=len(choices),
                text=output_text,
                reasoning_content=output['reasoning_content'],
                logprobs=None,
                finish_reason=None
            )
            choices.append(choice_data)

            num_generated_tokens += len(output["token_ids"])

            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )

        return CompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

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
            return ErrorResponse(message=str(e), code=5001)

        if request_prompt_ids is not None:
            request_prompts = request_prompt_ids
        num_choices = len(request_prompts)

        api_server_logger.info(f"start inference for request {num_choices}")

        try:
            for idx, prompt in enumerate(request_prompts):
                request_id_idx = f"{request_id}-{idx}"
                api_server_logger.info(f"{prompt}")
                current_req_dict = request.to_dict_for_infer(request_id_idx, prompt)
                try:
                    self.engine_client._format_and_add_data(current_req_dict)
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
                    return ErrorResponse(code=5002, message=str(e))

        except ValueError as e:
            return ErrorResponse(message=str(e), code=5002)


    async def completion_full_generator(self,
                             request: CompletionRequest,
                             num_choices: int,
                             request_id: str,
                             created_time: int,
                             model_name: str):
        """
        Process the full completion request.
        """
        try:
            tasks = []
            for i in range(num_choices):
                choice_request_id = f"{request_id}-{i}"
                task = self._process_single_choice(choice_request_id)
                tasks.append(task)

            valid_results = await asyncio.gather(*tasks)

            return self.request_output_to_completion_response(
                final_res_batch=valid_results,
                request=request,
                request_id=request_id,
                created_time=created_time,
                model_name=model_name
            )
        except Exception as e:
            api_server_logger.error(f"Error in completion_full_generator: {e}", exc_info=True)
            raise

    async def _process_single_choice(self, choice_request_id: str):
        """
        Process a single choice of the completion request.
        """
        while True:
            while (choice_request_id not in self.engine_client.req_output or
                not self.engine_client.req_output[choice_request_id]):
                await asyncio.sleep(0.02)

            result = self.engine_client.req_output[choice_request_id].pop()
            is_end = result.finished

            if is_end:
                del self.engine_client.req_output[choice_request_id]
                processed = self.engine_client.data_processor.process_response(result)
                return processed.todict()

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
            output_tokens = [0] * num_choices
            queue = asyncio.Queue()

            async def producer(idx: int):
                """
                Produce results for a single choice of the completion request.
                """
                req_id = f"{request_id}-{idx}"
                while True:
                    while req_id not in self.engine_client.req_output or not self.engine_client.req_output[req_id]:
                        await asyncio.sleep(0.01)

                    result = self.engine_client.req_output[req_id].pop()  # 改为pop(0确保顺序
                    is_end = result.finished

                    processed = self.engine_client.data_processor.process_response(result)
                    if processed is not None:
                        output = processed.todict()
                        await queue.put({"idx": idx, "data": output, "is_end": is_end})

                    if is_end:
                        del self.engine_client.req_output[req_id]
                        break

            tasks = [asyncio.create_task(producer(idx)) for idx in range(num_choices)]

            while num_choices > 0 :
                item = await queue.get()
                res = item["data"]
                if res['metrics']['model_forward_time'] is None:
                    arrival_time = res['metrics']['first_token_time']
                else:
                    arrival_time = res['metrics']['model_forward_time']
                output = res["outputs"]
                chunk = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[CompletionResponseStreamChoice(
                        index=item["idx"],
                        text=output["text"],
                        reasoning_content=output["reasoning_content"],
                        arrival_time=arrival_time
                    )]
                )

                # 更新token计数
                output_tokens[0] += 1

                # 流式输出
                # api_server_logger.info(f"{chunk}")
                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

                if item["is_end"]:
                    num_choices -= 1
                    if getattr(request, "stream_options", None) and request.stream_options.include_usage:
                        usage_chunk = CompletionStreamResponse(
                            id=request_id,
                            created=created_time,
                            model=model_name,
                            usage=UsageInfo(
                                prompt_tokens=len(res.get("prompt_token_ids", [])),
                                completion_tokens=output_tokens[0]
                            )
                        )
                        yield f"data: {usage_chunk.model_dump_json(exclude_unset=True)}\n\n"

                    continue


                queue.task_done()

            # 所有任务完成后发送结束标记
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {ErrorResponse(message=str(e), code=5002).model_dump_json(exclude_unset=True)}\n\n"
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

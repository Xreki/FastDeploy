# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from typing import Optional, Union, cast, TypeVar, List
import uuid
from fastapi import Request

# yapf: disable
from fastdeployllm.entrypoints.openai.protocol import ErrorResponse, CompletionRequest, CompletionResponse, CompletionStreamResponse, CompletionResponseStreamChoice, CompletionResponseChoice,UsageInfo
from fastdeployllm.utils import api_server_logger 

from asyncio import FIRST_COMPLETED, AbstractEventLoop, Task
from fastdeployllm.engine.request import RequestOutput


async def async_wrapper(sync_gen):
    """正确转换同步生成器的异步包装器"""
    while True:
        try:
            # 直接传递 next 和生成器对象
            item = await asyncio.get_event_loop().run_in_executor(
                None, 
                next,  # 直接使用 next 函数
                sync_gen  # 传递生成器对象
            )
            yield item
            if item.get("finished", False):
                break
        except StopIteration:  # 显式捕获同步结束信号
            api_server_logger.info("Sync generator has been fully traversed.")
            break

class OpenAIServingCompletion:
    def __init__(self, engine_client):
        self.engine_client = engine_client

    async def create_completion(self, request: CompletionRequest):
        """重构后的异步处理方法"""
        created_time = int(time.time())
        request_id = f"cmpl-{uuid.uuid4()}"
        request_prompt_ids = None
        request_prompts = None

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

        if request_prompt_ids is not None:
            request_prompts = request_prompt_ids

        current_req_dict = request.to_dict_for_infer(request_id)
        
        try:
            # 创建异步生成器列表
            async_generators = []
            for idx, prompt in enumerate(request_prompts):
                # 创建同步生成器
                if request_prompt_ids is not None:
                    current_req_dict["prompt_token_ids"] = request_prompt_ids[idx]
                else:
                    current_req_dict["prompt"] = request_prompts[idx]
                sync_gen = self.engine_client.generate(
                    current_req_dict,
                    request.stream
                )
                # 转换为异步生成器

                async_gen = async_wrapper(sync_gen)
                async_generators.append(async_gen)

            # 合并生成器
            merged_generator = self.merge_async_generators(async_generators)

            if request.stream:
                return self.completion_stream_generator(
                    request=request,
                    generator=merged_generator,
                    request_id=request_id,
                    created_time=created_time,
                    model_name=request.model
                )
            else:
                return await self.handle_non_streaming(
                    merged_generator,
                    request,
                    request_id,
                    created_time,
                    model_name=request.model
                )

        except ValueError as e:
            return ErrorResponse(message=str(e), code=400)

    async def merge_async_generators(self, generators: List[AsyncGenerator]):
        """合并多个异步生成器为一个异步生成器"""
        task_to_index = {}
        # 初始化任务池
        for idx, gen in enumerate(generators):
            task = asyncio.create_task(gen.__anext__())
            task_to_index[task] = idx

        while task_to_index:
            # 等待至少一个任务完成
            done, pending = await asyncio.wait(
                task_to_index.keys(),
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done:
                # 获取对应的生成器索引
                idx = task_to_index.pop(task)
                
                try:
                    result = await task  # 显式获取结果
                    yield idx, result
                    
                    # 重新调度该生成器
                    new_task = asyncio.create_task(generators[idx].__anext__())
                    task_to_index[new_task] = idx
                except StopAsyncIteration:  # 正确捕获异步结束信号
                    del generators[idx]
                    api_server_logger.info("Sync generator %s has been fully traversed.", idx)
                except Exception as e:
                    # 处理其他异常
                    api_server_logger.exception(e)
        

    async def handle_non_streaming(self, 
                             generator: AsyncGenerator,
                             request: CompletionRequest,
                             request_id: str,
                             created_time: int,
                             model_name: str):
        final_outputs = dict()
        try:
            async for idx, res in generator:
                # 确保结果按生成器顺序存储
                final_outputs[idx] = res

            # 过滤可能的空值（当生成器数量不固定时）
            valid_results = [res for idx, res in final_outputs.items() if res is not None]
            
            return self.request_output_to_completion_response(
                final_res_batch=valid_results,
                request=request,
                request_id=request_id,
                created_time=created_time,
                model_name=model_name
            )
        except Exception as e:
            api_server_logger.info(f"{e}")

    async def completion_stream_generator(self,
                                        request: CompletionRequest,
                                        generator: AsyncGenerator,
                                        request_id: str,
                                        created_time: int,
                                        model_name: str):
        try:
            async for idx, res in generator:
                # 处理每个响应块
                output = res['outputs']
                chunk = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[CompletionResponseStreamChoice(
                        index=output['index'],
                        text=output['text'],
                        reasoning_content=output['reasoning_content']
                    )]
                )
                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
            
            # 最终统计信息
            if request.stream_options and request.stream_options.include_usage:
                usage_chunk = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    usage=UsageInfo(
                        prompt_tokens=sum(res['prompt_tokens']),
                        completion_tokens=sum(res['completion_tokens'])
                    )
                )
                yield f"data: {usage_chunk.model_dump_json(exclude_unset=True)}\n\n"
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"{ErrorResponse(message=str(e), code=400)}\n\n"
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


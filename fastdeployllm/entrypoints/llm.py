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
from __future__ import annotations
import sys
import traceback
import uuid
import time
from typing import Optional, Dict, List, Any, Union, overload
from tqdm import tqdm

from fastdeployllm.engine.args_utils import EngineArgs
from fastdeployllm.engine.engine import LLMEngine
from fastdeployllm.engine.sampling_params import SamplingParams

from fastdeployllm.utils import model_server_logger


class LLM:
    """
    Initializes a Language Model instance.

    Args:
        model (str):
            The name of the language model to use. Supported models are listed in
            `LLMEngine.SUPPORTED_MODELS`.
        tokenizer (Optional[str], optional):
            The name of the tokenizer to use. Defaults to None. If not specified, the
            default tokenizer for the selected model will be used.
        **kwargs (optional):
            Additional keyword arguments to pass to the `EngineArgs` constructor. See
            `EngineArgs.__init__` for details. Defaults to {}.

    Raises:
        ValueError:
            If `model` is not in `LLMEngine.SUPPORTED_MODELS`.
    """
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        **kwargs,
    ):

        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            **kwargs,
        )

        # Create the Engine
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args)

        self.default_sampling_params = SamplingParams(max_tokens = self.llm_engine.cfg.max_seq_len)

        self.llm_engine.start()


    def generate(
        self,
        prompts: Union[str, list[str], list[int], list[list[int]],
                       dict[str, Any], list[dict[str, Any]]],
        sampling_params: Optional[Union[SamplingParams,
                                        list[SamplingParams]]] = None,
        use_tqdm: bool = True,
    ):
        """
        Generate function for the LLM class.

        Args:
            prompts (Union[str, list[str], list[int], list[list[int]], dict[str, Any], list[dict[str, Any]]]):
                The prompt to use for generating the response.
            sampling_params (Optional[Union[SamplingParams, list[SamplingParams]]], optional):
                The sampling parameters to use for generating the response. Defaults to None.
            use_tqdm (bool, optional): Whether to use tqdm for the progress bar. Defaults to True.

        Returns:
            Union[str, list[str]]: The generated response.
        """

        if sampling_params is None:
            sampling_params = self.default_sampling_params

        if isinstance(sampling_params, SamplingParams):
            sampling_params_len = 1
        else:
            sampling_params_len = len(sampling_params)

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(prompts, list) and isinstance(prompts[0], int):
            prompts = [prompts]


        if isinstance(prompts, dict):
            if "prompts" not in prompts:
                raise ValueError("prompts must be a input dict")
            text = prompts.pop("prompt")
            prompts = [text]
            sampling_params = SamplingParams.from_dict(prompts)
        

        if sampling_params_len != 1 and len(prompts) != sampling_params_len:
            raise ValueError("prompts and sampling_params must be the same length.")

        req_ids = self._add_request(
            prompts=prompts,
            sampling_params=sampling_params
        )


        # get output
        outputs = self._run_engine(req_ids, use_tqdm=use_tqdm)
        return outputs




    def _add_request(
        self,
        prompts,
        sampling_params,
    ):
        """
            添加一个请求到 LLM Engine，并返回该请求的 ID。
        如果请求已经存在于 LLM Engine 中，则不会重复添加。

        Args:
            prompts (str): 需要处理的文本内容，类型为字符串。

        Returns:
            None: 无返回值，直接修改 LLM Engine 的状态。
        """
        if prompts is None:
            raise ValueError("prompts and prompt_ids cannot be both None.")

        prompts_len = len(prompts)
        req_ids = []
        for i in range(prompts_len):
            request_id = str(uuid.uuid4())
            if isinstance(prompts[i], str):
                tasks = {
                    "prompt": prompts[i],
                    "req_id": request_id,
                }
            elif isinstance(prompts[i], list) and isinstance(prompts[i][0], int):
                tasks = {
                    "prompt_token_ids": prompts[i],
                    "req_id": request_id,
                }
            elif isinstance(prompts[i], dict):
                tasks = prompts[i]
                tasks["req_id"] = request_id
            else:
                raise TypeError(
                    f"Invalid type for 'prompt': {type(prompts[i])}, expected one of ['str', 'list', 'dict']."
                )
            req_ids.append(request_id)
            if isinstance(sampling_params, list):
                sampling_params = sampling_params[i]
            self.llm_engine.add_requests(tasks, sampling_params)
        return req_ids


    def _run_engine(
        self, req_ids: list[str], use_tqdm: bool
    ):
        """
            运行引擎，并返回结果列表。

        Args:
            use_tqdm (bool, optional): 是否使用tqdm进度条，默认为False。

        Returns:
            list[Dict[str, Any]]: 包含每个请求的结果字典的列表，字典中包含以下键值对：
                    - "text": str, 生成的文本；
                    - "score": float, 得分（可选）。

        Raises:
            无。
        """
        # Initialize tqdm.

        if use_tqdm:
            num_requests = len(req_ids)
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, "
                         f"output: {0:.2f} toks/s"),
            )

        output = []
        while num_requests:
            for req_id in req_ids:
                try:
                    result = self.llm_engine.get_result(req_id)
                    if result is None:
                        time.sleep(0.01)
                        continue
                    is_end = result.finished
                    result = self.llm_engine.data_processor.process_response(result)
                    model_server_logger.debug(f"Send result to client under push mode: {result}")
                    if is_end:
                        output.append(result)
                        num_requests -= 1
                        req_ids.remove(req_id)
                        model_server_logger.debug("Request id: {} has been completed.".format(req_id))
                        if use_tqdm:
                            pbar.update(1)
                except Exception as e:
                        model_server_logger.error("Unexcepted error happend: {}".format(e))
        if use_tqdm:
            pbar.close()
        return output


if __name__ == "__main__":
    # llm = LLM(model="llama_model")
    # output = llm.generate(prompts="who are you？", use_tqdm=True)
    # print(output)
    llm = LLM(model="/opt/baidu/paddle_internal/FastDeploy/fastdeployllm/llama_model", tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=30)
    output = llm.generate(prompts="who are you？", use_tqdm=True, sampling_params=sampling_params)
    print(output)


    output = llm.generate(prompts=["who are you？", "what can you do？"], sampling_params = SamplingParams(temperature=1, max_tokens=50), use_tqdm=True)
    print(output)

    output = llm.generate(prompts=["who are you？", "what can you do？"], sampling_params = [SamplingParams(temperature=1, max_tokens=50), SamplingParams(temperature=1, max_tokens=20)], use_tqdm=True)
    print(output)

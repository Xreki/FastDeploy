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

import sys
import traceback
import uuid
import time
from typing import Optional, Dict, List, Any, Union
from tqdm import tqdm

from fastdeployllm.engine.args_utils import EngineArgs
from fastdeployllm.engine.engine import LLMEngine
# from server.engine.request import Request

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

        self.llm_engine.start()


    # self.default_sampling_params=SamplingParams()



    def generate(
        self,
        prompts: Union[str, List[int]],
        use_tqdm: bool = True,
    ):
        """Generates the completions for the input prompts.
        """



        self._add_request(
            prompts=prompts)


        # get output
        outputs = self._run_engine(use_tqdm=use_tqdm)
        return outputs




    def _add_request(
        self,
        prompts: Union[str, List[int]],
    ):
        """
            添加一个请求到 LLM Engine，并返回该请求的 ID。
        如果请求已经存在于 LLM Engine 中，则不会重复添加。
        
        Args:
            prompts (str): 需要处理的文本内容，类型为字符串。
        
        Returns:
            None: 无返回值，直接修改 LLM Engine 的状态。
        """
        request_id = str(uuid.uuid4())
        if isinstance(prompts, str):
            tasks = {
                "text": prompts,
                "req_id": request_id,
            }
        else:
            tasks = {
                "input_ids": prompts,
                "req_id": request_id,
            }

        self.llm_engine.add_requests(tasks)

    def _run_engine(
        self, *, use_tqdm: bool
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

        while self.llm_engine.unfinished_requests_num() == 0:
            time.sleep(1)
        if use_tqdm:
            num_requests = self.llm_engine.unfinished_requests_num()
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, "
                         f"output: {0:.2f} toks/s"),
            )
        output = []


        while self.llm_engine.unfinished_requests_num():
            try:
                batch_result = self.llm_engine.cached_generated_tokens.get()
                for result in batch_result:
                    is_end = result.get("is_end", 0)
                    result = self.llm_engine.data_processor.process_response(result)
                    model_server_logger.debug(f"Send result to client under push mode: {result}")
                    # TODO 输出格式
                    if is_end:
                        output.append(result)
                        if use_tqdm:
                            pbar.update(1)
            except Exception as e:
                    model_server_logger.error("Unexcepted error happend: {}, {}".format(e, str(traceback.format_exc())))
        if use_tqdm:
            pbar.close()
        return output


if __name__ == "__main__":
    llm = LLM(model="llama_model")
    output = llm.generate(prompts="who are you？", use_tqdm=True)
    print(output)
#    llm = LLM(model="llama_model", tensor_parallel_size=4)
#    output = llm.generate(prompts=[100273, 94065 , 94125 , 93983 , 93957 , 93949 , 5     , 93938 , 93939 ,
#         1859  , 94029 , 94125 , 93983 , 93978 , 93949 , 6     , 93938 , 369   ,
#         93919 , 5     , 1859  , 93964 , 94341 , 16882 , 42799 , 94735 , 94022 ,
#         2326  , 2087  , 712   , 1779  , 2254  , 7     , 1248  , 93956 , 94604 ,
#         93983 , 93939 , 93983 , 43690 , 93986 , 94110 , 269   , 94098 , 23    ,
#         23    , 93980 , 3058  , 93974 , 9128  , 93973 , 355   , 95159 , 5756  ,
#         93919 , 6     , 93983 , 408   , 95159 , 6     , 361   , 95159 , 5756  ,
#         93919 , 4     , 93983 , 412   , 95159 , 4     , 576   , 438   , 93974 ,
#         9128  , 93973 , 23    , 8296  , 94395 , 1159  , 93956 , 21558 , 3458  ,
#         1676  , 10621 , 6     , 8231  , 3240  , 94035 , 100272], use_tqdm=True)
#    print(output)

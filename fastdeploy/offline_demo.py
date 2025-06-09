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

import paddle
import torch
from fastdeploy.entrypoints.llm import LLM
from fastdeploy.engine.sampling_params import SamplingParams
model_name_or_path = "/root/paddlejob/workspace/env_run/output/kaiyuan_45Tw4a8quant"
#model_name_or_path = "/root/paddlejob/workspace/env_run/output/chenjianye/eb45t02"

batch_size=1

# 超参设置
sampling_params = SamplingParams(top_p=0, temperature=0.95, max_tokens=128)
llm = LLM(model=model_name_or_path, tensor_parallel_size=4, max_num_seqs=batch_size, num_gpu_blocks_override=1000)

input_text = ["北京天安门广场在哪里?\n"] * batch_size
#input_token = [[3991, 94112, 94286, 94398, 12877, 20298, 94009, 23]]
# for i in range(1):
#     output = llm.generate(prompts=input_text, sampling_params=sampling_params, use_tqdm=True)

# paddle.framework.core.nvprof_start()
for i in range(1):
    output = llm.generate(prompts=input_text, 
                          sampling_params=sampling_params, 
                          use_tqdm=True)
# paddle.framework.core.nvprof_stop()



print(output)


# output = llm.generate(prompts=["who are you？", "what can you do？"], use_tqdm=True)
# print(output)

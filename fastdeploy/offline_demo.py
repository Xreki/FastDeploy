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
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.entrypoints.llm import LLM

model_name_or_path = "/root/paddlejob/workspace/env_run/output/chenjianye/models/ERNIE-4.5-Lite-step-682000"

# 超参设置
sampling_params = SamplingParams(temperature=0.8, max_tokens=30)
llm = LLM(model=model_name_or_path,
          tensor_parallel_size=1,
          num_gpu_blocks_override=1500)
output = llm.generate(prompts="张三问，今天天气怎么样",
                      sampling_params=sampling_params,
                      use_tqdm=True)
print(output)

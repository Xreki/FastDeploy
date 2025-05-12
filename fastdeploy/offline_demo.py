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


from fastdeploy.entrypoints.llm import LLM
from fastdeploy.engine.sampling_params import SamplingParams
# model_name_or_path = "/home/zexuli/baidu/paddle_internal/EfficientLLM/ErnieBot-Lite"
model_name_or_path = "/home/zexuli/baidu/paddle_internal/EfficientLLM/Llama-2-7b-chat"

# 超参设置
sampling_params = SamplingParams(temperature=0.1, max_tokens=100)
llm = LLM(model=model_name_or_path, tensor_parallel_size=1,max_num_seqs=1,engine_worker_queue_port=8002)
output = llm.generate(prompts="北京天安门广场在哪里?\n",use_tqdm=True)
print(output)

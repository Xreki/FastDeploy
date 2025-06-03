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

export NVIDIA_TF32_OVERRIDE=0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
export FLAGS_enable_pir_api=1
export FLAGS_use_gemm_dequant=1
export FLAGS_use_append_attn=1
export ELLM_DYNAMIC_MODE=0

model_path=${1:-"/path/to/model"}

python export_generation_model.py \
        --model_name_or_path ${model_path} \
        --output_path ${model_path}/export \
        --use_efficientllm True \
        --dtype bfloat16 \
        # --export_model_type "w8a8c8"

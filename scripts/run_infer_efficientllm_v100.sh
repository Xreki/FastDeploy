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
export NCCL_ALGO=Tree
export FLAGS_allocator_strategy=naive_best_fit
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_gemm_use_half_precision_compute_type=False
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
export FLAGS_enable_pir_api=1

# export FLAGS_enable_blaslt_global_search=1
# exprot FLAGS_cublaslt_device_best_config=/path/to/cublaslt_device_best_config.csv

# export FLAGS_use_cutlass_device_best_config_path=/path/to/cutlass_device_best_config.json

infer_model_path=${1:-"/path/to/model"}

python infer_generation.py \
    --model_name_or_path ${infer_model_path}\
    --dtype float16\
    --input_file "./data/query-answers-list.jsonl" \
    --use_efficientllm True \

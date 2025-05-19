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
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_use_cutlass_fmha=1
export FLAGS_fmha_mode=flash_attention_v2
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
export FLAGS_dynamic_static_unified_comm=1
export FLAGS_enable_pir_api=1
export FLAGS_allocator_strategy=auto_growth
export DEVICES=0,1,2,3,4,5,6,7
# export AIPE_SECURITY_SERVER_HOST=10.151.16.26
# export FASTDEPLOY_EP_PRODUCT_NAME=safety-query-safety
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

infer_model_path=${1:-"/path/to/model"}

python -m paddle.distributed.launch \
    --gpus ${DEVICES} \
    infer_generation.py \
    --model_name_or_path ${infer_model_path} \
    --input_file "./data/query-answers-list.jsonl" \
    --dtype bfloat16 \
    --use_efficientllm True \
    --data_format "pt" \
    --append_bos_token "False" \
    --top_p 0 \
    --batch_size 2 \

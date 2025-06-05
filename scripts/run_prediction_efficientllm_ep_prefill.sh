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

rm -rf log
rm -f core*

export NVIDIA_TF32_OVERRIDE=0
export NCCL_ALGO=Tree
export NCCL_NVLS_ENABLE=0
export FLAGS_allocator_strategy=auto_growth
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_gemm_use_half_precision_compute_type=False
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
export FLAGS_enable_pir_api=0
export FLAGS_use_append_attn=1
export FLAGS_use_fa3=1

export devices=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=${devices}
export ENABLE_EFFICIENTLLM_LOAD_MODEL_CONCURRENCY=0

export PREDICT_MODEL_TYPE=${PREDICT_MODEL_TYPE:-"W8A16Cfp8"}
export MOE_QUANT_TYPE=${MOE_QUANT_TYPE-"fp8"}
export EP_SCALE_DIR=${EP_SCALE_DIR:-"/path/to/scale_dir"} # scale json文件所在的目录
# export FLAGS_enable_blaslt_global_search=1
# export FLAGS_cublaslt_device_best_config=/path/to/cublaslt_device_best_config.csv

# export FLAGS_use_cutlass_device_best_config_path=/path/to/cutlass_device_best_config.json

model_path=${1:-"/root/paddlejob/workspace/env_run/bh_test_quantized_fp8"}
# model_path=/root/paddlejob/workspace/env_run/EB45T0332kMODEL_0425_v1


for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
unset ${name}
done
export PADDLE_TRAINER_ID=0
export PADDLE_TRAINERS_NUM=1
export TRAINER_INSTANCES_NUM=1
export TRAINER_INSTANCES=`hostname -i`
self_ip=`hostname -i`

python -m paddle.distributed.launch \
        --gpus ${devices} \
        predict_generation.py \
        --model_name_or_path ${model_path} \
        --input_file "../data/query-answers-list.jsonl" \
        --output_file ./predict_out.json \
        --predict_model_type $PREDICT_MODEL_TYPE \
        --dtype bfloat16 \
        --data_format "sft" \
        --append_bos_token "False" \
        --max_dec_len ${MAX_DEC_LEN:-128} \
        --top_p 0 \
        --moe_quant_type $MOE_QUANT_TYPE \
        --use_ep "True" \
        --generation_phase 1 \
        --use_cache_kv_int8 "False" \
        --scale_dir ${EP_SCALE_DIR} \
        --use_safetensors "False" \
        --use_offline_quant "False"

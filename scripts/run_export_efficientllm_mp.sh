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
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
export export_model_type=WINT8
export FLAGS_enable_pir_api=0
# export AIPE_SECURITY_SERVER_HOST=10.151.16.26
# export FASTDEPLOY_EP_PRODUCT_NAME=safety-query-safety
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export ELLM_DYNAMIC_MODE=0

model_path=${1:-"/path/to/weight_splited_model"}
export_with_pad_vocab=True

# example for mp 2
# export device="0,1"
# export CUDA_VISIBLE_DEVICES=${device}

# python -m paddle.distributed.launch \
#     --gpus ${device} \
#     export_generation_model.py \
#     --model_name_or_path ${model_path} \
#     --output_path ${model_path}/export_wint8_tp2 \
#     --dtype bfloat16 \
#     --export_model_type=${export_model_type} \
#     --pad_vocab ${export_with_pad_vocab} \
#     --use_efficientllm True \

# example for mp4
export device="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES=${device}
export_with_pad_vocab=False

export FLAGS_use_append_attn=1
export FLAGS_enable_pir_api=1
export GLOG_v=0
python -m paddle.distributed.launch \
    --gpus ${device} \
    export_generation_model.py \
    --model_name_or_path ${model_path} \
    --output_path ${model_path}/export_wint8_efficient_pir_8k \
    --dtype bfloat16 \
    --export_model_type=${export_model_type} \
    --pad_vocab ${export_with_pad_vocab} \
    --use_efficientllm True \
    --max_seq_len 8192 \
    --max_dec_len 8192 \
    --moe_quant_type "weight_only_int4"

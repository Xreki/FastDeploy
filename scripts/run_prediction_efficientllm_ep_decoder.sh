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
set -ex

rm -rf log
rm -f core*

# export NVSHMEM_HCA_LIST=mlx5_0:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
# export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=xgbe0
echo "设置网卡信息"
export $(bash get_rdma_nics.sh gpu)
export $(bash get_rdma_nics.sh cpu)
echo "KV_CACHE_SOCKET_IFNAME=${KV_CACHE_SOCKET_IFNAME}"
export KV_CACHE_SOCKET_IFNAME=${KV_CACHE_SOCKET_IFNAME}
export NVSHMEM_HCA_LIST=${KVCACHE_RDMA_NICS}
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${KV_CACHE_SOCKET_IFNAME}
export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET
export NVSHMEM_IBGDA_NUM_RC_PER_PE=1
export NVSHMEM_SYMMETRIC_HEAP_SIZE=32G
export NVSHMEM_IB_ENABLE_IBGDA=true
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_DISABLE_P2P=true
export NVSHMEM_IBGDA_NUM_RC_PER_PE=1  # 注意这里重复了，但按原内容保留
export NVSHMEM_IB_TRAFFIC_CLASS=130
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_NVLS_ENABLE=0

export ELLM_DYNAMIC_MODE=1

export FLAGS_call_stack_level=2
export GLOG_logtostderr=true
export GLOG_v=0
export NVIDIA_TF32_OVERRIDE=0
# export NCCL_ALGO=Tree
export FLAGS_allocator_strategy=auto_growth
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_gemm_use_half_precision_compute_type=False
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
export PYTHONPATH=$(dirname $(pwd))/custom_ops/gpu_ops/fp8_deep_gemm/:$PYTHONPATH
export FLAGS_enable_pir_api=0
export FLAGS_use_append_attn=1

export devices=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=${devices}

export PREDICT_MODEL_TYPE=${PREDICT_MODEL_TYPE:-"W8A16"}
export MOE_QUANT_TYPE=${MOE_QUANT_TYPE-"fp8"}

export EP_SCALE_DIR=${EP_SCALE_DIR:-"/path/to/scale_dir"} # scale json文件所在的目录
# export CACHE_PARAMS=${CACHE_PARAMS:-"/path/to/scale_dir/cache_params.pdparams"} # cache所需要的pdparams文件路径




echo "最终的量化方式为 $PREDICT_MODEL_TYPE"

model_path=${1:-"/path/to/model"}
NUM_NODES=${2:-1}

if [ $NUM_NODES -eq 1 ];then

echo "SINGLE NODE EP"

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
        --input_file "./data/query-answers-list.jsonl" \
        --output_file ./predict_out.json \
        --predict_model_type $PREDICT_MODEL_TYPE \
        --dtype bfloat16 \
        --data_format "pt" \
        --append_bos_token "False" \
        --max_dec_len ${MAX_DEC_LEN:-128} \
        --max_seq_len ${MAX_SEQ_LEN:-8192} \
        --top_p 0 \
        --moe_quant_type $MOE_QUANT_TYPE \
        --use_ep "True" \
        --generation_phase 2 \
        --batch_size ${3:-1} \
        --use_micro_batch ${4:-"False"} \
        --use_cache_kv_int8 ${USE_CACHE_KV_INT8:-"False"} \
        --use_cache_kv_int4 ${USE_CACHE_KV_INT4:-"False"} \
        --scale_dir ${EP_SCALE_DIR}

elif [ $NUM_NODES -gt 1 ];then

echo "${NUM_NODES} NODE EP"

for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F"=" '{print $1}'`; do
unset ${name}
done

which python
IP_LIST=${5}
echo ${IP_LIST}


python -m paddle.distributed.launch \
        --nnodes ${NUM_NODES} \
        --gpus ${devices} \
        --ips ${IP_LIST} \
        predict_generation.py \
        --model_name_or_path ${model_path} \
        --input_file "./data/query-answers-list.jsonl" \
        --output_file ./predict_out.json \
        --predict_model_type $PREDICT_MODEL_TYPE \
        --dtype bfloat16 \
        --data_format "pt" \
        --append_bos_token "False" \
        --max_dec_len ${MAX_DEC_LEN:-128} \
        --max_seq_len ${MAX_SEQ_LEN:-8192} \
        --top_p 0 \
        --moe_quant_type $MOE_QUANT_TYPE \
        --use_ep "True" \
        --generation_phase 2 \
        --batch_size ${3:-1} \
        --use_micro_batch ${4:-"False"} \
        --use_cache_kv_int8 ${USE_CACHE_KV_INT8:-"False"} \
        --use_cache_kv_int4 ${USE_CACHE_KV_INT4:-"False"} \
        --scale_dir ${EP_SCALE_DIR}

fi

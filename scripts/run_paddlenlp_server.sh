#!/bin/bash

# source ./fastdeploy/agent/prepare.sh
source "$FASTDEPLOY_ENV_NAME/bin/activate"


# paddlenlp配置
export FLAGS_set_to_1d=False
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_dataloader_use_file_descriptor=False
export HF_DATASETS_DOWNLOAD_TIMEOUT=1
export FLAGS_gemm_use_half_precision_compute_type=False
export FLAGS_force_cublaslt_no_reduced_precision_reduction=True
export FLAGS_custom_allreduce=0
export FLAGS_mla_use_tensorcore=0
export FLAGS_cascade_attention_max_partition_size=2048

unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
export CUDA_VISIBLE_DEVICES=6,7

export INFERENCE_MSG_QUEUE_ID="778910"
export FD_LOG_DIR="./log_${INFERENCE_MSG_QUEUE_ID}"
export FD_MODEL_NAME="paddlenlp_model"


rm -rf ./log*

python fastdeploy/entrypoints/openai/api_server.py --config fastdeploy/agent_work.yaml --port 9809 --engine-worker-queue-port 9094 --tensor-parallel-size 2

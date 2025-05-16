#!/bin/bash

# 默认的启动worker脚本,当启动agent时没有指定 --start_job_by_agent_sh 时的默认脚本，与physical_env/start_agent_1_8 一致

export FLAGS_set_to_1d=False
export NVIDIA_TF32_OVERRIDE=0
export FLAGS_dataloader_use_file_descriptor=False
export HF_DATASETS_DOWNLOAD_TIMEOUT=1
export FLAGS_gemm_use_half_precision_compute_type=False
export FLAGS_force_cublaslt_no_reduced_precision_reduction=True
export FLAGS_custom_allreduce=0
export FLAGS_mla_use_tensorcore=0
export FLAGS_cascade_attention_max_partition_size=2048


source "/root/paddlejob/workspace/env_run/gaoziyuan/miniconda3/bin/activate" \
    "/root/paddlejob/workspace/env_run/gaoziyuan/miniconda3/envs/gaoziyuanpy310" || exit 1

cd $ROLLOUT_WORKER_ROOT

unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
export OPEN_SOURCE=1
export PYTHONPATH=/root/paddlejob/workspace/env_run/gaoziyuan/develop_nlp/PaddleNLP:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$4
export INFERENCE_MSG_QUEUE_ID=$4
export FD_MODEL_NAME="1"
export FD_LOG_DIR="./log_${INFERENCE_MSG_QUEUE_ID}"

# 注意调整 tensor-parallel-size
python fastdeploy/entrypoints/openai/api_server.py --config fastdeploy/agent_work.yaml --tensor-parallel-size $5 --port $2 --engine-worker-queue-port $3  1>> $4-stdout.log 2>> $4-stderr.log
#!/bin/bash

source "/root/paddlejob/workspace/env_run/gaoziyuan/miniconda3/bin/activate" \
    "/root/paddlejob/workspace/env_run/gaoziyuan/miniconda3/envs/gaoziyuanpy310" || exit 1

cd /root/paddlejob/workspace/env_run/gaoziyuan/gaoziyuan_fd_nlp/FastDeploy

unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
export OPEN_SOURCE=1
export PYTHONPATH=/root/paddlejob/workspace/env_run/gaoziyuan/gaoziyuan_fd_nlp/PaddleNLP:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$4
export INFERENCE_MSG_QUEUE_ID=$4
export FD_LOG_DIR="./log_${INFERENCE_MSG_QUEUE_ID}"

rm -rf ./log_${INFERENCE_MSG_QUEUE_ID}

# 注意调整 tensor-parallel-size
python fastdeploy/entrypoints/openai/api_server.py --config fastdeploy/test.yaml --tensor-parallel-size 2 --port $2 --engine-worker-queue-port $3

#!/bin/bash
rm -rf ./log*


export FLAGS_use_append_attn=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export INFERENCE_MSG_QUEUE_ID="1111" # 随便给个名字
export FD_LOG_DIR="./log_${INFERENCE_MSG_QUEUE_ID}" #不同实例的log路径
export FD_MODEL_NAME="eb45t"
# 本地测试需要
export RUN_MODE="test"

# tensor-parallel-size注意是8
python fastdeploy/entrypoints/openai/api_server.py --config training/agent_work_45T.yaml --port 9809 --engine-worker-queue-port 9091 --tensor-parallel-size 8

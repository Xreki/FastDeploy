#!/bin/bash
rm -rf ./log*

# efficient_llm 需要开启这个参数
export FLAGS_use_append_attn=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export INFERENCE_MSG_QUEUE_ID="1111" # 随便给个名字
export FD_LOG_DIR="./log_${INFERENCE_MSG_QUEUE_ID}" #不同实例的log路径
export FD_MODEL_NAME="eb45t" # 随便给个名字

# 不需要，已安装了efficientllm whl包
# export PYTHONPATH=/root/paddlejob/workspace/env_run/gaoziyuan/develop_nlp/PaddleNLP/:/root/paddlejob/workspace/env_run/gaoziyuan/develop_nlp/EfficientLLM:$PYTHONPATH

# tensor-parallel-size注意是8，yaml还使用agent_work.yaml即可
python fastdeploy/entrypoints/openai/api_server.py --config fastdeploy/test.yaml --port 9809 --engine-worker-queue-port 9091 --tensor-parallel-size 8
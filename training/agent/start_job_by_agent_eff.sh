#!/bin/bash


cd $ROLLOUT_WORKER_ROOT/training
source ${ROLLOUT_WORKER_ROOT}/training//agent/build_env.sh
source "${ROLLOUT_WORKER_ROOT}/${FASTDEPLOY_ENV_NAME}/bin/activate"

unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
export FLAGS_use_append_attn=1
export CUDA_VISIBLE_DEVICES=$4
export INFERENCE_MSG_QUEUE_ID=$4
export FD_MODEL_NAME="eb45t"
export FD_LOG_DIR="./log_${INFERENCE_MSG_QUEUE_ID}"

# 注意调整 tensor-parallel-size
python -m fastdeploy.entrypoints.openai.api_server --config agent_work_45T.yaml --tensor-parallel-size $5 --port $2 --engine-worker-queue-port $3 --metrics-port $6 1>> $4-stdout.log 2>> $4-stderr.log

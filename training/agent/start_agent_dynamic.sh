#!/bin/bash
set -x

# 环境变量设置
export ROLLOUT_WORKER_ROOT="${PWD%/*/*}"
export ROLLOUT_CONTROLLER_HOST=${rollout_controller_host:-"http://10.11.155.41:8771"}
source ${ROLLOUT_WORKER_ROOT}/training/agent/build_env.sh
source "${ROLLOUT_WORKER_ROOT}/${FASTDEPLOY_ENV_NAME}/bin/activate"

# 参数检查
if [ $# -ne 3 ]; then
    echo "用法: $0 <卡数> <实例数> <任务ID>"
    echo "卡数: 1-8, 表示每个实例使用的GPU数量"
    echo "实例数: 要启动的实例数量"
    echo "任务ID: job id"
    exit 1
fi



CARDS_PER_INSTANCE=$1
NUM_INSTANCES=$2
JOB_ID=$3

# 验证卡数参数
if [ $CARDS_PER_INSTANCE -lt 1 ] || [ $CARDS_PER_INSTANCE -gt 8 ]; then
    echo "卡数必须在1-8之间"
    exit 1
fi

cd ${ROLLOUT_WORKER_ROOT}/training/agent

# 计算需要的端口总数
TOTAL_PORTS=$((NUM_INSTANCES * 3))

# 获取空闲端口
ports=`sh get_free_ports.sh $TOTAL_PORTS`

# 启动实例
for ((i=0; i<$NUM_INSTANCES; i++)); do
    # 生成device_id字符串,每个实例递增
    DEVICE_ID=""
    for ((j=0; j<$CARDS_PER_INSTANCE; j++)); do
        curr_id=$((i * CARDS_PER_INSTANCE + j))
        if [ $j -eq 0 ]; then
            DEVICE_ID="$curr_id"
        else
            DEVICE_ID="${DEVICE_ID},$curr_id"
        fi
    done

    # 计算端口索引
    AP_IDX=$((i * 3 + 1))
    IP_IDX=$((i * 3 + 2))
    QP_IDX=$((i * 3 + 3))

    # 获取对应端口
    AP=`echo $ports | awk -v idx=$AP_IDX '{print $idx}'`
    IP=`echo $ports | awk -v idx=$IP_IDX '{print $idx}'`
    QP=`echo $ports | awk -v idx=$QP_IDX '{print $idx}'`

    # 启动agent
    python rollout-worker-agent_dynamic.py --device_id $DEVICE_ID -j $JOB_ID -p $CARDS_PER_INSTANCE -ap $AP -ip $IP -qp $QP &

    # 等待5秒确保启动
    sleep 5
done

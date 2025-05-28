#!/bin/bash

# 环境变量设置
# Check CE_root_path environment variable
if [ -n "$CE_root_path" ]; then
    export ROLLOUT_WORKER_ROOT="$CE_root_path/third_party/FastDeploy"
else
    export ROLLOUT_WORKER_ROOT=`pwd`
fi
export ROLLOUT_CONTROLLER_HOST=${rollout_controller_host:-"http://10.11.155.41:8771"}


source ${ROLLOUT_WORKER_ROOT}/fastdeploy/agent/build_env.sh
source "${ROLLOUT_WORKER_ROOT}/${FASTDEPLOY_ENV_NAME}/bin/activate"

FILES=("agent_work.yaml" "agent_work_45T.yaml")

# 写入agent_work.yaml 和 agent_work_45T.yaml 的 scheduler 段
for YAML_FILE in "${FILES[@]}"; do
  echo "处理文件: $YAML_FILE"

  SKIP_SCHEDULER=false

  # 控制开关
  if [ "$SCHEDULER_SWITCH" != "true" ]; then
    echo "SCHEDULER_SWITCH 不是 true，跳过写入 $YAML_FILE。"
    SKIP_SCHEDULER=true
  fi

  # 如果已有 scheduler 段，则跳过写入
  if grep -qE '^\s*scheduler\s*:' "$YAML_FILE"; then
    echo "文件 $YAML_FILE 中已存在 scheduler 段，跳过写入。"
    SKIP_SCHEDULER=true
  fi

  # 写入 scheduler 段
  if [ "$SKIP_SCHEDULER" != "true" ]; then
    {
      echo ""
      echo "scheduler:"
      [ -n "$SCHEDULER_NAME" ] && echo "  name: $SCHEDULER_NAME"
      [ -n "$SCHEDULER_TTL" ] && echo "  ttl: $SCHEDULER_TTL"
      [ -n "$SCHEDULER_WAIT_RESPONSE_TIMEOUT" ] && echo "  wait_response_timeout: $SCHEDULER_WAIT_RESPONSE_TIMEOUT"
      [ -n "$SCHEDULER_HOST" ] && echo "  host: $SCHEDULER_HOST"
      [ -n "$SCHEDULER_PORT" ] && echo "  port: $SCHEDULER_PORT"
      [ -n "$SCHEDULER_DB" ] && echo "  db: $SCHEDULER_DB"
      [ -n "$SCHEDULER_PASSWORD" ] && echo "  password: $SCHEDULER_PASSWORD"
      [ -n "$SCHEDULER_TOPIC" ] && echo "  topic: $SCHEDULER_TOPIC"
      [ -n "$SCHEDULER_REMOTE_WRITE_TIME" ] && echo "  remote_write_time: $SCHEDULER_REMOTE_WRITE_TIME"
    } >> "$YAML_FILE"

    echo "指定 scheduler 配置已追加到 $YAML_FILE"
  fi

  echo ""
done

# 参数检查
if [ $# -lt 3 ] || [ $# -gt 4 ]; then
    echo "用法: $0 <卡数> <实例数> <任务ID> [运行模式]"
    echo "卡数: 1-8, 表示每个实例使用的GPU数量"
    echo "实例数: 要启动的实例数量"
    echo "任务ID: job id"
    echo "运行模式: (可选) 指定运行模式，默认为空"
    exit 1
fi

CARDS_PER_INSTANCE=$1
NUM_INSTANCES=$2
JOB_ID=$3
RUN_MODE=${4:-""}  # 如果未提供第四个参数，则默认为空字符串

export RUN_MODE="$RUN_MODE"

# 验证卡数参数
if [ $CARDS_PER_INSTANCE -lt 1 ] || [ $CARDS_PER_INSTANCE -gt 8 ]; then
    echo "卡数必须在1-8之间"
    exit 1
fi

cd $ROLLOUT_WORKER_ROOT/fastdeploy/agent

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

    # 启动agent，根据MODEL环境变量选择启动脚本
    if [ "$MODEL" = "eb45t" ]; then
        python rollout-worker-agent_dynamic.py --device_id $DEVICE_ID -j $JOB_ID -p $CARDS_PER_INSTANCE -ap $AP -ip $IP -qp $QP -s start_job_by_agent_eff.sh &
    else
        python rollout-worker-agent_dynamic.py --device_id $DEVICE_ID -j $JOB_ID -p $CARDS_PER_INSTANCE -ap $AP -ip $IP -qp $QP &
    fi

    # 等待5秒确保启动
    sleep 5
done

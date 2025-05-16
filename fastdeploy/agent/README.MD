# agent/ 维护rollout-worker-agent.py工具
|-- physical_env 维护rollout-worker-agent.py启停工具，支持在物理机裸机启动，按单机启动X个实例、每个实例Y卡拆分脚本
`-- virtual_env 维护rollout-worker-agent.py启停工具，支持在虚拟环境启动，按单机启动X个实例、每个实例Y卡拆分脚本

# python文件启动命令
usage: rollout-worker-agent.py [-h] [-d DEVICE_ID] [-j JOB_ID] [-s START_JOB_BY_AGENT_SH]

这是一个Agent

options:
  -h, --help            show this help message and exit
  -d, --device_id DEVICE_ID
                        指定显卡ID
  -j, --job_id JOB_ID   job id
  -s, --start_job_by_agent_sh START_JOB_BY_AGENT_SH
                        start_job_by_agent.sh, 相对ROLLOUT_WORKER_ROOT/agent 的路径

# 脚本启动命令

sh physical_env/start_agent_1_8.sh
1 代表1卡， 8代表启动8个agent
或者
sh virtual_env/start_agent_1_8.sh

# 清理脚本
sh stop_agent.sh
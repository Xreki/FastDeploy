export ROLLOUT_WORKER_ROOT=/root/paddlejob/workspace/env_run/gaoziyuan/gaoziyuan_fd_nlp/FastDeploy
export ROLLOUT_CONTROLLER_HOST=http://10.11.155.41:8771

# set env
for i in {0..7}
do
    python agent/rollout-worker-agent.py --device_id $i &
        sleep 5
done

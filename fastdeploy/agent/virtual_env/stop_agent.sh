ps -ef | grep rollout-worker-agent | grep -v grep | awk '{print $2}' | xargs kill

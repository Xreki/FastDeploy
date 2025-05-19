#!/bin/bash

function kill_process() {
    ps -ef | grep rollout-worker-agent | grep -v grep | awk '{print $2}' | xargs kill -9
    pkill -ef '(api_server.*)'
    pkill -ef '(rollout-worker-agent.*)'
}

# kill agent worker forcely
kill_process || true

# delete instances on rollout_controller
rollout_controller_host=${rollout_controller_host:-"http://10.178.25.54:8080"}
job_id=${job_id:-"test"}
echo "Deleting job:"
curl -X DELETE \
    ${rollout_controller_host}/api/v1/job/${job_id} \
    -H "Content-Type: application/json"

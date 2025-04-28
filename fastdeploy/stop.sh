api_server_pids=$(ps auxww | grep "api_server" | grep -v grep | awk '{print $2}')
echo $api_server_pids
for in_pid in ${api_server_pids[@]}; do
    kill -9 ${in_pid}
done
echo 'end kill api server'

# coding=utf-8
# !/usr/bin/env python3

"""A simple management agent"""

from flask import Flask, request
import json
import os
import threading
import time
import requests
import subprocess
import logging
from typing import Dict, Any, Optional
import socket
import signal
import atexit
import sys
import argparse

port = 8098
app = Flask(__name__)
instance_id = "none"
device_use_id = str(0)
job_id = 1
start_job_by_agent_sh = os.getenv('START_JOB_BY_AGENT_SH', "start_job_by_agent.sh")
parallel_degree = 2

rollout_controller_host = os.getenv('ROLLOUT_CONTROLLER_HOST', "http://10.174.138.88:8839")
start_status_upload_api = f"{rollout_controller_host}/model/report_model_version"
stop_status_upload_api = f"{rollout_controller_host}/status/serving_stop"
#  /api/v{version}/job/{job_id}/instance
register_api = f"{rollout_controller_host}/api/v1/job/1/instance" 
unregister_api = f"{rollout_controller_host}/model/unregister"
registered = False

rollout_worker_host = "http://127.0.0.1"
rollout_worker_http_port = int(os.getenv('ROLLOUT_WORKER_HTTP_PORT', "8146"))
rollout_worker_queue_port = int(os.getenv('ROLLOUT_WORKER_QUEUE_PORT', "8147"))
rollout_worker_health_api = f"{rollout_worker_host}:{rollout_worker_http_port}/health"
rollout_worker_root = os.getenv('ROLLOUT_WORKER_ROOT', "/root/paddlejob/")  
#rollout_worker_yaml = "test.yaml"
#rollout_worker_start_cmd = (
#    f"python fastdeployllm/openai/api_server.py --config "
#    f"{rollout_worker_yaml} --port {rollout_worker_port}"
#)


def get_local_ip() -> str:
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logging.error(f"Failed to get local IP: {str(e)}")
        return "127.0.0.1"

def background_start(job_id: str, model_path: str, model_version: str) -> None:
    """Construct Start Command"""
    if model_path == "":
        model_path = "/"
    start_cmd = (
            f"cd {rollout_worker_root}/fastdeploy/agent && bash {start_job_by_agent_sh} "
            f"{model_path} {rollout_worker_http_port} {rollout_worker_queue_port} {device_use_id} {parallel_degree} &"
            )
    logging.info(f"check start command: {start_cmd}")
    os.system(start_cmd)

    ip = get_local_ip()

    cnt = 1
    status = 0
    while True:
        logging.info(f"check worker's starting status [{cnt} times] ...")
        try:
            response = requests.get(f"{rollout_worker_host}:{rollout_worker_http_port}/health")
            if response.status_code == 200:
                logging.info("worker is ready!")
                status = 1
                break
        except requests.exceptions.RequestException as e:
            logging.error(str(e))
        cnt += 1
        time.sleep(10)

        if cnt >= 20:
            break
    if status:
        logging.info("notify rl controller")
        try:
            headers = {
                    "Content-Type": "application/json"
                    }
            requests.put(
                f"{rollout_controller_host}/api/v1/job/{job_id}/instance/{instance_id}",
                data=json.dumps({
                    "ip": ip,
                    "infer_port": rollout_worker_http_port,
                    "agent_port": port,
                    "worker_status": "started",
                    "model_version": int(model_version)
                }),
                headers=headers)
        except requests.exceptions.RequestException as e:
            logging.error(str(e))
    else:
        logging.error("worker start failed!")

@app.route('/infer/start', methods=['POST'])
def start() -> str:
    """Start Infer Engine"""
    req = request.get_data()
    info = json.loads(req.decode('utf-8'))

    logging.info(f"receive start request: {info}")

    thread = threading.Thread(
        target=background_start,
        kwargs={
            "job_id": str(info["job_id"]),
            "model_path": str(info["model_info"]["local_path"]),
            "model_version": str(info["model_info"]["model_version"])
        }
    )
    thread.start()

    return json.dumps({'msg': 'ok', 'status': 0, 'data': {}}, indent=2)

def background_stop(job_id: str) -> None:
    """Construct Stop Command"""
    # 通过端口号找到 PID 并 kill
    # stop_cmd = f"kill -9 $(lsof -t -i:{rollout_worker_http_port})"
    # stop_cmd = f"ps -ef | grep fastdeploy | grep {rollout_worker_http_port} | awk '{{print $2}}' | xargs kill -9"
    stop_cmd = f"pkill -ef '(fastdeploy.*{rollout_worker_http_port})'"
    print(f"stop_cmd: {stop_cmd}")

    os.system(stop_cmd)

    logging.info(f"check stop command: {stop_cmd}")
    ip = get_local_ip()

    cnt = 1
    status = 0
    while True:
        logging.info(f"check worker's stopping status [{cnt} times] ...")
        try:
            # 使用 netstat 检查端口状态
            # "ps -ef | grep fastdeploy | grep -v grep | grep -v vim",
            result = subprocess.run(
                f"netstat -tunlp 2>/dev/null | grep :{rollout_worker_http_port}",
                shell=True,
                capture_output=True,
                text=True
            )
            """No Valid Process"""
            if len(result.stdout) < 2:
                status = 1
                break
        except subprocess.SubprocessError as e:
            logging.error(str(e))

        cnt += 1
        time.sleep(1)

        if cnt >= 300:
            break
    if status:
        logging.info("notify rollout controller")
        try:
            headers = {
                    "Content-Type": "application/json"
                    }
            requests.put(
                f"{rollout_controller_host}/api/v1/job/{job_id}/instance/{instance_id}",
                 data=json.dumps({
                    "ip": ip,
                    "infer_port": rollout_worker_http_port,
                    "agent_port": port,
                    "worker_status": "stopped",
                }),
                headers=headers
                )
        except requests.exceptions.RequestException as e:
            logging.error(str(e))
    else:
        logging.error("worker stop failed!")

@app.route('/infer/stop', methods=['POST'])
def stop() -> str:
    """Stop Infer Engine"""
    req = request.get_data()
    info = json.loads(req.decode('utf-8'))

    logging.info(f"receive stop request: {info}")

    thread = threading.Thread(
        target=background_stop,
        kwargs={"job_id": str(info["job_id"])}
    )
    thread.start()

    return json.dumps({'msg': 'ok', 'status': 0, 'data': {}}, indent=2)

def get_available_port() -> int:
    """Get an available port"""
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def get_unique_ports(count: int = 3) -> list:
    """获取指定数量的唯一端口
    
    Args:
        count: 需要获取的端口数量
        
    Returns:
        list: 唯一端口列表
    """
    ports = set()
    while len(ports) < count:
        p = get_available_port()
        if p > 10118:
            ports.add(p)
    return list(ports)

def register(port: Optional[int] = None, 
            http_port: Optional[int] = None, 
            queue_port: Optional[int] = None) -> bool:
    """Register to Rollout Controller synchronously"""
    try:
        # 如果没有提供端口，则生成新的端口
        if port is None:
            port = get_available_port()
        if http_port is None:
            http_port = get_available_port()
        if queue_port is None:
            queue_port = get_available_port()
            
        local_ip = get_local_ip()
        logging.info(f"Registering with IP: {local_ip}, Port: {port}, HTTP Port: {http_port}, Queue Port: {queue_port}")
        
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "ip": local_ip,
            "agent_port": port,
            "infer_port": http_port,
            "worker_status": "initialized"
        }
        
        response = requests.post(
            f"{rollout_controller_host}/api/v1/job/{job_id}/instance" ,
            data=json.dumps(data),
            headers=headers
        )
        
        if response.status_code == 200:
            logging.info(f"Successfully registered with IP: {local_ip}, Port: {port}")
             # 更新全局端口变量
            global instance_id, registered, rollout_worker_http_port, rollout_worker_queue_port
            rollout_worker_http_port = http_port
            rollout_worker_queue_port = queue_port
            registered = True
            instance_id = response.json().get('result').get('id')  
            return True
        else:
            logging.error(f"Registration failed with status code: {response.status_code}")
            return False
            
    except Exception as e:
        logging.error(f"Registration failed: {str(e)}")
        return False

def async_register(port: Optional[int] = None, 
            http_port: Optional[str] = None, 
            queue_port: Optional[str] = None) -> None:
    """Asynchronous registration with retry mechanism"""
    def _register_with_retry():
        retry_count = 0
        max_retries = 1000
        warning_threshold = 100
        retry_interval = 5  # seconds

       

        while retry_count < max_retries:
            if register(port=port, http_port=http_port, queue_port=queue_port):
                logging.info(f"Registration successful after {retry_count} attempts")   
                break
            
            retry_count += 1
            if retry_count == warning_threshold:
                logging.warning(f"Regist failed {warning_threshold} times, continue retrying")
            
            if retry_count < max_retries:
                logging.info(f"Retrying in {retry_interval} s (attempt {retry_count + 1}/{max_retries})")
                time.sleep(retry_interval)
        
        if retry_count >= max_retries:
            logging.error(f"Regist failed after {max_retries} attempts")

    # Start the registration process in a separate thread
    thread = threading.Thread(target=_register_with_retry)
    thread.daemon = True  # 设置为守护线程，这样主程序退出时线程也会退出
    thread.start()

def unregister() -> bool:
    """Unregister from Rl Controller"""
    try:
        local_ip = get_local_ip()
        logging.info(f"Unregistering with IP: {local_ip}, Port: {port}")
        
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "ip": local_ip,
            "port": port,
            "http_port": rollout_worker_http_port,
            "queue_port": rollout_worker_queue_port,
        }
        
        response = requests.delete(
            f"{rollout_controller_host}/api/v1/job/{job_id}/instance/{instance_id}",
            data=json.dumps(data),
            headers=headers
        )
        
        if response.status_code == 200:
            logging.info(f"Successfully unregistered with IP: {local_ip}, Port: {port}")
            return True
        else:
            logging.error(f"Unregistration failed with status code: {response.status_code}")
            return False
            
    except Exception as e:
        logging.error(f"Unregistration failed: {str(e)}")
        return False

def cleanup():
    """Cleanup function to be called on exit"""
    logging.info("Starting cleanup process...")
    if registered:
        unregister()
    logging.info("Cleanup completed")

def set_device_id(id: str):
    """Set device id"""
    global device_use_id
    device_use_id = id

def set_job_id(id: int):
    """Set device id"""
    global job_id
    job_id = id

def set_start_job_by_agent_sh(sh: str):
    """Set start job by agent sh"""
    global start_job_by_agent_sh
    start_job_by_agent_sh = sh

def set_parallel_degree(degree: str):
    """Set start job by agent sh"""
    global parallel_degree
    parallel_degree = degree


if __name__ == '__main__':
    # 注册清理函数
    atexit.register(cleanup)
    
    # # 注册信号处理
    signal.signal(signal.SIGTERM, lambda signo, frame: sys.exit(0))
    signal.signal(signal.SIGINT, lambda signo, frame: sys.exit(0))

    # parser = argparse.ArgumentParser(description='device id')
    # parser.add_argument('--device_id', type=int, help='device id to use')
    # args = parser.parse_args()
    # if args.device_id:
    #     device_use_id = int(args.device_id)
    #     print(f"device_use_id: {device_use_id}")
    #     set_device_id(device_use_id)

    parser = argparse.ArgumentParser(description='这是一个Agent')
    parser.add_argument('-d', '--device_id', help='指定显卡ID')
    parser.add_argument('-j', '--job_id', help='job id')
    parser.add_argument('-s', '--start_job_by_agent_sh', help='start_job_by_agent.sh, 相对ROLLOUT_WORKER_ROOT/agent 的路径')
    parser.add_argument('-p', '--parallel_degree', help='parallel_degree') 
    parser.add_argument('-ap', '--agent_port', help='agent_port') 
    parser.add_argument('-ip', '--infer_port', help='infer_port') 
    parser.add_argument('-qp', '--queue_port', help='queue_port') 
    args = parser.parse_args()
    if args.device_id:
        set_device_id(args.device_id)
    if args.job_id:
        set_job_id(args.job_id)
    if args.start_job_by_agent_sh:
        set_start_job_by_agent_sh(args.start_job_by_agent_sh)
    if args.parallel_degree:
        set_parallel_degree(args.parallel_degree)
    if args.agent_port:
        port = int(args.agent_port)
    if args.infer_port:
        rollout_worker_http_port = int(args.infer_port)
    if args.queue_port:
        rollout_worker_queue_port = int(args.queue_port)
    print(f"device_use_id: {device_use_id}, job_id: {job_id},"
            "start_job_by_agent_sh: {start_job_by_agent_sh}, parallel_degree: {parallel_degree}")
    
    # 获取三个不重复的端口
    #ports = get_unique_ports(3)
    
    #device_num = int(device_use_id.split(',')[-1])
    #port = ports[0] + device_num
    #rollout_worker_http_port = ports[1] + device_num 
    #rollout_worker_queue_port = ports[2] + device_num

    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=f'rollout-worker-agent_{device_use_id}.log',
        filemode='w'
    )
    
    print(f"port: {port}, http_port: {rollout_worker_http_port}, queue_port: {rollout_worker_queue_port}")

    logging.info(f"port: {port}, http_port: {rollout_worker_http_port}, queue_port: {rollout_worker_queue_port}")
    
    
    if not register(port=port, http_port=rollout_worker_http_port, queue_port=rollout_worker_queue_port):
        # 如果第一次失败，启动异步注册，使用相同的端口进行重试
        logging.info("First registration failed, starting asynchronous registration...")
        async_register(port=port, http_port=rollout_worker_http_port, queue_port=rollout_worker_queue_port)
    
    # debug = true 动态分配端口不生效
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False 
    )
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

# 全局变量标记start_cmd是否已执行
start_cmd_executed = False

def background_start(job_id: str, model_path: str, model_version: str) -> None:
    """Start worker by calling downstream HTTP APIs"""
    if model_path == "":
        model_path = "./"

    ip = get_local_ip()
    max_retries = 3
    retry_interval = 1  # seconds
    
    global start_cmd_executed
    if not start_cmd_executed:
        # 执行start_cmd
        start_cmd = (
            f"cd {rollout_worker_root}/fastdeploy/agent && bash {start_job_by_agent_sh} "
            f"./ {rollout_worker_http_port} {rollout_worker_queue_port} {device_use_id} {parallel_degree} &"
        )
        logging.info(f"Executing start command: {start_cmd}")
        print(f"Executing start command: {start_cmd}")
        os.system(start_cmd)
        
        # 健康检查（除非指定跳过）
        if not args.skip_health_check:
            cnt = 1
            status = 0
            # 线下测试本地加载需要43次
            while cnt <= 60:
                logging.info(f"Health check [{cnt}/20] ...")
                try:
                    response = requests.get(f"{rollout_worker_host}:{rollout_worker_http_port}/health", timeout=5)
                    if response.status_code == 200:
                        logging.info("Worker is healthy!")
                        status = 1
                        start_cmd_executed = True
                        break
                except requests.exceptions.RequestException as e:
                    logging.error(f"Health check failed: {str(e)}")
                    print(f"Health check failed: {str(e)},cnt: {cnt}")
                cnt += 1
                time.sleep(5)

            if not status:
                logging.error("Worker failed to start after 20 health checks")
                print("Worker failed to start after 20 health checks")
                stop_cmd = f"pkill -ef '(fastdeploy.*{rollout_worker_http_port})'"
                logging.info(f"Executing stop command: {stop_cmd}")
                os.system(stop_cmd)
        else:
            logging.info("Skipping health check as requested")
    
    try:
        # Call update_model_weight API with 500s timeout (不重试)
        # eb45 加载较长
        print(f"Starting worker with job_id: {job_id}, model_path: {model_path}, model_version: {model_version}")   
        update_response = requests.get(
            f"{rollout_worker_host}:{rollout_worker_http_port}/update_model_weight",
            timeout=500
        )
        
        if update_response.status_code != 200:
            logging.error(f"Failed to update model weight: {update_response.text}")
            print(f"Failed to update model weight: {update_response.text}, {update_response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling update_model_weight: {str(e)}")
        print("Failed to update model weight", str(e))
        return

    logging.info("Successfully updated model weight")
    print("Successfully updated model weight") 
    
    # 仅对通知controller的请求进行重试
    for attempt in range(max_retries):
        try:
            # Skip notifying controller if health check is skipped
            if args.skip_health_check:
                logging.info("Skipping controller notification due to skip_health_check")
                print("Skipping controller notification due to skip_health_check")
                return
                
            headers = {"Content-Type": "application/json"}
            requests.put(
                f"{rollout_controller_host}/api/v1/job/{job_id}/instance/{instance_id}",
                data=json.dumps({
                    "ip": ip,
                    "infer_port": rollout_worker_http_port,
                    "agent_port": port,
                    "worker_status": "started",
                    "model_version": int(model_version)
                }),
                headers=headers,
                timeout=10
            )
            print("Successfully notify rollout controller when reload finished")  
            return
        except requests.exceptions.RequestException as e:
            logging.error(f"Error notifying controller (attempt {attempt + 1}/{max_retries}): {str(e)}")
            print(f"Failed to notify rollout controller (attempt {attempt + 1}/{max_retries})", str(e))
            if attempt < max_retries - 1:
                time.sleep(retry_interval)



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
    """Stop worker by calling downstream HTTP APIs"""
    ip = get_local_ip()
    max_retries = 3
    retry_interval = 1  # seconds
    
    try:
        # Call clear_load_weight API with 300s timeout (不重试)
        print(f"Stopping worker with job_id: {job_id}") 
        clear_response = requests.get(
            f"{rollout_worker_host}:{rollout_worker_http_port}/clear_load_weight",
            timeout=300
        )
            
        if clear_response.status_code != 200:
            logging.error(f"Failed to clear load weight: {clear_response.text} {clear_response.status_code}")
            print(f"Failed to clear load weight: {clear_response.text}, {clear_response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling clear_load_weight: {str(e)}")
        print("Failed to clear load weight", str(e))
        return

    logging.info("Successfully cleared load weight")
    print("Successfully cleared load weight")   
    
    # 仅对通知controller的请求进行重试
    for attempt in range(max_retries):
        try:
            headers = {"Content-Type": "application/json"}
            requests.put(
                f"{rollout_controller_host}/api/v1/job/{job_id}/instance/{instance_id}",
                data=json.dumps({
                    "ip": ip,
                    "infer_port": rollout_worker_http_port,
                    "agent_port": port,
                    "worker_status": "stopped",
                }),
                headers=headers,
            )
            return
        except requests.exceptions.RequestException as e:
            logging.error(f"Error notifying controller (attempt {attempt + 1}/{max_retries}): {str(e)}")
            print(f"Failed to notify rollout controller (attempt {attempt + 1}/{max_retries})", str(e))
            if attempt < max_retries - 1:
                time.sleep(retry_interval)


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
    
    # 执行stop_cmd
    stop_cmd = f"pkill -ef '(fastdeploy.*{rollout_worker_http_port})'"
    logging.info(f"Executing stop command: {stop_cmd}")
    os.system(stop_cmd)
    
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
    
    # 注册信号处理
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
    parser.add_argument('-skip', '--skip-health-check', action='store_true', help='跳过健康检查')

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

    print(f"device_use_id: {device_use_id},\n"
             f"job_id: {job_id},\n"
             f"start_job_by_agent_sh: {start_job_by_agent_sh},\n"
             f"parallel_degree: {parallel_degree}")

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

    logging.info("Starting service")
    
    
    if not register(port=port, http_port=rollout_worker_http_port, queue_port=rollout_worker_queue_port):
        # 如果第一次失败，启动异步注册，使用相同的端口进行重试
        logging.info("First registration failed, starting asynchronous registration...")
        async_register(port=port, http_port=rollout_worker_http_port, queue_port=rollout_worker_queue_port)
    
    # # 启动健康监控线程（除非跳过健康检查）
    # if not args.skip_health_check:
    #     def health_monitor():
    #         """Monitor worker health and restart if needed"""
    #         failure_count = 0
    #         max_failures = 5
    #         check_interval = 10  # seconds
            
    #         while True:
    #             try:
    #                 response = requests.get(
    #                     f"{rollout_worker_host}:{rollout_worker_http_port}/health", 
    #                     timeout=5
    #                 )
    #                 if response.status_code == 200:
    #                     failure_count = 0
    #                     logging.info("Worker health check passed")
    #                 else:
    #                     failure_count += 1
    #                     logging.warning(f"Worker health check failed (count: {failure_count})")
    #             except requests.exceptions.RequestException as e:
    #                 failure_count += 1
    #                 logging.warning(f"Worker health check error (count: {failure_count}): {str(e)}")
                
    #             if failure_count >= max_failures:
    #                 logging.error("Worker unhealthy, attempting to restart...")
    #                 # 执行清理
    #                 # stop_cmd = f"pkill -ef '(fastdeploy.*{rollout_worker_http_port})'"
    #                 # logging.info(f"Executing stop command: {stop_cmd}")
    #                 # os.system(stop_cmd)

    #                 # 重新启动
    #                 logging.info(f"Executing restart command: {start_cmd}")
    #                 os.system(start_cmd)
    #                 failure_count = 0
                
    #             time.sleep(check_interval)
        
    #     monitor_thread = threading.Thread(target=health_monitor)
    #     monitor_thread.daemon = True
    #     monitor_thread.start()
    #     logging.info("Health monitor thread started")
    # else:
    #     logging.info("Skipping health monitor thread as requested")

    # debug = true 动态分配端口不生效
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False 
    )
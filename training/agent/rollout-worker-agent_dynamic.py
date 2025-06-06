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
from subprocess import Popen
import multiprocessing
import yaml
import glob
import tempfile

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
metrics_port = int(8000)
rollout_worker_health_api = f"{rollout_worker_host}:{rollout_worker_http_port}/health"
rollout_worker_root = os.getenv('ROLLOUT_WORKER_ROOT', "/root/paddlejob/")
# rollout_worker_yaml = "test.yaml"
# rollout_worker_start_cmd = (
#    f"python fastdeployllm/openai/api_server.py --config "
#    f"{rollout_worker_yaml} --port {rollout_worker_port}"
# )
# job_id -> Process
update_procs = {}
kill_cmd = "lsof /dev/nvidia* | awk '{print $2}' | xargs -I {} kill -9 {}"


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


def fault_tolerance(job_id):
    """fault tolerance for oom error"""
    logging.error("Worker Exited unexpectedly or timed out, start fault tolerance...")
    print("Worker Exited unexpectedly or timed out, start fault tolerance...")
    global start_cmd_executed
    start_cmd_executed = False
    os.system(kill_cmd)
    notice_controller(job_id, "", "stopped", "fault_tolerance_stop")


def init_child_logger(log_file: str):
    """Initialize child logger"""
    # 清空原有的 handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a'
    )


def update_weight_and_controller(job_id: str, model_path: str, model_version: str) -> None:
    """Update the weights of a model and notify the controller that it has been successfully loaded."""
    if model_path == "":
        model_path = "./"
    init_child_logger(f'rollout-worker-agent_child_{device_use_id}.log')

    try:
        # Call update_model_weight API with 300s timeout (不重试)
        # eb45 加载较长
        print(f"Starting worker with job_id: {job_id}, model_path: {model_path}, model_version: {model_version}")
        logging.info(f"Starting worker with job_id: {job_id}, model_path: {model_path}, model_version: {model_version}")
        update_response = requests.get(
            f"{rollout_worker_host}:{rollout_worker_http_port}/update_model_weight",
            timeout=300
        )

        if update_response.status_code != 200:
            logging.error(f"Failed to update model weight: {update_response.text}")
            print(f"Failed to update model weight: {update_response.text}, {update_response.status_code}")
            return
    except requests.exceptions.Timeout as e:
        logging.error(f"calling update_model_weight timeout: {str(e)}")
        print(f"calling update_model_weight timeout: {str(e)}")
        fault_tolerance(job_id)
        return
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling update_model_weight: {str(e)}")
        print("Failed to update model weight", str(e))
        return
    except Exception as e:
        logging.error(f"Unknown error occurred while updating model weight: {str(e)}")
        print("Failed to update model weight", str(e))
        return

    logging.info("Successfully updated model weight")
    print("Successfully updated model weight")

    # 仅对通知controller的请求进行重试
    notice_controller(job_id, model_version, "started", "")


def notice_controller(job_id: str, model_version: str, status: str, reason: str) -> None:
    """Notify the controller about the current state of the worker"""
    ip = get_local_ip()
    max_retries = 3
    retry_interval = 1  # seconds
    # 仅对通知controller的请求进行重试
    for attempt in range(max_retries):
        try:
            # Skip notifying controller if health check is skipped
            if args.skip_health_check:
                logging.info("Skipping controller notification due to skip_health_check")
                print("Skipping controller notification due to skip_health_check")
                return

            headers = {"Content-Type": "application/json"}
            data = json.dumps({
                "ip": ip,
                "infer_port": rollout_worker_http_port,
                "agent_port": port,
                "worker_status": status,
                "reason": reason
            })
            try:
                model_version = int(model_version)
                data = json.dumps({
                    "ip": ip,
                    "infer_port": rollout_worker_http_port,
                    "agent_port": port,
                    "worker_status": status,
                    "model_version": model_version,
                    "reason": reason
                })
            except ValueError:
                pass

            requests.put(
                f"{rollout_controller_host}/api/v1/job/{job_id}/instance/{instance_id}",
                data=data,
                headers=headers,
                timeout=10
            )
            logging.info("Successfully notify rollout controller when reload finished")
            print("Successfully notify rollout controller when reload finished")
            return
        except requests.exceptions.RequestException as e:
            logging.error(f"Error notifying controller (attempt {attempt + 1}/{max_retries}): {str(e)}")
            print(f"Failed to notify rollout controller (attempt {attempt + 1}/{max_retries})", str(e))
            if attempt < max_retries - 1:
                time.sleep(retry_interval)


def monitor_worker(job_id: str, proc: Popen, model_version: str, stop_event: threading.Event):
    """Monitor the worker process and report its status"""
    # 每隔 1s 检查一次进程状态
    try:
        while not stop_event.is_set():
            ret = proc.poll()
            if ret is not None:
                # ret 就是子进程的退出码
                logging.error(f"[{job_id}] Worker exited with error (code={ret})")
                time.sleep(5)
                # 仅对通知controller的请求进行重试
                notice_controller(job_id, model_version, "stopped", "abnormal_stop")
                global start_cmd_executed
                start_cmd_executed = False
                break
            time.sleep(30)
    except Exception as e:
        logging.error(f"Monitor worker thread failed: {e}")

def health_check() -> bool:
    """Perform health check on the worker process"""
    try:
        response = requests.get(f"{rollout_worker_host}:{rollout_worker_http_port}/health", timeout=10)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Health check failed: {str(e)}")
    return False


worker_proc = None
worker_monitor_thread = None
worker_stop_event = threading.Event()

def cleanup_worker(job_id):
    """clean up rollout worker process"""
    global worker_proc, worker_monitor_thread, worker_stop_event
    #  监控线程
    if worker_monitor_thread and worker_monitor_thread.is_alive():
        worker_stop_event.set()
        worker_monitor_thread.join(timeout=5)

    #  推理进程
    if worker_proc and worker_proc.poll() is None:
        logging.info("Killing previous worker process")
        os.killpg(os.getpgid(worker_proc.pid), signal.SIGTERM)
        try:
            worker_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logging.warning("Worker process did not exit in time")

    #  更新进程
    p = update_procs.pop(job_id, None)
    if p and p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            os.kill(p.pid, signal.SIGKILL)

def background_start(job_id: str, model_path: str, model_version: str, modify_max_model_len: bool) -> None:
    """Start worker by calling downstream HTTP APIs"""
    global start_cmd_executed
    try:
        #  如果max_model_len被修改，需要重启进程
        if modify_max_model_len and start_cmd_executed:
            cleanup_worker(job_id)
            start_cmd_executed = False
    except Exception as e:
        logging.error(f"Failed to restart worker process: {str(e)}")

    logging.info(f"Creating background_start thread, start_cmd status is {start_cmd_executed}")
    if not start_cmd_executed:
        # 执行start_cmd
        start_cmd = [
            "bash",
            f"{rollout_worker_root}/training/agent/{start_job_by_agent_sh}",
            "./",
            str(rollout_worker_http_port),
            str(rollout_worker_queue_port),
            device_use_id,
            str(parallel_degree),
            str(metrics_port)
        ]
        # Popen 时加 preexec_fn=os.setsid，让它在新的进程组里启动
        proc = Popen(start_cmd, cwd=f"{rollout_worker_root}/training/agent",
                     preexec_fn=os.setsid)
        global worker_proc
        worker_proc = proc
        logging.info(f"Executing start command: {start_cmd}")
        print(f"Executing start command: {start_cmd}")
        start_cmd_executed = True

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
                start_cmd_executed = False
        else:
            logging.info("Skipping health check as requested")

        # 启动一个守护线程，监听进程，并上报
        logging.info(f"Creating monitor_worker thread")
        global worker_monitor_thread
        global worker_stop_event
        worker_stop_event = threading.Event()
        worker_monitor_thread = threading.Thread(
            target=monitor_worker,
            kwargs={
                "job_id": job_id,
                "proc": proc,
                "model_version": model_version,
                "stop_event": worker_stop_event
            },
            daemon=True
        )
        worker_monitor_thread.start()

    p = update_procs.pop(job_id, None)
    if p and p.is_alive():
        logging.info(f"Terminating subprocess")
        p.terminate()
        p.join(5)
        if p.is_alive():
            logging.info(f"Terminating subprocess by os")
            os.kill(p.pid, signal.SIGKILL)

    logging.info(f"Creating update_weight_and_controller subprocess")
    p = multiprocessing.Process(
        target=update_weight_and_controller,
        args=(job_id, model_path, model_version),
        daemon=True
    )
    p.start()
    update_procs[job_id] = p


max_model_len = 0

@app.route('/infer/start', methods=['POST'])
def start() -> str:
    """Start Infer Engine"""
    req = request.get_data()
    info = json.loads(req.decode('utf-8'))

    logging.info(f"receive start request: {info}")
    modify_max_model_len = False
    try:
        global max_model_len
        new_max_model_len = int(info["max_model_len"])
        if max_model_len != new_max_model_len:
            set_max_model_len(new_max_model_len)
            modify_max_model_len = True
            max_model_len = new_max_model_len
    except Exception as e:
        logging.error(f"set max_model_len failed: {str(e)}")

    thread = threading.Thread(
        target=background_start,
        kwargs={
            "job_id": str(info["job_id"]),
            "model_path": str(info["model_info"]["local_path"]),
            "model_version": str(info["model_info"]["model_version"]),
            "modify_max_model_len": modify_max_model_len
        }
    )
    thread.start()

    ret = json.dumps({'msg': 'ok', 'status': 0, 'data': {}}, indent=2)
    logging.info(f"send start response: {ret}")
    return ret


def background_stop(job_id: str) -> None:
    """Stop worker by calling downstream HTTP APIs"""
    max_retries = 3
    try:
        # Call clear_load_weight API with 30s timeout (重试3次)
        print(f"Stopping worker with job_id: {job_id}")
        logging.info(f"Stopping worker with job_id: {job_id}")
        cnt = 0
        # health check
        is_health = health_check()
        if not is_health:
            fault_tolerance(job_id)
            return

        while cnt < max_retries:
            clear_response = requests.get(
                f"{rollout_worker_host}:{rollout_worker_http_port}/clear_load_weight",
                timeout=300
            )

            if clear_response.status_code != 200:
                logging.error(f"Failed to clear load weight: {clear_response.text} {clear_response.status_code}")
                print(f"Failed to clear load weight: {clear_response.text}, {clear_response.status_code}")
            else:
                logging.info("Successfully cleared load weight")
                print("Successfully cleared load weight")
                # 仅对通知controller的请求进行重试
                notice_controller(job_id, "", "stopped", "normal_stop")
                return
            cnt += 1
    except requests.exceptions.Timeout as e:
        logging.error(f"calling clear_load_weight timeout: {str(e)}")
        print(f"calling clear_load_weight timeout: {str(e)}")
        fault_tolerance(job_id)
        return
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling clear_load_weight: {str(e)}")
        print("Failed to clear load weight", str(e))
        return


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

    logging.info(f"clear subprocess of current job")
    p = update_procs.pop(str(info["job_id"]), None)
    if p and p.is_alive():
        logging.info(f"Terminating subprocess")
        p.terminate()
        p.join(5)
        if p.is_alive():
            logging.info(f"Terminating subprocess by os")
            os.kill(p.pid, signal.SIGKILL)
    logging.info(f"clear done")

    ret = json.dumps({'msg': 'ok', 'status': 0, 'data': {}}, indent=2)
    logging.info(f"send stop response: {ret}")
    return ret


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
            f"{rollout_controller_host}/api/v1/job/{job_id}/instance",
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


def set_max_model_len(max_model_len: int):
    """Set max model length in all agent_work YAML files"""
    try:
        pattern = os.path.join(rollout_worker_root, "training", "agent_work*.yaml")
        yaml_files = glob.glob(pattern)

        if not yaml_files:
            print(f"未找到匹配的 YAML 文件：{pattern}")
            logging.warning(f"未找到匹配的 YAML 文件：{pattern}")
            return

        for yaml_path in yaml_files:
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)

                if not isinstance(data, dict):
                    print(f"{yaml_path} 内容不是字典，跳过")
                    logging.warning(f"{yaml_path} 内容不是字典，跳过")
                    continue

                if 'max_model_len' in data:
                    old_value = data['max_model_len']
                    data['max_model_len'] = max_model_len
                    print(f"{yaml_path}: max_model_len 从 {old_value} 修改为 {max_model_len}")
                    logging.info(f"{yaml_path}: max_model_len 从 {old_value} 修改为 {max_model_len}")
                else:
                    print(f"{yaml_path}: 不存在 max_model_len 字段，跳过")
                    logging.info(f"{yaml_path}: 不存在 max_model_len 字段，跳过")
                    continue

                # 原子写入，防止并发读到一半的内容
                atomic_write_yaml(data, yaml_path)

            except Exception as fe:
                print(f"{yaml_path}: 修改失败 - {str(fe)}")
                logging.error(f"{yaml_path}: 修改失败 - {str(fe)}")

    except Exception as e:
        print(f"Set max_model_len failed: {str(e)}")
        logging.error(f"Set max_model_len failed: {str(e)}")

def atomic_write_yaml(data, target_path):
    """原子方式写入 YAML 文件，防止写一半被读到"""
    dir_name = os.path.dirname(target_path)
    with tempfile.NamedTemporaryFile('w', delete=False, dir=dir_name) as tf:
        yaml.dump(data, tf, default_flow_style=False, sort_keys=False)
        temp_name = tf.name
    os.replace(temp_name, target_path)  # 原子替换

if __name__ == '__main__':
    # multiprocessing.set_start_method("spawn", force=True)
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
    parser.add_argument('-s', '--start_job_by_agent_sh',
                        help='start_job_by_agent.sh, 相对ROLLOUT_WORKER_ROOT/agent 的路径')
    parser.add_argument('-p', '--parallel_degree', help='parallel_degree')
    parser.add_argument('-ap', '--agent_port', help='agent_port')
    parser.add_argument('-ip', '--infer_port', help='infer_port')
    parser.add_argument('-qp', '--queue_port', help='queue_port')
    parser.add_argument('-mp', '--metrics_port', help='metrics_port')
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
    if args.metrics_port:
        metrics_port = int(args.metrics_port)

    print(f"device_use_id: {device_use_id},\n"
          f"job_id: {job_id},\n"
          f"start_job_by_agent_sh: {start_job_by_agent_sh},\n"
          f"parallel_degree: {parallel_degree}")

    # 获取三个不重复的端口
    # ports = get_unique_ports(3)

    # device_num = int(device_use_id.split(',')[-1])
    # port = ports[0] + device_num
    # rollout_worker_http_port = ports[1] + device_num
    # rollout_worker_queue_port = ports[2] + device_num

    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=f'rollout-worker-agent_{device_use_id}.log',
        filemode='a'
    )

    print(f"port: {port}, http_port: {rollout_worker_http_port}, "
          f"queue_port: {rollout_worker_queue_port}, metrics_port: {metrics_port}")

    logging.info(f"port: {port}, http_port: {rollout_worker_http_port}"
                 f", queue_port: {rollout_worker_queue_port}, metrics_port: {metrics_port}")

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
        debug=False,
        use_reloader=False
    )

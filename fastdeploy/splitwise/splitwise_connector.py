"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import threading
import time
import os
import zmq
from concurrent.futures import ThreadPoolExecutor

from fastdeploy.inter_communicator import EngineWorkerQueue
from fastdeploy.utils import get_logger

logger = get_logger("splitwise_connector", "splitwise_connector.log")


class SplitwiseConnector:
    """

    """
    def __init__(self, cfg, scheduler, worker_queue, resource_manager):
        self.cfg = cfg
        self.scheduler = scheduler
        self.engine_worker_queue = worker_queue
        self.resource_manager = resource_manager
        self.connect_innode_instances = {}
        self.temp_cache_info = dict()

    def send_splitwise_tasks(self, tasks):
        """
        send splitwise tasks to specific port, temporary dispatch splitwise tasks
        """
        tasks_status = "mixed"
        current_port = -1
        is_changable = os.getenv("PD_CHANGEABLE", "0") == "1"
        while True:
            for port in self.cfg.innode_prefill_ports:
                current_port = self.send_splitwise_tasks_innode(tasks, port)
                if current_port != -1:
                    tasks_status = "decode"
                    break
            if current_port != -1 or is_changable:
                break
            else:
                time.sleep(0.005)
                   
        if tasks_status == "decode":
            for task in tasks:
                task.disaggregate_info = {"role": tasks_status, "port": current_port}

    def send_splitwise_tasks_innode(self, tasks, port):
        """
        send splitwise tasks to specific port
        """
        current_port = -1
        if port not in self.connect_innode_instances:
            self.create_connection(port)
        if self.connect_innode_instances[port].get_prefill() == 1:
            for task in tasks:
                task.disaggregate_info = {"role": "prefill", "port": self.cfg.engine_worker_queue_port}
            self.connect_innode_instances[port].put_splitwise_tasks(("prefill", tasks))
            current_port = port
        return current_port


    def send_first_token(self, port, tasks_list):
        """
        send first token to specific port
        """
        if port not in self.connect_innode_instances:
            self.create_connection(port)
        self.connect_innode_instances[port].put_splitwise_tasks(("decode", tasks_list))

    def create_connection(self, port):
        """
        create connection to specific port
        """
        self.connect_innode_instances[port] = EngineWorkerQueue(
                                        address=("0.0.0.0", int(port)),
                                        num_client=self.cfg.tensor_parallel_size, 
                                        client_id=0)

    def send_cache_infos(self, tasks):
        """
        send cache info to specific port
        """
        is_decode = False
        temp_cache_info = dict()
        for i in range(len(tasks)):
            if tasks[i].disaggregate_info is None:
                continue
            logger.info(f"{tasks[i].disaggregate_info}")
            if tasks[i].disaggregate_info["role"] == "decode":
                if tasks[i].disaggregate_info["port"] not in temp_cache_info:
                    temp_cache_info[tasks[i].disaggregate_info["port"]] = [{
                        "request_id": tasks[i].request_id,
                        "device_ids": self.cfg.device_ids.split(","),
                        "dest_block_ids": tasks[i].disaggregate_info["block_tables"],
                    }]
                else:
                    temp_cache_info[tasks[i].disaggregate_info["port"]].append({
                        "request_id": tasks[i].request_id,
                        "device_ids": self.cfg.device_ids.split(","),
                        "dest_block_ids": tasks[i].disaggregate_info["block_tables"],
                    })
                is_decode = True
            else:
                if tasks[i].disaggregate_info["port"] not in temp_cache_info:
                    temp_cache_info[tasks[i].disaggregate_info["port"]] = [{
                        "request_id": tasks[i].request_id,
                        "src_block_ids": tasks[i].block_tables,
                    }]
                else:
                    temp_cache_info[tasks[i].disaggregate_info["port"]].append({
                        "request_id": tasks[i].request_id,
                        "src_block_ids": tasks[i].block_tables,
                    })

        if not is_decode and len(temp_cache_info):
            for k, v in temp_cache_info.items():
                self.engine_worker_queue.put_cache_info(v)
        else:
            for k, v in temp_cache_info.items():
                self.connect_innode_instances[k].put_cache_info(v)
        return is_decode


"""
跨机跨进程通信队列
"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import os
import threading
import socket
import time
import numpy as np
from multiprocessing.managers import (AcquirerProxy, BaseManager, ListProxy,
                                      Value, ValueProxy)
from queue import Queue
from fastdeployllm.utils import model_server_logger


class EngineWorkerQueue:
    """
    Cross-machine and cross-process communication queue between Engine and Worker.
    Manages shared resources using multiprocessing managers for inter-process communication.
    """

    def __init__(self,
                 address=('0.0.0.0', 5000),
                 authkey=b'secret_key',
                 is_server=False,
                 num_client=1,
                 client_id=-1):
        """
        Initialize the communication queue.

        Args:
            address (tuple): Network address (IP, port) for the queue server
            authkey (bytes): Authentication key for secure connection
            is_server (bool): Whether this instance acts as a server
            num_client (int): Total number of expected clients
            client_id (int): Unique identifier for client instances
        """
        self.address = address
        self.authkey = authkey
        self.num_client = num_client
        self.client_id = client_id

        # Custom QueueManager for proxy object registration
        class QueueManager(BaseManager):
            pass

        if is_server:
            # Server-side initialization for shared resources
            self.tasks_init = list()
            self.client_read_flag_init = [1] * self.num_client
            self.lock_init = threading.Lock()
            self.read_finish_flag_init = Value("i", 0)
            self.connected_client_counter_init = Value("i", 0)

            # Register shared objects with proxy types
            QueueManager.register("get_tasks",
                                  callable=lambda: self.tasks_init,
                                  proxytype=ListProxy)
            QueueManager.register("get_client_read_flag",
                                  callable=lambda: self.client_read_flag_init,
                                  proxytype=ListProxy)
            QueueManager.register("get_lock",
                                  callable=lambda: self.lock_init,
                                  proxytype=AcquirerProxy)
            QueueManager.register("get_read_finish_flag",
                                  callable=lambda: self.read_finish_flag_init,
                                  proxytype=ValueProxy)
            QueueManager.register(
                "get_connected_client_counter",
                callable=lambda: self.connected_client_counter_init,
                proxytype=ValueProxy)

            self.manager = QueueManager(address=self.address,
                                        authkey=self.authkey)
            self.manager.start()
        else:
            # Client-side connection setup
            assert self.client_id >= 0 and self.client_id < self.num_client, (
				f"self.client_id={self.client_id}, self.num_client={self.num_client}")
            QueueManager.register("get_tasks")
            QueueManager.register("get_client_read_flag")
            QueueManager.register("get_lock")
            QueueManager.register("get_read_finish_flag")
            QueueManager.register("get_connected_client_counter")
            self.manager = QueueManager(address=self.address,
                                        authkey=self.authkey)
            self._connect_with_retry()

        # Get proxy objects for shared resources
        self.tasks = self.manager.get_tasks()
        self.client_read_flag = self.manager.get_client_read_flag()
        self.lock = self.manager.get_lock()
        self.read_finish_flag = self.manager.get_read_finish_flag()
        self.connected_client_counter = self.manager.get_connected_client_counter(
        )
        assert self.num_client == len(self.client_read_flag)

        if is_server:
            model_server_logger.info(f"EngineWorkerQueue server started.")
        else:
            # Update client connection counter
            self.lock.acquire()
            self.connected_client_counter.set(
                self.connected_client_counter.get() + 1)
            self.lock.release()
            model_server_logger.info((
         		f"Connected EngineWorkerQueue client_id: {self.client_id}, number "
         		f"of connected clients: {self.connected_client_counter.get()}"
            ))

    def _connect_with_retry(self, max_retries=5, interval=3):
        """
        Connect to the server with retry mechanism.

        Args:
            max_retries (int): Maximum connection attempts
            interval (int): Retry interval in seconds

        Raises:
            ConnectionError: If all connection attempts fail
        """
        for _ in range(max_retries):
            try:
                self.manager.connect()
                return
            except ConnectionRefusedError:
                time.sleep(interval)
        raise ConnectionError(f"TaskQueue cannot connect {self.address}")

    def put_tasks(self, tasks):
        """
        Add tasks to the shared queue in a thread-safe manner.
        Waits until all clients have read previous tasks before adding new ones.

        Args:
            tasks (list): Tasks to be added to the queue
        """
        self.lock.acquire()
        while sum(self.client_read_flag) < self.num_client:
            self.lock.release()
            time.sleep(0.001)
            self.lock.acquire()

        self.tasks[:] = list()
        self.client_read_flag[:] = [0] * self.num_client
        self.tasks.append(tasks)
        self.lock.release()

    def get_tasks(self):
        """
        Retrieve tasks from the shared queue and update read status.

        Returns:
            tuple: (list of tasks, bool indicating if all clients have read)
        """
        tasks = list()
        self.lock.acquire()
        tasks.extend(self.tasks)
        self.client_read_flag[self.client_id] = 1
        all_client_read = np.sum(self.client_read_flag) == self.num_client
        if all_client_read:
            self.tasks[:] = list()
        self.lock.release()
        return tasks, all_client_read

    def num_tasks(self):
        """
        Get current number of tasks in the queue.

        Returns:
            int: Total number of tasks
        """
        self.lock.acquire()
        total_num = len(self.tasks)
        self.lock.release()
        return total_num

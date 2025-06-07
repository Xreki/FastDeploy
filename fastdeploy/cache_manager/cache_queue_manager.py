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

import os
import threading
import time
from multiprocessing.managers import (
    AcquirerProxy,
    BaseManager,
    ListProxy,
    Value,
    ValueProxy,
)
from queue import Queue

from fastdeploy.utils import get_logger

logger = get_logger("cache_queue_manager", "cache_queue_manager.log")


class QueueManager(BaseManager):
    """
    BaseManager
    """
    pass

class CacheQueueManager(object):
    """
    multiprocessing manager for cache queue
    """

    def __init__(self, rank=0, mp_num=8, port=56666):
        """
        init
        """
        QueueManager.register("get_transfer_task_queue") 
        QueueManager.register("get_tansfer_done_queue") 
        QueueManager.register("get_cache_sync_value") 
        QueueManager.register("get_transfer_task_lock")
        QueueManager.register("get_transfer_task_done_lock")
        QueueManager.register('get_barrier1')
        QueueManager.register('get_barrier2')
        QueueManager.register('get_barrier3')
        QueueManager.register('get_swap_to_cpu_barrier1')
        QueueManager.register('get_swap_to_cpu_barrier2')
        QueueManager.register('get_swap_to_gpu_barrier1')
        QueueManager.register('get_swap_to_gpu_barrier2')


        self.client_manager = QueueManager(
            address=("127.0.0.1", port), authkey=b"cache_queue_service"
        )
        self.client_manager.connect()
        self.mp_num = mp_num
        self.rank = rank
        self.position = 1 << rank
        self.total_num = (1 << self.mp_num) - 1

        # Swap task
        self.transfer_task_queue = self.client_manager.get_transfer_task_queue()
        self.tansfer_done_queue = self.client_manager.get_tansfer_done_queue()
        self.task_sync_value = self.client_manager.get_cache_sync_value()
        self.task_lock = self.client_manager.get_transfer_task_lock()
        self.task_done_lock = self.client_manager.get_transfer_task_done_lock()
        self.barrier1 = self.client_manager.get_barrier1()
        self.barrier2 = self.client_manager.get_barrier2()
        self.barrier3 = self.client_manager.get_barrier3()

        logger.info(f"init cache queue manager successful, rank: {rank}")

        # completion sync flags
        

    def put_transfer_task(self, item):
        """
        put swap task
        """
        self.task_lock.acquire()
        if 0 < self.task_sync_value.get() < self.total_num:
            self.task_lock.release()
            while 0 < self.task_sync_value.get() < self.total_num:
                time.sleep(0.001)
            self.task_lock.acquire()
        self.task_sync_value.set(0)
        self.transfer_task_queue.append(item)
        logger.info(f"put_transfer_task: put swap task {item[-1]} to queue successful")
        self.task_lock.release()

    def get_transfer_task(self):
        """
        get swap task
        """
        data = None
        read_finish = False
        self.task_lock.acquire()
        if (
            self.task_sync_value.get() & self.position == 0
            and len(self.transfer_task_queue) > 0
        ):
            data = self.transfer_task_queue[0]
            logger.debug(
                f"get_transfer_task: Get {data} by {self.rank} from queue successful"
            )
            set_value = self.task_sync_value.get() | self.position
            logger.info("get_transfer_task: rank: {0} set_value: {1}".format(self.rank, set_value))
            if set_value >= self.total_num:
                self.transfer_task_queue.pop(0)
                set_value = 0
                read_finish = True
            self.task_sync_value.set(set_value)
        self.task_lock.release()
        return data, read_finish

    def put_transfer_done_signal(self, item):
        """
        put swap result
        """
        self.task_done_lock.acquire()
        self.tansfer_done_queue.append(item)
        self.task_done_lock.release()
        logger.info(f"put_transfer_done_signal: put swap task {item[-1]} finished signal to queue successful")

    def get_transfer_done_signal(self):
        """
        get swap result
        """
        data = None
        self.task_done_lock.acquire()
        if len(self.tansfer_done_queue) > 0:
            data = self.tansfer_done_queue.pop(0)
            logger.info(f"get_transfer_done_signal: Get swap task {data[-1]} finished signal from queue successful")
        self.task_done_lock.release()
        return data

    def empty(self):
        """
        check if queue is empty
        """
        try:
            return len(self.transfer_task_queue) == 0
        except Exception as e:
            logger.error(f"empty function meets error: {e}")
            raise e
    
def launch_queue_service(port, num_workers):
    """
    launch_queue_service
    """
    try:
        logger.info(f"start to launch cache queue service, port:{port}")
        cache_task_queue = list()  
        cache_task_done_queue = list() 
        cache_task_sync_lock = threading.Lock() 
        cache_task_done_sync_lock = threading.Lock()
        cache_sync_value = Value("i", 0) 


        QueueManager.register(
            "get_transfer_task_queue",
            callable=lambda: cache_task_queue,
            proxytype=ListProxy,
        )
        QueueManager.register(
            "get_tansfer_done_queue",
            callable=lambda: cache_task_done_queue,
            proxytype=ListProxy,
        )
        QueueManager.register(
            "get_transfer_task_lock", callable=lambda: cache_task_sync_lock
        )
        QueueManager.register(
            "get_transfer_task_done_lock", callable=lambda: cache_task_done_sync_lock
        )
        QueueManager.register(
            "get_cache_sync_value",
            callable=lambda: cache_sync_value,
            proxytype=ValueProxy,
        )


        barrier1 = threading.Barrier(num_workers)
        QueueManager.register('get_barrier1', callable=lambda: barrier1)
        barrier2 = threading.Barrier(num_workers)
        QueueManager.register('get_barrier2', callable=lambda: barrier2)
        barrier3 = threading.Barrier(num_workers)
        QueueManager.register('get_barrier3', callable=lambda: barrier3)


        swap_to_cpu_barrier1 = threading.Barrier(num_workers)
        QueueManager.register('get_swap_to_cpu_barrier1', callable=lambda: swap_to_cpu_barrier1)
        swap_to_cpu_barrier2 = threading.Barrier(num_workers)
        QueueManager.register('get_swap_to_cpu_barrier2', callable=lambda: swap_to_cpu_barrier2)
        swap_to_gpu_barrier1 = threading.Barrier(num_workers)
        QueueManager.register('get_swap_to_gpu_barrier1', callable=lambda: swap_to_gpu_barrier1)
        swap_to_gpu_barrier2 = threading.Barrier(num_workers)
        QueueManager.register('get_swap_to_gpu_barrier2', callable=lambda: swap_to_gpu_barrier2)


        m = QueueManager(address=("127.0.0.1", port), authkey=b"cache_queue_service")
        s = m.get_server()
        logger.info("launch queue service successful")
        s.serve_forever()
        logger.info("finish queue service")
    except Exception as e:
        logger.error(f"launch queue service failed, error_msg: {e}")
        raise e


if __name__ == "__main__":
    port = int(os.getenv("IPC_CACHE_QUEUE_PORT", "56668"))
    mp_num = int(os.getenv("MP_NUM", "1"))
    launch_queue_service(port, mp_num)

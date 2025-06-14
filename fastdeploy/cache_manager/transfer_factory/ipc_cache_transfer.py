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

import math
import os
import socket
import threading
import time
from multiprocessing import shared_memory
import ctypes
import os

from queue import Queue

import numpy as np
import paddle

import socket
from fastdeploy.utils import get_logger
from fastdeploy.inter_communicator import IPCSignal
from fastdeploy.inter_communicator import EngineWorkerQueue

use_pip_eff_llm = os.getenv('USE_PIP_EFF_LLM')
if use_pip_eff_llm is None:
    from fastdeploy.model_executor.ops.gpu import get_data_ptr_ipc
    from fastdeploy.model_executor.ops.gpu import ipc_sent_key_value_cache_by_remote_ptr
    from fastdeploy.model_executor.ops.gpu import ipc_sent_key_value_cache_by_remote_ptr_block_sync
else:
    from efficientllm.ops.gpu import get_data_ptr_ipc
    from efficientllm.ops.gpu import ipc_sent_key_value_cache_by_remote_ptr
    from efficientllm.ops.gpu import ipc_sent_key_value_cache_by_remote_ptr_block_sync


logger = get_logger(f"cache_messager", f"cache_messager.log")


class IPCConnector:
    """
    IPC communication class.
    """
    def __init__(self, rank_id_, remote_gpu_id_, layer_num, local_gpu_id_):
        """
        Args:
        rank_id_: rank id
        remote_gpu_id_: remote gpu id
        
        """
        self.remote_key_tensor_ptr_list = []
        self.remote_value_tensor_ptr_list = []
        self.remote_gpu_id = int(remote_gpu_id_)
        self.rank_id = rank_id_
        self.local_gpu_id = int(local_gpu_id_)
        tmp = paddle.ones([1, 1])
        logger.info(f"init ipc rank{self.rank_id} with remote {self.remote_gpu_id} {self.local_gpu_id}")
        for layer_id in range(layer_num):
            key_unique_name =  f"key_caches_{layer_id}_rank{self.rank_id}.device{self.remote_gpu_id}"
            value_unique_name = f"value_caches_{layer_id}_rank{self.rank_id}.device{self.remote_gpu_id}"
            self.remote_key_tensor_ptr_list.append(get_data_ptr_ipc(tmp, key_unique_name))
            self.remote_value_tensor_ptr_list.append(get_data_ptr_ipc(tmp, value_unique_name))
        self.write_stream = paddle.device.Stream(f'gpu:{self.local_gpu_id}')
        self.finish_event = paddle.device.Event()


class IPCCommManager:
    """
    IPC communication manager, used to initialize ipc and cache transmission.
    """
    def __init__(self, rank_id_, gpu_idx_,
                 local_key_cache_tensor_list,   # tensor list
                 local_value_cache_tensor_list, # tensor
                 ):
        self.rank_id = rank_id_
        self.gpu_idx = gpu_idx_
        # local cache to tensor
        self.local_key_cache_tensor_list = local_key_cache_tensor_list
        self.local_value_cache_tensor_list = local_value_cache_tensor_list
        self.layer_num = len(self.local_key_cache_tensor_list)
        # record connected ipc info
        self.comm_map = {}


    def connect(self, remote_gpu_id_=0):
        """
        Connect to remote gpu.
        """
        logger.info(f"{self.rank_id}: connect to remote_gpu_id:{remote_gpu_id_} {self.layer_num} {self.gpu_idx}")
        if self.is_connected(remote_gpu_id_):
            return True
        else:
            self.comm_map[remote_gpu_id_] = IPCConnector(self.rank_id, remote_gpu_id_, self.layer_num, self.gpu_idx)
            return True


    def is_connected(self, remote_gpu_id_=0):
        """
        Check if remote gpu is connected.
        """
        if remote_gpu_id_ in self.comm_map.keys():
            return True
        else:
            return False

    def write_cache(self, remote_gpu_id, local_block_ids, remote_block_ids, layer_idx):
        """
        Connect to remote gpu and write cache.
        """
        block_num = len(local_block_ids)
        if not self.is_connected(remote_gpu_id):
            self.connect(remote_gpu_id)
        comm = self.comm_map[remote_gpu_id]

        with paddle.device.stream_guard(comm.write_stream):
            ipc_sent_key_value_cache_by_remote_ptr(
                self.local_key_cache_tensor_list[layer_idx],  
                self.local_value_cache_tensor_list[layer_idx], 
                local_block_ids, 
                remote_block_ids, 
                comm.remote_key_tensor_ptr_list[layer_idx],
                comm.remote_value_tensor_ptr_list[layer_idx],
                block_num,
                self.gpu_idx,
                comm.remote_gpu_id,
                comm.write_stream.stream_base.cuda_stream
            )

    def write_block_by_sync(self, remote_gpu_id):
        """
        check finish event and wait for it
        """
        paddle.set_device(f'gpu:{self.gpu_idx}')
        comm = self.comm_map[remote_gpu_id]
        ipc_sent_key_value_cache_by_remote_ptr_block_sync(
                self.local_key_cache_tensor_list[0],  #tensor no use
                self.local_value_cache_tensor_list[0], #tensor no use
                comm.write_stream.stream_base.cuda_stream)



class IPCCacheTransfer(object):
    """
    IPC cache messager, used to initialize ipc and cache transmission.
    """
    def __init__(self, engine_worker_queue_port, gpu_cache_kvs, rank, nranks, num_layers, gpu_id=0):
        paddle.set_device(f'gpu:{gpu_id}')
        self.gpu_cache_kvs = gpu_cache_kvs

        logger.info(f"rank: {rank} , gpu_id: {gpu_id}")

        self.rank = rank
        self.nranks = nranks
        address = ('0.0.0.0', engine_worker_queue_port)
        self.engine_worker_queue = EngineWorkerQueue(
            address=address, is_server=False, num_client=self.nranks, client_id=self.rank)

        self.num_layers = num_layers
        cache_k = []
        cache_v = []
        for layer_idx in range(self.num_layers):
            cache_k.append(self.gpu_cache_kvs[f'key_caches_{layer_idx}_rank{self.rank}_device{gpu_id}'])
            cache_v.append(self.gpu_cache_kvs[f'value_caches_{layer_idx}_rank{self.rank}_device{gpu_id}'])

        logger.info(f"get cache kv tensor layer{self.num_layers}")
        local_device_id = int(str(cache_k[0].place)[-2])
        logger.info(f"creating ipc_comm with local_device_id:{local_device_id}, ")
        self.cache_info = dict()

        self.gpu_id = gpu_id

        try:
            self.messager = IPCCommManager(
                                    self.rank, 
                                    gpu_id,
                                    cache_k,
                                    cache_v,
                                    )
            logger.info(f"done create ipc_comm with local_device_id:{local_device_id}, ")
            
            
            self.cache_info = dict()     
            self.last_step_idx = -1
            layerwise_send_cache_thread = threading.Thread(target=self._prefill_layerwise_send_cache_thread)
            layerwise_send_cache_thread.daemon = True
            layerwise_send_cache_thread.start()

            logger.info(f"start prefill layerwise send cache thread")

        except Exception as e:
            logger.info(f"create ipc_comm failed, {e}")
            raise e

        logger.info(f"cache messager init finished, use ipc")


    def _prefill_layerwise_send_cache_thread(self):
        """
        prefill layerwise send cache thread
        """
        logger.info(f"prefill layerwise send cache thread start")
        try:
            prefilled_step_idx_data = np.zeros(shape=[1], dtype=np.int32)
            prefilled_layer_idx_data = np.zeros(shape=[1], dtype=np.int32)
            try:
                step_shm_value = IPCSignal(name=f"splitwise_complete_prefilled_step_{self.rank}",
                                            array=prefilled_step_idx_data,
                                            dtype=np.int32,
                                            suffix=self.gpu_id,
                                            create=True)
                layer_shm_value = IPCSignal(name=f"splitwise_complete_prefilled_layer_{self.rank}",
                                            array=prefilled_layer_idx_data,
                                            dtype=np.int32,
                                            suffix=self.gpu_id,
                                            create=True)
            except:
                step_shm_value = IPCSignal(name=f"splitwise_complete_prefilled_step_{self.rank}",
                                            array=prefilled_step_idx_data,
                                            dtype=np.int32,
                                            suffix=self.gpu_id,
                                            create=False)
                layer_shm_value = IPCSignal(name=f"splitwise_complete_prefilled_layer_{self.rank}",
                                            array=prefilled_layer_idx_data,
                                            dtype=np.int32,
                                            suffix=self.gpu_id,
                                            create=False)

            step_shm_value.value[0] = -1
            layer_shm_value.value[0] = -1

            self.last_step_idx = -1
            self.last_layer_idx = -1
            logger.info(f"prefilled_step_idx: {self.last_step_idx}")

            cache_sent_set = set()
            while True:
                cache_info = self.engine_worker_queue.get_cache_info()
                current_cache_info = []
                if not cache_info:
                    time.sleep(0.001)
                    continue
                else:
                    for info in cache_info:
                        if info['request_id'] in self.cache_info:
                            current_info = self.cache_info[info["request_id"]]
                            current_info.update(info)
                            src_block_ids = current_info["src_block_ids"][-len(current_info["dest_block_ids"]):]
                            current_info["src_block_ids"] = src_block_ids
                            current_info["current_layer_ids"] = 0
                            assert "src_block_ids" in current_info and "dest_block_ids" in current_info
                            current_cache_info.append(current_info)
                            del self.cache_info[info["request_id"]]
                        else:
                            self.cache_info[info["request_id"]] = info

                if len(current_cache_info) == 0:
                    time.sleep(0.001)
                    continue

                logger.info(f"current_cache_info:{current_cache_info}")
                logger.info(f"current_step_idx:{step_shm_value.value[0]} {self.last_step_idx}")
                logger.info(f"current_layer_idx:{layer_shm_value.value[0]} {self.last_layer_idx}")
                while step_shm_value.value[0] < 0 or self.last_step_idx == step_shm_value.value[0] \
                    or self.last_layer_idx == layer_shm_value.value[0]:
                    time.sleep(0.001)
                    continue
                
                while 1:
                    prefilled_layer_idx = layer_shm_value.value[0]
                    for item in current_cache_info:
                        for layer_idx in range(self.last_layer_idx + 1, prefilled_layer_idx + 1):
                            target_gpu_id = int(item['device_ids'][self.rank])
                            src_block_ids = paddle.to_tensor(item['src_block_ids'], dtype='int32', place='cpu')
                            dest_block_ids = paddle.to_tensor(item['dest_block_ids'], dtype='int32', place='cpu')
                            self.messager.write_cache(target_gpu_id, src_block_ids, dest_block_ids, layer_idx)
                        cache_sent_set.add((int(item['device_ids'][self.rank]), item['request_id']))
                    self.last_layer_idx = prefilled_layer_idx
 
                    if (self.last_layer_idx + 1) == self.num_layers:
                        break

                if (self.last_layer_idx + 1) == self.num_layers:
                    logger.info(f"num of finish write cache task: {len(cache_sent_set)}")
                    for item in cache_sent_set:
                        self.messager.write_block_by_sync(item[0])
                        logger.info(f"finish write cache {item[1]}")
                    self.last_layer_idx = -1
                    self.engine_worker_queue.finish_request_barrier.wait()
                    if self.rank == 0:
                        finished_req = []
                        for item in cache_sent_set:
                            logger.info(f"put finished req: {item[1]}")
                            finished_req.append(item[1])
                        self.engine_worker_queue.put_finished_req(finished_req)
                    self.last_step_idx += 1
                    cache_sent_set = set()
                    logger.info(f"prefilled_step_idx: {self.last_step_idx}")
        except Exception as e:
            logger.error(f"prefill layerwise send cache thread has exception: {e}")

    
    def connect(self, device_id=0):
        """
        connect to remote device
        """
        if self.messager.is_connected(device_id):
            logger.info(f"devices {device_id} is already connected")
            return True

        logger.info(f"connect device_id: {device_id}")
        flag = self.messager.connect(device_id)
        return flag


    

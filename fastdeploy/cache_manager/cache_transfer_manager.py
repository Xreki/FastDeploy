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

import argparse
import os
import concurrent.futures
from multiprocessing import shared_memory
import json
import queue
import time
import threading
from enum import Enum
import paddle
import numpy as np


from fastdeploy.utils import get_logger
from fastdeploy.cache_manager.cache_queue_manager import CacheQueueManager
from fastdeploy.inter_communicator import IPCSignal
# TODO 不显式 import 初始化


use_pip_eff_llm = os.getenv('USE_PIP_EFF_LLM')
if use_pip_eff_llm is None:
    from fastdeploy.model_executor.ops.gpu import set_data_ipc
    from fastdeploy.model_executor.ops.gpu import swap_cache_all_layers
    from fastdeploy.model_executor.ops.gpu import cuda_host_alloc
else:
    from efficientllm.gpu import set_data_ipc
    from efficientllm.gpu import swap_cache_all_layers
    from efficientllm.gpu import cuda_host_alloc


def parse_args():
    """
    从命令行解析参数
    """
    parser = argparse.ArgumentParser("Cache transfer manager")
    parser.add_argument("--rank", type=int, default=0, help="分布式训练中的rank ID")
    parser.add_argument("--device_id", type=int, default=0, help="GPU设备ID")
    parser.add_argument("--num_layers", type=int, default=1, help="Transformer层数")
    parser.add_argument("--num_attention_heads", type=int, default=1, help="注意力头数")
    parser.add_argument("--hidden_size", type=int, default=1, help="隐藏层维度")
    parser.add_argument("--kv_num_head", type=int, default=1, help="Key/Value的头数")
    parser.add_argument("--mp_num", type=int, default=1, help="模型并行度")
    parser.add_argument("--protocol", type=str, default="ipc", 
                       help="通信协议，目前支持ipc")
    parser.add_argument("--enable_splitwise", type=int, default=0,
                       help="是否启用分片模式 (0/1)")
    parser.add_argument("--cache_queue_port", type=int, default=9923,
                       help="缓存队列通信端口")
    parser.add_argument("--engine_worker_queue_port", type=int, default=9923,
                       help="引擎工作队列端口")
    parser.add_argument("--engine_pid", type=int, default=None,
                       help="引擎进程PID（用于IPC信号同步）")
    
    # 以下是新增的参数
    parser.add_argument("--num_gpu_blocks", type=int, default=1,
                       help="每层GPU缓存块数量")
    parser.add_argument("--num_cpu_blocks", type=int, default=4,
                       help="每层CPU缓存块数量")
    parser.add_argument("--block_size", type=int, default=64,
                       help="每个缓存块的序列长度")
    parser.add_argument("--bytes_per_layer_per_block", type=int, default=1024,
                       help="每层每个块的字节数")
    parser.add_argument("--cache_dtype", type=str, default="float16",
                       choices=["float16", "bfloat16", "float32"],
                       help="缓存数据类型")
    
    args = parser.parse_args()
    return args

class CacheStatus(Enum):
    """
    Cache状态枚举类"""

    GPU = 0  # 在GPU中
    SWAP2CPU = 1  # 从GPU交换到CPU
    SWAP2GPU = 2  # 从CPU交换到GPU
    CPU = 3  # 在CPU中


class SSDEvent(Enum):
    """
    SSD事件枚举类"""

    READ = 100
    WRITE = 101
    UPDATE = 102
    DELETE = 103


# 初始化分布式环境


class CacheTransferManager:
    """
    管理CPU和GPU之间缓存的交换传输
    """

    def __init__(self, args):
        """
        初始化CacheTransferManager
        """

        device = args.device_id
        rank = args.rank
        paddle.set_device(f"gpu:{device}")
        self.gpu_cache_kvs = {}  # GPU上的cache存储空间
        self.cpu_cache_kvs = {}  # CPU上的cache存储空间，可以设置为GPU上存储空间的N倍
        self.gpu_cache_k_tensors = []
        self.gpu_cache_v_tensors = []
        # 用来并行执行多卡的传输任务
        self.read_ssd_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.write_ssd_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        )
        self.swap_to_cpu_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        )
        self.swap_to_gpu_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        )
        self.transfer_task_queue = queue.Queue()  # 用来接收传输任务
        self.tansfer_done_queue = queue.Queue()  # 用来告知任务执行完毕
        self.n_ranks = args.mp_num
        self.rank = rank
        self.device = device
        self.cache_task_queue = CacheQueueManager(
            rank=rank,
            mp_num=args.mp_num,
            port=args.cache_queue_port,
        )

        self.num_cpu_blocks = args.num_cpu_blocks

        cache_type = args.cache_dtype

        for i in range(args.num_layers):
            # 创建gpu cache kv 缓存的显存空间
            self.gpu_cache_kvs[
                "key_caches_{}_rank{}_device{}".format(i, rank, device)
            ] = paddle.full(
                shape=[
                    args.num_gpu_blocks,
                    args.kv_num_head,
                    args.block_size,
                    args.hidden_size
                    // args.num_attention_heads,
                ],
                fill_value=0,
                dtype=cache_type,
            )
            self.gpu_cache_k_tensors.append(
                self.gpu_cache_kvs["key_caches_{}_rank{}_device{}".format(i, rank, device)]
            )
            self.gpu_cache_kvs[
                "value_caches_{}_rank{}_device{}".format(i, rank, device)
            ] = paddle.full(
                shape=[
                    args.num_gpu_blocks,
                    args.kv_num_head,
                    args.block_size,
                    args.hidden_size
                    // args.num_attention_heads,
                ],
                fill_value=0,
                dtype=cache_type,
            )
            self.gpu_cache_v_tensors.append(
                self.gpu_cache_kvs["value_caches_{}_rank{}_device{}".format(i, rank, device)]
            )
            # 将分配的gpu显存share到infer进程
            set_data_ipc(
                self.gpu_cache_kvs["key_caches_{}_rank{}_device{}".format(i, rank, device)],
                "key_caches_{}_rank{}.device{}".format(i, rank, device)
            )
            set_data_ipc(
                self.gpu_cache_kvs["value_caches_{}_rank{}_device{}".format(i, rank, device)],
                "value_caches_{}_rank{}.device{}".format(i, rank, device)
            )
        cache_kv_size_byte = sum([tmp.numel() * 1 for key, tmp in self.gpu_cache_kvs.items()])
        logger.info(f"device :{self.device}")
        logger.info(f"cache_kv_size_byte : {cache_kv_size_byte}")
        logger.info(f"done init cache (full) gmem alloc : {paddle.device.cuda.memory_allocated()}")

        paddle.set_device("cpu")
        self.k_dst_ptrs = []
        self.v_dst_ptrs = []
        for i in range(args.num_layers):
            # 创建cpu cache kv 缓存的空间
            self.cpu_cache_kvs[
                "key_caches_{}_rank{}".format(i, rank)
            ] = cuda_host_alloc(
                args.num_cpu_blocks * args.bytes_per_layer_per_block
            )
            self.k_dst_ptrs.append(
                self.cpu_cache_kvs["key_caches_{}_rank{}".format(i, rank)]
            )
            self.cpu_cache_kvs[
                "value_caches_{}_rank{}".format(i, rank)
            ] = cuda_host_alloc(
                args.num_cpu_blocks * args.bytes_per_layer_per_block
            )
            self.v_dst_ptrs.append(
                self.cpu_cache_kvs["value_caches_{}_rank{}".format(i, rank)]
            )
        # 标记共享内存中的flag表明已经cache初始化完毕
        
        cache_ready_signal_data = np.zeros(
            shape=[args.mp_num], dtype=np.int32)
        self.cache_ready_signal = IPCSignal(name="cache_ready_signal",
                                             array=cache_ready_signal_data,
                                             dtype=np.int32,
                                             suffix=args.engine_pid,
                                             create=False)
        self.cache_ready_signal.value[self.rank] = 1

        # 创建CacheMessager，负责跨实例传输Cache
        paddle.set_device(f"gpu:{device}")
        if args.enable_splitwise:
            logger.debug("create cache messager...")

            commu_protocol = args.protocol.split(",")
            assert len(commu_protocol) == 1
            assert commu_protocol[0] in ["ipc"], f"not support protocol: {args.protocol}"
            logger.info(f"{args}")
            from fastdeploy.cache_manager.transfer_factory.ipc_cache_messager import IPCCacheMessager
            self.cache_messager = IPCCacheMessager(engine_worker_queue_port=args.engine_worker_queue_port, 
                        gpu_cache_kvs=self.gpu_cache_kvs,
                        rank=self.rank, 
                        nranks=args.mp_num, 
                        num_layers=args.num_layers, 
                        gpu_id=self.device)
            logger.info("successfully create cache messager")
        logger.info(f"done init CacheMessager gmem alloc : {paddle.device.cuda.memory_allocated()}")

        # 多进程间获取数据的同步

        cache_task_broadcast_data = np.zeros(
            shape=[1], dtype=np.int32)
        self.cache_task_broadcast_signal = IPCSignal(name="cache_task_broadcast_signal",
                                             array=cache_task_broadcast_data,
                                             dtype=np.int32,
                                             suffix=args.engine_pid,
                                             create=False)


    def _do_swap_to_cpu_task(
        self, swap_node_ids, gpu_block_id, cpu_block_id, event_type, transfer_task_id
    ):
        """
        执行GPU->CPU
        """
        self.cache_task_queue.swap_to_cpu_barrier1.wait()
        if self.rank == 0:
            self.cache_task_queue.swap_to_cpu_barrier1.reset()
        result = self._transfer_data(
            swap_node_ids,
            gpu_block_id,
            cpu_block_id,
            event_type,
            transfer_task_id,
        )
        self.cache_task_queue.swap_to_cpu_barrier2.wait()
        if self.rank == 0:
            self.cache_task_queue.swap_to_cpu_barrier2.reset()
            self.cache_task_queue.put_transfer_done_signal(result)
            logger.debug(f"_do_swap_to_cpu_task: put_transfer_done_signal {result}")
            logger.info(
                f"_do_swap_to_cpu_task: put_transfer_done_signal for transfer_task_id {transfer_task_id}"
            )

    def _do_swap_to_gpu_task(
        self, swap_node_ids, gpu_block_id, cpu_block_id, event_type, transfer_task_id
    ):
        """
        执行CPU->GPU
        """
        self.cache_task_queue.swap_to_gpu_barrier1.wait()
        if self.rank == 0:
            self.cache_task_queue.swap_to_gpu_barrier1.reset()
        result = self._transfer_data(
            swap_node_ids,
            gpu_block_id,
            cpu_block_id,
            event_type,
            transfer_task_id,
        )
        self.cache_task_queue.swap_to_gpu_barrier2.wait()
        if self.rank == 0:
            self.cache_task_queue.swap_to_gpu_barrier2.reset()
            self.cache_task_queue.put_transfer_done_signal(result)
            logger.debug(f"_do_swap_to_gpu_task: put_transfer_done_signal {result}")
            logger.info(
                f"_do_swap_to_gpu_task: put_transfer_done_signal for transfer_task_id {transfer_task_id}"
            )

    def do_data_transfer(self):
        """
        执行数据传输任务
        """
        while True:
            try:
                if self.rank == 0:
                    # 队列不为空, 可取出数据
                    if not self.cache_task_queue.empty():
                        self.cache_task_broadcast_signal.value[0] = 1
                if self.n_ranks > 1:
                    self.cache_task_queue.barrier1.wait()
                    if self.rank == 0:
                        self.cache_task_queue.barrier1.reset()
                if self.cache_task_broadcast_signal.value[0] == 1:
                    data, read_finish = self.cache_task_queue.get_transfer_task()
                    logger.debug(f"transfer data: get_transfer_task {data}")
                    if read_finish:
                        self.cache_task_broadcast_signal.value[0] = 0
                    (
                        swap_node_ids,
                        gpu_block_id,
                        cpu_block_id,
                        event_type,
                        transfer_task_id,
                    ) = data
                    if event_type.value == CacheStatus.SWAP2CPU.value:
                        self.swap_to_cpu_thread_pool.submit(
                            self._do_swap_to_cpu_task,
                            swap_node_ids,
                            gpu_block_id,
                            cpu_block_id,
                            event_type,
                            transfer_task_id,
                        )
                    else:
                        self.swap_to_gpu_thread_pool.submit(
                            self._do_swap_to_gpu_task,
                            swap_node_ids,
                            gpu_block_id,
                            cpu_block_id,
                            event_type,
                            transfer_task_id,
                        )
                else:
                    if self.n_ranks > 1:
                        self.cache_task_queue.barrier2.wait()
                        if self.rank == 0:
                            self.cache_task_queue.barrier2.reset()
                    continue

                if self.n_ranks > 1:
                    self.cache_task_queue.barrier3.wait()
                    if self.rank == 0:
                        self.cache_task_queue.barrier3.reset()
            except Exception as e:
                logger.info(f"do_data_transfer: error: {e}")

    def _transfer_data(
        self,
        swap_node_ids,
        task_gpu_block_id,
        task_cpu_block_id,
        event_type,
        transfer_task_id,
    ):
        """
        传输数据
        task_gpu_block_id格式 [[block_id0, [fold_block_id0, fold_block_id1]],
            [block_id1, [fold_block_id0, fold_block_id1]], ...]
        """
        logger.debug(
            f"transfer data: transfer_task_id {transfer_task_id}: swap_node_ids {swap_node_ids}"
            + f"task_gpu_block_id {task_gpu_block_id} task_cpu_block_id {task_cpu_block_id} event_type {event_type}"
        )
        start_time = time.time()
        try:
            # transform block id
            assert len(task_gpu_block_id) == len(task_cpu_block_id)
            gpu_block_ids = task_gpu_block_id
            cpu_block_ids = task_cpu_block_id

            if event_type.value == CacheStatus.SWAP2CPU.value:
                swap_cache_all_layers(
                    self.gpu_cache_k_tensors,
                    self.k_dst_ptrs,
                    self.num_cpu_blocks,
                    gpu_block_ids,
                    cpu_block_ids,
                    self.device,
                    0,
                )
                swap_cache_all_layers(
                    self.gpu_cache_v_tensors,
                    self.v_dst_ptrs,
                    self.num_cpu_blocks,
                    gpu_block_ids,
                    cpu_block_ids,
                    self.device,
                    0,
                )

            elif event_type.value == CacheStatus.SWAP2GPU.value:
                swap_cache_all_layers(
                    self.gpu_cache_k_tensors,
                    self.k_dst_ptrs,
                    self.num_cpu_blocks,
                    gpu_block_ids,
                    cpu_block_ids,
                    self.device,
                    1,
                )
                swap_cache_all_layers(
                    self.gpu_cache_v_tensors,
                    self.v_dst_ptrs,
                    self.num_cpu_blocks,
                    gpu_block_ids,
                    cpu_block_ids,
                    self.device,
                    1,
                )
            else:
                logger.warning(
                    f"transfer data: Get unexpected event type {event_type}, only SWAP2CPU and SWAP2GPU supported"
                )
        except Exception as e:
            logger.error(f"transfer data: error: {e}")
            raise e
        end_time = time.time()
        elasped_time = end_time - start_time
        logger.info(
            f"transfer data: transfer_task_id {transfer_task_id} event_type {event_type}: "
            + f"transfer {len(gpu_block_ids)} blocks done  elapsed_time {elasped_time:.4f}"
        )
        return (
            swap_node_ids,
            task_gpu_block_id,
            task_cpu_block_id,
            event_type,
            transfer_task_id,
        )


def main():
    """
    启动cache manager
    """

    cache_manager = CacheTransferManager(args)
    # 开启数据传输任务的监听线程
    transfer_thread = threading.Thread(target=cache_manager.do_data_transfer)
    transfer_thread.start()



if __name__ == "__main__":

    args = parse_args()
    logger = get_logger(
        f"cache_transfer_manager", f"cache_transfer_manager.log"
    )
    main()

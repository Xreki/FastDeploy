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
import heapq
import time
import threading
from collections import defaultdict
from threading import Lock, Event
import subprocess
import uuid
import multiprocessing
from multiprocessing import shared_memory
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from fastdeploy.cache_manager.cache_queue_manager import CacheQueueManager, launch_queue_service
from fastdeploy.cache_manager.cache_transfer_manager import CacheStatus
from fastdeploy.inter_communicator import IPCSignal
from fastdeploy.utils import get_logger

logger = get_logger("prefix_cache_manager", "prefix_cache_manager.log")



class CacheMetrics:
    """
     Cache Metrics used to record the cache hit time, token num, request num, etc.
    """
    def __init__(self):
        self.total_match_time = 0.0  # 匹配时间的总和
        self.avg_match_time = 0.0  # 匹配时间
        self.min_match_time = 1e9
        self.max_match_time = 0.0
        # 请求数角度
        self.req_count = 0  # 请求的总数
        self.hit_req_count = 0  # 能够命中缓存的请求数
        self.hit_req_ratio = 0.0  # 请求命中率 = 命中请求数 / 请求总数
        # token数角度
        self.total_gpu_matched_token_num = 0  # gpu中命中的token数
        self.total_cpu_matched_token_num = 0  # cpu中命中的token数
        self.total_ssd_matched_token_num = 0  # ssd中命中的token数
        self.matched_token_num = 0  # 命中的token数
        self.total_token_num = 0  # 总token数
        self.hit_token_ratio = 0.0  # 总token命中率 = 命中的token / 总token
        self.cpu_hit_token_ratio = 0.0
        self.gpu_hit_token_ratio = 0.0
        self.ssd_hit_token_ratio = 0.0

    def _update_history_hit_metrics(self):
        """
        update hit ratio
        """
        self.hit_req_ratio = self.hit_req_count / self.req_count
        self.hit_token_ratio = self.matched_token_num / self.total_token_num
        self.cpu_hit_token_ratio = (
            self.total_cpu_matched_token_num / self.total_token_num
        )
        self.gpu_hit_token_ratio = (
            self.total_gpu_matched_token_num / self.total_token_num
        )
        self.ssd_hit_token_ratio = (
            self.total_ssd_matched_token_num / self.total_token_num
        )
        logger.info(
            f"Metrics for all requests: req_count {self.req_count} hit_req_count {self.hit_req_count}"
            + f" hit_req_ratio {self.hit_req_ratio:.2f} hit_token_ratio {self.hit_token_ratio:.2f}"
            + f" gpu_hit_token_ratio {self.gpu_hit_token_ratio:.2f}"
            + f" cpu_hit_token_ratio {self.cpu_hit_token_ratio:.2f}"
            + f" ssd_hit_token_ratio {self.ssd_hit_token_ratio:.2f}"
            + f" total_gpu_matched_token_num {self.total_gpu_matched_token_num}"
            + f" total_cpu_matched_token_num {self.total_cpu_matched_token_num}"
            + f" total_ssd_matched_token_num {self.total_ssd_matched_token_num}"
            + f" total_matched_token_num {self.matched_token_num}"
            + f" total_token_num {self.total_token_num}"
        )

    def calculate_hit_metrics(
        self,
        req_id,
        current_query_cpu_match_token_num,
        current_query_gpu_match_token_num,
        current_ssd_match_token_num,
        current_query_token_num,
    ):
        """
        计算当前query命中率
        """
        # 当前query的命中率
        cpu_cache_match_ratio = (
            current_query_cpu_match_token_num / current_query_token_num
        )
        gpu_cache_match_ratio = (
            current_query_gpu_match_token_num / current_query_token_num
        )
        ssd_cache_match_ratio = current_ssd_match_token_num / current_query_token_num
        total_match_ratio = (
            cpu_cache_match_ratio + gpu_cache_match_ratio + ssd_cache_match_ratio
        )

        # 计算历史累计指标
        self.total_cpu_matched_token_num += (
            current_query_cpu_match_token_num  # cpu中命中的token数
        )
        self.total_gpu_matched_token_num += (
            current_query_gpu_match_token_num  # gpu中命中的token数
        )
        self.total_ssd_matched_token_num += current_ssd_match_token_num
        self.matched_token_num += (
            current_query_cpu_match_token_num
            + current_query_gpu_match_token_num
            + current_ssd_match_token_num
        )  # 命中的token数
        self.total_token_num += current_query_token_num  # 总token数
        logger.info(
            f"Metrics for req_id {req_id}: token_num {current_query_token_num}"
            + f" cpu_cache_match_ratio {cpu_cache_match_ratio}"
            + f" gpu_cache_match_ratio {gpu_cache_match_ratio}"
            + f" ssd_cache_match_ratio {ssd_cache_match_ratio} total_match_ratio {total_match_ratio}"
        )




class PrefixCacheManager:
    """
    管理复用system prompt
    """

    def __init__(self, cache_config, tensor_parallel_size, splitwise_role="mixed"):
        """
        初始化前缀树管理类
        """

        self.metrics = CacheMetrics()

        if splitwise_role != "mixed":
            self.enable_splitwise = 1
        else:
            self.enable_splitwise = 0

        self.cache_config = cache_config
        # GPU和CPU的总block list
        self.num_gpu_blocks = cache_config.prefill_kvcache_block_num
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        self.gpu_free_block_list = list(
            range(self.num_gpu_blocks - 1, -1, -1)
        )  # 服务端管理的GPU上剩余的block id
        if self.num_cpu_blocks > 0:
            self.cpu_free_block_list = list(
                range(self.num_cpu_blocks - 1, -1, -1)
            )  # CPU上剩余的block id
        else:
            self.cpu_free_block_list = []
        heapq.heapify(self.gpu_free_block_list)
        heapq.heapify(self.cpu_free_block_list)
        self.node_id_pool = list(
            range(self.num_gpu_blocks + self.num_cpu_blocks)
        )

        self.radix_tree_root = BlockNode(-1, [], 0, 0, -1, 0, None, None, None)  # 根节点

        # radix_tree_root中保留的Node要么是存在于GPU中，要么是存在于CPU中
        # 如果CPU和GPU中都不存在，则代表节点需要从树中移除，这种情况一般发生在cpu上的cache已经满了，需要缓存新cache时候按照lru将旧cache移除
        # 下列数据结构用于gpu缓存管理
        self.gpu_lru_leaf_heap = []  # 保存gpu上叶子节点的堆
        self.gpu_lru_leaf_set = set()  # 保存gpu上叶子节点的集合

        # 下列数据结构用于cpu缓存管理
        self.cpu_lru_leaf_heap = []  # 保存在cpu上叶子节点的lru堆
        self.cpu_lru_leaf_set = set()  # 保存缓存在cpu上的lru堆

        # 下列数据结构用于交换缓存管理
        self.request_release_lock = Lock()  # 用于request和release过程的数据同步
        self.task_swapping_event = {}  # 保存req_id 和 一个Event事件，用于同步
        self.task_ssd_event = {}  # 保存ssd任务id和一个Event事件，用于同步
        self.task_ssd_result = {}  # 保存ssd任务id和执行结果的返回值


        # 辅助数据结构
        self.node_map = {}  # {node_id: Node}  保存node id和Node的映射关系
        self.req_leaf_map = (
            {}
        )  # {request_id: leaf node}   保存任务req_id和对应radix tree中最末端节点node的映射关系
        self.leaf_req_map = defaultdict(set)  # 保存最末端节点node和任务req_id的映射关系
        self.unfilled_req_block_map = defaultdict(
            list
        )  # 保存输入token数量小于block_size所分配的block id

        self.executor_pool = ThreadPoolExecutor(max_workers=1)  # 执行release异步操作的线程池
        self.free_gpu_executor_pool = ThreadPoolExecutor(
            max_workers=1
        )  # 执行free gpu异步操作的线程池
        self.free_cpu_executor_pool = ThreadPoolExecutor(
            max_workers=1
        )  # 执行free cpu异步操作的线程池
        self.gpu_free_task_future = None # 当前正在异步执行的swap out任务
        self.cache_status_lock = Lock()  # 用于同步cache状态的锁


        logger.info(
            f"num_gpu_blocks_server_owned {self.num_gpu_blocks} num_cpu_blocks "
            + f"{self.num_cpu_blocks}, bytes_per_layer_per_block {self.cache_config.bytes_per_layer_per_block}"
        )

    

    def launch_cache_manager(self, cache_config, tensor_parallel_size, \
                    device_ids, engine_worker_queue_port, pid_suffix):

        """
        launch_cache_manager function used to initialize the cache manager.
        """
        broadcast_cache_task_flag_array = np.zeros([1], dtype=np.int32)

        self.shm_cache_task_flag_broadcast = IPCSignal(name="cache_task_broadcast_signal",
                                             array=broadcast_cache_task_flag_array,
                                             dtype=np.int32,
                                             suffix=pid_suffix,
                                             create=True)


        multiprocessing.Process(
            target=launch_queue_service,
            args=(cache_config.cache_queue_port, tensor_parallel_size),
        ).start()

        current_dir_path = os.path.split(os.path.abspath(__file__))[0]
        filename = "cache_transfer_manager.py"
        py_path = os.path.join(current_dir_path, filename)

        device_ids = device_ids.split(",")


        if (
            hasattr(cache_config.model_cfg, "num_key_value_heads")
            and hasattr(cache_config.model_cfg, "num_key_value_heads")
            and cache_config.model_cfg.num_key_value_heads is not None
            and int(cache_config.model_cfg.num_key_value_heads) > 0
        ):
            kv_num_head = int(cache_config.model_cfg.num_key_value_heads) // tensor_parallel_size
        else:
            kv_num_head = cache_config.model_cfg.num_attention_heads // tensor_parallel_size


        py_launcher = "python3"
        flag_array = np.zeros([tensor_parallel_size], dtype=np.int32)

        cache_ready_signal_data = np.zeros(
            shape=[tensor_parallel_size], dtype=np.int32)
        self.cache_ready_signal = IPCSignal(name="cache_ready_signal",
                                             array=cache_ready_signal_data,
                                             dtype=np.int32,
                                             suffix=pid_suffix,
                                             create=True)
        log_dir = os.getenv("FD_LOG_DIR", "log")
        for i in range(tensor_parallel_size):
            launch_cmd = (
                f"FLAGS_allocator_strategy=auto_growth CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
                + " NCCL_MAX_NCHANNELS=1 NCCL_BUFFSIZE=0"
                + f" {py_launcher} {py_path}"
                + f" --device_id {int(device_ids[i])}"
                + f" --rank {i}"
                + f" --num_layers {cache_config.model_cfg.num_layers}"
                + f" --num_attention_heads {cache_config.model_cfg.num_attention_heads}"
                + f" --hidden_size {cache_config.model_cfg.hidden_size}"
                + f" --kv_num_head {kv_num_head}"
                + f" --mp_num {tensor_parallel_size}"
                + f" --cache_dtype {cache_config.cache_dtype}"
                + f" --cache_queue_port {cache_config.cache_queue_port}"
                + f" --enable_splitwise {int(self.enable_splitwise)}"
                + f" --engine_worker_queue_port {engine_worker_queue_port}"
                + f" --num_gpu_blocks {cache_config.total_block_num}"
                + f" --num_cpu_blocks {cache_config.num_cpu_blocks}"
                + f" --bytes_per_layer_per_block {cache_config.bytes_per_layer_per_block}"
                + f" --block_size {cache_config.block_size}"
                + f" --engine_pid {pid_suffix}"
                + f" >{log_dir}/launch_cache_manager_{int(device_ids[i])}.log 2>&1"
            )
            logger.info(f"Launch cache transfer manager, command:{launch_cmd}")
            cache_manager_process = subprocess.Popen(
                launch_cmd, shell=True, preexec_fn=os.setsid
            )
        # 等待cache初始化完毕
        logger.info(f"Waiting for cache transfer manager ready...")
        while np.sum(self.cache_ready_signal.value) != tensor_parallel_size:
            time.sleep(1)
        exit_code = cache_manager_process.poll()
        if exit_code is None:
            logger.info(f"Launch cache transfer manager successful")
        else:
            logger.info(
                f"Launch cache transfer manager failed, see launch_cache_manager.log for more information"
            )

        if cache_config.enable_hierarchical_cache and self.num_cpu_blocks > 0:
            logger.info("Enable hierarchical cache.")
            self._enable_cpu_cache(tensor_parallel_size)

    def update_cache_config(self, cache_config):
        """
        update cache config
        """
        self.cache_config = cache_config
        self.num_gpu_blocks = cache_config.prefill_kvcache_block_num
        self.gpu_free_block_list = list(
            range(self.num_gpu_blocks - 1, -1, -1)
        )  # 服务端管理的GPU上剩余的block id

        heapq.heapify(self.gpu_free_block_list)
        self.node_id_pool = list(
            range(self.num_gpu_blocks + self.num_cpu_blocks)
        )


    def _enable_cpu_cache(self, tensor_parallel_size):
        """
        开启cpu缓存
        """

        ipc_cache_queue_port = self.cache_config.cache_queue_port
        self.cache_task_queue = CacheQueueManager(
            rank=0,
            mp_num=tensor_parallel_size,
            port=ipc_cache_queue_port,
        )
        # 开启获取传输任务结果的监听线程
        self.transfer_recv_thread = threading.Thread(
            target=self.recv_data_transfer_result
        )
        self.transfer_recv_thread.start()

    def allocate_gpu_blocks(self, num_blocks):
        """
        分配GPU上的block
        参数：
            num_blocks: 需要分配的block数量
        """
        assert num_blocks <= len(
            self.gpu_free_block_list
        ), f"gpu free block num: {len(self.gpu_free_block_list)} < needed number {num_blocks}"
        allocated_block_ids = [
            heapq.heappop(self.gpu_free_block_list) for i in range(num_blocks)
        ]
        logger.info(
            f"allocate_gpu_blocks: {allocated_block_ids}, len(self.gpu_free_block_list) {len(self.gpu_free_block_list)}"
        )
        return allocated_block_ids

    def recycle_gpu_blocks(self, gpu_block_ids):
        """
        回收GPU上的block
        参数：
            gpu_block_ids: 需要回收的block id list
        """
        logger.info(
            f"recycle_gpu_blocks: {gpu_block_ids}, len(self.gpu_free_block_list) {len(self.gpu_free_block_list)}"
        )
        if isinstance(gpu_block_ids, list):
            for gpu_block_id in gpu_block_ids:
                heapq.heappush(self.gpu_free_block_list, gpu_block_id)
        else:
            heapq.heappush(self.gpu_free_block_list, gpu_block_ids)

    def allocate_cpu_blocks(self, num_blocks):
        """
        分配CPU上的block
        参数：
            num_blocks: 需要分配的block数量
        """
        assert num_blocks <= len(
            self.cpu_free_block_list
        ), f"cpu free block num: {len(self.cpu_free_block_list)} < needed number {num_blocks}"
        allocated_block_ids = [
            heapq.heappop(self.cpu_free_block_list) for i in range(num_blocks)
        ]
        logger.info(
            f"allocate_cpu_blocks: {allocated_block_ids}, len(self.cpu_free_block_list) {len(self.cpu_free_block_list)}"
        )
        return allocated_block_ids

    def recycle_cpu_blocks(self, cpu_block_ids):
        """
        回收CPU上的block
        参数：
            cpu_block_ids: 需要回收的block id list
        """
        logger.info(
            f"recycle_cpu_blocks: {cpu_block_ids}, len(self.cpu_free_block_list) {len(self.cpu_free_block_list)}"
        )
        if isinstance(cpu_block_ids, list):
            for cpu_block_id in cpu_block_ids:
                heapq.heappush(self.cpu_free_block_list, cpu_block_id)
        else:
            heapq.heappush(self.cpu_free_block_list, cpu_block_ids)


    def issue_swap_task(
        self,
        transfer_task_id,
        swap_node_ids,
        gpu_block_ids,
        cpu_block_ids,
        event_type,
        is_sync=True,
    ):
        """
        发起数据交换任务
        参数：
            transfer_task_id: 传输任务id
            swap_node_ids:    待交换节点的node id list
            gpu_block_ids:    待交换的gpu block id list
            cpu_block_ids:    待交换的cpu block id list
            event_type:       交换类型，CacheStatus.SWAP2GPU or CacheStatus.SWAP2CPU
            is_sync:          是否同步等待数据传输完成
        """

        self.task_swapping_event[transfer_task_id] = Event()
        self.cache_task_queue.put_transfer_task(
            (
                swap_node_ids,
                gpu_block_ids,
                cpu_block_ids,
                event_type,
                transfer_task_id,
            )
        )  # 发起数据传输任务
        if is_sync:
            self.sync_swap_task(transfer_task_id)
        return

    def sync_swap_task(self, transfer_task_id):
        """
        同步数据交换任务
        当issue_swap_task中设置is_sync为False时需主动调用该函数同步结果
        """
        self.task_swapping_event[transfer_task_id].wait()
        del self.task_swapping_event[transfer_task_id]

    def _check_validity(self, req_id, match_gpu_blocks_num, expected_block_num):
        """
        检查是否有足够的GPU内存来分配cache
        match_gpu_blocks_num: 在前缀树中命中的GPU block数量
        expected_block_num:   请求需要的总GPU block数量
        """
        if expected_block_num - match_gpu_blocks_num > len(self.gpu_free_block_list):
            msg = (
                f"request_block_ids: request block for req_id {req_id} failed. "
                + f"matched gpu block num: {match_gpu_blocks_num} require extra gpu block num: "
                + f"{expected_block_num - match_gpu_blocks_num} > free block num: {len(self.gpu_free_block_list)}"
            )
            logger.info(msg)
            raise Exception("Not enough GPU memory to allocate cache")  # 不够分配的情况下，报异常

    
    def _prepare_cpu_cache(self, req_id, swap_node_ids, gpu_recv_block_ids, \
                cpu_recv_block_ids, match_cpu_block_ids, ssd_read_block_num):
        """
        将cpu cache转移到GPU
        """
        transfer_task_id = req_id
        need_transfer_task_gpu_block_ids = []
        need_transfer_task_cpu_block_ids = []

        for tmp_gpu_block_id in gpu_recv_block_ids:
            need_transfer_task_gpu_block_ids.append(tmp_gpu_block_id)
        for tmp_cpu_block_id in match_cpu_block_ids:
            need_transfer_task_cpu_block_ids.append(tmp_cpu_block_id)
        if ssd_read_block_num > 0:
            for tmp_cpu_block_id in cpu_recv_block_ids[:ssd_read_block_num]:
                swap_node_ids.append(None)
                need_transfer_task_cpu_block_ids.append(
                    tmp_cpu_block_id
                )
        assert len(need_transfer_task_gpu_block_ids) == len(
            need_transfer_task_cpu_block_ids
        )
        logger.info(
            f"request_block_ids: req_id {req_id} issue_swap_task transfer_task_id {transfer_task_id}"
        )
        self.issue_swap_task(
            transfer_task_id,
            swap_node_ids,
            need_transfer_task_gpu_block_ids,
            need_transfer_task_cpu_block_ids,
            CacheStatus.SWAP2GPU,
            True,
        )

    def _prepare_cache(self, req_id, input_ids, block_size, \
        expected_block_num, match_gpu_block_ids, match_cpu_block_ids, match_node_ids):
        """
        准备可复用cache到GPU中
        """
        ssd_read_block_num = 0
        ssd_match_token_num = 0
        match_gpu_blocks_num = len(match_gpu_block_ids)
        match_cpu_blocks_num = len(match_cpu_block_ids)
        matched_block_num = match_gpu_blocks_num + match_cpu_blocks_num
        
        cpu_recv_block_ids = []  # 用来接收ssd上缓存的cpu block
        gpu_recv_block_ids = []  # 用来接收cpu上缓存的gpu block
        gpu_extra_block_ids = []
        

        # 分配用来接收cpu上匹配到的cache的gpu block
        if match_cpu_blocks_num > 0:
            gpu_recv_block_ids = self.allocate_gpu_blocks(
                match_cpu_blocks_num
            )
        # 分配用来接收未匹配到的部分的block
        gpu_extra_block_num = expected_block_num - matched_block_num
        if gpu_extra_block_num > 0:
            gpu_extra_block_ids = self.allocate_gpu_blocks(
                gpu_extra_block_num
            )

        if len(gpu_recv_block_ids) > 0:
            # 发起swap操作
            self._prepare_cpu_cache(req_id, match_node_ids, gpu_recv_block_ids, \
                        cpu_recv_block_ids, match_cpu_block_ids, ssd_read_block_num)
        
        return gpu_recv_block_ids, gpu_extra_block_ids, ssd_match_token_num
        

    def request_block_ids(self, task, block_size, dec_token_num, *args):
        """
        为任务申请block。
        该接口为同步接口，如果发生cpu到gpu的数据交换，则会阻塞等待数据同步完成。调用方如果需要实现异步效果，请使用线程池来调用。
        参数：
            task: 任务的dict
            block_size: 每个block的大小
            dec_token_num: 在server侧给解码预留的token数量
        返回：
            common_block_ids: 匹配上的公共block list
            unique_block_ids: 单独分配的block list
        """
        with self.request_release_lock:
            try:
                hit_info = {}
                hit_info["gpu_cache_blocks"] = 0
                hit_info["cpu_cache_blocks"] = 0
                hit_info["ssd_cache_blocks"] = 0
                self.metrics.req_count += 1
                input_ids = task.prompt_token_ids
                req_id = task.request_id
                logger.info(
                    f"request_block_ids: start to allocate blocks for req_id {req_id}"
                )
                input_token_num = len(input_ids)  
                common_block_ids = []
                unique_block_ids = []
                # 1. 匹配可复用的block
                (
                    match_gpu_block_ids,
                    match_cpu_block_ids,
                    swap_node_ids,
                    match_block_node,
                    gpu_match_token_num,
                    cpu_match_token_num,
                ) = self.match_block(req_id, input_ids, block_size)
                match_gpu_blocks_num = len(match_gpu_block_ids)
                match_cpu_blocks_num = len(match_cpu_block_ids)
                matched_block_num = match_gpu_blocks_num + match_cpu_blocks_num
                matched_token_num_in_cpu_and_gpu = gpu_match_token_num + cpu_match_token_num
                # 检查合法性
                block_num = (
                    input_token_num + block_size - 1 + dec_token_num
                ) // block_size
                self._check_validity(req_id, matched_block_num, block_num)
                # 更新共享节点的信息
                current_time = time.time()
                self._update_matched_node_info(req_id, match_block_node, current_time)
                # 将可复用的cache移到GPU
                gpu_recv_block_ids, gpu_extra_block_ids, ssd_match_token_num = self._prepare_cache(req_id, \
                    input_ids, block_size, block_num, match_gpu_block_ids, match_cpu_block_ids, swap_node_ids)
                # 更新matched_block_num （加上ssd读取到的部分）
                matched_block_num = (
                    gpu_match_token_num + cpu_match_token_num + ssd_match_token_num
                )
                # 2. 为不在前缀树中的token分配node
                common_block_ids = match_gpu_block_ids + gpu_recv_block_ids
                unique_block_ids = gpu_extra_block_ids
                
                dec_block_num = dec_token_num // block_size
                left_input_ids = input_ids[matched_token_num_in_cpu_and_gpu:]  # 没在前缀树中的token
                gpu_build_path_block_ids = []

                gpu_build_path_block_ids = gpu_extra_block_ids

                leaf_node = self.build_path(
                    req_id,
                    current_time,
                    input_ids,
                    left_input_ids,
                    gpu_build_path_block_ids,
                    block_size,
                    match_block_node,
                    dec_block_num
                )  # 建立剩余节点的路径，返回叶子节点
                self.req_leaf_map[
                    req_id
                ] = leaf_node  # 申请block的时候，创建好当前的req_id 和 叶子节点的映射关系，由于有block没有在树中展开，后续匹配的时候如果展开了节点这一关系需要发生更新
                self.leaf_req_map[leaf_node].add(req_id)
                # 3. 更新统计指标
                if matched_block_num > 0:
                    self.metrics.hit_req_count += 1
                self.metrics.calculate_hit_metrics(
                    req_id,
                    cpu_match_token_num,
                    gpu_match_token_num,
                    ssd_match_token_num,
                    input_token_num,
                )
                hit_info["gpu_cache_blocks"] = gpu_match_token_num // block_size
                hit_info["cpu_cache_blocks"] = cpu_match_token_num // block_size
                hit_info["ssd_cache_blocks"] = ssd_match_token_num // block_size
                self.metrics._update_history_hit_metrics()
                if self.metrics.req_count % 10000 == 0:
                    self.metrics._init_histroy_hit_metrics()  # 每10000个请求重置一次指标计算
                logger.info(
                    f"request_block_ids: request block for req_id {req_id}: common_block_ids "
                    + f"{common_block_ids}, unique_block_ids {unique_block_ids}"
                )
                return common_block_ids, unique_block_ids, hit_info
            except Exception as e:
                logger.error(f"request_block_ids: error: {type(e)} {e}")
                raise e

    def release_block_ids_async(self, task):
        """
        异步接口
        """
        return self.executor_pool.submit(self.release_block_ids, task)

    def release_block_ids(self, task):
        """
        释放任务的block。该函数只负责降低节点的共享计数。
        shared_count是0时候，叶子节点可以放入gpu lru。
        参数：
            task: 任务的dict
        """
        with self.request_release_lock:
            try:
                req_id = task.request_id
                leaf_node = self.req_leaf_map.pop(req_id)  # 找到当前req_id对应的叶子节点
                if leaf_node in self.leaf_req_map:
                    self.leaf_req_map[leaf_node].remove(req_id)
                    if not (self.leaf_req_map[leaf_node]):
                        del self.leaf_req_map[leaf_node]
                node = leaf_node
                while node != self.radix_tree_root:
                    if req_id in node.req_id_set:
                        # 适配decoding加速
                        node.req_id_set.remove(req_id)
                    node.decrement_shared_count()
                    node = node.parent

                logger.info(f"release_block_ids: req_id {req_id} leaf_node {leaf_node}")

                if leaf_node == self.radix_tree_root:
                    # 直接回收block id
                    self.recycle_gpu_blocks(self.unfilled_req_block_map[req_id])
                    del self.unfilled_req_block_map[req_id]
                    return

                # 已经结束的任务放入lru中，等待后续被free
                if leaf_node in self.gpu_lru_leaf_set:  # 已经存在了
                    return
                if (
                    leaf_node.shared_count == 0
                    and leaf_node.is_gpu_leaf_node
                    and leaf_node.is_persistent is False
                ):
                    self.gpu_lru_leaf_set.add(leaf_node)
                    heapq.heappush(self.gpu_lru_leaf_heap, leaf_node)
                logger.info(
                    f"release_block_ids: req_id {req_id} has been finished, "
                    + f"current gpu_lru_leaf_heap length {len(self.gpu_lru_leaf_heap)}"
                )
                return
            except Exception as e:
                logger.error(f"release_block_ids: error: {type(e)} {e}")
                raise e
    
    def _handle_free_gpu_node_without_cpu(self, node):
        """
        单级缓存下驱逐gpu node
        """
        node.cache_status = CacheStatus.CPU  # 更改node状态
        # 回收当前节点的node_id
        self.node_id_pool.append(node.node_id)
        if node.node_id in self.node_map:
            del self.node_map[node.node_id]
        logger.info(
            f"free_block_ids_async: free node {node}"
        )
        # 回收分配出去的block id
        self.recycle_gpu_blocks(node.reverved_dec_block_ids)
        node.reverved_dec_block_ids = []
        self.recycle_gpu_blocks(node.block_id)

    def _handle_free_gpu_node_with_cpu(self, node, hash_value_input_ids_map, \
        hash_value_depth_map, need_recycle_gpu_block_ids, hash_value_gpu_block_ids_map, hash_value_swap_node_ids_map):
        """
        多级缓存下驱逐gpu node
        """

        # 给dec预留的block直接回收
        self.recycle_gpu_blocks(node.reverved_dec_block_ids)
        node.reverved_dec_block_ids = []

        # 准备数据传输任务
        need_recycle_gpu_block_ids.append(node.block_id)
        hash_value_gpu_block_ids_map[node.input_hash_value].append(
            node.block_id
        )
        hash_value_swap_node_ids_map[node.input_hash_value].append(
            node.node_id
        )

    def _evict_cache_async(self, future, total_gpu_free_count, \
        hash_value_gpu_block_ids_map, hash_value_block_ids_map, \
        hash_value_swap_node_ids_map, hash_value_input_ids_map, hash_value_depth_map):
        """
        异步执行GPU->CPU的swap out
        """
        if future is not None:
            future.result()  # 等待cpu上驱逐任务结束（cpu->ssd)
        transfer_task_id = str(
            uuid.uuid4()
        )  # 因为free时没有req_id, 生成一个唯一id作为传输任务的唯一id
        swap_node_ids = []
        need_transfer_task_gpu_block_ids = []
        need_transfer_task_cpu_block_ids = []
        cpu_block_ids = self.allocate_cpu_blocks(total_gpu_free_count)
        for input_hash_value in hash_value_gpu_block_ids_map.keys():
            need_transfer_task_gpu_block_ids.extend(
                reversed(hash_value_gpu_block_ids_map[input_hash_value])
            )
            all_allocated_cpu_block_ids = []
            for _ in reversed(
                hash_value_gpu_block_ids_map[input_hash_value]
            ):
                cpu_block_id_t = cpu_block_ids.pop(0)
                all_allocated_cpu_block_ids.append(
                    cpu_block_id_t
                )
                need_transfer_task_cpu_block_ids.append(
                    cpu_block_id_t
                )

            swap_node_ids.extend(
                reversed(hash_value_swap_node_ids_map[input_hash_value])
            )
        logger.info(
            f"free_block_ids_async: issue transfer task: "
            + f"transfer_task_id {transfer_task_id}: "
            + f"swap_node_ids {swap_node_ids} need_transfer_task_gpu_block_ids "
            + f"{need_transfer_task_gpu_block_ids}, need_transfer_task_cpu_block_ids "
            + f"{need_transfer_task_cpu_block_ids}, CacheStatus.SWAP2CPU"
        )
        self.issue_swap_task(
            transfer_task_id,
            swap_node_ids,
            need_transfer_task_gpu_block_ids,
            need_transfer_task_cpu_block_ids,
            CacheStatus.SWAP2CPU,
            True,
        )

        logger.info(
            f"free_block_ids_async: after free, "
            + f"len(self.gpu_free_block_list) {len(self.gpu_free_block_list)}"
        )
        return

    def free_block_ids_async(self, need_block_num):
        """
        异步清理已经分配出去的block，清理最多need_block_num个gpu block
        参数：
            need_query_block_num: 需要驱逐的gpu block数量
        返回：
            Event
        """
        with self.request_release_lock:
            if self.gpu_free_task_future is not None:
                if not self.gpu_free_task_future.done():
                    return
                else:
                    self.gpu_free_task_future.result()
                    self.gpu_free_task_future = None
            try:
                need_recycle_gpu_block_ids = []

                hash_value_input_ids_map = {}
                hash_value_block_ids_map = defaultdict(list)
                hash_value_depth_map = {}
                # 用于加速swap，将连续的block id尽可能放到一起
                hash_value_swap_node_ids_map = defaultdict(list)
                hash_value_gpu_block_ids_map = defaultdict(list)
                total_gpu_free_count = 0

                # 清理lru中未被使用的节点，清理need_block_num个block
                while True:
                    if len(self.gpu_lru_leaf_heap) == 0:
                        # 没有可以被删除的路径了
                        break
                    if total_gpu_free_count >= need_block_num:
                        break
                    # 弹出lru的叶子节点
                    node = heapq.heappop(self.gpu_lru_leaf_heap)
                    self.gpu_lru_leaf_set.remove(node)
                    if (
                        not self.cache_config.enable_hierarchical_cache
                    ):  # 没开多级cache存储，直接回收block
                        if node.shared_count == 0 and node.is_gpu_leaf_node:  # 直接回收
                            self._handle_free_gpu_node_without_cpu(node)
                            total_gpu_free_count += 1
                            cur_node = node
                            node = node.parent
                            if cur_node.hash_value in node.children:
                                del node.children[cur_node.hash_value]  # 父节点中删除当前子节点
                            if not node.children:  # 没有孩子节点了，是新的叶子节点
                                # 将新的叶子节点入堆
                                if node in self.gpu_lru_leaf_set:  # 已经存在了
                                    continue
                                if (
                                    node != self.radix_tree_root
                                    and node.shared_count == 0
                                    and node.is_gpu_leaf_node
                                    and node.is_persistent is False
                                ):
                                    heapq.heappush(self.gpu_lru_leaf_heap, node)
                                    self.gpu_lru_leaf_set.add(node)
                        else:
                            continue
                    else:
                        if node.shared_count == 0 and node.is_gpu_leaf_node:  # 可以被调度出去
                            node.cache_status = CacheStatus.SWAP2CPU  # 更改node状态
                        else:  # 有引用，不能被调度出去
                            continue
                        self._handle_free_gpu_node_with_cpu(node, hash_value_input_ids_map, \
                            hash_value_depth_map, need_recycle_gpu_block_ids, \
                            hash_value_gpu_block_ids_map, hash_value_swap_node_ids_map)
                        total_gpu_free_count += 1
                        
                        # 将新的gpu节点入堆
                        node = node.parent
                        if node in self.gpu_lru_leaf_set:  # 已经存在了
                            continue
                        if (
                            node != self.radix_tree_root
                            and node.shared_count == 0
                            and node.is_gpu_leaf_node
                            and node.is_persistent is False
                        ):
                            heapq.heappush(self.gpu_lru_leaf_heap, node)
                            self.gpu_lru_leaf_set.add(node)

                # 2. 发起异步GPU->CPU的驱逐任务
                if hash_value_gpu_block_ids_map:
                    # 1. 判断需要转移的节点，是否cpu缓存空间足够
                    cpu_free_future = None
                    if total_gpu_free_count > len(self.cpu_free_block_list):
                        # 需要释放部分cpu缓存空间
                        cpu_free_count = total_gpu_free_count
                        if cpu_free_count < need_block_num:
                            cpu_free_count = need_block_num
                        cpu_free_future = self.free_cpu_executor_pool.submit(
                            self.free_cpu_block_ids, cpu_free_count
                        )
                    self.gpu_free_task_future = self.free_gpu_executor_pool.submit(
                        self._evict_cache_async, cpu_free_future, total_gpu_free_count, \
                        hash_value_gpu_block_ids_map, hash_value_block_ids_map, \
                        hash_value_swap_node_ids_map, hash_value_input_ids_map, hash_value_depth_map
                    )
                else:
                    self.gpu_free_task_future = None
            except Exception as e:
                logger.error(f"free_block_ids_async: error: {type(e)} {e}")
                raise e

    def free_cpu_block_ids(self, need_block_num):
        """
        驱逐cpu block，至少need_block_num个block
        参数：
            need_block_num: 需要驱逐的cpu block数量
        返回：
            freed_block_num: 驱逐的cpu block数量
        """
        hash_value_input_ids_map = {}
        hash_value_block_ids_map = defaultdict(list)
        hash_value_depth_map = {}
        need_recycle_cpu_block_ids = []
        total_cpu_free_count = 0
        with self.request_release_lock:
            while True:
                if len(self.cpu_lru_leaf_heap) == 0:
                    # 没有可以被删除的节点了
                    break
                if total_cpu_free_count >= need_block_num:
                    break
                # 弹出lru的叶子节点
                node = heapq.heappop(self.cpu_lru_leaf_heap)
                self.cpu_lru_leaf_set.remove(node)
                tmp_block_ids = []
                if (
                    node.shared_count == 0
                    and node.cache_status == CacheStatus.CPU
                    and node.is_cpu_leaf_node
                ):
                    if self.cache_config.enable_ssd_cache:
                        tmp_block_ids.append(node.block_id)
                        hash_value_input_ids_map[node.input_hash_value] = node.input_ids
                        hash_value_depth_map[
                            node.input_hash_value
                        ] = node.depth  # 最后更新的是越靠近树根的节点
                        need_recycle_cpu_block_ids.append(node.block_id)
                    else:
                        self.recycle_cpu_blocks(node.block_id)
                    hash_value_block_ids_map[node.input_hash_value].extend(
                        reversed(tmp_block_ids)
                    )  # 从叶子到树根的节点
                    logger.info(f"free_cpu_block_ids: free node {node}")
                    # 回收当前节点的node_id
                    self.node_id_pool.append(node.node_id)
                    total_cpu_free_count += 1
                    if node.node_id in self.node_map:
                        del self.node_map[node.node_id]
                    cur_node = node
                    node = node.parent
                    if cur_node.hash_value in node.children:
                        del node.children[cur_node.hash_value]  # 父节点中删除当前子节点
                    if not node.children:  # 没有孩子节点了，是新的叶子节点
                        if node in self.cpu_lru_leaf_set:  # 已经存在了（可能是从别的路径放入的）
                            continue
                        # 将新的叶子节点入堆
                        if (
                            node != self.radix_tree_root
                            and node.shared_count == 0
                            and node.is_cpu_leaf_node
                            and node.cache_status == CacheStatus.CPU
                        ):
                            heapq.heappush(self.cpu_lru_leaf_heap, node)
                            self.cpu_lru_leaf_set.add(node)
        logger.info(
            f"free_cpu_block_ids: after free, "
            + f"len(self.cpu_free_block_list) {len(self.cpu_free_block_list)}"
        )
        return total_cpu_free_count

    def cal_block_hash(self, block):
        """
        block: input_ids组成的block
        """
        return hash(tuple(block))

    def match_block(self, req_id, input_ids, block_size):
        """
        匹配input_ids在前缀树里的公共部分
        参数：
            req_id: 任务的req_id
            input_ids: 输入的token ids
            block_size: 每个block的大小
        返回：
            match_gpu_block_ids:   匹配到的公共gpu block id list
            match_cpu_block_ids:   匹配到的公共cpu block id list
            swap_node_ids: 需要做swap交换的node id list
            match_block_node: 匹配到的最后一个节点
            gpu_match_token_num: gpu匹配到的token数
            cpu_match_token_num: cpu匹配到的token数
        """

        total_token_num = len(input_ids)
        current_match_node = self.radix_tree_root  # 从根节点开始搜
        match_gpu_block_ids = []
        match_cpu_block_ids = []
        match_node_ids = []
        match_token_num = 0
        cpu_match_token_num = 0
        gpu_match_token_num = 0
        swap_node_ids = []
        matche_nodes = []
        has_modified_gpu_lru_leaf_heap = False
        has_modified_cpu_lru_leaf_heap = False

        with self.cache_status_lock:
            while match_token_num < total_token_num:
                token_block = input_ids[match_token_num : match_token_num + block_size]
                token_num = len(token_block)
                if token_num != block_size:
                    break
                hash_value = self.cal_block_hash(token_block)
                if hash_value in current_match_node.children:  # 匹配上节点
                    child = current_match_node.children[hash_value]
                    matche_nodes.append(child)
                    match_node_ids.append(child.node_id)
                    if (
                        child in self.gpu_lru_leaf_set
                    ):  # 之前在lru中的叶子节点匹配上了，删掉lru中的这个node, 因为它肯定不会是叶子节点了
                        self.gpu_lru_leaf_set.remove(child)
                        self.gpu_lru_leaf_heap.remove(child)
                        has_modified_gpu_lru_leaf_heap = True
                    elif (
                        child in self.cpu_lru_leaf_set
                    ):  # 之前在lru中的叶子节点匹配上了，删掉lru中的这个node, 因为它肯定不会是叶子节点了
                        self.cpu_lru_leaf_set.remove(child)
                        self.cpu_lru_leaf_heap.remove(child)
                        has_modified_cpu_lru_leaf_heap = True
                    if child.has_in_gpu:  # 在gpu上
                        match_gpu_block_ids.append(child.block_id)
                        gpu_match_token_num += block_size
                    else:
                        if child.cache_status == CacheStatus.SWAP2CPU:  # 在从GPU到CPU转移
                            logger.info(f"match_block: req_id {req_id} matched node"
                                         + f" {child.node_id} which is being SWAP2CPU")
                            child.cache_status = CacheStatus.GPU  # 状态置为GPU
                            match_gpu_block_ids.append(child.block_id)
                            gpu_match_token_num += block_size
                        elif child.cache_status == CacheStatus.CPU:  # 在cpu上
                            child.cache_status = CacheStatus.SWAP2GPU
                            match_cpu_block_ids.append(child.block_id)
                            cpu_match_token_num += block_size
                            swap_node_ids.append(child.node_id)
                    match_token_num = match_token_num + block_size  # 当前匹配都的token数
                    current_match_node = child
                else:  # 没有匹配的节点了
                    break

        if has_modified_gpu_lru_leaf_heap:  # 重新构建gpu lru heap
            heapq.heapify(self.gpu_lru_leaf_heap)  # 重新构建堆
        if has_modified_cpu_lru_leaf_heap:  # 重新构建cpu lru heap
            heapq.heapify(self.cpu_lru_leaf_heap)
        
        logger.info(f"match_block: req_id {req_id} matched nodes: {match_node_ids}")
        return (
            match_gpu_block_ids,
            match_cpu_block_ids,
            swap_node_ids,
            current_match_node,
            gpu_match_token_num,
            cpu_match_token_num,
        )

    def _update_matched_node_info(self, req_id, last_node, current_time):
        """
        更新匹配节点的信息
        """
        node = last_node
        while node != self.radix_tree_root:
            node.increment_shared_count()
            node.last_used_time = current_time
            node.req_id_set.add(req_id)
            node = node.parent


    def build_path(
        self,
        req_id,
        current_time,
        input_ids,
        left_input_ids,
        gpu_block_ids,
        block_size,
        last_node,
        reverved_dec_block_num
    ):
        """
        公共前缀之外的block建立路径
        参数:
            req_id: 任务的req_id
            left_input_ids: 剩下的没在前缀树中的输入
            gpu_block_ids:  构建新路径时候可供新节点分配的gpu block id列表
            block_size: 每个block的token数
            last_node: 最后一个匹配成功的节点
            reverved_dec_block_num: 预留给解码的block数量
        返回：
            leaf_node: 叶子节点
        """
        gpu_block_ids = gpu_block_ids.copy()
        node = last_node
        reverved_dec_block_ids = []
        input_hash_value = self.cal_block_hash(input_ids)
        logger.info(f"{gpu_block_ids} {input_hash_value}")
        
        # 为剩余token 分配节点、建立路径：填充hash值、时间戳
        token_num = len(left_input_ids)
        if token_num == 0:
            for i in range(reverved_dec_block_num):
                reverved_dec_block_ids.append(gpu_block_ids.pop(0))
            last_node.reverved_dec_block_ids.extend(reverved_dec_block_ids)
            return last_node
        node = last_node
        unique_node_ids = []
        new_last_node = last_node
        has_unfilled_block = False

        for i in range(0, token_num, block_size):
            current_block = left_input_ids[i : i + block_size]
            current_block_size = len(current_block)  # 最后一个block可能没填满
            if current_block_size != block_size:
                has_unfilled_block = True
            else:
                hash_value = self.cal_block_hash(current_block)
                allocated_block_id = gpu_block_ids.pop(0)
                node_id = self.node_id_pool.pop()
                unique_node_ids.append(node_id)
                new_last_node = BlockNode(
                    node_id,
                    input_ids,
                    input_hash_value,
                    node.depth + 1,
                    allocated_block_id,
                    current_block_size,
                    hash_value,
                    current_time,
                    parent=node,
                    shared_count=1,
                    reverved_dec_block_ids=[]
                )
                new_last_node.req_id_set.add(req_id)
                self.node_map[node_id] = new_last_node
                node.children[hash_value] = new_last_node
                node = new_last_node
        if has_unfilled_block is True:
            reverved_dec_block_ids.append(gpu_block_ids.pop(0))  # 最后一个未填满的block
        # 给解码分配block
        for i in range(reverved_dec_block_num):
            reverved_dec_block_ids.append(gpu_block_ids.pop(0))
        if new_last_node == self.radix_tree_root:  # 输入token数量就小于block size
            self.unfilled_req_block_map[req_id] = reverved_dec_block_ids
        else:
            new_last_node.reverved_dec_block_ids.extend(reverved_dec_block_ids)
        logger.info(
            f"build_path: allocate unique node ids {unique_node_ids} for req_id {req_id}"
        )
        # 3. 返回叶子节点
        return new_last_node


    def _handle_swap_result(
        self, swap_node_id, task_gpu_block_id, task_cpu_block_id, event_type
    ):
        """
        处理swap后的结果
        """
        if swap_node_id is None:  # 是None的时候，说明不需要处理Node，只单纯做block的swap
            return
        with self.cache_status_lock:
            if (
                event_type.value == CacheStatus.SWAP2CPU.value
            ):  # 只有在请求处理完毕的时候可能会触发GPU->CPU
                # block转变成存储到CPU
                # 1. 找到该节点
                gpu_block_id = task_gpu_block_id
                cpu_block_id = task_cpu_block_id
                node = self.node_map[swap_node_id]
                if node.cache_status.value == CacheStatus.GPU.value: # 在SWAP2CPU时，节点被新进来的query复用
                    # 回收cpu block id
                    logger.info(f"recv_data_transfer_result: node {node.node_id} "
                                    + f"has been reused when SWAP2CPU, recycle cpu block id {cpu_block_id}")
                    self.recycle_cpu_blocks(cpu_block_id)  # 回收分配的cpu block id
                else:
                    node.cache_status = CacheStatus.CPU  # 代表已在CPU上
                    node.block_id = cpu_block_id  # 记录为cpu的block id
                    # 将新的叶子节点入堆
                    if (
                        node != self.radix_tree_root
                        and node.shared_count == 0
                        and node.is_cpu_leaf_node
                        and node.cache_status == CacheStatus.CPU
                    ):
                        if node not in self.cpu_lru_leaf_set:
                            heapq.heappush(self.cpu_lru_leaf_heap, node)
                            self.cpu_lru_leaf_set.add(node)
                    # 回收gpu block id
                    self.recycle_gpu_blocks(gpu_block_id)
                    logger.info(
                        f"recv_data_transfer_result: after SWAP2CPU, node {node}"
                    )

            elif (
                event_type.value == CacheStatus.SWAP2GPU.value
            ):  # 只有在请求打进来的时候可能会触发CPU->GPU
                # block转变成存储到GPU
                gpu_block_id = task_gpu_block_id
                cpu_block_id = task_cpu_block_id
                # 1. 找到该节点
                node = self.node_map[swap_node_id]
                node.cache_status = CacheStatus.GPU  # 代表已在GPU上
                node.block_id = gpu_block_id  # 记录为gpu的block id
                # 回收cpu block id
                self.recycle_cpu_blocks(cpu_block_id)
                logger.info(f"recv_data_transfer_result: after SWAP2GPU, node {node}")
            else:
                logger.warning(
                    f"recv_data_transfer_result: Get unexpected event type {event_type}"
                    + f", only SWAP2CPU and SWAP2GPU supported"
                )

    def recv_data_transfer_result(self):
        """
        接收数据的传输结果
        """
        while True:
            # 异步资源管理的原则
            # 1. 如果是需要分配资源，在异步传输之前分配
            # 2. 如果是需要回收资源，在异步传输完成后回收
            try:
                data = self.cache_task_queue.get_transfer_done_signal()
                if data is None:
                    time.sleep(0.001)
                    continue
                (
                    swap_node_ids,
                    task_gpu_block_id,
                    task_cpu_block_id,
                    event_type,
                    transfer_task_id,
                ) = data
                length = len(task_gpu_block_id)
                for i in range(length):
                    self._handle_swap_result(
                        swap_node_ids[i],
                        task_gpu_block_id[i],
                        task_cpu_block_id[i],
                        event_type,
                    )
                if transfer_task_id in self.task_swapping_event:
                    self.task_swapping_event[transfer_task_id].set()
                logger.info(
                    f"recv_data_transfer_result: transfer_task_id {transfer_task_id}: "
                    + f"task_node_ids {swap_node_ids} task_gpu_block_id {task_gpu_block_id} "
                    + f"task_cpu_block_id {task_cpu_block_id} event_type {event_type} done"
                )
            except Exception as e:
                logger.warning(f"recv_data_transfer_result: error: {e}")
                raise e


class BlockNode:
    """
    BlockNode类，用于存储每个节点的信息
    """
    def __init__(
        self,
        node_id,
        input_ids,
        input_hash_value,
        depth,
        block_id,
        token_num,
        hash_value,
        last_used_time,
        parent=None,
        shared_count=1,
        reverved_dec_block_ids=[],
        cache_status=CacheStatus.GPU,
        is_persistent=False,
        persistent_shared_count=0,
    ):
        """
        参数:
            node_id: 节点的标识符
            depth: 节点的深度
            block_id: 分配的block id (在cpu上则为cpu block id, 在gpu上则为gpu block id)
            token_num: 当前block的token数目
            hash_value: 当前block的hash值
            last_used_time: 最后一次使用的时间戳
            parent: 父节点
            shared_count: 该节点正在被使用中的请求计数
            reverved_dec_block_ids: 预分配保留给解码的block, 格式为[block_id, block_id,...]
            cache_status: 当前cache的状态，包括USING, SWAP2CPU, SWAP2GPU, FREE
            is_persistent: 是否是持久化存储的节点
            persistent_shared_count: 被持久化cache请求的计数
        """
        # 不可修改属性
        self.node_id = node_id
        self.depth = depth
        self.parent = parent
        self.hash_value = hash_value
        self.token_num = token_num
        self.input_ids = input_ids
        self.input_hash_value = input_hash_value

        # 可修改属性
        self.children = {}  # hash_value: node，保存子节点的hash值 和 对应的节点，方便匹配
        self.shared_count = shared_count  # 当请求进来匹配到时，+1， 当请求结束时 -1
        self.last_used_time = last_used_time  # 当请求进来匹配到时，更新时间戳
        self.block_id = block_id  # 在cpu上时，表示cpu的block id。在gpu上时，表示gpu的block id
        self.reverved_dec_block_ids = reverved_dec_block_ids  # 在GPU上时，保留给解码的block ids
        self.cache_status = cache_status  # 节点的状态，包括USING, SWAP2CPU, SWAP2GPU, FREE
        self.is_persistent = is_persistent  # 是否是持久化存储的节点
        self.persistent_shared_count = persistent_shared_count  # cache被持久化prompt引用的计数
        self.req_id_set = set()  # 记录该Node被哪些请求引用，可用于decoding加速

    def __lt__(self, other):
        """
        <支持，因为会放入堆中, 要支持比较
        """
        if self.last_used_time < other.last_used_time:
            return True
        elif self.last_used_time > other.last_used_time:
            return False
        else:
            return self.depth > other.depth  # block数量多的放前面

    def __str__(self):
        """
        返回描述节点的字符串信息
        """
        if self.parent is not None:
            parent_node_id = self.parent.node_id
        else:
            parent_node_id = None
        return (
            f"node_id {self.node_id}: depth {self.depth} hash_value {self.hash_value}"
            + f" shared_count {self.shared_count} is_gpu_leaf_node {self.is_gpu_leaf_node}"
            + f" is_cpu_leaf_node {self.is_cpu_leaf_node} block_id {self.block_id} "
            + f"has_in_gpu {self.has_in_gpu} "
            + f"cache_status {self.cache_status}  parent {parent_node_id} with children number "
            + f"{len(self.children)} req_id_set {self.req_id_set}"
        )

    @property
    def has_in_gpu(self):
        """
        是否在gpu上
        除了GPU状态，其余状态has_in_gpu都表示为False
        """
        return self.cache_status == CacheStatus.GPU

    def increment_shared_count(self):
        """
        增加共享计数
        """
        self.shared_count += 1

    def decrement_shared_count(self):
        """
        减少共享计数
        """
        self.shared_count -= 1

    @property
    def is_cpu_leaf_node(self):
        """
        是否是cpu上的叶子节点
        """
        if (self.cache_status == CacheStatus.CPU) and (len(self.children) == 0):
            return True
        return False

    @property
    def is_gpu_leaf_node(self):
        """
        是否是gpu上的叶子节点
        """
        if self.has_in_gpu is False:
            return False
        else:
            if len(self.children) == 0:  # 没有子节点
                return True
            for child in self.children.values():
                if child.has_in_gpu is True:  # 存在gpu
                    return False
            return True
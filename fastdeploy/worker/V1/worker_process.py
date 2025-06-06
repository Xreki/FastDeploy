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
import time
from typing import List

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet

from fastdeploy.config import LLMConfig
from fastdeploy.inter_communicator import EngineWorkerQueue as TaskQueue
from fastdeploy.inter_communicator import IPCSignal
from fastdeploy.utils import get_logger
from fastdeploy.worker.V1.GpuWorker import GpuWorker

logger = get_logger("worker_process", )


class PaddleDisWorkerProc():
    """
    Paddle Distrubuted wrapper for fastdeploy.worker.Worker,
        for handling single-node multi-GPU tensor parallel.
    The wrapper internally executea an event loop that continuously executes requests
        in the task queue. Control flow is transmitted by IPC.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
    ):
        self.llm_config = llm_config
        self.parallel_config = llm_config.parallel_config

        # Initialize distributed enviroment
        (self.rank, self.local_rank) = self.init_distributed_enviroment()

        # TODO(gongshaotian): Use worker factory to get worker
        self.worker = GpuWorker(llm_config=llm_config,
                                local_rank=self.local_rank,
                                rank=self.rank)

        # Initialize task queue
        task_address = ('0.0.0.0',
                        self.parallel_config.engine_worker_queue_port)
        self.task_queue = TaskQueue(address=task_address,
                                    is_server=False,
                                    num_client=self.rank,
                                    client_id=self.local_rank)
        # Initialize health status
        self.init_health_status()

    def init_health_status(self):
        """
        Initialize the health status of the worker.
        Worker Status:
            worker_ready_singnal:
            worker_healthy_live_signal:
            exist_task_signal:
            exist_swapped_task_signal:
            model_weights_status:
        """
        # init worker_ready_singnal
        workers_ready = np.zeros(shape=[self.rank], dtype=np.int32)
        self.worker_ready_singnal = IPCSignal(
            name="worker_ready_singnal",
            array=workers_ready,
            dtype=np.int32,
            suffix=self.parallel_config.engine_pid,
            create=False)
        self.worker_ready_singnal.value[self.local_rank] = 1

        # init worker_healthy_live_signal
        workers_alive = np.zeros(shape=[self.rank], dtype=np.int32)
        self.worker_healthy_live_signal = IPCSignal(
            name="worker_healthy_live_signal",
            array=workers_alive,
            dtype=np.int32,
            suffix=self.parallel_config.engine_pid,
            create=False)
        self.worker_healthy_live_signal.value[self.local_rank] = int(
            time.time())

        # init exist_task_signal
        workers_exist_task = np.zeros([1], dtype=np.int32)
        self.exist_task_signal = IPCSignal(
            name="exist_task_signal",
            array=workers_exist_task,
            dtype=np.int32,
            suffix=self.parallel_config.engine_pid,
            create=False)

        # init exist_swapped_task_signal
        workers_swapped_task = np.zeros(shape=[1], dtype=np.int32)
        self.exist_swapped_task_signal = IPCSignal(
            name="exist_swapped_task_signal",
            array=workers_swapped_task,
            dtype=np.int32,
            suffix=self.parallel_config.engine_pid,
            create=False)

        # init model_weights_status
        workers_model_weights = np.zeros(shape=[1], dtype=np.int32)
        self.model_weights_status = IPCSignal(
            name="model_weights_status",
            array=workers_model_weights,
            dtype=np.int32,
            suffix=self.parallel_config.engine_pid,
            create=False)

    def event_loop_normal(self):
        """ Main event loop for Paddle Distrubuted Workers.
        TODO(gongshaotian): support remote calling of functions that control worker.
        """
        # Currently, only support single node
        self.nnode = 1

        while True:
            if self.ranks > 1:
                # Synchronize before updating weights
                paddle.distributed.barrier()

            self.insert_step = False
            self.worker_healthy_live_signal.value[self.local_rank] = int(
                time.time())

            # The first worker detects whether there are tasks in the task queue
            mp_num_per_node = self.rank / self.nnode
            if self.local_rank % mp_num_per_node == 0:
                if self.task_queue.num_tasks() > 0:
                    if self.nnode > 1:
                        self.task_queue.read_finish_flag.set(1)
                    else:
                        self.exist_task_signal.value[0] = 1
            if self.rank > 1:
                # Synchronize the signal for other workers
                paddle.distributed.barrier()

            if self.exist_task_signal.value[
                    0] == 1 or self.task_queue.read_finish_flag.get() == 1:
                logger.info(f"Rank: {self.local_rank} Detected new requests.")
                self.insert_step = True

                tasks, read_finish = self.task_queue.get_tasks()
                if read_finish:
                    # Ensure that every worker get the task
                    self.task_queue.value[0] = 0
                    self.task_queue.read_finish_flag.set(0)

                req_dicts = []
                for req_dict, bsz in tasks:
                    num_running_requests = int(bsz)
                    req_dicts.extend(req_dict)
                logger.info(f"Rank: {self.local_rank}, num_running_requests: {num_running_requests}, " \
                            f"num_insert_requests: {len(req_dicts)}")

                # Process prefill inputs
                self.worker.preprocess_new_task(req_dicts)

            # Execute model to generate token. The generated token will be written to the buffer.
            # These generated tokens can be obtained through get_output op.
            self.worker.execute_model()

    def init_distributed_enviroment(self, seed=20) -> List[int]:
        """ Initialize Paddle Fleet and get rank of worker """
        # Global rank
        self.rank = dist.get_world_size()
        dist_strategy = fleet.DistributedStrategy()

        dist_strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": self.rank,
            "pp_degree": 1,
            "sharding_degree": 1,
        }

        # Set control in tensor parallel
        dist_strategy.tensor_parallel_configs = {"tensor_init_seed": seed}
        fleet.init(is_collective=True, strategy=dist_strategy)

        # Local rank
        self.local_rank = fleet.worker_index()

        return self.rank, self.local_rank

    def determine_num_available_blocks(self):
        """
        """
        # 1. Get available memory(bytes)
        available_kv_cache_memory = self.worker.determine_available_memory()

        # 2. Calculate the appropriate number of blocks
        kv_cache_spec_list = self.model_runner.get_kv_cache_spec()
        merged_layer_spec = kv_cache_spec_list[0].merge(kv_cache_spec_list)
        num_blocks_local = int(available_kv_cache_memory //
                               merged_layer_spec.block_memory_used)

        # 3. Send IPCSignal
        get_profile_block_num = np.zeros(shape=[self.rank], dtype=np.int32)
        self.get_profile_block_num_signal = IPCSignal(
            name="get_profile_block_num",
            array=get_profile_block_num,
            dtype=np.int32,
            suffix=self.parallel_config.engine_pid,
            create=False)
        self.get_profile_block_num_signal.value[
            self.local_rank] = num_blocks_local
        # wait all worker send the signal
        while np.any(self.get_profile_block_num_signal.value <= 0):
            time.sleep(0.01)
        num_blocks_global = self.get_profile_block_num_signal.value.min().item(
        )
        self.get_profile_block_num_signal.value[self.rank] = num_blocks_global

        # 4. Updata share inputs
        self.model_runner._update_share_input_block_num(
            block_num=num_blocks_global)

    def init_device(self):
        """ """
        self.worker.init_device()


def parse_args():
    """
    Parse args from command line
    """
    parser = argparse.ArgumentParser("FastDeploy LLM Inference")
    parser.add_argument("-m",
                        "--model_name_or_path",
                        type=str,
                        default="./output",
                        help="model dir")
    parser.add_argument("-mbs",
                        "--max_num_seqs",
                        type=int,
                        default=34,
                        help="max batch size")
    parser.add_argument("--max_block_num", type=int, default=2000)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--engine_worker_queue_port", type=int, default=9923)
    parser.add_argument("--max_model_len",
                        type=int,
                        default=3072,
                        help="max model len")
    parser.add_argument("--device_ids",
                        type=str,
                        default="0",
                        help="cuda visible devices")
    parser.add_argument("--dtype",
                        type=str,
                        default="bfloat16",
                        help="input dtype")
    parser.add_argument("--enc_dec_block_num",
                        type=int,
                        default=1,
                        help="encoder's decoder num")
    parser.add_argument("--kv_cache_ratio",
                        type=float,
                        default=0.7,
                        help="kv cache ratio for input")
    parser.add_argument("--first_token_id",
                        type=int,
                        default=1,
                        help="first token id")
    parser.add_argument("--gpu_memory_utilization",
                        type=float,
                        default=0.9,
                        help="gpu memory utilization")
    parser.add_argument("--engine_pid",
                        type=int,
                        default=None,
                        help="Process ID of engine")
    parser.add_argument("--do_profile",
                        type=int,
                        default=0,
                        help="do profile or not")
    parser.add_argument("--dynamic_load_weight",
                        type=int,
                        default=0,
                        help="dynamic load weight or not")
    parser.add_argument("--pad_token_id",
                        type=int,
                        default=-1,
                        help="pad token id")
    parser.add_argument("--eos_tokens_lens",
                        type=int,
                        default=2,
                        help="eos token lens")
    parser.add_argument("--enable_chunked_prefill",
                        action='store_true',
                        help="enable chunked prefill")
    parser.add_argument(
        "--speculate_method",
        default=None,
        type=str,
        choices=[
            "autoregressive",
            "inference_with_reference",
            "draft_model",
            "hydra",
            "eagle",
        ],
    )
    parser.add_argument(
        "--attention_backend",
        default="APPEND_ATTN",
        type=str,
        choices=[
            "APPEND_ATTN",
        ],
    )
    parser.add_argument("--speculate_max_draft_tokens", type=int, default=1)

    args = parser.parse_args()
    return args


def run_worker_proc():
    """
    start worker process
    """
    # Get args form Engine
    args = parse_args()
    # Initialize LLMConfig
    # TODO(gongshaotian): Unified all configs to LLMConfig
    llm_config = LLMConfig()
    llm_config.parallel_config.engine_pid = args.engine_pid
    llm_config.parallel_config.model_name_or_path = args.model_name_or_path
    llm_config.parallel_config.max_num_seqs = args.max_num_seqs
    llm_config.parallel_config.max_block_num = args.max_block_num
    llm_config.parallel_config.block_size = args.block_size
    llm_config.parallel_config.engine_worker_queue_port = args.engine_worker_queue_port
    llm_config.parallel_config.max_model_len = args.max_model_len
    llm_config.parallel_config.device_ids = args.device_ids
    llm_config.parallel_config.dtype = args.dtype
    llm_config.parallel_config.enc_dec_block_num = args.enc_dec_block_num
    llm_config.parallel_config.kv_cache_ratio = args.kv_cache_ratio
    llm_config.parallel_config.first_token_id = args.first_token_id
    llm_config.parallel_config.gpu_memory_utilization = args.gpu_memory_utilization
    llm_config.parallel_config.engine_pid = args.engine_pid
    llm_config.parallel_config.do_profile = args.do_profile
    llm_config.parallel_config.dynamic_load_weight = args.dynamic_load_weight
    llm_config.parallel_config.pad_token_id = args.pad_token_id
    llm_config.parallel_config.eos_tokens_lens = args.eos_tokens_lens
    llm_config.parallel_config.enable_chunked_prefill = args.enable_chunked_prefill
    llm_config.parallel_config.speculate_method = args.speculate_method
    llm_config.parallel_config.attention_backend = args.attention_backend
    llm_config.parallel_config.speculate_max_draft_tokens = args.speculate_max_draft_tokens

    worker_proc = PaddleDisWorkerProc(llm_config)
    worker_proc.init_device()
    if llm_config.parallel_config.do_profile:
        worker_proc.determine_num_available_blocks()
    worker_proc.event_loop_normal()


if __name__ == "__main__":
    run_worker_proc()

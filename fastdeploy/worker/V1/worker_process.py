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

from fastdeploy.config import (AdditionalConfig, DecodingConfig, DeviceConfig,
                               FDConfig, GraphOptimizationConfig,
                               KVCacheConfig, LoadConfig, ModelConfig,
                               MoEConfig, ParallelConfig, SpeculativeConfig,
                               TmpConfig)
from fastdeploy.inter_communicator import EngineWorkerQueue as TaskQueue
from fastdeploy.inter_communicator import IPCSignal
from fastdeploy.model_executor.layers.quantization import \
    get_quantization_config
from fastdeploy.model_executor.models.utils import parser_quant_type
from fastdeploy.utils import get_logger
from fastdeploy.worker.V1.gpu_worker import GpuWorker

logger = get_logger("worker_process", "worker_process.log")


class PaddleDisWorkerProc():
    """
    Paddle Distrubuted wrapper for fastdeploy.worker.Worker,
        for handling single-node multi-GPU tensor parallel.
    The wrapper internally executea an event loop that continuously executes requests
        in the task queue. Control flow is transmitted by IPC.
    """

    def __init__(
        self,
        fd_config: FDConfig,
    ):
        self.fd_config = fd_config
        self.parallel_config = fd_config.parallel_config

        # Initialize distributed enviroment
        (self.rank, self.local_rank) = self.init_distributed_enviroment()
        self.fd_config.parallel_config.tensor_parallel_rank = self.local_rank
        self.fd_config.model_config.tensor_parallel_rank = self.local_rank
        self.fd_config.parallel_config.tensor_parallel_degree = self.rank
        self.fd_config.model_config.tensor_parallel_degree = self.rank
        self.fd_config.parallel_config.mp_size = self.rank
        self.fd_config.parallel_config.ep_size = 1
        self.fd_config.parallel_config.column_cut = False

        # TODO(gongshaotian): Use worker factory to get worker
        self.worker = GpuWorker(fd_config=fd_config,
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
            if self.rank > 1:
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
                    self.exist_task_signal.value[0] = 0
                    self.task_queue.read_finish_flag.set(0)

                req_dicts = []
                for req_dict, bsz in tasks:
                    num_running_requests = int(bsz)
                    req_dicts.extend(req_dict)
                logger.info(f"Rank: {self.local_rank}, num_running_requests: {num_running_requests}, " \
                            f"num_insert_requests: {len(req_dicts)}")

                # Process prefill inputs
                self.worker.preprocess_new_task(req_dicts)

            if not self.worker.model_runner.not_need_stop():
                if self.rank > 1:
                    paddle.distributed.barrier()

                time.sleep(0.001)
                continue

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
        print(
            f"------- available_kv_cache_memory:{available_kv_cache_memory / 1024**3} GB --------"
        )

        # 2. Calculate the appropriate number of blocks
        model_block_memory_used = self.worker.cal_theortical_kvcache()
        num_blocks_local = int(available_kv_cache_memory //
                               model_block_memory_used)
        print(
            f"------- model_block_memory_used:{model_block_memory_used} --------"
        )
        print(f"------- num_blocks_local:{num_blocks_local} --------")

        # 3. Send IPCSignal
        if self.fd_config.parallel_config.do_profile:
            get_profile_block_num = np.zeros(shape=[self.rank], dtype=np.int32)
            self.get_profile_block_num_signal = IPCSignal(
                name="get_profile_block_num",
                array=get_profile_block_num,
                dtype=np.int32,
                suffix=self.parallel_config.engine_pid,
                create=False)
            self.get_profile_block_num_signal.value[
                self.local_rank] = num_blocks_local

            # Wait all worker send the signal
            while np.any(self.get_profile_block_num_signal.value <= 0):
                time.sleep(0.01)
            num_blocks_global = self.get_profile_block_num_signal.value.min(
            ).item()
            self.get_profile_block_num_signal.value[
                self.local_rank] = num_blocks_global
        else:
            num_blocks_global = num_blocks_local

        # 4. Updata share inputs
        self.worker.reinitialize_kv_cache(num_gpu_blocks=num_blocks_global)

    def init_device(self):
        """ """
        self.worker.init_device()

    def load_model(self):
        """ """
        self.worker.load_model()


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
    parser.add_argument("--total_block_num", type=int, default=2000)
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
                        action='store_true',
                        help="do profile or not")
    parser.add_argument("--dynamic_load_weight",
                        action='store_true',
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

    parser.add_argument("--max_num_batched_tokens",
                        type=int,
                        default=2048,
                        help="max num batched tokens")
    parser.add_argument("--enable_prefix_caching",
                        action='store_true',
                        help="enable prefix cache")
    parser.add_argument("--splitwise_role",
                        type=str,
                        default="mixed",
                        help="splitwise role")
    parser.add_argument("--ori_vocab_size", type=int, default=None)

    args = parser.parse_args()
    return args


def initialize_fd_config(args) -> FDConfig:
    """Initialize FDConfig
    TODO(gongshaotian): Unified all configs to FDConfig
    """
    # NOTE(gongshaotian): From build stream line model
    config, _ = ModelConfig.get_config_dict(args.model_name_or_path)
    config["head_dim"] = config.get(
        "head_dim", config["hidden_size"] // config["num_attention_heads"])
    config["rope_theta"] = config.get("rope_theta", 10000.0)
    model_config = ModelConfig.from_dict(config)
    # TODO Set `head_dim` again. Because `ModelConfig` class doesn't support feeding head_dim at all!
    model_config.head_dim = config["head_dim"] 
    paddle.set_default_dtype(args.dtype)

    device_config = DeviceConfig()
    # model_config = ModelConfig()
    kv_cache_config = KVCacheConfig()

    cachekv_dtype = config.get("cache_quant_type", None)
    if cachekv_dtype is not None:
        logger.info(
            f"cachekv is set to [{cachekv_dtype}] according to your config file's cache_quant_type field"
        )
        kv_cache_config.cache_quant_dtype = config["cache_quant_type"]
    decoding_config = DecodingConfig()
    decoding_config = MoEConfig()
    tmp_config = TmpConfig()
    additional_config = AdditionalConfig()
    speculative_config = SpeculativeConfig()
    parallel_config = ParallelConfig()
    load_config = LoadConfig()
    moe_config = MoEConfig()
    graph_opt_config = GraphOptimizationConfig()

    # Note(tangbinhan): used for load_checkpoint
    model_config.tensor_parallel_rank = parallel_config.tensor_parallel_rank
    model_config.use_ep = parallel_config.use_ep
    model_config.is_mtp = speculative_config.is_mtp

    group_size = config.get("group_size", -1)
    num_key_value_heads = config.get("num_key_value_heads", -1)
    if num_key_value_heads is None:
        num_key_value_heads = -1

    if config.get("ffn_hidden_size", None) is not None:
        ffn_hidden_size = config["ffn_hidden_size"]
    elif config.get("intermediate_size", None) is not None:
        ffn_hidden_size = config["intermediate_size"]
    else:
        ffn_hidden_size = 4 * config["hidden_size"]
        if config["hidden_act"].lower() == "swiglu":
            if paddle.distributed.get_world_size() > 1:
                multiple_of = 8 * config["num_attention_heads"]
            else:
                multiple_of = 4 * config["num_attention_heads"]
            ffn_hidden_size = multiple_of * (
                (int(2 * ffn_hidden_size / 3) + multiple_of - 1) //
                multiple_of)

    num_layers = config.get("num_layers", None) or config.get(
        "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError(f"num_layers<{num_layers}> is invalid")

    use_moe = config.get("moe_layer_start_index", num_layers) < num_layers

    model_config.ffn_hidden_size = ffn_hidden_size
    model_config.num_layers = num_layers

    model_config.group_size = group_size
    model_config.use_rmsnorm = config.get("use_rmsnorm", True)
    model_config.num_key_value_heads = num_key_value_heads
    model_config.export_model_type = config.get("predict_model_type",
                                                "weight_only_int8")
    tmp_config.has_zero_point = config.get("has_zero_point", False)
    tmp_config.is_channel_wise = config.get("is_channel_wise", False),
    model_config.start_layer_index = config.get("start_layer_index", 0)
    moe_config.num_experts = config.get("moe_num_experts", None)
    moe_config.moe_intermediate_size = config.get("moe_intermediate_size",
                                                  None)
    moe_config.moe_use_gate_correction_bias = config.get(
        "moe_use_gate_correction_bias", True)
    moe_config.moe_every2 = config.get("moe_every2", False)
    moe_config.top_k = config.get("moe_topk", 8)
    moe_config.moe_num_shared_experts = config.get("moe_num_shared_experts", 0)
    moe_config.moe_layer_start_index = config.get("moe_layer_start_index", 0)
    moe_config.moe_use_ffn_shared_weight_and_bias = config.get(
        "moe_use_ffn_shared_weight_and_bias", False)
    moe_config.use_moe = use_moe
    moe_config.moe_group = config.get("moe_group", False)
    moe_config.moe_quant_type = config.get("moe_quant_type",
                                           "weight_only_int4")
    tmp_config.weight_block_size = config.get("weight_block_size", [-1, -1])
    model_config.ori_vocab_size = config.get("vocab_size", -1)
    if "ErnieBotLMHeadModel" in config.get("architectures"):
        model_config.ori_vocab_size = args.ori_vocab_size

    weight_dtype, act_dtype, cachekv_dtype = parser_quant_type(
        model_config.export_model_type)
    model_config.weight_dtype = weight_dtype
    act_dtype = args.dtype if (act_dtype != args.dtype) else act_dtype
    model_config.act_dtype = act_dtype  # set as args.dtype from engine
    logger.info(
        f"quant_type: weight[{weight_dtype}], act[{act_dtype}] -> act[{args.dtype}], cachekv[{cachekv_dtype}]"
    )

    if weight_dtype == "int8" and act_dtype in ["bfloat16", "float16"]:
        quant_cls = get_quantization_config("weight_only")
        quant_config = quant_cls.from_config({
            "weight_only_linear_arch": None,
            "algo": "weight_only_int8"
        })
        quant_config.quant_max_bound = 0
        quant_config.quant_min_bound = 0
        quant_config.quant_round_type = 0
        model_config.use_smooth_quant = False
    elif weight_dtype == "int4" and act_dtype in ["bfloat16", "float16"]:
        quant_cls = get_quantization_config("weight_only")
        quant_config = quant_cls.from_config({
            "weight_only_linear_arch": None,
            "algo": "weight_only_int4"
        })
        quant_config.quant_max_bound = 0
        quant_config.quant_min_bound = 0
        quant_config.quant_round_type = 0
        model_config.use_smooth_quant = False
    else:
        quant_config = None

    model_config.architectures = config.get("architectures")

    # Update parallel config
    parallel_config.engine_pid = args.engine_pid
    parallel_config.model_name_or_path = args.model_name_or_path
    parallel_config.max_num_seqs = args.max_num_seqs
    parallel_config.max_block_num = args.total_block_num
    parallel_config.block_size = args.block_size
    parallel_config.engine_worker_queue_port = args.engine_worker_queue_port
    parallel_config.max_model_len = args.max_model_len
    model_config.max_seq_len = args.max_model_len
    model_config.max_length = args.max_model_len
    parallel_config.device_ids = args.device_ids
    parallel_config.dtype = args.dtype
    parallel_config.enc_dec_block_num = args.enc_dec_block_num
    parallel_config.kv_cache_ratio = args.kv_cache_ratio
    parallel_config.first_token_id = args.first_token_id
    parallel_config.gpu_memory_utilization = args.gpu_memory_utilization
    parallel_config.engine_pid = args.engine_pid
    parallel_config.do_profile = args.do_profile
    parallel_config.dynamic_load_weight = args.dynamic_load_weight
    parallel_config.pad_token_id = args.pad_token_id
    parallel_config.eos_tokens_lens = args.eos_tokens_lens
    parallel_config.enable_chunked_prefill = args.enable_chunked_prefill
    parallel_config.speculate_method = args.speculate_method
    parallel_config.attention_backend = args.attention_backend
    parallel_config.speculate_max_draft_tokens = args.speculate_max_draft_tokens
    parallel_config.max_num_batched_tokens = args.max_num_batched_tokens
    parallel_config.enable_prefix_caching = args.enable_prefix_caching

    fd_config = FDConfig(model_config=model_config,
                         parallel_config=parallel_config,
                         speculative_config=speculative_config,
                         device_config=device_config,
                         additional_config=additional_config,
                         load_config=load_config,
                         tmp_config=tmp_config,
                         moe_config=moe_config,
                         decoding_config=decoding_config,
                         quant_config=quant_config,
                         kv_cache_config=kv_cache_config,
                         graph_opt_config=graph_opt_config)

    return fd_config


def run_worker_proc():
    """
    start worker process
    """
    # Get args form Engine
    args = parse_args()

    # Get fd_config
    fd_config = initialize_fd_config(args)

    # Start event loop
    worker_proc = PaddleDisWorkerProc(fd_config)
    worker_proc.init_device()
    worker_proc.load_model()
    worker_proc.determine_num_available_blocks()
    worker_proc.event_loop_normal()


if __name__ == "__main__":
    run_worker_proc()

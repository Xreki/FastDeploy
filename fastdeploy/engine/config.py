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

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from fastdeploy.scheduler import SchedulerConfig
from fastdeploy.utils import (check_unified_ckpt, get_host_ip,
                              is_port_available, llm_logger)

TaskOption = Literal["generate"]


class ModelConfig:
    """
    Configuration class for the model.

    Attributes:
        model_dir (str): Directory path to the model.
        is_unified_ckpt (bool): Flag indicating if the checkpoint is unified.
        model_name_or_path (str): Name or path of the model.
    """

    def __init__(self,
                 model_name_or_path: str,
                 config_json_file: str = "config.json",
                 dynamic_load_weight: int = 0,
                 download_dir: Optional[str] = None):
        """
        Initialize the ModelConfig class.

        Args:
            model_name_or_path (str): Name or path of the model.
            config_json_file (str): Path to the configuration JSON file. Default is 'config.json'.
            download_dir (Optional[str]): Directory to download model files. Default is None.
        """
        self.model_dir = model_name_or_path
        self.is_unified_ckpt = check_unified_ckpt(self.model_dir)
        self.dynamic_load_weight = dynamic_load_weight

        config_file = os.path.join(model_name_or_path, config_json_file)
        if os.path.isfile(model_name_or_path):
            try:
                from paddlenlp.transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_name_or_path)
                config_dict = {
                    k: v
                    for k, v in vars(config).items() if not k.startswith('_')
                }
                for key, value in config_dict.items():
                    setattr(self, key, value)
            except Exception:
                llm_logger.error(
                    "Don't support the current model, you can use `paddlenlp` to register your model."
                )
                raise ValueError(
                    "Don't support the current model, you can use `paddlenlp` to register your model."
                )
        else:
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
                for key, value in config_dict.items():
                    try:
                        setattr(self, key, value)
                    except Exception:
                        continue

        if isinstance(self.architectures, list):
            self.architectures = self.architectures[0]
        self.model_name_or_path = model_name_or_path
        self.override_name_from_config()
        self.read_from_env()

    def override_name_from_config(self):
        """
        Override attribute names from the exported model's configuration.
        """
        if not self.is_unified_ckpt and hasattr(self, "infer_model_mp_num"):
            self.tensor_parallel_size = self.infer_model_mp_num
            del self.infer_model_mp_num

        if hasattr(self, "num_hidden_layers"):
            if hasattr(self, "remove_tail_layer"):
                if self.remove_tail_layer is True:
                    self.num_hidden_layers -= 1
                elif isinstance(self.remove_tail_layer, int):
                    self.num_hidden_layers -= self.remove_tail_layer

            self.num_layers = self.num_hidden_layers
            del self.num_hidden_layers

        if not hasattr(self, "mla_use_absorb"):
            self.mla_use_absorb = False
        if not hasattr(self, "head_dim"):
            assert hasattr(self, "hidden_size") and hasattr(
                self, "num_attention_heads")
            self.head_dim = self.hidden_size // self.num_attention_heads

    def read_from_env(self):
        """
        Read configuration information from environment variables and update the object's attributes.

        If an attribute is not present or is an empty string in the environment variables, use the default value.
        """
        self.max_stop_seqs_num = int(os.getenv("MAX_STOP_SEQS_NUM", "5"))
        self.stop_seqs_max_len = int(os.getenv("STOP_SEQS_MAX_LEN", "8"))

        self.ellm_dynamic_quant_type = os.getenv("ELLM_DYNAMIC_QUANT_TYPE",
                                                 "default")
        # 动态图推理是否使用停止序列
        self.ellm_dynamic_use_stop_seqs = int(
            os.getenv("ELLM_DYNAMIC_USE_STOP_SEQS", "0")) == 1

        def reset_config_value(key, value):
            if not hasattr(self, key.lower()):
                if os.getenv(key, None):
                    value = eval(os.getenv(key))
                    llm_logger.info(
                        f"Get parameter `{key}` = {value} from environment.")
                else:
                    llm_logger.info(
                        f"Parameter `{key}` will use default value {value}.")
                setattr(self, key.lower(), value)

        if "ErnieForCausalLM" in self.architectures and not hasattr(
                self, "model_name"):
            self.model_name = os.getenv("FD_MODEL_NAME")
            assert self.model_name is not None, (
                "There is no parameter model_name in config.json or "
                "FD_MODEL_NAME in environment variables.")

        reset_config_value("COMPRESSION_RATIO", 1.0)
        reset_config_value("ROPE_THETA", 10000)

    def _get_download_model(self, model_name, model_type="default"):
        # TODO: Provide dynamic graph for self-downloading and save to the specified download directory.
        pass

    def print(self):
        """
        Print all configuration information.
        """
        llm_logger.info("Model Configuration Information :")
        for k, v in self.__dict__.items():
            llm_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        llm_logger.info(
            "=============================================================")




class CacheConfig:
    """
    Configuration for the KV cache.

    Attributes:
        block_size (int): Size of a cache block in number of tokens.
        gpu_memory_utilization (float): Fraction of GPU memory to use for model execution.
        cache_dtype (str): Data type for kv cache storage. Default is 'bfloat16'.
        num_gpu_blocks_override (Optional[int]): Number of GPU blocks to use. Overrides profiled num_gpu_blocks if provided.
        kv_cache_ratio (float): Ratio for calculating the maximum block number.
        enc_dec_block_num (int): Number of encoder-decoder blocks.
        enable_prefix_caching (bool): Flag to enable prefix caching.
    """

    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cache_dtype: str = "bfloat16",
        num_gpu_blocks_override: Optional[int] = None,
        cpu_offload_gb: Optional[int] = None,
        kv_cache_ratio: float = 0.75,
        enc_dec_block_num: int = 2,
        tensor_parallel_size: int = 1,
        enable_prefix_caching=False,
        enable_ssd_cache=False,
        model_cfg=None,
        cache_queue_port=None,
        enable_chunked_prefill=False,
    ):
        """
        Initialize the CacheConfig class.

        Args:
            block_size (int): Size of a cache block in number of tokens.
            gpu_memory_utilization (float): Fraction of GPU memory to use.
            cache_dtype (str): Data type for cache storage. Default is 'bfloat16'.
            num_gpu_blocks_override (Optional[int]): Override for number of GPU blocks.
            num_cpu_blocks (Optional[int]): Number of CPU blocks.
            kv_cache_ratio (float): Ratio for max block calculation.
            enc_dec_block_num (int): Number of encoder-decoder blocks.
            enable_prefix_caching (bool): Enable prefix caching.
        """
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.num_gpu_blocks_override = num_gpu_blocks_override
        self.kv_cache_ratio = kv_cache_ratio
        self.enc_dec_block_num = enc_dec_block_num
        self.cache_dtype = cache_dtype
        if hasattr(model_cfg, "kvcache_quant_type"):
            self.cache_dtype = self.model_cfg.kvcache_quant_type

        self.enable_chunked_prefill = enable_chunked_prefill

        self.enable_prefix_caching = enable_prefix_caching
        if cpu_offload_gb is None:
            self.enable_hierarchical_cache = False
        else:
            self.enable_hierarchical_cache = True

        self.enable_ssd_cache = enable_ssd_cache
        self.model_cfg = model_cfg
        self.cache_queue_port = cache_queue_port
        self.cpu_offload_gb = cpu_offload_gb

        if (hasattr(self.model_cfg, "num_key_value_heads")
            and hasattr(self.model_cfg, "num_key_value_heads")
            and self.model_cfg.num_key_value_heads is not None
            and int(self.model_cfg.num_key_value_heads) > 0):
            kv_num_head = int(
                self.model_cfg.num_key_value_heads)
        else:
            kv_num_head = self.model_cfg.num_attention_heads
        self.model_cfg.kv_num_head = kv_num_head


        # TODO check name
        if self.cache_dtype.lower() == "wint4":
            byte_size = 0.5
        elif self.cache_dtype.lower() == "wint8":
            byte_size = 1
        else:
            byte_size = 2

        self.each_token_cache_space = int(
            self.model_cfg.num_layers
            * kv_num_head
            * self.model_cfg.head_dim
            * byte_size
        ) 
        self.bytes_per_block = int(
            self.each_token_cache_space * self.block_size
        ) 
        self.bytes_per_layer_per_block = int(
            self.block_size
            * self.model_cfg.kv_num_head
            * self.model_cfg.head_dim // tensor_parallel_size
            * byte_size
        )

        if self.cpu_offload_gb is None:
            self.num_cpu_blocks = 0
        else:
            self.num_cpu_blocks = int(self.cpu_offload_gb * 1024**3 / self.bytes_per_block)
        self._verify_args()

    def metrics_info(self):
        """Convert cache_config to dict(key: str, value: str) for prometheus metrics info."""
        return {key: str(value) for key, value in self.__dict__.items()}

    def _verify_args(self):
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}.")
        if self.kv_cache_ratio > 1.0:
            raise ValueError("KV cache ratio must be less than 1.0. Got "
                             f"{self.kv_cache_ratio}.")

    def postprocess(self, num_total_tokens, number_of_tasks):
        """
        calculate block num
        """
        self.dec_token_num = self.enc_dec_block_num * self.block_size
        if self.num_gpu_blocks_override is not None:
            self.total_block_num = self.num_gpu_blocks_override
            self.prefill_kvcache_block_num = int(self.total_block_num * self.kv_cache_ratio)
        else:
            length = num_total_tokens // number_of_tasks
            block_num = (length + self.block_size - 1 + self.enc_dec_block_num) // self.block_size
            self.total_block_num =  block_num * number_of_tasks
            self.prefill_kvcache_block_num = self.total_block_num
            llm_logger.info(f"Doing profile, the total_block_num:{self.total_block_num}")
        

    def reset(self, num_gpu_blocks):
        """
        reset gpu block number
        """
        self.total_block_num  = num_gpu_blocks
        self.prefill_kvcache_block_num = int(self.total_block_num * self.kv_cache_ratio)
        llm_logger.info((f"Reset block num, the total_block_num:{self.total_block_num},"
            f" prefill_kvcache_block_num:{self.prefill_kvcache_block_num}"))

    def print(self):
        """
        print all config

        """
        llm_logger.info("Cache Configuration Information :")
        for k, v in self.__dict__.items():
            llm_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        llm_logger.info(
            "=============================================================")


class Config:
    """
    Initial configuration class.

    Attributes:
        model_config (ModelConfig): Model configuration object.
        cache_config (CacheConfig): Cache configuration object.
        model_name_or_path (str): Directory path to the model or the model name.
        tokenizer (Optional[str]): Default is the model.
        max_num_batched_tokens (Optional[int]): Maximum number of batched tokens.
        tensor_parallel_size (int): Tensor parallel size.
        nnode (int): Number of nodes.
        max_model_len (int): Maximum model length. Default is 8192.
        max_num_seqs (int): Maximum number of sequences. Default is 8.
        mm_processor_kwargs (Optional[Dict[str, Any]]): Additional arguments for multi-modal processor.
        speculative_config (Optional[Dict[str, Any]]): Speculative execution configuration.
        use_warmup (bool): Flag to use warmup.
        engine_worker_queue_port (int): Port for engine worker queue.
        enable_mm (bool): Flag to enable multi-modal processing.
        splitwise_role (str): Splitwise role.
        innode_prefill_ports (Optional[List[int]]): Innode prefill ports. 
            Temporary configuration, will be removed in the future.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
        model_name_or_path: str = None,
        tokenizer: str = None,
        tensor_parallel_size: int = 8,
        nnode: int = 1,
        max_model_len: int = 8192,
        max_num_seqs: int = 8,
        max_num_batched_tokens: Optional[int] = None,
        pod_ips: Optional[List[str]] = None,
        speculative_config: Optional[Dict[str, Any]] = None,
        use_warmup: bool = False,
        engine_worker_queue_port: int = 8002,
        limit_mm_per_prompt: Optional[Dict[str, Any]] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        enable_mm: bool = False,
        splitwise_role: str = "mixed",
        innode_prefill_ports: Optional[List[int]] = None,
        max_num_partial_prefills: int = 1,
        max_long_partial_prefills: int = 1,
        long_prefill_token_threshold: int = 0,
    ):
        """
        Initialize the Config class.

        Args:
            model_config (ModelConfig): Model configuration object.
            cache_config (CacheConfig): Cache configuration object.
            scheduler_config (SchedulerConfig): Scheduler configuration object.
            model_name_or_path (str): Model directory path or model name.
            tokenizer (str): Default is the model.
            tensor_parallel_size (int): Tensor parallel size. Default is 8.
            nnode (int): Number of nodes. Default is 1.
            max_model_len (int): Maximum model length. Default is 8192.
            max_num_seqs (int): Maximum number of sequences. Default is 8.
            max_num_batched_tokens (Optional[int]): Maximum number of batched tokens. Default is None.
            pod_ips (Optional[List[str]]): List of POD IPs. Default is None.
            mm_processor_kwargs (Optional[Dict[str, Any]]): Additional arguments for multi-modal processor. Default is None.
            speculative_config (Optional[Dict[str, Any]]): Speculative execution configuration. Default is None.
            use_warmup (bool): Flag to use warmup. Default is False.
            engine_worker_queue_port (int): Engine worker queue port. Default is 8002.
            enable_mm (bool): Flag to enable multi-modal processing. Default is False.
            splitwise_role (str): Splitwise role. Default is "mixed".
            innode_prefill_ports (Optional[List[int]]): Innode prefill ports. Default is None.
        """
        self.model_config = model_config
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config
        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        self.max_num_batched_tokens = max_num_batched_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.nnode = nnode
        self.pod_ips = pod_ips
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.limit_mm_per_prompt = limit_mm_per_prompt
        self.mm_processor_kwargs = mm_processor_kwargs
        self.enable_mm = enable_mm
        self.speculative_config = speculative_config
        self.use_warmup = use_warmup
        self.splitwise_role = splitwise_role
        self.innode_prefill_ports = innode_prefill_ports
        self.max_num_partial_prefills = max_num_partial_prefills
        self.max_long_partial_prefills = max_long_partial_prefills
        self.long_prefill_token_threshold = long_prefill_token_threshold

        assert self.splitwise_role in ["mixed", "prefill", "decode"]

        # TODO: Temporary configuration, will be removed in the future.
        if innode_prefill_ports is None:
            assert self.splitwise_role in ["mixed", "prefill"], \
                " `innode_prefill_ports` can only support in decode mode"
        # TODO
        self.max_prefill_batch = 3
        if enable_mm:
            self.max_prefill_batch = 1  # TODO:当前多模prefill阶段只支持并行度为1,待优化


        self.engine_worker_queue_port = engine_worker_queue_port
        self.device_ids = ",".join(
            [str(i) for i in range(self.tensor_parallel_size)])
        self.device_ids = os.getenv("CUDA_VISIBLE_DEVICES", self.device_ids)

        self.read_from_config()
        self.postprocess()
        self.check()
        self.print()

    def postprocess(self):
        """
        calculate some parameters
        """
        if len(self.device_ids.split(',')) > self.tensor_parallel_size:
            self.device_ids = ",".join(
                self.device_ids.split(',')[:self.tensor_parallel_size:])
        assert len(
            self.device_ids.split(',')
        ) == self.tensor_parallel_size, f"The number of available GPUs is {len(self.device_ids.split(','))}, which is less than the tensor parallel required {self.tensor_parallel_size}."

        assert self.tensor_parallel_size % self.nnode == 0, f"tensor_parallel_size: {self.tensor_parallel_size} should be divisible by nnode: {self.nnode}"
        self.tp_num_per_node = self.tensor_parallel_size // self.nnode
        self.host_ip = get_host_ip()

        import paddle
        self.paddle_commit_id = paddle.version.commit

        if self.max_num_batched_tokens is None:
            if self.cache_config.enable_chunked_prefill:
                self.max_num_batched_tokens = 2048
            else:
                self.max_num_batched_tokens = self.max_model_len
        
        if self.long_prefill_token_threshold == 0:
            self.long_prefill_token_threshold = int(self.max_model_len * 0.04)

        self.cache_config.postprocess(self.max_num_batched_tokens, self.max_num_seqs)
        self.cache_config.max_block_num_per_seq = int(self.max_model_len // self.cache_config.block_size)


    def check(self):
        """
        check the legality of config
        """
        assert (
            self.max_num_seqs <= 256
        ), "The parameter `max_num_seqs` is not allowed to exceed 256, " "but now it's {}.".format(
            self.max_num_seqs)
        assert (
            is_port_available('0.0.0.0', self.engine_worker_queue_port)
        ), f"The parameter `engine_worker_queue_port`:{self.engine_worker_queue_port} is already in use."
        assert (
            8 >= self.tensor_parallel_size > 0
        ), f"tensor_parallel_size: {self.tensor_parallel_size} should be between 1 and 8"
        assert (self.nnode >= 1), f"nnode: {self.nnode} should no less than 1"
        assert (self.max_model_len >= 16), f"max_model_len: {self.max_model_len} should be larger than 16"
        assert (self.max_num_seqs >= 1), f"max_num_seqs: {self.max_num_seqs} should be larger than 1"
        assert (self.max_num_batched_tokens >= self.max_num_seqs), f"max_num_batched_tokens: {self.max_num_batched_tokens} should be larger than or equal to max_num_seqs: {self.max_num_seqs}"
        assert (self.max_num_batched_tokens <= self.max_model_len * self.max_num_seqs), f"max_num_batched_tokens: {self.max_num_batched_tokens} should be larger" \
                f"than or equal to max_num_seqs: {self.max_num_seqs} * max_model_len: {self.max_model_len}"
        assert (self.max_num_partial_prefills >= 1), f"max_num_partial_prefills: {self.max_num_partial_prefills} should be larger than or equal to 1"

        assert (self.max_long_partial_prefills >= 1), f"max_long_partial_prefills: {self.max_long_partial_prefills} should be larger than or equal to 1"
        assert (self.max_long_partial_prefills <= self.max_num_partial_prefills), f"max_long_partial_prefills: {self.max_long_partial_prefills} should " \
                f"be less than or equal to max_num_partial_prefills: {self.max_num_partial_prefills}"

        if not self.cache_config.enable_chunked_prefill:
            assert (self.max_num_batched_tokens >= self.max_model_len), f"max_num_batched_tokens: {self.max_num_batched_tokens} should be larger than or equal to max_model_len: {self.max_model_len}"

        if self.max_num_partial_prefills > 1:
            assert (self.enable_chunked_prefill is True), f"Chunked prefill must be enabled to set max_num_partial_prefills > 1"
            assert (self.long_prefill_token_threshold < self.max_model_len), f"long_prefill_token_threshold: {self.long_prefill_token_threshold} should be less than max_model_len: {self.max_model_len}"

        self.scheduler_config.check()

    def print(self, file=None):
        """
        print all config

        Args:
            file (str): the path of file to save config
        """
        llm_logger.info(
            "=================== Configuration Information ===============")
        for k, v in self.__dict__.items():
            if k == "generation_config" and v is not None:
                for gck, gcv in v.to_dict().items():
                    llm_logger.info("{:<20}:{:<6}{}".format(gck, "", gcv))
            elif k == "cache_config" or k == "model_config" or k == "scheduler_config":
                v.print()
            else:
                llm_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        llm_logger.info(
            "=============================================================")
        if file is not None:
            f = open(file, "a")
            now_time = datetime.now()
            f.write(f"{now_time} configuration information as below,\n")
            for k, v in self.__dict__.items():
                f.write("{:<20}:{:<6}{}\n".format(k, "", v))
            f.close()

    def read_from_config(self):
        """
        reset model config from json file
        """

        def reset_value(cls, value_name, key):
            if hasattr(cls, key):
                value = getattr(cls, key)
                setattr(cls, value_name, value)
                llm_logger.info(
                    f"Reset parameter {value_name} = {value} from configuration."
                )

        reset_value(self.cache_config, "block_size", "infer_model_block_size")
        reset_value(self.model_config, "return_full_hidden_states", "return_full_hidden_states")
        reset_value(self.cache_config, "cache_dtype", "infer_model_dtype")

    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent=4)

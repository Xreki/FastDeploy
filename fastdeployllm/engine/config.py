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
"""

import json
import os
from datetime import datetime
import re
import uuid
from fastdeployllm.utils import llm_logger
from fastdeployllm.download_model import download_from_txt

from typing import Literal,Optional,Dict,List,Any


TaskOption = Literal["generate"]

class ModelConfig:
    def __init__(self, 
        model_name_or_path: str,
        config_json_file: str = "config.json",
        paddle_model_name: Optional[str] = None,
        download_dir: Optional[str] = None,
        use_tqdm_on_load: bool = True,
        ):

        self.model_dir = model_name_or_path

        config_file = os.path.join(model_name_or_path, config_json_file)
        if os.path.isfile(model_name_or_path):
            try:
                from paddlenlp.transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_name_or_path)

                config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
                for key, value in config_dict.items():
                    setattr(self, key, value)
            except:
                llm_logger.error("Don't support the current model, you can use `paddlenlp` to register your model.")
                raise ValueError("Don't support the current model, you can use `paddlenlp` to register your model.")
        else:
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
                for key, value in config_dict.items():
                    try:
                        setattr(self, key, value)
                    except Exception as e:
                        continue


        self.model_name_or_path = model_name_or_path

        self.override_name_from_config()

        self.read_from_env()


    def override_name_from_config(self):
        """
        从导出模型的配置文件中加载
        """

        if hasattr(self, "infer_model_mp_num"):
            self.mp_num = self.infer_model_mp_num
            del self.infer_model_mp_num
        if hasattr(self, "num_hidden_layers"):
            self.num_layers = self.num_hidden_layers
            del self.num_hidden_layers
        if not hasattr(self, "mla_use_absorb"):
            self.mla_use_absorb = False

        



    def read_from_env(self):
        """
            从环境变量中读取配置信息，并更新当前对象的属性值。
        如果某个属性在环境变量中不存在或为空字符串，则使用默认值。

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        self.max_stop_seqs_num = int(os.getenv("MAX_STOP_SEQS_NUM", "5"))
        self.stop_seqs_max_len = int(os.getenv("STOP_SEQS_MAX_LEN", "8"))
        self.bad_tokens = str(os.getenv("BAD_TOKENS", "-1"))
        self.first_token_id = int(os.getenv("FIRST_TOKEN_ID", "1"))

        self.ellm_dynamic_quant_type = os.getenv("ELLM_DYNAMIC_QUANT_TYPE", "default")
        # 动态图推理是否使用停止序列
        self.ellm_dynamic_use_stop_seqs = int(os.getenv("ELLM_DYNAMIC_USE_STOP_SEQS", "0")) == 1
        
        def reset_config_value(key, value):
            if not hasattr(self, key.lower()):
                if os.getenv(key, None):
                    value = eval(os.getenv(key))
                    llm_logger.info("Get parameter `{}` = {} from environment.".format(key, value))
                else:
                    llm_logger.info("Parameter `{}` will use default value {}.".format(key, value))
                setattr(self, key.lower(), value)
                
        reset_config_value("COMPRESSION_RATIO", 1.0)
        reset_config_value("ROPE_THETA", 10000)


    def _get_download_model(self, model_name, model_type="default"):
        # TODO
        # 提供动态图进行自行下载
        # 保存至指定的 download dir
        pass

    def print(self):
        """
        print all config

        """
        llm_logger.info("Model Configuration Information :")
        for k, v in self.__dict__.items():
                llm_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        llm_logger.info("=============================================================")



class CacheConfig:
    """Configuration for the KV cache.

    Args:
        block_size: Size of a cache block in number of tokens.
        gpu_memory_utilization: Fraction of GPU memory to use for the model execution.
        cache_dtype: Data type for kv cache storage.
        num_gpu_blocks_override: Number of GPU blocks to use. This overrides the
            profiled num_gpu_blocks if None. 
        enable_prefix_caching: Whether to enable prefix caching.
    """
    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cache_dtype: str = "bfloat16",
        num_gpu_blocks_override: Optional[int] = None,
        block_ratio: float = 0.75,
        enc_dec_block_num: int = 2,
        enable_prefix_caching: bool = False,
    ):
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.num_gpu_blocks_override = num_gpu_blocks_override
        self.block_ratio = block_ratio
        self.enc_dec_block_num = enc_dec_block_num
        self.cache_dtype = cache_dtype
        self.enable_prefix_caching = enable_prefix_caching
        self._verify_args()

    def metrics_info(self):
        """Convert cache_config to dict(key: str, value: str) for prometheus metrics info."""
        return {key: str(value) for key, value in self.__dict__.items()}

    def _verify_args(self):
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}.")
        if self.block_ratio > 1.0:
            raise ValueError(
                "Block ratio must be less than 1.0. Got "
                f"{self.block_ratio}.")


    def postprocess(self, num_total_tokens, number_of_tasks):
        """
        calculate block num
        """
        self.dec_token_num = self.enc_dec_block_num * self.block_size
        if self.num_gpu_blocks_override is not None:
            self.total_block_num = self.num_gpu_blocks_override
        else:
            length = num_total_tokens // number_of_tasks
            block_num = (length + self.block_size - 1 + self.enc_dec_block_num) // self.block_size 
            self.total_block_num =  block_num * number_of_tasks
            llm_logger.info(f"Doing profile, the total_block_num:{self.total_block_num}")
        self.max_block_num = int(self.total_block_num * self.block_ratio)

    def reset(self, num_gpu_blocks):
        """
        reset gpu block number
        """
        self.total_block_num  = num_gpu_blocks
        self.max_block_num = int(self.total_block_num * self.block_ratio)
        llm_logger.info((f"Reset block num, the total_block_num:{self.total_block_num},"
            f" max_block_num:{self.max_block_num}"))

    def print(self):
        """
        print all config

        """
        llm_logger.info("Cache Configuration Information :")
        for k, v in self.__dict__.items():
                llm_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        llm_logger.info("=============================================================")


class Config:
    """
    initial configuration
    """

    def __init__(self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        model: str = None,
        download_dir: str = None,
        tensor_parallel_size: int = 8,
        nnode: int = 1,
        max_cached_task_num: int = 128,
        max_model_len: int = 8192,
        max_cache_task_num: int = 128,
        max_num_seqs: int = 8,
        max_num_batched_tokens: Optional[int] = None,
        pod_ips: Optional[List[str]] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        speculative_config: Optional[Dict[str, Any]] = None,
        use_warmup: bool = False,
        use_tqdm_on_load: bool = True,
        ):

        self.model_config = model_config
        self.cache_config = cache_config
        self.model_dir = model
        self.max_num_batched_tokens = max_num_batched_tokens
        self.download_dir = download_dir
        self.mp_num = tensor_parallel_size
        self.nnode = nnode
        self.pod_ips = pod_ips
        self.max_seq_len = max_model_len
        self.max_batch_size = max_num_seqs
        self.mm_processor_kwargs = mm_processor_kwargs
        self.max_cached_task_num = max_cached_task_num

 
        self.speculative_config = speculative_config
        self.use_warmup = use_warmup
        self.use_tqdm_on_load = use_tqdm_on_load
        self.max_prefill_batch = 3

        self.infer_port = int(os.getenv("INFER_PORT", "5289"))
        self.device_ids = ",".join([str(i) for i in range(self.mp_num)])
        self.device_ids = os.getenv("CUDA_VISIBLE_DEVICES",
                                    self.device_ids)

        self.read_from_config()
        self.postprocess()
        self.check()
        self.print()



    def postprocess(self):
        """
        calculate some parameters
        """

        assert self.mp_num % self.nnode == 0, f"mp_num: {self.mp_num} should be divisible by nnode: {self.nnode}"
        self.mp_num_per_node = self.mp_num // self.nnode
        self.host_ip = os.getenv("HOST_IP", "127.0.0.1")
        if self.nnode > 1:
            self.ips = os.getenv("POD_IPS")


        import paddle
        self.paddle_commit_id = paddle.version.commit

        if self.max_num_batched_tokens is None:
            self.max_num_batched_tokens = self.max_seq_len

        self.cache_config.postprocess(self.max_num_batched_tokens, self.max_batch_size)



    def check(self):
        """
        check the legality of config
        """
        import math

        assert (
            self.max_batch_size <= 256
        ), "The parameter `max_batch_size` is not allowed to exceed 256, " "but now it's {}.".format(
            self.max_batch_size
        )

    def print(self, file=None):
        """
        print all config

        Args:
            file (str): the path of file to save config
        """
        llm_logger.info("=================== Configuration Information ===============")
        for k, v in self.__dict__.items():
            if k == "generation_config" and v is not None:
                for gck, gcv in v.to_dict().items():
                    llm_logger.info("{:<20}:{:<6}{}".format(gck, "", gcv))
            elif k == "cache_config" or k == "model_config":
                v.print()
            else:
                llm_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        llm_logger.info("=============================================================")
        if file is not None:
            f = open(file, "a")
            now_time = datetime.now()
            f.write(f"{now_time} configuration information as below,\n")
            for k, v in self.__dict__.items():
                f.write("{:<20}:{:<6}{}\n".format(k, "", v))
            f.close()


    def get_model_config(self):
        """
        load config file

        Returns:
            ModelConfig
        """

        return self.model_config

    def read_from_config(self):
        """
        reset model config from json file
        """

        config = self.get_model_config()
        def reset_value(cls, value_name, key):
            if hasattr(config, key):
                value = getattr(config, key)
                setattr(cls, value_name, value)
                llm_logger.info(f"Reset parameter {value_name} = {value} from configuration.")

        reset_value(self.cache_config, "block_size", "infer_model_block_size")
        reset_value(self, "max_seq_len", "infer_model_max_seq_len")
        reset_value(self, "return_full_hidden_states", "return_full_hidden_states")
        reset_value(self.cache_config, "cache_dtype", "infer_model_dtype")



    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent=4)

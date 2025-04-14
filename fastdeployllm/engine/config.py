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
from fastdeployllm.utils import model_server_logger
from fastdeployllm.download_model import download_from_txt
from paddlenlp.experimental.transformers import SpeculateArgument

from typing import Literal,Optional,Dict,List,Any


TaskOption = Literal["generate"]

class ModelConfig:
    def __init__(self,
        model_name_or_path: str,
        config_json_file: str = "config.json"
        ):
        """
            Args:
            model_name_or_path (str): Name of the pre-trained model or path to its directory.
                Defaults to "bert-base-uncased".
            config_json_file (str, optional): Path to a JSON file containing the configuration parameters.
                Defaults to "config.json".
        """

        self.model_dir = None


        from paddlenlp.transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name_or_path)

        config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
        for key, value in config_dict.items():
            setattr(self, key, value)


        self.model_name_or_path = model_name_or_path

        self.override_from_config()

        self.read_from_env()

    def override_from_config(self):
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


class Config:
    """
    initial configuration
    """

    def __init__(self,
        model_config: ModelConfig,
        model: str = None,
        tensor_parallel_size: int = 8,
        nnode: int = 1,
        max_cached_task_num: int = 128,
        kv_cache_dtype: str = 'bfloat16',
        max_model_len: int = 8192,
        block_bs: float = 4,
        block_ratio: float = 0.75,
        block_size: int = 64,
        enc_dec_block_num: int = 2,
        max_num_seqs: int = 8,
        pod_ips: Optional[List[str]] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        speculative_config: Optional[Dict[str, Any]] = None,
        use_warmup: bool = False,
        enable_prefix_caching: bool = False,
        use_tqdm_on_load: bool = True,
        ):

        self.model_config = model_config
        self.model_dir = model
        self.mp_num = tensor_parallel_size
        self.block_bs = block_bs
        self.block_ratio = block_ratio
        self.nnode = nnode
        self.pod_ips = pod_ips
        self.block_size = block_size
        self.enc_dec_block_num = enc_dec_block_num
        self.dtype = kv_cache_dtype
        self.max_seq_len = max_model_len
        self.max_batch_size = max_num_seqs
        self.mm_processor_kwargs = mm_processor_kwargs
        self.max_cached_task_num = max_cached_task_num


        self.speculative_config = speculative_config
        self.use_warmup = use_warmup
        self.enable_prefix_caching = enable_prefix_caching
        self.use_tqdm_on_load = use_tqdm_on_load
        self.max_prefill_batch = 3

        self.infer_port = int(os.getenv("INFER_PORT", "5289"))
        self.device_ids = os.getenv("CUDA_VISIBLE_DEVICES", "0")

        self.postprocess()
        self.check()



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

        if self.block_ratio >= 1.0:
            self.enc_dec_block_num = (self.max_seq_len + self.block_size - 1) // self.block_size
        self.max_query_block_num = (self.max_seq_len + self.block_size - 1) // self.block_size
        self.dec_token_num = self.enc_dec_block_num * self.block_size
        self.total_block_num = int(self.block_bs * self.max_query_block_num)
        self.max_block_num = int(self.total_block_num * self.block_ratio)
        model_server_logger.info(f"max_block_num:{self.max_block_num}")

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


        # max_output_token_num
        max_output_token_num = (
            self.total_block_num - self.max_block_num
        ) * self.block_size + self.enc_dec_block_num * self.block_size
        assert max_output_token_num >= self.max_seq_len, (
            f"The available output token number of the service is {max_output_token_num}, "
            f"which is less than the setting MAX_DEC_LEN:{self.max_seq_len}. "
        )

        # Maximum input length of a single query that the service can handle
        max_input_token_num = int(math.floor(self.max_block_num * self.block_size - self.dec_token_num))
        assert max_input_token_num >= self.max_seq_len, (
            f"The available input token number of the service is {max_input_token_num}, "
            f"which is less than the setting MAX_SEQ_LEN:{self.max_seq_len}. "
        )

    def print(self, file=None):
        """
        print all config

        Args:
            file (str): the path of file to save config
        """
        model_server_logger.info("=================== Configuration Information ===============")
        for k, v in self.__dict__.items():
            if k == "generation_config" and v is not None:
                for gck, gcv in v.to_dict().items():
                    model_server_logger.info("{:<20}:{:<6}{}".format(gck, "", gcv))
            else:
                model_server_logger.info("{:<20}:{:<6}{}".format(k, "", v))
        model_server_logger.info("=============================================================")
        if file is not None:
            f = open(file, "a")
            now_time = datetime.now()
            f.write(f"{now_time} configuration information as below,\n")
            for k, v in self.__dict__.items():
                f.write("{:<20}:{:<6}{}\n".format(k, "", v))
            f.close()


    def _get_download_model(self, model_type="default"):
        """
            获取下载模型，支持的模型类型为"default"和"speculate"。
        如果模型名称不在支持列表中，将抛出ValueError异常。

        Args:
            model_type (str, optional): 模型类型，默认为"default"，可选值为"default"或"speculate"。 Default to "default".

        Raises:
            ValueError: 如果模型名称不在支持列表中。

        Returns:
            None.
        """
        env = os.environ
        model_name = env.get("model_name")
        # Define supported model patterns
        supported_patterns = [
            r".*Qwen.*",
            r".+Llama.+",
            r".+Mixtral.+",
            r".+DeepSeek.+",
        ]

        # Check if model_name matches any supported pattern
        if not any(re.match(pattern, model_name) for pattern in supported_patterns):
            raise ValueError(
                f"{model_name} is not in the supported list. Currently supported models: Qwen, Llama, Mixtral, DeepSeek. Please check the model name from this document https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/server/docs/static_models.md"
            )
        model_server_logger.info(f"Start downloading model: {model_name}")
        tag=env.get("tag")
        base_url=f"https://paddlenlp.bj.bcebos.com/models/static/{tag}/{model_name}"
        if self.nnode == 1:
            # single node model
            temp_file = "model"
        elif env.get("POD_0_IP", "127.0.0.1") == self.host_ip:
            # Master node model
            temp_file = "node1"
        else:
            temp_file = "node2"
        if model_type == "default":
            path = self.model_dir
        elif model_type == "speculate":
            path = os.getenv("SPECULATE_MODEL_PATH")
            temp_file="mtp"
        model_url = base_url+f"/{temp_file}"
        download_from_txt(model_url, path)



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
        def reset_value(self, value_name, key):
            if hasattr(config, key):
                value = getattr(config, key)
                setattr(self, value_name, value)
                model_server_logger.info(f"Reset parameter {value_name} = {value} from configuration.")

        reset_value(self, "block_size", "infer_model_block_size")
        reset_value(self, "max_seq_len", "infer_model_max_seq_len")
        reset_value(self, "return_full_hidden_states", "return_full_hidden_states")
        reset_value(self, "kv_cache_dtype", "infer_model_dtype")



    def __str__(self) -> str:
        """
            将对象转换为字符串，返回一个JSON格式的字符串。
        该方法用于在打印或显示对象时，将其转换为可读性更高的字符串形式。

        Args:
            None

        Returns:
            str (str): JSON格式的字符串，包含对象所有属性和值。
        """
        return json.dumps(self.__dict__, indent=4)

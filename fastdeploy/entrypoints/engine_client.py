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

import zmq
import time
from random import randint
import uuid
import numpy as np

from fastdeploy.input.preprocess import InputPreprocessor
from fastdeploy.engine.request import Request
from fastdeploy.inter_communicator import ZmqClient, IPCSignal
from fastdeploy.utils import api_server_logger, EngineError


class EngineClient:
    """
    EngineClient is a class that handles the communication between the client and the server.
    """
    def __init__(self, tokenizer, max_model_len, tensor_parallel_size, pid):
        input_processor =  InputPreprocessor(tokenizer)
        self.data_processor = input_processor.create_processor()
        self.max_model_len = max_model_len
        self.worker_healthy_live_recorded_time_array = np.zeros(shape=[tensor_parallel_size], dtype=np.float32)
        self.worker_healthy_live_signal = IPCSignal(name="worker_healthy_live_signal",
                    array=self.worker_healthy_live_recorded_time_array,
                    dtype=np.float32,
                    suffix=pid,
                    create=False)

    def create_zmq_client(self, model, mode):
        """
        Create a ZMQ client.
        """
        self.zmq_client = ZmqClient(model, mode)
        self.zmq_client.connect()

    def format_and_add_data(self, prompts: dict):
        """
        Format the request data and send the request to the server.
        """
        if "request_id" in prompts:
            prompts["req_id"] = prompts["request_id"]

        if "req_id" not in prompts:
            request_id = str(uuid.uuid4())
            prompts["req_id"] = request_id
        query_list = []

        if "context" in prompts:
            for item in prompts["context"]:
                if item["role"] == "system":
                    prompts["system"] = item["utterance"]
                elif item["role"] in ["user", "assistant"]:
                    query_list.append(item["utterance"])
                    prompts["prompt"] = query_list

        if "max_tokens" not in prompts:
            prompts["max_tokens"] = self.max_model_len

        self.add_requests(prompts)

    def add_requests(self, task):
        """
        Add a new request to the queue.

        Args:
            task: Request A dictionary representing the request.
            sampling_params: A dictionary representing the sampling parameters.

        Returns:
            None
        """

        task["preprocess_start_time"] = time.time()

        self.data_processor.process_request_dict(task, self.max_model_len)

        task["prompt_token_ids_len"] = len(task["prompt_token_ids"])
        input_ids_len = task["prompt_token_ids_len"]
        task["max_tokens"] = min(self.max_model_len - input_ids_len , task.get("max_tokens"))
        min_tokens = task.get("min_tokens")
        if input_ids_len + min_tokens >= self.max_model_len:
            error_msg = (
                f"Input text is too long, input_ids_len ({input_ids_len}) "
                f"+ min_dec_len ({min_tokens}) >= max_model_len "
            )
            api_server_logger.error(error_msg)
            raise EngineError(error_msg, error_code=400)

        if input_ids_len > self.max_model_len:
            error_msg = (
                f"Length of input token({input_ids_len}) exceeds the limit max_model_len({self.max_model_len})."
            )
            api_server_logger.error(error_msg)
            raise EngineError(error_msg, error_code=400)

        task["preprocess_end_time"] = time.time()
        preprocess_cost_time = task["preprocess_end_time"] - task["preprocess_start_time"]
        api_server_logger.info(
            f"Cache request with req_id ({task.get('request_id')}), "
            f"cost {time.time() - preprocess_cost_time}"
        )
        api_server_logger.debug(f"Recieve task: {task}")
        self.zmq_client.send_json(task)


    def check_health(self, time_interval_threashold=30):
        """
        Check the health of the model server by checking whether all workers are alive.

        """
        if self.worker_healthy_live_signal.value[0]:
            elapsed_time = time.time() - self.worker_healthy_live_signal.value[0]
            if elapsed_time > time_interval_threashold:
                return False, "Worker Service Not Healthy"

        return True, ""

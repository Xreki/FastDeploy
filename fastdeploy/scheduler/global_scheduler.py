"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from typing import List, Optional
import time
import redis
from fastdeploy.engine.request import Request, RequestOutput
from fastdeploy.scheduler.data import ScheduledRequest, ScheduledResponse
from fastdeploy.utils import llm_logger


class GlobalScheduler(object):
    """
    GlobalScheduler class
    """

    def __init__(self,
                 host: str,
                 port: int,
                 db: int,
                 password: Optional[str],
                 topic: str,
                 ttl: int,
                 remote_write_time: int,
                 wait_response_timeout: float
                 ):

        self.topic = topic
        self.ttl = ttl
        self.remote_write_time = remote_write_time
        self.wait_response_timeout = wait_response_timeout
        self.wait_request_timeout = 10

        self.client = redis.Redis(
            host=host, port=port, db=db, password=password)

    def _request_queue_name(self):
        return f"{self.topic}.request"

    def _response_queue_name(self, id: str):
        return f"{self.topic}.response.{id}"

    def _unique_key_name(self, id: str):
        return f"{self.topic}.unique.{id}"

    @staticmethod
    def calc_required_blocks(token_num, block_size):
        """calculate required blocks for given token number"""
        return (token_num + block_size - 1) // block_size

    def put_requests(self, requests: List[Request]):
        """
            add requests to shared cache
        """
        requests: List[ScheduledRequest] = [
            ScheduledRequest(request) for request in requests]

        # check the uniqueness of the request_id
        valided_keys = list()
        duplicated_ids = list()
        for request in requests:
            unique_key = self._unique_key_name(request.id)
            if self.client.set(unique_key, "", ex=self.ttl, nx=True):
                valided_keys.append(unique_key)
            else:
                duplicated_ids.append(request.id)

        if len(duplicated_ids) > 0:
            self.client.delete(*valided_keys)
            raise ValueError(
                f"Request_id is duplicated (ids={duplicated_ids})")

        # add to request queue
        serialized_requests = [request.serialize() for request in requests]
        self.client.rpush(self._request_queue_name(), *serialized_requests)
        llm_logger.debug(f"Global cached requests: {requests}")

    def get_requests(self, available_blocks, block_size, reserved_output_blocks, \
        max_num_batched_tokens, batch=1) -> List[Request]:
        """
            get requests blocked from shared cache
        """

        if available_blocks <= reserved_output_blocks or batch < 1:
            return []

        serialized_requests = self.client.lpop(
            self._request_queue_name(), batch)

        if serialized_requests is None or len(serialized_requests) == 0:
            blocked_data = self.client.blpop(
                self._request_queue_name(), self.wait_request_timeout)
            if blocked_data is None:
                return []
            serialized_requests = blocked_data[1:]

        required_total_blocks = 0
        current_prefill_tokens = 0
        remaining_request = []
        requests = []
        for serialized_request in serialized_requests:
            if len(remaining_request) > 0:
                remaining_request.append(serialized_request)
                continue

            request: ScheduledRequest = ScheduledRequest.unserialize(
                serialized_request)
            if (time.time() - request.scheduled_time) > self.ttl:
                llm_logger.info(f"Request_id ({request.id}) has expired")
                continue

            required_input_blocks = self.calc_required_blocks(
                request.size, block_size)
            current_prefill_tokens += request.size
            required_total_blocks += required_input_blocks + reserved_output_blocks
            if required_total_blocks > available_blocks or current_prefill_tokens > max_num_batched_tokens:
                remaining_request.append(serialized_request)
                continue
            requests.append(request.raw)
        llm_logger.debug(f"Global get requests:{len(requests)}")

        if len(remaining_request) > 0:
            self.client.lpush(self._request_queue_name(), *remaining_request)
        return requests

    def put_results(self, results: List[RequestOutput]):
        """
            add results to shared cache
        """
        responses: List[ScheduledResponse] = [
            ScheduledResponse(result) for result in results]
        sorted_responses = sorted(
            responses, key=lambda response: f"{response.id}.{response.index}")

        group = dict()
        for response in sorted_responses:
            serialized_response = response.serialize()
            if response.id not in group:
                group[response.id] = [serialized_response]
                continue
            group[response.id].append(serialized_response)

        for response_id, responses in group.items():
            ttl = self.client.ttl(self._unique_key_name(
                response_id)) - self.remote_write_time
            if ttl <= 0:
                llm_logger.info(
                    f"Output of request_id ({response_id}) has expired")
                continue

            with self.client.pipeline() as pipe:
                pipe.multi()
                pipe.rpush(self._response_queue_name(response_id), *responses)
                pipe.expire(self._response_queue_name(response_id), ttl)
                pipe.execute()

    def get_results(self, request_id: str) -> List[RequestOutput]:
        """
            get results blocked from shared cache
        """
        key = self._response_queue_name(request_id)
        size = self.client.llen(key)

        serialized_responses = self.client.lpop(key, size)
        if serialized_responses is None or len(serialized_responses) == 0:
            ttl = self.client.ttl(self._unique_key_name(request_id))
            wait_time = self.wait_response_timeout if ttl <= 0 else min(ttl, self.wait_response_timeout)
            blocked_data = self.client.blpop(key, wait_time)
            if blocked_data is None:
                return []
            serialized_responses = blocked_data[1:]

        output = []
        for serialized_response in serialized_responses:
            response = ScheduledResponse.unserialize(serialized_response)
            output.append(response.raw)
        return output

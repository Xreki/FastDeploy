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

from typing import List
import time
import redis
from fastdeploy.engine.request import Request, RequestOutput
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.scheduler.data import ScheduledRequest, ScheduledResponse
from fastdeploy.utils import llm_logger


class GlobalScheduler(object):
    """
    GlobalScheduler class
    """

    def __init__(self):
        self.client = redis.Redis(
            host='10.178.5.194', port=6379, db=1, password="aurora_123")
        self.topic = "fd_reqs"
        self.ttl = 180
        self.redundant_ttl = 30
        self.unique_key_ttl = self.ttl + self.redundant_ttl
        self.wait_request_timeout = 10
        self.wait_response_timeout = 5  # required: wait_response_timeout < redundant_ttl

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
            if self.client.set(unique_key, "", ex=self.unique_key_ttl, nx=True):
                valided_keys.append(unique_key)
            else:
                duplicated_ids.append(request.id)

        if len(duplicated_ids) > 0:
            self.client.delete(*valided_keys)
            raise ValueError(
                f"request_id is duplicated (ids={duplicated_ids})")

        # add to request queue
        serialized_requests = [request.serialize() for request in requests]
        self.client.rpush(self._request_queue_name(), *serialized_requests)
        llm_logger.debug(f"global cached requests: {requests}")
        main_process_metrics.num_requests_waiting.inc(len(requests))

    def get_requests(self, available_blocks, block_size, reserved_output_blocks, batch=1) -> List[Request]:
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
        remaining_request = []
        requests = []
        for serialized_request in serialized_requests:
            if len(remaining_request) > 0:
                remaining_request.append(serialized_request)
                continue

            request: ScheduledRequest = ScheduledRequest.unserialize(
                serialized_request)
            if (time.time() - request.scheduled_time) > self.ttl:
                llm_logger.info(f"request_id ({request.id}) has expired")
                continue

            required_input_blocks = self.calc_required_blocks(
                request.size, block_size)
            required_total_blocks += required_input_blocks + reserved_output_blocks
            if required_total_blocks > available_blocks:
                remaining_request.append(serialized_request)
                continue
            requests.append(request.raw)
        llm_logger.debug(f"global get requests:{len(requests)}")

        if len(remaining_request) > 0:
            self.client.lpush(self._request_queue_name(), *remaining_request)
        main_process_metrics.num_requests_running.inc(len(requests))
        main_process_metrics.num_requests_waiting.dec(len(requests))
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
                response_id)) - self.redundant_ttl
            if ttl <= 0:
                llm_logger.info(
                    f"output of request_id ({response_id}) has expired")
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
            ttl = self.client.ttl(self._unique_key_name(
                request_id)) - self.redundant_ttl
            if ttl <= 0:
                raise ValueError(
                    f"output of request_id ({request_id}) has expired")

            blocked_data = self.client.blpop(key, self.wait_response_timeout)
            if blocked_data is None:
                return []
            serialized_responses = blocked_data[1:]

        output = []
        for serialized_response in serialized_responses:
            response = ScheduledResponse.unserialize(serialized_response)
            output.append(response.raw)
        return output

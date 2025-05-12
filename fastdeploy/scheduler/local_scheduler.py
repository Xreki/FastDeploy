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


from typing import Dict, List, Set
import threading
import time

from fastdeploy.utils import llm_logger
from fastdeploy.engine.request import Request, RequestOutput
from fastdeploy.scheduler.data import ScheduledRequest, ScheduledResponse


class LocalScheduler(object):
    """
    LocalScheduler Class
    """

    def __init__(self):
        self.mutex = threading.Lock()
        self.max_size = 10000000
        self.ttl = 1800000
        self.ids: Set[str] = set()

        self.request_read_cursor = 0
        self.requests: List[ScheduledRequest] = list()
        self.responses: Dict[str, List[ScheduledResponse]] = dict()

        self.wait_request_timeout = 10
        self.wait_response_timeout = 0.001  # required: wait_response_timeout < ttl

        self.requests_not_empty = threading.Condition(self.mutex)
        self.responses_not_empty = threading.Condition(self.mutex)

    def _recycle(self):
        """
            recycle memory
        """
        if len(self.requests) <= self.max_size:
            return

        now = time.time()
        expired_ids = []
        for request in self.requests:
            if (now - request.scheduled_time < self.ttl):
                break
            expired_ids.append(request.id)

        for i, expired_id in enumerate(expired_ids):
            self.ids.discard(expired_id)
            self.requests.pop(i)
            self.responses.pop(expired_id, None)

        if len(expired_ids) > 0:
            if len(expired_ids) - 1 >= self.request_read_cursor:
                self.request_read_cursor = 0
            else:
                self.request_read_cursor -= len(expired_ids)

    def put_requests(self, requests: List[Request]):
        """  submit requests to scheduler
             Args: 
                 requests: List[Request]
        """
        requests: List[ScheduledRequest] = [
            ScheduledRequest(request) for request in requests]
        with self.mutex:
            if len(self.requests) + len(requests) > self.max_size:
                self._recycle()
            if len(self.requests) + len(requests) > self.max_size:
                raise OverflowError(
                    f"exceeding the max length of the local scheduler (max_size={self.max_size})")

            duplicated_ids = [
                request.id for request in requests if request.id in self.ids]
            if len(duplicated_ids) > 0:
                raise ValueError(
                    f"request_id is duplicated (ids={duplicated_ids})")

            self.requests += requests
            self.ids.update([request.id for request in requests])
            self.requests_not_empty.notify_all()

            llm_logger.debug(f"local cached requests: {requests}")
            

    def calc_required_blocks(self, token_num, block_size):
        """calculate required blocks for given token number"""
        return (token_num + block_size - 1) // block_size

    def get_requests(self, available_blocks, block_size, reserved_output_blocks, batch=1) -> List[Request]:
        """get requests from local cache
            Args: 
                available_blocks: int
                block_size: int
                reserved_output_blocks: int
                batch: int
        """
        if available_blocks <= reserved_output_blocks or batch < 1:
            return []

        with self.requests_not_empty:
            batch_requests = self.requests_not_empty.wait_for(
                lambda: self.requests[self.request_read_cursor:
                                      self.request_read_cursor + batch], self.wait_request_timeout)

            required_total_blocks = 0
            requests: List[Request] = []
            for request in batch_requests:
                required_input_blocks = self.calc_required_blocks(
                    request.size, block_size)
                required_total_blocks += required_input_blocks + reserved_output_blocks
                if required_total_blocks > available_blocks:
                    break
                requests.append(request.raw)

            self.request_read_cursor += len(requests)
            llm_logger.debug(f"local get requests: {len(requests)}")
            return requests

    def put_results(self, results: List[RequestOutput]):
        """put results into local cache"""
        responses: List[ScheduledResponse] = [
            ScheduledResponse(result) for result in results]
        with self.mutex:
            for response in responses:
                if response.id not in self.ids:
                    llm_logger.info(
                        f"output of request_id({response.id} is expired)")
                    continue

                if response.id not in self.responses:
                    self.responses[response.id] = [response]
                    continue
                self.responses[response.id].append(response)
            self.responses_not_empty.notify_all()

    def get_results(self, request_id: str) -> List[RequestOutput]:
        """get results from local cache"""
        with self.responses_not_empty:
            if request_id not in self.ids:
                raise ValueError(f"output of request_id {request_id} has expired")
            
            responses = self.responses_not_empty.wait_for(
                lambda: self.responses.get(request_id, []), self.wait_response_timeout)
            self.responses.pop(request_id, None)
            return [response.raw for response in responses]

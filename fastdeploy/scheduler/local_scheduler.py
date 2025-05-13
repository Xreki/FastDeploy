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


from typing import Dict, List, Optional
import threading
import time

from fastdeploy.utils import llm_logger
from fastdeploy.engine.request import Request, RequestOutput
from fastdeploy.scheduler.data import ScheduledRequest, ScheduledResponse


class LocalScheduler(object):
    """
    LocalScheduler Class
    """

    def __init__(self,
                 max_size: int,
                 ttl: int,
                 wait_response_timeout: float):
        self.max_size = max_size
        self.ttl = ttl

        self.mutex = threading.Lock()
        self.ids_read_cursor = 0
        self.ids: List[str] = list()

        self.requests: Dict[str, ScheduledRequest] = dict()
        self.responses: Dict[str, List[ScheduledResponse]] = dict()

        self.wait_request_timeout = 10
        self.wait_response_timeout = wait_response_timeout

        self.requests_not_empty = threading.Condition(self.mutex)
        self.responses_not_empty = threading.Condition(self.mutex)

    def _recycle(self, request_id: Optional[str] = None):
        """
            recycle memory
        """
        if request_id is not None:
            self.requests.pop(request_id, None)
            self.responses.pop(request_id, None)
            self.ids.pop(self.ids.index(request_id))
            self.ids_read_cursor -= 1
            return

        if self.max_size <= 0:
            return

        if len(self.requests) <= self.max_size:
            return

        now = time.time()
        expired_ids = []
        for request_id in self.ids:
            request = self.requests[request_id]
            if (now - request.scheduled_time < self.ttl):
                break
            expired_ids.append(request.id)

        for i, expired_id in enumerate(expired_ids):
            self.requests.pop(expired_id, None)
            self.responses.pop(expired_id, None)
            self.ids.pop(i)

        if len(expired_ids) > 0:
            if len(expired_ids) - 1 >= self.ids_read_cursor:
                self.ids_read_cursor = 0
            else:
                self.ids_read_cursor -= len(expired_ids)

    def put_requests(self, requests: List[Request]):
        """  submit requests to scheduler
             Args:
                 requests: List[Request]
        """
        scheduled_requests: Dict[str, ScheduledRequest] = dict()
        scheduled_ids: List[str] = list()
        for request in requests:
            scheduled_request = ScheduledRequest(request)
            scheduled_requests[scheduled_request.id] = scheduled_request
            scheduled_ids.append(scheduled_request.id)

        with self.mutex:
            self._recycle()
            if self.max_size > 0 and len(self.requests) + len(scheduled_requests) > self.max_size:
                raise OverflowError(
                    f"exceeding the max length of the local scheduler (max_size={self.max_size})")

            duplicated_ids = [
                scheduled_id for scheduled_id in scheduled_ids if scheduled_id in self.requests]
            if len(duplicated_ids) > 0:
                raise ValueError(
                    f"request_id is duplicated (ids={duplicated_ids})")

            self.requests.update(scheduled_requests)
            self.ids += scheduled_ids
            self.requests_not_empty.notify_all()

            llm_logger.debug(f"local cached requests: {scheduled_ids}")

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
            batch_ids = self.requests_not_empty.wait_for(
                lambda: self.ids[self.ids_read_cursor:
                                 self.ids_read_cursor + batch], self.wait_request_timeout)

            required_total_blocks = 0
            requests: List[Request] = []
            for request_id in batch_ids:
                request = self.requests[request_id]
                required_input_blocks = self.calc_required_blocks(
                    request.size, block_size)
                required_total_blocks += required_input_blocks + reserved_output_blocks
                if required_total_blocks > available_blocks:
                    break
                requests.append(request.raw)

            self.ids_read_cursor += len(requests)
            llm_logger.debug(f"local get requests: {len(requests)}")
            return requests

    def put_results(self, results: List[RequestOutput]):
        """put results into local cache"""
        responses: List[ScheduledResponse] = [
            ScheduledResponse(result) for result in results]
        with self.mutex:
            for response in responses:
                if response.id not in self.requests:
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
            if request_id not in self.requests:
                raise ValueError(
                    f"output of request_id {request_id} is expired")

            responses = self.responses_not_empty.wait_for(
                lambda: self.responses.get(request_id, []), self.wait_response_timeout)
            self.responses.pop(request_id, None)

            finished = False
            results = []
            for response in responses:
                results.append(response.raw)
                finished |= response.finished

            if finished:
                self._recycle(request_id)
            return results

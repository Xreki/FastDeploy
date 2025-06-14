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
import os
import threading
import time
import traceback
import weakref
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from paddlenlp.utils.env import MAX_BSZ
from paddlenlp.utils.env import MAX_DRAFT_TOKENS
from paddlenlp.utils.env import SPECULATE_MAX_BSZ

from fastdeploy.engine.request import CompletionOutput
from fastdeploy.engine.request import RequestMetrics
from fastdeploy.engine.request import RequestOutput
from fastdeploy.inter_communicator import IPCSignal

from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.utils import llm_logger


class TokenProcessor(object):
    """
    get Token/Score from Paddle inference engine
    """

    def __init__(self, cfg, cached_generated_tokens, engine_worker_queue,
                 split_connector):
        import paddle

        paddle.device.set_device("cpu")
        self.cfg = cfg
        self.cached_generated_tokens = cached_generated_tokens
        self.resource_manager = None
        self.engine_worker_queue = engine_worker_queue
        self.tokens_counter = Counter()
        self.split_connector = split_connector

        self.is_speculate_decoding = False
        if self.is_speculate_decoding:
            self.output_tokens = paddle.full(shape=[
                SPECULATE_MAX_BSZ * MAX_DRAFT_TOKENS + SPECULATE_MAX_BSZ + 2, 1
            ],
                                             fill_value=2,
                                             dtype="int64")
        else:
            self.output_tokens = paddle.full(shape=[MAX_BSZ + 2, 1],
                                             fill_value=2,
                                             dtype="int64")
        self.worker = None

        self.statics_start_time = time.time()
        self.number_of_tasks = 0
        self.number_of_input_tokens = 0
        self.number_of_output_tokens = 0
        self.total_step = 0
        prefill_time_data = np.zeros([100], dtype=np.float32)
        self.prefill_time_signal = IPCSignal(
            name="prefill_time_signal",
            array=prefill_time_data,
            dtype=np.float32,
            suffix=os.getpid(),
            create=True)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._finalizer = weakref.finalize(self, self._cleanup_resources)

    def _cleanup_resources(self):
        """Cleaning up shared memory resources"""
        if hasattr(self, 'prefill_time_signal'):
            self.prefill_time_signal.clear()

        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

    def set_resource_manager(self, resource_manager):
        """
        set ResourceManager

        Args:
            resource_manager (ResourceManager)
        """
        assert self.resource_manager is None, "The resource manager is not None, cannot set again."
        self.resource_manager = resource_manager

    def run(self):
        """
        start thread to get tokens
        """
        assert self.resource_manager is not None, "The resource manager is None, cannot run."
        if self.worker is not None:
            raise Exception("Worker is already running!")

        self.worker = threading.Thread(target=self.process_sampling_results,
                                       args=())
        self.worker.daemon = True
        self.worker.start()

    def process_sampling_results(self):
        """
        read tokens from paddle inference engine and process
        """
        from fastdeploy.model_executor.models import \
            inference_runner_supported_models
        if not any(self.cfg.model_config.architectures.startswith(model) for model in inference_runner_supported_models) \
            and not self.cfg.model_config.architectures.startswith("ErnieMoEVLForCausalLM") \
            and "ErnieBotLMHeadModel" not in self.cfg.model_config.architectures:
            from paddlenlp_ops import get_output, speculate_get_output
        else:
            os.environ["ELLM_LOG_LEVEL"] = "3"
            use_pip_eff_llm = os.getenv('USE_PIP_EFF_LLM')
            if use_pip_eff_llm is None:
                from fastdeploy.model_executor.ops.gpu import (
                    get_output, speculate_get_output)
            else:
                from efficientllm.ops.gpu import (get_output,
                                                  speculate_get_output)

        while True:
            try:
                rank_id = 0
                is_blocking = True
                if self.is_speculate_decoding:
                    speculate_get_output(self.output_tokens, rank_id,
                                         is_blocking)
                else:
                    get_output(self.output_tokens, rank_id, is_blocking)
                if self.output_tokens[0, 0] == -2:
                    continue
                self._process_prefill_metrics()
                self._process_batch_output()
            except Exception as e:
                llm_logger.info("while get input_data error: {0} {1}".format(
                    e, str(traceback.format_exc())))


    def _process_prefill_metrics(self):
        """Asynchronous processing prefill time indicators"""
        def process_metrics():
            try:
                current_index = 0
                while current_index < len(self.prefill_time_signal.value):
                    prefill_time = self.prefill_time_signal.value[current_index]
                    if prefill_time > 0:
                        main_process_metrics.request_prefill_time.observe(prefill_time)
                        self.prefill_time_signal.value[current_index] = 0
                    current_index += 1
            except Exception as e:
                llm_logger.error(f"Error processing prefill metrics: {e}")
        self.executor.submit(process_metrics)
    def postprocess(self, batch_result):
        """
        single post-processing function

        Args:
            batch_result (list): batch results
        """
        self.cached_generated_tokens.put_results(batch_result)

    def _recycle_resources(self, task_id, index, task, is_prefill=False):
        """
        recycle resources
        """
        if is_prefill and not self.resource_manager.cache_transfer_finished[
                task_id]:
            wait_for_all_finish = time.time()
            while 1:
                finished_task_ids = self.engine_worker_queue.get_finished_req()
                if len(finished_task_ids) > 0:
                    for finished_task_id in finished_task_ids:
                        llm_logger.info(
                            f"finished_task_id: {finished_task_id}")
                        self.resource_manager.cache_transfer_finished[
                            finished_task_id] = True
                    if self.resource_manager.cache_transfer_finished[task_id]:
                        break
                else:
                    time.sleep(0.001)
            llm_logger.info(
                f"recycle_resources cost time: {time.time() - wait_for_all_finish}"
            )

        if task_id in self.resource_manager.cache_transfer_finished and self.resource_manager.cache_transfer_finished[
                task_id]:
            del self.resource_manager.cache_transfer_finished[task_id]

        self.resource_manager.stop_flags[index] = True
        self.resource_manager.tasks_list[index] = None
        self.resource_manager._recycle_block_tables(task)
        if task_id in self.tokens_counter:
            del self.tokens_counter[task_id]

    def _process_batch_output(self):
        """
        batch post-processing function
        """
        tokens = self.output_tokens.numpy()
        batch = self.output_tokens[1, 0]
        if not self.is_speculate_decoding:
            tokens = tokens[2:batch + 2]
        else:
            accept_num = tokens[2:batch + 2]

        batch_result = list()
        prefill_batch_result = list()
        prefill_port = -1
        for i in range(batch):
            if self.resource_manager.stop_flags[i]:
                continue

            if not self.is_speculate_decoding:
                token_ids = [int(tokens[i, 0])]
            else:
                token_ids = tokens[
                    2 + SPECULATE_MAX_BSZ + i * MAX_DRAFT_TOKENS:2 +
                    SPECULATE_MAX_BSZ + i * MAX_DRAFT_TOKENS +
                    accept_num[i, 0],
                    0,
                ].tolist()
            if any(token_id < 0 for token_id in token_ids):
                continue

            task = self.resource_manager.tasks_list[i]

            if task.get("prefill_chunk_info", None) is not None:
                prefill_chunk_num = task.get("prefill_chunk_num", 0)
                task.prefill_chunk_num = prefill_chunk_num + 1
                
                if task.prefill_chunk_num < len(task.prefill_chunk_info):
                    continue

            task_id = task.request_id

            self.total_step += 1
            current_time = time.time()
            if self.tokens_counter[task_id] == 0:
                metrics = RequestMetrics(
                    arrival_time=task.arrival_time,
                    inference_start_time=task.inference_start_time,
                    first_token_time=time.time() - task.inference_start_time,
                    time_in_queue=task.schedule_start_time -
                    task.preprocess_end_time,
                    preprocess_cost_time=task.preprocess_end_time -
                    task.preprocess_start_time)

                self._record_first_token_metrics(task, current_time)

            else:
                metrics = RequestMetrics(
                    arrival_time=time.time(),
                    request_start_time=task.arrival_time,
                )
            self.number_of_output_tokens += len(token_ids)
            self._record_metrics(task, current_time, token_ids)
            result = RequestOutput(request_id=task_id,
                                   outputs=CompletionOutput(index=i,
                                                            token_ids=[]),
                                   finished=False,
                                   metrics=metrics)
            if self.tokens_counter[task_id] == 0:
                if task.messages is not None:
                    result.prompt = task.messages
                result.prompt_token_ids = task.prompt_token_ids
                result.num_cached_tokens = task.num_cached_tokens

            is_prefill = task.disaggregate_info is not None and task.disaggregate_info[
                "role"] == "prefill"

            for token_id in token_ids:
                self.tokens_counter[task_id] += 1
                result.outputs.token_ids.append(token_id)
                if token_id in task.eos_token_ids or is_prefill:
                    result.finished = True
                    result.prompt = task.prompt
                    result.prompt_token_ids = task.prompt_token_ids
                    llm_logger.info(
                        f"Request: {task_id} finished, number of "
                        f"generated tokens: {self.tokens_counter[task_id]}.")
                    llm_logger.info(
                        f"Request: {task_id} token ratio: {self.tokens_counter[task_id] / (time.time() - task.inference_start_time)}"
                    )
                    llm_logger.info(f"{self.resource_manager.info()}")
                    llm_logger.info(
                        f"Speculate accept ratio: {1 - self.total_step * 1.0 / self.number_of_output_tokens}"
                        f" total step: {self.total_step}. total_output_token_num: {self.number_of_output_tokens}"
                    )
                    if not is_prefill:
                        self._record_completion_metrics(task, current_time)
                    self._recycle_resources(task_id, i, task, is_prefill)
                    if is_prefill:
                        prefill_batch_result.append(result)
                        prefill_port = task.disaggregate_info['port']
                    break
            if not is_prefill:
                batch_result.append(result)
        if len(prefill_batch_result) > 0:
            self.split_connector.send_first_token(prefill_port,
                                                  prefill_batch_result)
        self.postprocess(batch_result)

    def _record_metrics(self, task, current_time, token_ids):
        """Record all metrics for a task"""
        if hasattr(task, 'last_token_time') and task.last_token_time is not None:
            token_gen_time = current_time - task.last_token_time
            main_process_metrics.time_per_output_token.observe(token_gen_time)
        task.last_token_time = current_time

        # Record generation metrics
        main_process_metrics.generation_tokens_total.inc(len(token_ids))

    def _record_first_token_metrics(self, task, current_time):
        """Record metrics for first token"""
        task.first_token_time = current_time
        main_process_metrics.time_to_first_token.observe(current_time - task.inference_start_time)
        main_process_metrics.request_queue_time.observe(task.schedule_start_time - task.preprocess_end_time)

    def _record_completion_metrics(self, task, current_time):
        """Record metrics when request completes"""
        if hasattr(task, 'first_token_time'):
            decode_time = current_time - task.first_token_time
            main_process_metrics.request_decode_time.observe(decode_time)

        main_process_metrics.num_requests_running.dec(1)
        main_process_metrics.request_success_total.inc()
        main_process_metrics.request_inference_time.observe(current_time - task.inference_start_time)
        main_process_metrics.request_generation_tokens.observe(self.tokens_counter[task.request_id])

class WarmUpTokenProcessor(TokenProcessor):
    """
    Warmup Processor
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._is_running = True
        self._is_blocking = True

    def postprocess(self, batch_result):
        pass

    def process_sampling_results(self):
        """
        get output from model and process it
        """
        from fastdeploy.model_executor.models import \
            inference_runner_supported_models
        if not any(self.cfg.model_config.architectures.startswith(model) for model in inference_runner_supported_models) \
            and not self.cfg.model_config.architectures.startswith("ErnieMoEVLForCausalLM"):
            from paddlenlp_ops import get_output, speculate_get_output
        else:
            os.environ["ELLM_LOG_LEVEL"] = "3"
            use_pip_eff_llm = os.getenv('USE_PIP_EFF_LLM')
            if use_pip_eff_llm is None:
                from fastdeploy.model_executor.ops.gpu import (
                    get_output, speculate_get_output)
            else:
                from efficientllm.ops.gpu import (get_output,
                                                  speculate_get_output)

        while self._is_running:
            try:
                rank_id = 0
                if self.is_speculate_decoding:
                    speculate_get_output(self.output_tokens, rank_id,
                                         self._is_blocking)
                else:
                    get_output(self.output_tokens, rank_id, self._is_blocking)

                if self.output_tokens[0, 0] == -2:
                    continue
                self._process_batch_output()
            except Exception as e:
                llm_logger.info("while get input_data error: {0} {1}".format(
                    e, str(traceback.format_exc())))

    def stop(self):
        """
        stop warm up thread
        """
        self._is_running = False
        self.worker.join()
        llm_logger.info("warm up thread stop")
        del self.worker

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

from __future__ import annotations
import sys
import asyncio
import multiprocessing
import os
import signal
import queue
import subprocess
import time
import uuid
import traceback
import weakref
from datetime import datetime
from multiprocessing import shared_memory
from collections import deque
import threading
import numpy as np

from fastdeploy.input.preprocess import InputPreprocessor
from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.request import Request
from fastdeploy.engine.resource_manager import ResourceManager
from fastdeploy.inter_communicator import EngineWorkerQueue
from fastdeploy.output.token_processor import TokenProcessor, WarmUpTokenProcessor
from fastdeploy.inter_communicator import IPCSignal
from fastdeploy.utils import llm_logger


class LLMEngine(object):
    """
    Engine class responsible for managing the Large Language Model (LLM) operations.

    Attributes:
        cfg (Config): Configuration object containing all the parameters.
        cached_generated_tokens (queue.Queue): Queue to store generated tokens.
        cached_task_deque (collections.deque): Deque to store cached tasks.
        req_output (dict): Dictionary to store request outputs.
        req_output_completion (dict): Dictionary to track request completion status.
        input_processor (InputPreprocessor): Preprocessor for input data.
        resource_manager (ResourceManager): Manager for resource allocation.
        token_processor (TokenProcessor): Processor for token generation.
        engine_worker_queue (EngineWorkerQueue): Queue for communication between engine and workers.
        is_started (bool): Flag indicating if the engine has started.
        do_profile (int): Flag indicating if profiling is enabled.
    """

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs):
        """
        Creates an LLM engine from the provided engine arguments.

        Args:
            engine_args (EngineArgs): Engine arguments object.

        Returns:
            LLMEngine: Instance of the LLMEngine class.
        """
        # Create the engine configs.
        config = engine_args.create_engine_config()
        # Create the LLMEngine.
        return cls(cfg=config)

    def __init__(self, cfg):
        """
        Initializes the LLMEngine with the provided configuration.

        Args:
            cfg (Config): Config object containing all the configuration parameters.
        """
        self.cfg = cfg

        self.cached_generated_tokens = queue.Queue()
        self.cached_task_deque = deque()
        self.req_output = dict()
        self.req_output_completion = dict()

        self.input_processor = InputPreprocessor(cfg.tokenizer)
        self.resource_manager = ResourceManager(cfg.max_num_seqs, cfg.cache_config)

        self.token_processor = TokenProcessor(cfg=self.cfg, cached_generated_tokens=self.cached_generated_tokens)
        self.token_processor.set_resource_manager(self.resource_manager)
        time.sleep(1)  # TODO: Investigate the purpose of this sleep.

        # TODO
        # 1. 增加engine hostname
        address = ('0.0.0.0', self.cfg.engine_worker_queue_port)
        self.engine_worker_queue = EngineWorkerQueue(
            address=address, is_server=True, num_client=self.cfg.tensor_parallel_size)

        self.is_started = False

        if self.cfg.cache_config.num_gpu_blocks_override is None:
            self.do_profile = 1
        else:
            self.do_profile = 0

        self._init_worker_signals()
        self._finalizer = weakref.finalize(self, self._exit_sub_services)

    def start(self):
        """
        Initializes the engine and starts its sub-services.
        """
        assert not self.is_started, "The engine is already started."
        start_time = time.time()

        self.data_processor = self.input_processor.create_processor()

        self.worker_proc = self._start_worker_service()
        llm_logger.info("Waitting worker processes ready...")
        time.sleep(5)
        while not self._worker_processes_ready():
            if self.worker_proc.poll() is not None:
                llm_logger.error("The worker process is not alive, check log/worklog.* for more details.")
                return False


        # Start warmup if enabled
        if self.cfg.use_warmup:
            llm_logger.info("Starting warmup")
            self._set_warmup_token_processor()
            self.warmup()
            self._del_warmup_token_processor()
            llm_logger.info("Warmup finished")

        self.token_processor.tasks_queue = self.engine_worker_queue

        self.insert_task_to_worker_thread = threading.Thread(target=self._insert_task_to_worker, args=())
        self.insert_task_to_worker_thread.daemon = True
        self.insert_task_to_worker_thread.start()

        # Start TokenProcessor thread
        self.token_processor.run()

        self._receive_output_thread = threading.Thread(target=self._recieve_output, args=())
        self._receive_output_thread.daemon = True
        self._receive_output_thread.start()

        if self.do_profile:
            self._stop_profile()
        llm_logger.info("Worker processes are launched with {} seconds.".format(time.time() - start_time))
        return True



    def _recieve_output(self):
        """
        Recieve output from token processor and store them in cache
        """
        while True:
            try:
                batch_result = self.cached_generated_tokens.get()
                for result in batch_result:
                    if result.request_id not in self.req_output:
                        self.req_output[result.request_id] = deque()
                        self.req_output_completion[result.request_id] = result
                    else:
                        self.req_output_completion[result.request_id].add(result)
                        result.metrics.model_forward_time = \
                            self.req_output_completion[result.request_id].metrics.model_forward_time

                    if result.finished:
                        result = self.req_output_completion[result.request_id]
                        del self.req_output_completion[result.request_id]

                    self.req_output[result.request_id].appendleft(result)

            except Exception as e:
                llm_logger.error("Unexcepted error happend: {}, {}".format(e, str(traceback.format_exc())))

    def get_result(self, req_id):
        """
        Get result from cache
        """
        if req_id not in self.req_output:
            return None
        if self.req_output[req_id]:
            return self.req_output[req_id].pop()
        else:
            return None


    def _get_tasks(self, num=1):
       """
       Get tasks from cached_task_deque.
       """
       if len(self.cached_task_deque) == 0:
           return []

       task_num = min(len(self.cached_task_deque), num)
       tasks = []
       need_block_num = 0
       for i in range(task_num):
           num_input_token = self.cached_task_deque[-1].prompt_token_ids_len
           need_block_num += self.resource_manager.get_required_block_number(num_input_token)
           if need_block_num > self.resource_manager.availabel_block_num():
               break
           tasks.append(self.cached_task_deque.pop())
       return tasks

    def _insert_task_to_worker(self):
        """
        Insert task to engine thread, monitor cached_task_deque.
        if the engine has resource, insert task to engine
        """
        try:
            while 1:
                if self.resource_manager.available_batch() == 0:
                    time.sleep(0.001)
                    continue
                if self.engine_worker_queue.num_tasks() > 0:
                    time.sleep(0.001)
                    continue

                num_prefill_batch = min(self.resource_manager.available_batch(), self.cfg.max_prefill_batch)
                tasks = self._get_tasks(num_prefill_batch)
                if len(tasks) == 0:
                    time.sleep(0.001)
                    continue

                try:
                    self.insert_tasks(tasks)
                except Exception as e:
                    err_msg = "Error happend while insert task to engine: {}, {}.".format(
                        e, str(traceback.format_exc())
                    )
                    llm_logger.error(err_msg)
            llm_logger.info("finish insert_task_to_worker thread")
        except Exception as e:
            llm_logger.error(
                "insert_task_to_worker thread exit " f"unexpectedly, {e}. {str(traceback.format_exc())}"
            )


    def add_requests(self, task, sampling_params=None):
        """
        Add a new request to the queue.

        Args:
            task: Request A dictionary representing the request.
            sampling_params: A dictionary representing the sampling parameters.

        Returns:
            None
        """
        # TODO 输入输出长度确认

        request = Request.from_dict(task)
        if sampling_params is not None:
            request.sampling_params = sampling_params
        request.preprocess_start_time = datetime.now()
        if int(task.get("enable_text_truncate", 1)):
            real_seq_len = self.cfg.max_model_len - task.get("max_dec_len", 800)
            self.data_processor.process_request(request, max_model_len=real_seq_len)
        else:
            self.data_processor.process_request(request, self.cfg.max_model_len)


        request.prompt_token_ids_len = len(request.prompt_token_ids)
        input_ids_len = request.prompt_token_ids_len
        request.set("max_tokens", min(self.cfg.max_model_len - input_ids_len , request.get("max_tokens")))
        min_tokens = request.get("min_tokens")
        if input_ids_len + min_tokens >= self.cfg.max_model_len:
            error_msg = (
                f"Input text is too long, input_ids_len ({input_ids_len}) "
                f"+ min_dec_len ({min_tokens}) >= max_model_len "
            )
            llm_logger.error(error_msg)
            return

        if input_ids_len > self.cfg.max_model_len:
            error_msg = (
                f"Length of input token({input_ids_len}) exceeds the limit max_model_len({self.cfg.max_model_len})."
            )
            llm_logger.error(error_msg)
            return


        required_block_num = self.resource_manager.get_required_block_number(input_ids_len)
        if required_block_num > self.resource_manager.total_block_number():
            error_msg = f"The input task required resources is exceed the limit, task={task}."
            llm_logger.error(error_msg)
            return

        request.preprocess_end_time = datetime.now()
        self.cached_task_deque.appendleft(request)
        llm_logger.info(
            f"cache task with req_id ({request.get('request_id')}), "
            f"cached_task_num: {len(self.cached_task_deque)}."
        )
        llm_logger.debug(f"cache task: {request}")


    def warmup(self):
        """
        construct test tasks and avoid out of memory problem in the worker process
        """
        # get eos_token_id
        pass

    def insert_tasks(self, tasks):
        """
        Insert tasks to engine.
        """
        if not isinstance(tasks, list):
            tasks = [tasks]

        for item in tasks:
            item.schedule_start_time = datetime.now()

        available_batch = np.sum(self.resource_manager.stop_flags)
        if len(tasks) > available_batch:
            llm_logger.error("Inserting batch:{} exceeds the available batch:{}.".format(
                len(tasks), available_batch))
            llm_logger.error("The exceeded part will be ignored!")
            tasks = tasks[:available_batch]

        tasks = self.resource_manager.allocate_resources_for_new_tasks(tasks)
        if not tasks:
            return False

        self.token_processor.number_of_tasks += len(tasks)
        for i in range(len(tasks)):
            self.token_processor.number_of_input_tokens += tasks[i].prompt_token_ids_len

        req_ids = [t.request_id for t in tasks]
        llm_logger.info(f"Tasks are sent to engine, req_ids={req_ids}")
        self.engine_worker_queue.put_tasks((tasks, self.resource_manager.real_bsz))
        return True

    def task_is_finished(self, index):
        """
        judge if the task is finished
        """
        assert index < len(self.resource_manager.stop_flags)
        return self.resource_manager.stop_flags[index]


    def all_tasks_finished(self):
        """
        judge if all tasks are finished
        """
        return np.sum(self.resource_manager.stop_flags) == len(self.resource_manager.stop_flags)




    def _set_warmup_token_processor(self):
        """
        set token_processor for warmup
        """
        self.token_processor_backup = self.token_processor
        self.token_processor = WarmUpTokenProcessor(self.cfg)
        self.token_processor.set_resource_manager(self.resource_manager)
        self.token_processor.tasks_queue = self.engine_worker_queue

        # start TokenProcessor thread
        self.token_processor.run()

    def _del_warmup_token_processor(self):
        """
        delete token_processor for warmup
        """
        self.token_processor.stop()
        del self.token_processor

        # reset token_processor
        self.token_processor = self.token_processor_backup
        del self.token_processor_backup

    def _worker_processes_ready(self):
        """
        judge if all worker processes are ready

        """
        if np.sum(self.worker_ready_signal.value) == self.cfg.tp_num_per_node:
            return True
        return False

    def _init_worker_signals(self):
        """
        Initialize shared memory to indicate engine status
        """
        # worker_ready_signal 用于engine感知各worker进程是否Ready
        worker_ready_signal_data = np.zeros(shape=[self.cfg.tensor_parallel_size], dtype=np.int32)
        self.worker_ready_signal = IPCSignal(name="worker_ready_singnal",
                                             array=worker_ready_signal_data,
                      						 dtype=np.int32,
  											 suffix=os.getpid(),
											 create=True)

        # exist_task_signal 用于各worker进程感知是否有新Task需要处理
        exist_task_signal_data = np.zeros([1], dtype=np.int32)
        self.exist_task_signal = IPCSignal(name="exist_task_signal",
                                           array=exist_task_signal_data,
										   dtype=np.int32,
                                           suffix=os.getpid(),
										   create=True)

        # exist_swapped_task_signal 用于engine感知worker中是否存在swapped task
        exist_swapped_task_signal_data = np.zeros([1], dtype=np.int32)
        self.exist_swapped_task_signal = IPCSignal(
            name="exist_swapped_task_signal",
			array=exist_swapped_task_signal_data,
			dtype=np.int32,
            suffix=os.getpid(),
			create=True)


        # worker_live_signal 用于engine感知各worker进程是否存活，记录每个step 时间
        worker_healthy_live_recorded_time_array = np.zeros(shape=[self.cfg.tensor_parallel_size], dtype=np.float32)
        self.worker_healthy_live_signal = IPCSignal(name="worker_healthy_live_signal",
                    array=worker_healthy_live_recorded_time_array,
					dtype=np.float32,
                    suffix=os.getpid(),
					create=True)

        if self.do_profile:
            get_profile_block_num = np.zeros([self.cfg.tensor_parallel_size], dtype=np.int32)
            self.get_profile_block_num_signal = IPCSignal(
                name="get_profile_block_num",
				array=get_profile_block_num,
				dtype=np.int32,
                suffix=os.getpid(),
				create=True)

    def _exit_sub_services(self):
        """
        exit sub services
        """
        self.worker_ready_signal.clear()
        self.exist_task_signal.clear()
        self.exist_swapped_task_signal.clear()
        self.worker_healthy_live_signal.clear()
        if hasattr(self, "worker_proc") and self.worker_proc is not None:
            os.killpg(self.worker_proc.pid, signal.SIGTERM)

    def _start_worker_service(self):
        """
        start gpu worker service

        """
        current_file_path = os.path.abspath(__file__)
        current_dir_path = os.path.split(current_file_path)[0]
        pd_cmd = f"{sys.executable} -m paddle.distributed.launch "
        py_script = os.path.join(current_dir_path, "../model_executor/worker.py")
        arguments = (f" --nnodes {str(self.cfg.nnode)}"
                    f" --devices {self.cfg.device_ids} {py_script}"
                    f" --max_num_seqs {self.cfg.max_num_seqs} --max_model_len {self.cfg.max_model_len}"
                    f" --gpu_memory_utilization {self.cfg.cache_config.gpu_memory_utilization}"
                    f" --model_name_or_path {str(self.cfg.model_name_or_path)}"
                    f" --device_ids {self.cfg.device_ids}"
                    f" --engine_worker_queue_port {str(self.cfg.engine_worker_queue_port)}"
                    f" --max_block_num {self.cfg.cache_config.total_block_num} "
                    f"--block_size {self.cfg.cache_config.block_size}"
                    f" --enc_dec_block_num {self.cfg.cache_config.enc_dec_block_num}"
                    f" --eos_tokens_lens {self.data_processor.eos_token_id_len}"
                    f" --pad_token_id {self.data_processor.pad_token_id}"
                    f" --engine_pid {os.getpid()}"
                    f" --do_profile {self.do_profile}"
                    f" --block_ratio {self.cfg.cache_config.block_ratio} --dtype {self.cfg.cache_config.cache_dtype}")
        if self.cfg.nnode > 1:
            pd_cmd = pd_cmd + f" --ips {self.cfg.ips}"
        log_dir = os.getenv("FD_LOG_DIR", default="log")
        pd_cmd = pd_cmd + arguments + f" >{log_dir}/launch_worker.log 2>&1"
        llm_logger.info("Launch worker service command: {}".format(pd_cmd))
        p = subprocess.Popen(
            pd_cmd,
            shell=True,
            preexec_fn=os.setsid,
        )
        return p


    def _format_and_add_data(self, prompts: dict):

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
            prompts["max_tokens"] = self.cfg.max_model_len

        self.add_requests(prompts)
        return request_id

    def generate(self, prompts, stream):
        """
        Generates a response based on the given prompt using the model.

        Args:
            prompts (dict): The prompt to use for generating the response.
            stream (bool): Whether to stream the output or wait until completion.

        Yields:
            dict: The generated response.
        """
        llm_logger.info(f"Starting generation for prompt: {prompts}")
        req_id = self._format_and_add_data(prompts)

        while True:
            result = self.get_result(req_id)
            if result is None:
                time.sleep(0.01)  # Avoid busy waiting
                continue

            is_end = result.finished
            if stream:
                processed = self.data_processor.process_response(result)
                if processed is None:
                    continue
                output = processed.todict()
                if not is_end:
                    yield output

            # Exit loop if termination condition is met
            if is_end:
                processed = self.data_processor.process_response(result)
                del self.req_output[req_id]
                output = processed.todict()
                llm_logger.debug(f"Generate result: {output}")
                if not stream:
                    yield output
                else:
                    output["outputs"]["text"] = ""
                    output["outputs"]["reasoning_content"] = ""
                    yield output
                break


    def _stop_profile(self):
        """
        Stop profiling of the model server and reset variables.
        """
        self.do_profile = 0
        num_gpu_blocks = -1
        for i in range(self.cfg.tensor_parallel_size):
            while self.get_profile_block_num_signal.value[i] == 0:
                time.sleep(1)
            if num_gpu_blocks < 0:
                num_gpu_blocks = self.get_profile_block_num_signal.value[i]
            else:
                num_gpu_blocks = min(num_gpu_blocks, self.get_profile_block_num_signal.value[i])

        self.get_profile_block_num_signal.clear()
        llm_logger.info(f"Stop profile, num_gpu_blocks:  {num_gpu_blocks}")
        self.cfg.cache_config.reset(num_gpu_blocks)
        self.resource_manager.reset_cache_config(self.cfg.cache_config)

    def check_health(self, time_interval_threashold=30):
        """
        Check the health of the model server by checking whether all workers are alive.

        """
        if self.worker_healthy_live_signal.value[0]:
            elapsed_time = time.time() - self.worker_healthy_live_signal.value[0]
            if elapsed_time > time_interval_threashold:
                return False, "Worker Service Not Healthy"

        return True, ""

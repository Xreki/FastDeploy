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


import sys

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
from fastdeployllm.input.preprocess import InputPreprocessor



from fastdeployllm.engine.args_utils import EngineArgs
from fastdeployllm.checker import add_default_params, check_basic_params
from fastdeployllm.engine.resource_manager import ResourceManager
from fastdeployllm.inter_communicator.engine_worker_queue import EngineWorkerQueue
from fastdeployllm.output.token_processor import TokenProcessor, WarmUpTokenProcessor
from fastdeployllm.utils import model_server_logger


class LLMEngine(object):
    """
    Engine Class
    """

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
    ):
        """Creates an LLM engine from the engine arguments."""

        # Create the engine configs.
        config = engine_args.create_engine_config()
        # Create the LLMEngine.
        return cls(cfg=config)

    def __init__(self, cfg):
        """
            Args:
            cfg (Config): Config object containing all the configuration parameters.

        Raises:
            None

        Returns:
            None
        """
        self.cfg = cfg

        self.cached_generated_tokens = queue.Queue()
        self.cached_task_deque = deque()
        self.req_output = dict()

        self.input_processor = InputPreprocessor(cfg.model_dir)
        self.resource_manager = ResourceManager(self.cfg)

        self.token_processor = TokenProcessor(cfg=self.cfg, cached_generated_tokens=self.cached_generated_tokens)
        self.token_processor.set_resource_manager(self.resource_manager)
        time.sleep(1) # TODO ????

        # TODO
        # 1. 增加engine hostname
        # 2. self.cfg.infer_port -> self.cfg.engine_worker_queue_port
        address = ('0.0.0.0', self.cfg.infer_port)
        self.engine_worker_queue = EngineWorkerQueue(address=address, is_server=True, num_client=self.cfg.mp_num)

        self.is_started = False

        self._init_engine_flags()
        self._finalizer = weakref.finalize(self, self._exit_sub_services)


    def start(self):
        """
        initialize engine and start sub services
        """
        assert not self.is_started, "The engine is already started.!"
        start_time = time.time()


        self.data_processor = self.input_processor.create_processor()

        self.infer_proc = self._start_infer_service()
        model_server_logger.info("Waitting infer processes ready...")
        while not self._infer_processes_ready():
            time.sleep(1)
        self.is_started = True

        # start warmup
        if self.cfg.use_warmup:
            model_server_logger.info("Start warmup")
            self._set_warmup_token_processor()
            self.warmup()
            self._del_warmup_token_processor()
            model_server_logger.info("Warmup finish")


        self.token_processor.tasks_queue = self.engine_worker_queue

        self.insert_task_to_engine_thread = threading.Thread(target=self._insert_task_push_mode, args=())
        self.insert_task_to_engine_thread.daemon = True
        self.insert_task_to_engine_thread.start()

        # start TokenProcessor thread
        self.token_processor.run()

        self.start_push_sender_thread()
        model_server_logger.info("Infer processes are launched with {} seconds.".format(time.time() - start_time))


    def start_push_sender_thread(self):
        """
            启动推送模式发送线程，该线程会不断地向服务器发送数据。
        当客户端处于推送模式时，需要定期将数据发送到服务器以保持连接的有效性。
        该函数会在客户端初始化后自动调用一次。

        Args:
            无参数。

        Returns:
            无返回值，通过修改类成员变量 push_mode_sender_thread 来实现。
        """
        self.push_mode_sender_thread = threading.Thread(target=self._push_mode_sender_thread, args=())
        self.push_mode_sender_thread.daemon = True
        self.push_mode_sender_thread.start()

    def _push_mode_sender_thread(self):
        """
        push mode sender thread
        """
        while True:
            try:
                batch_result = self.cached_generated_tokens.get()
                for result in batch_result:
                    if result["req_id"] not in self.req_output:
                        self.req_output[result["req_id"]] = deque()
                    self.req_output[result["req_id"]].appendleft(result)
            except Exception as e:
                model_server_logger.error("Unexcepted error happend: {}, {}".format(e, str(traceback.format_exc())))

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


    def _insert_task_push_mode(self):
        """
        Insert task to engine thread, monitor cached_task_deque.
        if the engine has resource, insert task to engine
        """
        try:
            while 1:
                if self.resource_manager.available_batch() == 0:
                    time.sleep(0.001)
                    continue
                if len(self.cached_task_deque) == 0:
                    time.sleep(0.001)
                    continue
                if self.engine_worker_queue.num_tasks() > 0:
                    time.sleep(0.001)
                    continue

                i_bs = 0
                for _ in range(self.cfg.max_prefill_batch):
                    if len(self.cached_task_deque) == 0:
                        break
                    if self.resource_manager.available_batch() == 0:
                        break

                    input_token_num = len(self.cached_task_deque[-1]["input_ids"])
                    if not self.resource_manager.is_resource_sufficient(input_token_num):
                        break
                    task = self.cached_task_deque.pop()
                    try:
                        self.insert_tasks([task])
                    except Exception as e:
                        err_msg = "Error happend while insert task to engine: {}, {}.".format(
                            e, str(traceback.format_exc())
                        )
                        model_server_logger.error(err_msg)
            model_server_logger.info("finish insert_task_push_mode thread")
        except Exception as e:
            model_server_logger.error(
                "insert_task_push_mode thread exit " f"unexpectedly, {e}. {str(traceback.format_exc())}"
            )

    def unfinished_requests_num(self):
        """
            返回还没有完成的请求数量，即最大批处理大小减去当前可用批处理大小。

        Returns:
            int -- 还没有完成的请求数量，即最大批处理大小减去当前可用批处理大小。
        """
        return self.cfg.max_batch_size - self.resource_manager.available_batch()



    def add_requests(self, task):
        """
            将请求添加到队列中，并进行相应的处理。如果启用了文本截断，则对输入文本进行截断；否则，使用默认参数进行处理。
        如果任务需要的资源超过限制，则不会缓存该任务。

        Args:
            task (dict): 包含请求信息的字典，其中必须包含以下键值对：
                - "input_text" (str): 输入文本。
                - "model_name" (str, optional): 模型名称，默认为None。
                - "enable_text_truncate" (int, optional): 是否启用文本截断，默认为1。
                - "max_dec_len" (int, optional): 最大解码长度，默认为800。
                - "min_dec_len" (int, optional): 最小解码长度，默认为20。
                - "req_id" (str, optional): 请求ID，默认为None。

        Returns:
            None, 如果任务需要的资源超过限制，则不会缓存该任务。

        Raises:
            None
        """

        task = add_default_params(task)

        if int(task.get("enable_text_truncate", 1)):
            real_seq_len = self.cfg.max_seq_len - task.get("max_dec_len", 800)
            task = self.data_processor.process_request(task, max_seq_len=real_seq_len)
        else:
            task = self.data_processor.process_request(task, self.cfg.max_seq_len)

        input_ids_len = len(task["input_ids"])
        if "max_dec_len" not in task:
            task["max_dec_len"] = self.cfg.max_seq_len - input_ids_len
        min_dec_len = task["min_dec_len"]
        if input_ids_len + min_dec_len >= self.cfg.max_seq_len:
            error_msg = (
                f"Input text is too long, input_ids_len ({input_ids_len}) "
                f"+ min_dec_len ({min_dec_len}) >= max_seq_len "
            )
            model_server_logger.error(error_msg)
            return

        if input_ids_len > self.cfg.max_seq_len:
            error_msg = (
                f"Length of input token({input_ids_len}) exceeds the limit MAX_SEQ_LEN({self.cfg.max_seq_len})."
            )
            model_server_logger.error(error_msg)
            return


        required_block_num = self.resource_manager.get_required_block_number(input_ids_len)
        if required_block_num > self.resource_manager.total_block_number():
            error_msg = f"The input task required resources is exceed the limit, task={task}."
            model_server_logger.error(error_msg)
            return

        task["preprocess_end_time"] = datetime.now()
        self.cached_task_deque.appendleft(task)
        model_server_logger.info(
            f"cache task with req_id ({task.get('req_id')}), "
            f"cached_task_num: {len(self.cached_task_deque)}."
        )
        model_server_logger.debug(f"cache task: {task}")

    def warmup(self):
        """
        construct test tasks and avoid out of memory problem in the infer process
        """
        # get eos_token_id
        pass
    def insert_tasks(self, tasks):
        """
        insert tasks to the engine

        Args:
            tasks: list of tasks

        Returns:
            return: True if success, False otherwise
        """
        if not isinstance(tasks, list):
            tasks = [tasks]

        for item in tasks:
            item["schedule_start_time"] = datetime.now()

        available_batch = np.sum(self.resource_manager.stop_flags)
        if len(tasks) > available_batch:
            model_server_logger.error("Inserting batch:{} exceeds the available batch:{}.".format(
                len(tasks), available_batch))
            model_server_logger.error("The exceeded part will be ignored!")
            tasks = tasks[:available_batch]

        for i in range(len(tasks)):
            req_id = tasks[i]["req_id"]
            input_token_num = len(tasks[i]["input_ids"])
            if input_token_num >= self.cfg.max_seq_len - 1:
                model_server_logger.warning(f"{req_id}: Input length:{input_token_num}, exceed the limits.")
                tasks[i]["input_ids"] = tasks[i]["input_ids"][:self.cfg.max_seq_len - 1]
            if "seq_len" in tasks[i] and "max_dec_len" not in tasks[i]:
                tasks[i]["max_dec_len"] = tasks[i]["seq_len"]
            if "max_dec_len" not in tasks[i]:
                tasks[i]["max_dec_len"] = self.cfg.max_seq_len

            # max_dec_len + input_token_num > MAX_SEQ_LEN
            if input_token_num + tasks[i]["max_dec_len"] > self.cfg.max_seq_len:
                tasks[i]["max_dec_len"] = self.cfg.max_seq_len - input_token_num
                model_server_logger.warning("Force max_dec_len to be {} for req_id={}.".format(
                    tasks[i]["max_dec_len"], tasks[i]["req_id"]))

            # min_dec_len + input_token_num > MAX_SEQ_LEN
            if input_token_num + tasks[i]["min_dec_len"] > self.cfg.max_seq_len:
                tasks[i]["min_dec_len"] = self.cfg.max_seq_len - input_token_num
                model_server_logger.warning("Force min_dec_len to be {} for req_id={}.".format(
                    tasks[i]["min_dec_len"], tasks[i]["req_id"]))

        tasks = self.resource_manager.allocate_resources_for_new_tasks(tasks)
        if not tasks:
            return False

        self.token_processor.number_of_tasks += len(tasks)
        for i in range(len(tasks)):
            self.token_processor.number_of_input_tokens += len(tasks[i]["input_ids"])

        req_ids = [t["req_id"] for t in tasks]
        model_server_logger.info(f"Tasks are sent to engine, req_ids={req_ids}")
        self.engine_worker_queue.put_tasks((tasks, self.resource_manager.real_bsz))
        return True

    def task_is_finished(self, index):
        """
        judge if the task is finished

        Args:
            index: task index

        Returns:
            return: True if finished, False otherwise
        """
        assert index < len(self.resource_manager.stop_flags)
        return self.resource_manager.stop_flags[index]


    def all_tasks_finished(self):
        """
        judge if all tasks are finished

        Returns:
            return: True if all finished, False otherwise
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

    def _infer_processes_ready(self):
        """
        judge if all infer processes are ready

        Returns:
            return: True if all ready, False otherwise
        """
        if np.sum(self.flag_ready_array) == self.cfg.mp_num_per_node:
            return True
        return False

    def _clear_engine_flags(self):
        """
        clear engine flags
        """
        try:
            self.shm_flag_ready.close()
            self.shm_flag_ready.unlink()
            self.shm_flag_has_block_step.close()
            self.shm_flag_has_block_step.unlink()
        except:
            pass

    def _init_engine_flags(self):
        """
        Initialize shared memory to indicate engine status
        """


        flag_array = np.zeros([self.cfg.mp_num], dtype=np.int32)
        try:
            tmp = shared_memory.SharedMemory(
                create=False, size=flag_array.nbytes, name="shm_flag_infer_ready"
            )
            tmp.close()
            tmp.unlink()
        except:
            pass
        self.shm_flag_ready = shared_memory.SharedMemory(
            create=True, size=flag_array.nbytes, name="shm_flag_infer_ready"
        )
        self.flag_ready_array = np.ndarray(
            flag_array.shape, dtype=flag_array.dtype, buffer=self.shm_flag_ready.buf
        )
        self.flag_ready_array[:] = 0

        # broadcast flag for engine
        broadcast_flag_array = np.zeros([1], dtype=np.int32)
        try:
            tmp = shared_memory.SharedMemory(
                create=False,
                size=broadcast_flag_array.nbytes,
                name="shm_pd_infer_flag_broadcast",
            )
            tmp.close()
            tmp.unlink()
        except:
            pass
        self.shm_flag_broadcast = shared_memory.SharedMemory(
            create=True, size=broadcast_flag_array.nbytes, name="shm_pd_infer_flag_broadcast"
        )
        self.flag_broadcast_array = np.ndarray(
            broadcast_flag_array.shape,
            dtype=broadcast_flag_array.dtype,
            buffer=self.shm_flag_broadcast.buf,
        )
        self.flag_broadcast_array[0] = 0

        has_block_step_flag_array = np.zeros([1], dtype=np.int32)
        try:
            tmp = shared_memory.SharedMemory(
                create=False,
                size=has_block_step_flag_array.nbytes,
                name="shm_flag_has_block_step")
            tmp.close()
            tmp.unlink()
        except:
            pass
        self.shm_flag_has_block_step = shared_memory.SharedMemory(
            create=True,
            size=has_block_step_flag_array.nbytes,
            name="shm_flag_has_block_step")
        self.flag_has_block_step_array = np.ndarray(
            has_block_step_flag_array.shape,
            dtype=has_block_step_flag_array.dtype,
            buffer=self.shm_flag_has_block_step.buf)
        self.flag_has_block_step_array[:] = 0

    def _exit_sub_services(self):
        """
        exit sub services
        """
        if hasattr(self, "queue_service") and self.queue_service is not None:
            self.queue_service.terminate()
            self.queue_service.join()
        if hasattr(self, "infer_proc") and self.infer_proc is not None:
            os.killpg(self.infer_proc.pid, signal.SIGTERM)

    def _start_gpu_infer_service(self):
        """
        start gpu infer service

        Returns:
            p: process handle
        """
        current_file_path = os.path.abspath(__file__)
        current_dir_path = os.path.split(current_file_path)[0]
        pd_cmd = f"{sys.executable} -m paddle.distributed.launch "
        py_script = os.path.join(current_dir_path, "../model_executor/worker.py")
        arguments = (f" --nnodes {str(self.cfg.nnode)}"
                    f" --devices {self.cfg.device_ids} {py_script}"
                    f" --max_batch_size {self.cfg.max_batch_size} --max_seq_len {self.cfg.max_seq_len}"
                    f" --max_dec_len {self.cfg.max_seq_len}"
                    f" --model_name_or_path {str(self.cfg.model_dir)}"
                    f" --infer_port {str(self.cfg.infer_port)}"
                    f" --max_block_num {self.cfg.total_block_num} --block_size {self.cfg.block_size}"
                    f" --enc_dec_block_num {self.cfg.enc_dec_block_num}"
                    f" --eos_tokens_lens {self.data_processor.eos_token_id_len}"
                    f" --pad_token_id {self.data_processor.pad_token_id}"
                    f" --block_ratio {self.cfg.block_ratio} --dtype {self.cfg.dtype}")
        if self.cfg.nnode > 1:
            pd_cmd = pd_cmd + f" --ips {self.cfg.ips}"
        log_dir = os.getenv("FD_LOG_DIR", default="log")
        pd_cmd = pd_cmd + arguments + f" >{log_dir}/launch_infer.log 2>&1"
        model_server_logger.info("Launch infer service command: {}".format(pd_cmd))
        p = subprocess.Popen(
            pd_cmd,
            shell=True,
            preexec_fn=os.setsid,
        )
        return p

    def _start_infer_service(self):
        """
        start infer service
        """
        return self._start_gpu_infer_service()


    def _format_and_add_data(self, prompts: dict):
        request_id = str(uuid.uuid4())

        prompts["req_id"] = request_id
        if "top_p" in prompts:
            prompts["topp"] = prompts["top_p"]
        query_list = []
        if "context" in prompts:
            for item in prompts["context"]:
                if item["role"] == "system":
                    prompts["system"] = item["utterance"]
                elif item["role"] in ["user", "assistant"]:
                    query_list.append(item["utterance"])
                    prompts["text"] = query_list
        if "prompt" in prompts:
            prompts["text"] = [prompts["prompt"]]
        tasks = prompts

        self.add_requests(tasks)
        return request_id

    def generate(self, prompts, stream):
        """
        Generate a response based on the given prompt using the model.
        
        Args:
            prompts (dict): The prompt to use for generating the response.
            stream (bool): Whether to stream the output or wait until completion.
        
        Yields:
            str: The generated response.
        """
        model_server_logger.info(f"Start generate prompt: {prompts}")
        req_id = self._format_and_add_data(prompts)
        
        while True:
            # 获取当前请求的结果
            result = self.get_result(req_id)
            if result is None:
                time.sleep(0.01)  # 避免忙等待
                continue
            
            is_end = result.get('is_end', 1)
        
            if stream:
                processed = self.data_processor.process_response(result)
                model_server_logger.info(f"Output: {processed}")
                yield processed
            
            # 遇到终止条件时退出循环
            if is_end:
                processed = self.data_processor.process_response(result)
                model_server_logger.info(f"Output: {processed}")
                yield processed  
                break

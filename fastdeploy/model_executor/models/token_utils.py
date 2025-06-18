# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
Script for token counting.
"""
import json
import logging
import os
import random
import sys
import time

import paddleformers
import psutil
from paddleformers.trainer.integrations import TrainerCallback
from paddleformers.utils.log import logger as nlp_logger

import paddle

token_filename = ".barsfs_implicit_metrics"
token_time = 120
local_rank = int(os.getenv("PADDLE_RANK_IN_NODE", 0))


def find_version_from_file(file_path):
    """Find the file version."""
    tool_version = "NULL"
    model_version = "NULL"
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            lines = file.readlines()
            if len(lines) >= 1:
                split_result = lines[0].strip().split(":")
                if len(split_result) >= 2:
                    tool_version = split_result[1].strip()

            if len(lines) >= 2:
                split_result = lines[1].strip().split(":")
                if len(split_result) >= 2:
                    model_version = split_result[1].strip()

    return tool_version, model_version


def generate_unique_log_id():
    """Generate a unique log ID."""
    timestamp = int(time.time() * 1000)
    random_num = random.randint(0, 999)
    log_id = f"{timestamp}-{random_num}"
    return log_id


def process_index():
    """The index of the current process used."""
    if local_rank != -1:
        return paddle.distributed.get_rank()
    return 0


class TokenTimer:
    """Timer for token counting."""

    def __init__(self, task, safe_dir, max_interval_time):
        self._started = False
        self._start_time = 0  # 开始时间
        self._max_interval_time = max_interval_time  # 最大时间
        self._token_nums = 0  # 训练token数量
        self._context_tokens = 0  # predict 上下文token数量
        self._generation_tokens = 0  # predict 生成token数量
        self._log_id = os.getenv("SYS_JOB_ID", generate_unique_log_id())
        if len(self._log_id) == 0:
            self._log_id = generate_unique_log_id()
        self._globalstep_last = 0
        self._task = task
        self._path = os.path.join(safe_dir, token_filename)
        self._binary_version, self._model_version = find_version_from_file(
            "./tool_version.txt"
        )
        self._paddle_version = paddle.version.commit
        self._paddleformers_version = paddleformers.version.commit
        nlp_logger.info(f"training mode: {self._task}")

    def start(self):
        """Start TokenTimer."""
        assert not self._started, "timer has already been started"
        self._start_time = time.time()
        self._cur_time = self._start_time
        self._started = True

    def record(self, cnt=1):
        """Record the number of tokens."""
        self._token_nums += cnt

    def set_task(self, task):
        """Update the task."""
        self._task = task

    def set_token_nums(self, token_nums):
        """Update the number of context_tokens."""
        self._token_nums = token_nums

    def set_context_tokens(self, context_tokens):
        """Update the number of context_tokens."""
        self._context_tokens = context_tokens

    def set_generation_tokens(self, generation_tokens):
        """Update the number of generation_tokens."""
        self._generation_tokens = generation_tokens

    def write_data(self, data, is_main_process):
        """Writing file."""
        if not is_main_process:
            return
        with open(self._path, "w") as json_file:
            json.dump(data, json_file)

    def check_and_write(self):
        """Check the interval_time and write data to json file."""
        curtime = time.time()
        interval_time = curtime - self._cur_time
        if interval_time >= self._max_interval_time:
            self._cur_time = curtime
            data = {
                "logId": self._log_id,
                "startTimestamp": int(self._start_time * 1000),
                "endTimestamp": int(self._cur_time * 1000),
                "taskType": self._task,
                "tokensNum": self._token_nums,
                "contextTokenNum": self._context_tokens,
                "generationTokenNum": self._generation_tokens,
                "properties": {
                    "binaryVersion": self._binary_version,
                    "modelVersion": self._model_version,
                    "paddleVersion": self._paddle_version,
                    "paddleformersVersion": self._paddleformers_version,
                },
            }
            self.write_data(data, process_index() == 0)

            del data

    def stop(self):
        """Stop TokenTimer."""
        assert self._started, "timer is not started"
        self._started = False

    def get_globalstep_last(self):
        """Return the _globalstep_last."""
        return self._globalstep_last

    def set_globalstep_last(self, globalstep_last):
        """Set the _globalstep_last."""
        self._globalstep_last = globalstep_last


class TokenCallback(TrainerCallback):
    """Callback for token counting."""

    def __init__(self, model_args, data_args, training_args, flag="train"):
        """
        初始化函数，用于初始化模型、数据和训练参数。
            如果flag为"train"，则表示进行训练；否则，表示进行其他任务（例如预测）。

            Args:
                model_args (obj): 模型相关参数的对象，包括模型名称、阶段等信息。
                data_args (obj): 数据相关参数的对象，包括最大序列长度等信息。
                training_args (obj): 训练相关参数的对象，包括输出目录等信息。
                flag (str, optional): 默认为"train"，表示进行训练；可以设置为其他字符串，表示进行其他任务。 Default: "train".

            Returns:
                None.
        """
        if "train" == flag:
            if model_args.stage == "PPT":
                task = "train_post_pretrain"
                self.max_len = data_args.max_seq_len
            elif model_args.stage == "RM":
                task = "train_rm"
                self.max_len = data_args.max_seq_len * data_args.num_comparisons
            elif model_args.stage == "DPO":
                task = "train_dpo"
                self.max_len = data_args.max_seq_len
            elif model_args.stage == "DPO_LoRA":
                task = "train_dpo_lora"
                self.max_len = data_args.max_seq_len
            elif model_args.stage == "KTO":
                task = "train_kto"
                self.max_len = data_args.max_seq_len
            elif model_args.stage == "KTO_LoRA":
                task = "train_kto_lora"
                self.max_len = data_args.max_seq_len
            elif model_args.stage == "PPO":
                task = "train_ppo"
                self.max_len = data_args.max_seq_len
            elif model_args.stage == "Embedding":
                task = "train_embedding"
                self.max_len = data_args.max_query_len + data_args.max_passage_len
            else:
                if model_args.lora and not model_args.loraga:
                    task = "train_lora"
                elif model_args.lora and model_args.loraga:
                    task = "train_loraga"
                elif model_args.prefix_tuning:
                    task = "train_ptuning"
                else:
                    task = "train_sft"
                self.max_len = data_args.max_seq_len
        else:
            task = "train_post_pretrain"
            self.max_len = data_args.max_seq_len
        nlp_logger.info(f"################ task:{task} ################")
        self.timer = TokenTimer(task, training_args.output_dir, token_time)
        self.timer.start()

    def on_train_begin(self, args, state, control, **kwargs):
        """Set global_step."""
        self.timer.set_globalstep_last(state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        """Record the status at the end of each training step."""
        total_train_batch_size = (
            args.train_batch_size
            * args.gradient_accumulation_steps
            * args.dataset_world_size
        )
        num_steps = state.global_step - self.timer.get_globalstep_last()
        num_samples = total_train_batch_size * num_steps
        self.timer.record(num_samples * self.max_len)
        self.timer.set_globalstep_last(state.global_step)
        self.timer.check_and_write()

    def on_train_end(self, args, state, control, **kwargs):
        """Training is over."""
        self.timer._max_interval_time = 0
        self.timer.check_and_write()
        self.timer.stop()


def filter_zombie_processes(processes):
    """Filter zombie processes"""
    filtered_processes = []

    for process in processes:
        try:
            if process.status() != psutil.STATUS_ZOMBIE:
                filtered_processes.append(process)
        except Exception:
            continue

    return filtered_processes


def sep_path(path):
    """Guaranteed to end with delimiter on all os"""
    return path.rstrip(os.sep) + os.sep


def get_barsfs_root():
    """Get the root directory of the barsfs."""
    # 获取 barsfs 进程
    barsfs_processes = [
        p
        for p in psutil.process_iter()
        if p.name() == "barsfs" or p.name() == "barsfs.ezpack"
    ]  # name前缀
    # 过滤掉僵尸进程
    processes = filter_zombie_processes(barsfs_processes)

    output_paths = []
    if len(processes) == 0:
        nlp_logger.error("Security directory startup failed")
        sys.exit(1)
    # check barsfs进程带的参数
    for pro in processes:
        command = pro.cmdline()
        # 非法参数
        if len(command) < 2:
            continue
        # '/fuse/aaa/../ -> '/fuse'
        command[1] = os.path.normpath(command[1])
        # fuse启动于: '/root/a/b', command[1]='..' - > '/root/a'
        if not os.path.isabs(command[1]):
            fuse_absolute_path = os.path.abspath(os.path.join(pro.cwd(), command[1]))
        else:
            fuse_absolute_path = command[1]
        # /root/a -> /root/a/ 避免后续认定/root/model 为 /root/m 的子目录
        output_paths.append(sep_path(fuse_absolute_path))

    if len(output_paths) == 0:
        nlp_logger.error("Secure directory startup command error.")
        sys.exit(1)

    return output_paths


def get_filtered_mount_points():
    """Get mount path"""
    # 获取所有分区
    partitions = psutil.disk_partitions(all=True)
    mount_points = []
    for partition in partitions:
        fstype = partition.fstype
        mounted_on = partition.mountpoint
        device = partition.device
        # 过滤掉非 barsfs 分区
        if (device == "barsfs" or device == "barsfs.ezpack") and (
            fstype == "fuse.barsfs.ezpack" or fstype == "fuse.barsfs"
        ):
            # /root/a -> /root/a/ 避免后续认定/root/model 为 /root/m 的子目录
            mount_points.append(sep_path(mounted_on))

    if len(mount_points) < 1:
        nlp_logger.error("Security directory mount not detected")
        sys.exit(1)

    return mount_points


def has_link_between_paths(safe_dir, output_path):
    """Check symbolic links."""
    current_dir = output_path

    while sep_path(current_dir) != sep_path(safe_dir):
        if os.path.islink(current_dir):
            return True

        current_dir = os.path.dirname(current_dir)

        if current_dir == os.path.dirname(current_dir):
            # 达到根目录，但仍未匹配到 safe_dir
            nlp_logger.error("The output directory cannot match the security directory")
            sys.exit(1)

    if os.path.islink(safe_dir):
        return True

    return False


def find_python_files(directory):
    """Find python files."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def check_barsfs(output_path):
    """Check whether the output path is in the barsfs secure directory."""
    # double check
    mount_points = get_filtered_mount_points()
    barsfs_roots = get_barsfs_root()
    safe_dirs = []
    safe_dirs.extend(set(mount_points) & set(barsfs_roots))
    # '/output/aaa/../ -> '/output'
    output_path = os.path.normpath(output_path)
    # 程序启动于: '/root/a/b', output_path='..' - > '/root/a'
    if not os.path.isabs(output_path):
        output_path = os.path.abspath(output_path)
    # /root/a -> /root/a/ 避免后续认定/root/model 为 /root/m 的子目录
    output_path = sep_path(output_path)

    for safe_dir in safe_dirs:
        if safe_dir in output_path:
            if has_link_between_paths(safe_dir, output_path):
                nlp_logger.error("There are symbolic links under the secure directory.")
                sys.exit(1)
            nlp_logger.info(f"The output directory is under:{safe_dir}")
            nlp_logger.logger.logLevel = "info"
            nlp_logger.logger.setLevel(logging.INFO)
            try:
                from paddleslim.utils.log import logger as slim_logger

                slim_logger.logger.logLevel = "info"
                slim_logger.logger.setLevel(logging.INFO)
            except Exception as e:
                nlp_logger.info(f"The logger of PaddleSlim is not available: {e!s}")
            return True
    return False


def check_python_safe():
    """Check .py"""
    py_list = find_python_files("./")
    if len(py_list) > 0:
        nlp_logger.error(f"invalid python files {py_list}, please remove them.")
        sys.exit(1)


def check_output(output_path):
    """Check whether the output path is in the secure directory."""
    return check_barsfs(output_path)


def find_python_files_without_cipher(directory):
    """Find python files without cipher."""
    export_python_files = []
    for root, dirs, files in os.walk(directory):
        if "ernie_bot_cipher" in dirs:
            dirs.remove("ernie_bot_cipher")

        for file in files:
            # file_path = os.path.join(root, file)
            if file.endswith(".py"):
                export_python_files.append(os.path.join(root, file))
    return export_python_files


def check_special_field(python_files):
    """Check whether the special field is in the python files."""
    special_field = "WjI1fQOvhN"
    for file in python_files:
        with open(file, "r") as f:
            for line_number, line in enumerate(f, start=1):
                if special_field in line:
                    break
            else:
                return False
    return True


def get_safe_python(path):
    """
    Get all python files in a directory.
    """
    python_filenames = []
    for filename in os.listdir(path):
        if filename.endswith(".py"):
            python_filenames.append(os.path.join(path, filename))
    return python_filenames


def check_python_safe_export():
    """Check .py for export"""
    required_filenames = []
    required_filenames.extend(get_safe_python("./ernie_bot/"))
    required_filenames.extend(get_safe_python("./ernie_bot/moe/"))
    required_filenames.extend(get_safe_python("./ernie_bot/moe/distributed/"))
    required_filenames.extend(get_safe_python("./ernie_bot/fastdeploy/"))
    required_filenames.extend(get_safe_python("./ernie_bot/fastdeploy/layers/"))

    python_files = find_python_files_without_cipher("./")
    assert set(python_files) == set(
        required_filenames
    ), f"invaild python files: {python_files}, please remove them."

    if not check_special_field(python_files):
        nlp_logger.error("Some ernie_bot files have been modified, check failed.")
        sys.exit(1)


def check_output_export(output_path):
    """Check .py & whether the output path is in the secure directory."""
    check_python_safe_export()
    return check_barsfs(output_path)


def save_stop_info(args, stop_step, outside_eval, outside_predict):
    """Save stop info into json file."""
    if process_index() != 0:
        return
    output_path = args.logging_dir
    eval_turns = 0 + outside_eval
    predict_turns = 0 + outside_predict
    if args.do_eval:
        eval_turns += stop_step // args.eval_steps

    data = {
        "stop_step": stop_step,
        "eval_turns": eval_turns,
        "predict_turns": predict_turns,
    }
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, "stop_step.json")
    with open(file_path, "w") as json_file:
        json.dump(data, json_file)
    nlp_logger.info(f"Saving stop info into {file_path}")
    return

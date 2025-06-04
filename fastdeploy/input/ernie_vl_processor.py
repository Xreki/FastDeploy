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

import numpy as np

from fastdeploy.input.mm_processor import DataProcessor, IDS_TYPE_FLAG
from fastdeploy.input.ernie_processor import ErnieProcessor
from fastdeploy.engine.request import Request
from fastdeploy.entrypoints.chat_utils import parse_chat_messages
from fastdeploy.input.ernie_processor import ErnieProcessor
from fastdeploy.utils import api_server_logger


class ErnieMoEVLProcessor(ErnieProcessor):
    """The processor class for ERNIE MoE VL models."""
    def __init__(self, model_name_or_path, limit_mm_per_prompt=None, mm_processor_kwargs=None):
        self.use_hf_tokenizer = False
        self.is_thinking = False

        if "merge_llm_model" in model_name_or_path:
            model_name_or_path = os.path.dirname(model_name_or_path)
        tokenizer_path = model_name_or_path
        preprocessor_path = model_name_or_path
        processor_kwargs = self._parse_processor_kwargs(mm_processor_kwargs)
        
        self.ernie_processor = DataProcessor(
            tokenizer_name=tokenizer_path,
            image_preprocessor_name=preprocessor_path,
            **processor_kwargs
        )

        self.decode_status = dict()
        self._load_tokenizer()
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        self.eos_token_id_len = len(self.eos_token_ids)
        self.pad_token_id = self.get_pad_id()
        self.limit_mm_per_prompt = self._parse_limits(limit_mm_per_prompt)


    def _parse_processor_kwargs(self, kwargs):
        """解析多模态处理器参数配置"""
        if not kwargs:
            return {}

        try:
            if not isinstance(kwargs, dict):
                raise ValueError("mm-processor-kwargs must be a dictionary")

            # 验证参数类型
            api_server_logger.info(f"kwargs:{kwargs}")
            expected_types = {
                "spatial_conv_size": int,
                "temporal_conv_size": int,
                "image_min_pixels": int,
                "image_max_pixels": int,
                "video_min_pixels": int,
                "video_max_pixels": int,
                "video_target_frames": int,
                "video_frames_sample": str,
                "video_max_frames": int,
                "video_min_frames": int,
                "video_fps": int
            }

            for key, value in kwargs.items():
                if key in expected_types and not isinstance(value, expected_types[key]):
                    raise ValueError(
                        f"Invalid type for {key}: expected {expected_types[key].__name__}, got {type(value).__name__}")

            return kwargs

        except Exception as e:
            api_server_logger.warning(f"Invalid mm-processor-kwargs format: {e}")
            return {}

    def get_pad_id(self):
        """get pad id"""
        return self.tokenizer.pad_token_id

    def _load_tokenizer(self):
        """
        load tokenizer

        Returns:
            tokenizer (AutoTokenizer)
        """
        self.tokenizer = self.ernie_processor.tokenizer

    def process_request(self, request, max_model_len=None):
        """process the input data"""
        task = request.to_dict()
        self.process_request_dict(task, max_model_len)
        request = Request.from_dict(task)

        return request

    def _parse_limits(self, limits):
        """解析多模态限制配置"""
        DEFAULT_LIMITS = {
            "image": 1,
            "video": 1,
            "audio": 1
        }

        if not limits:
            return DEFAULT_LIMITS

        try:
            if not isinstance(limits, dict):
                raise ValueError("limit-mm-per-prompt must be a dictionary")
            api_server_logger.info(f"_parse_limits:{limits}")
            return {**DEFAULT_LIMITS, **limits}
        except Exception as e:
            api_server_logger.warning(f"Invalid limit-mm-per-prompt format: {e}, using default limits")
            return DEFAULT_LIMITS

    def _check_mm_limits(self, messages):
        """检查多模态数据是否超过限制"""
        mm_data = {
            "image": [],
            "video": [],
            "audio": []
        }

        # 提取多模态数据
        for message in messages:
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item.get("type") == "image_url":
                        mm_data["image"].append(item)
                    elif item.get("type") == "video_url":
                        mm_data["video"].append(item)
                    elif item.get("type") in ["audio_url", "input_audio"]:
                        mm_data["audio"].append(item)

        # 检查限制
        for modality, items in mm_data.items():
            if modality in self.limit_mm_per_prompt:
                limit = self.limit_mm_per_prompt[modality]
                if len(items) > limit:
                    raise ValueError(
                        f"Too many {modality} items in prompt. "
                        f"Got {len(items)}, but limit is {limit}."
                    )

    def process_request_dict(self, request, max_model_len=None):
        """process the input data"""

        if request.get("eos_token_ids") is None or len(
                request.get("eos_token_ids")) == 0:
            request["eos_token_ids"] = self.eos_token_ids

        stop_sequences = request.get("stop", [])
        if stop_sequences is not None and len(stop_sequences) != 0:
            stop_seqs, stop_seqs_len = self.update_stop_seq(stop_sequences)
            request.set("stop_token_ids", stop_seqs)
            request.set("stop_seqs_len", stop_seqs_len)

        messages = request.get("messages")
        try:
            self._check_mm_limits(messages)
        except ValueError as e:
            api_server_logger.error(f"Multi-modal limit exceeded: {e}")
            raise
        messages = parse_chat_messages(messages)
        output = self.ernie_processor.process(messages)
        metadata = request.get("metadata")
        # 如果metadata包含之前输出的token，将这些token添加到input_ids末尾
        if metadata and metadata.get("generated_token_ids"):
            self.append_generated_tokens(output, metadata["generated_token_ids"])
        output = self.pack_outputs(output)
        request["prompt_token_ids"] = output["input_ids"]
        request["prompt_token_ids_len"] = len(request["prompt_token_ids"])
        request["multimodal_inputs"] = output

        # 截断超过长度限制的prompt
        if max_model_len is not None and len(
                request['prompt_token_ids']) > max_model_len:
            request['prompt_token_ids'] = request[
                'prompt_token_ids'][:max_model_len - 1]

        return request

    def append_generated_tokens(self, multimodal_inputs, generated_token_ids):
        "append already generated tokens"
        
        num_tokens = len(generated_token_ids)
        multimodal_inputs["input_ids"].extend(generated_token_ids)
        multimodal_inputs["token_type_ids"].extend([IDS_TYPE_FLAG["text"]] * num_tokens)

        start = multimodal_inputs["cur_position"]
        for i in range(num_tokens):
            multimodal_inputs["position_ids"].append([start + i] * 3)
        multimodal_inputs["cur_position"] += num_tokens

    def pack_outputs(self, outs):
        # Stack or nullify image-related fields
        if not outs["images"]:
            outs["images"] = None
            outs["grid_thw"] = None
            outs["image_type_ids"] = None
        else:
            outs["images"] = np.vstack(outs["images"])
            outs["grid_thw"] = np.vstack(outs["grid_thw"])
            outs["image_type_ids"] = np.array(outs["image_type_ids"])

        # Convert lists to arrays
        outs["input_ids"] = np.array(outs["input_ids"], dtype=np.int64)
        outs["token_type_ids"] = np.array(outs["token_type_ids"], dtype=np.int64)
        outs["position_ids"] = np.array(outs["position_ids"], dtype=np.int64)

        return outs

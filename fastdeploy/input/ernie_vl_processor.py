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
from fastdeploy.input.mm_processor import DataProcessor


class ErnieMoEVLProcessor(ErnieProcessor):
    """The processor class for ERNIE MoE VL models."""

    def __init__(self, model_name_or_path):
        self.use_hf_tokenizer = False
        self.is_thinking = False

        if "merge_llm_model" in model_name_or_path:
            model_name_or_path = os.path.dirname(model_name_or_path)
        tokenizer_path = model_name_or_path
        preprocessor_path = model_name_or_path

        self.ernie_processor = DataProcessor(
            tokenizer_name=tokenizer_path,
            image_preprocessor_name=preprocessor_path)
        self.decode_status = dict()
        self._load_tokenizer()
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        self.eos_token_id_len = len(self.eos_token_ids)
        self.pad_token_id = self.get_pad_id()

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

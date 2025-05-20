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
from fastdeploy.input.mm_processor import DataProcessor
from fastdeploy.input.ernie_processor import ErnieProcessor
from fastdeploy.entrypoints.chat_utils import parse_chat_messages

class ErnieMoEVLProcessor(ErnieProcessor):
    """The processor class for ERNIE MoE VL models."""
    def __init__(self, model_name_or_path):
        self.use_hf_tokenizer = False
        self.is_thinking = False

        model_name_or_path = os.path.dirname(model_name_or_path)
        tokenizer_path = model_name_or_path
        preprocessor_path = model_name_or_path
        
        self.ernie_processor = DataProcessor(
            tokenizer_name=tokenizer_path, 
            image_preprocessor_name=preprocessor_path
        )
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

    def process_request_dict(self, request, max_model_len=None):
        """process the input data"""

        if request.get("eos_token_ids") is None or len(request.get("eos_token_ids")) == 0:
            request["eos_token_ids"] = self.eos_token_ids

        stop_sequences = request.get("stop", [])
        if stop_sequences is not None and len(stop_sequences) != 0:
            stop_seqs, stop_seqs_len = self.update_stop_seq(stop_sequences)
            request.set("stop_token_ids", stop_seqs)
            request.set("stop_seqs_len", stop_seqs_len)

        messages = request.get("messages")
        messages = parse_chat_messages(messages)
        output = self.ernie_processor.process(messages)
        request["prompt_token_ids"] = output["input_ids"]
        request["prompt_token_ids_len"] = len(request["prompt_token_ids"])
        request["multimodal_inputs"] = output

        return request
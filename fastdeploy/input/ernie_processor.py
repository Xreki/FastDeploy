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
from string import Template
import re
from typing import Dict, List, Union, Optional, Tuple
import json
import uuid

from fastdeploy.utils import data_processor_logger
from paddleformers.generation import GenerationConfig

if os.getenv("FLAG_TOKENIZER_V1"):
    data_processor_logger.info(f"use TOKENIZER version V1")
    from fastdeploy.input.ernie_tokenizer_v1 import ErnieBotTokenizer
else:
    data_processor_logger.info(f"use TOKENIZER verison V2")
    from fastdeploy.input.ernie_tokenizer_v2 import ErnieBotTokenizer

from fastdeploy.input.text_processor import BaseDataProcessor
from fastdeploy.engine.config import ModelConfig
from fastdeploy.entrypoints.openai.protocol import (
    DeltaToolCall,
    DeltaFunctionCall,
    ToolCall,
    FunctionCall
)


class ErnieProcessor(BaseDataProcessor):
    """
    初始化模型实例。

    Args:
        model_name_or_path (str): 模型名称或路径。

    Attributes:
        model_name_or_path (str): 存储模型名称或路径。
        decode_status (dict): 存储解码状态信息。
        tokenizer (object): 存储分词器实例。
        eos_token_ids (list): 存储结束符号的token ID列表。
        eos_token_id_len (int): 存储结束符号的token ID列表的长度。
        pad_token_id (int): 存储填充符号的token ID。
    """

    def __init__(self, model_name_or_path):

        self.model_name_or_path = model_name_or_path
        data_processor_logger.info(f"model_name_or_path: {model_name_or_path}")
        self._init_config()
        self.model_name = ModelConfig(model_name_or_path).model_name

        self.is_thinking = (self.model_name == "x1")
        data_processor_logger.info(f"Thinking mode is {self.is_thinking}")
        self.decode_status = dict()
        self._load_tokenizer()
        data_processor_logger.info(f"tokenizer information: bos_token is {self.tokenizer.bos_token} \
                                   {self.tokenizer.bos_token_id}, \
                                   eos_token is {self.tokenizer.eos_token}, {self.tokenizer.eos_token_id} ")
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        self.eos_token_id_len = len(self.eos_token_ids)
        self.pad_token_id = self.get_pad_id()
        if self.is_thinking:
            self.reasoning_parser = ErnieX1ReasoningParser(self.tokenizer)

    def _init_config(self):
        self.use_hf_tokenizer = int(os.getenv("USE_HF_TOKENIZER", 0)) == 1

        # Generation config
        try:
            self.generation_config = GenerationConfig.from_pretrained(
                self.model_name_or_path)
        except Exception as e:
            data_processor_logger.warning(
                f"Can't find generation config, so it will not use "
                f"generation_config field in the model config, details={e}")
            self.generation_config = None

    def process_request(self, request, max_model_len=None):
        """
        Preprocess the request

        Args:
            request (Dict): may contain text and messages fields

        Returns:
            bool: Whether preprocessing is successful
            str: error message
        """
        request = self._apply_default_parameters(request)
        if request.get("eos_token_ids") is None or len(request.eos_token_ids) == 0:
            request.eos_token_ids = self.eos_token_ids
        stop_sequences = request.get("stop", [])
        if stop_sequences is not None and len(stop_sequences) != 0:
            stop_seqs, stop_seqs_len = self.update_stop_seq(stop_sequences)
            request.set("stop_token_ids", stop_seqs)
            request.set("stop_seqs_len", stop_seqs_len)

        if request.prompt_token_ids is None or len(request.prompt_token_ids) == 0:
            system = request.get("system")
            if request.prompt is None and request.messages is None:
                raise ValueError(
                    f"The request should have `input_ids`, `text` or `messages`: {request}.")
            if request.prompt is not None:
                prompt = request.prompt
                prompt = prompt[0] if isinstance(prompt, list) else prompt
                if self.model_name == "base":
                    request.prompt_token_ids = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(prompt, tokenize=True))
                else:
                    messages = [{"role": "user", "content": prompt}]
                    request.prompt_token_ids = self.messages2ids(messages)
            else:
                request.prompt_token_ids = self.messages2ids(
                    request)

            if self.model_name == "base":
                assert (
                        system is None or system == ""
                ), "The loadding model is a base model, `system` is not supported."
                assert request.messages is None, "The loadding model is a base model, `messages` is not supported."

        if max_model_len is not None and len(
                request.prompt_token_ids) > max_model_len:
            request.prompt_token_ids = request.prompt_token_ids[:
                                                                max_model_len -
                                                                1]
        if request.get("max_tokens") is None:
            request.set("max_tokens", max(1, max_model_len - len(request.prompt_token_ids)))
        data_processor_logger.info(f"Processed request {request}")
        return request

    def process_request_dict(self, request, max_model_len=None):
        """
        Preprocess the request

        Args:
            request (Dict): may contain text and messages fields

        Returns:
            bool: Whether preprocessing is successful
            str: error message
        """
        request = self._apply_default_parameters(request)
        if not request.get('eos_token_ids'):
            request['eos_token_ids'] = self.eos_token_ids
        # 处理stop_sequences
        stop_sequences = request.get('stop', [])
        if stop_sequences:
            stop_seqs, stop_seqs_len = self.update_stop_seq(stop_sequences)
            request['stop_token_ids'] = stop_seqs
            request['stop_seqs_len'] = stop_seqs_len

        system = request.get("system")
        # 处理prompt_token_ids
        if not request.get('prompt_token_ids'):
            if request.get('prompt') is None and request.get('messages') is None:
                raise ValueError(
                    f"Request must contain 'prompt_token_ids', 'prompt', or 'messages': {request}")
            if request.get('prompt'):
                prompt = request.get('prompt')
                prompt = prompt[0] if isinstance(prompt, list) else prompt
                if self.model_name == "base":
                    request['prompt_token_ids'] = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(prompt, tokenize=True))
                else:
                    messages = [{"role": "user", "content": prompt}]
                    request['prompt_token_ids'] = self.messages2ids(messages)
            else:
                request['prompt_token_ids'] = self.messages2ids(
                    request)

        if self.model_name == "base":
            assert isinstance(
                request['prompt'], str
            ), "the loadding model is a base model, `prompt` must be a string type."
            assert (
                    system is None or system == ""
            ), "The loadding model is a base model, `system` is not supported."
            assert request.get(
                'messages'
            ) is None, "The loadding model is a base model, `messages` is not supported."

        # 截断超过长度限制的prompt
        if max_model_len is not None and len(
                request['prompt_token_ids']) > max_model_len:
            request['prompt_token_ids'] = request[
                'prompt_token_ids'][:max_model_len - 1]
        if request.get("max_tokens") is None:
            request["max_tokens"] = max(1, max_model_len - len(request['prompt_token_ids']))
        data_processor_logger.info(f"Processed request {request}")
        
        return request

    def process_response(self, response_dict, **kwargs):
        """
        Preprocess the response

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """

        is_end = response_dict.finished
        req_id = response_dict.request_id
        token_ids = response_dict.outputs.token_ids
        
        if is_end and len(token_ids) > 0:
            if token_ids[-1] == self.tokenizer.eos_token_id:
                token_ids = token_ids[:-1]
        if self.is_thinking:
            cur_decode_status = self.decode_status.get(req_id, None)
            response_content, reasoning_content, tool_call_content, cur_decode_status = \
                self.reasoning_parser.extract_reasoning_content_streaming(
                    cur_decode_status, token_ids, req_id)
            self.decode_status[req_id] = cur_decode_status
            response_dict.outputs.text = response_content
            response_dict.outputs.reasoning_content = reasoning_content
            response_dict.outputs.tool_call_content = None
            if tool_call_content:
                tool_name = tool_call_content.get("name", None)
                tool_args = tool_call_content.get("arguments", None)
                if tool_name and not tool_args:
                    delta_tool_call = DeltaToolCall(
                        id=tool_call_content.get("id", None),
                        type="function",
                        index=tool_call_content.get("index", 0),
                        function=DeltaFunctionCall(name=tool_name, arguments=tool_args)
                    )
                else:
                    delta_tool_call = DeltaToolCall(
                        index=tool_call_content.get("index", 0),
                        function=DeltaFunctionCall(name=tool_name,
                                                   arguments=tool_call_content.get("partical_output", ""))
                    )
                response_dict.outputs.tool_call_content = [delta_tool_call]
        else:
            response_dict.outputs.text = self.ids2tokens(token_ids, req_id)
        response_dict.usage = {
            "completion_tokens": response_dict.outputs.index + 1}
        if is_end:
            if self.is_thinking:
                cur_decode_status = self.decode_status.get(req_id, None)
                response_content, reasoning_content, tool_call_content = \
                    self.reasoning_parser.extract_reasoning_content(
                        cur_decode_status)
                response_dict.outputs.text = response_content
                response_dict.outputs.reasoning_content = reasoning_content
                response_dict.outputs.tool_call_content = None
                if tool_call_content:
                    for idx, tool_call in enumerate(tool_call_content):
                        tool_name = tool_call.get("name", None)
                        tool_args = tool_call.get("arguments", None)
                        tool_call_id_index_pair = self.reasoning_parser.tool_call_ids_indices_pairs[idx]
                        delta_tool_call = ToolCall(
                            index=tool_call_id_index_pair[0],
                            id=tool_call_id_index_pair[1],
                            function=FunctionCall(name=tool_name, arguments=tool_args)
                        )
                        if not response_dict.outputs.tool_call_content:
                            response_dict.outputs.tool_call_content = []
                        response_dict.outputs.tool_call_content.append(delta_tool_call)
            else:
                response_content, reasoning_content = self.clear_request_status(req_id)
                response_dict.outputs.text = response_content
            del self.decode_status[req_id]
            data_processor_logger.debug("Request id: {} has been completed.".format(token_ids))
        if response_dict.outputs.text == "" and \
                response_dict.outputs.reasoning_content == "" and \
                response_dict.outputs.tool_call_content == []:
            return None
        return response_dict

    def process_response_dict(self, response_dict, **kwargs):
        """
        Preprocess the response

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """

        stream = kwargs.get("stream", True)
        enable_thinking = kwargs.get("enable_thinking", True)
        is_end = response_dict["finished"]
        req_id = response_dict["request_id"]
        token_ids = response_dict["outputs"]["token_ids"]

        if is_end and len(token_ids) > 0:
            if token_ids[-1] == self.tokenizer.eos_token_id:
                token_ids = token_ids[:-1]
        if self.is_thinking:
            cur_decode_status = self.decode_status.get(req_id, None)
            response_content, reasoning_content, tool_call_content, cur_decode_status = \
                self.reasoning_parser.extract_reasoning_content_streaming(
                    cur_decode_status, token_ids, req_id)
            self.decode_status[req_id] = cur_decode_status
            response_dict["outputs"]["text"] = response_content
            response_dict["outputs"]["reasoning_content"] = reasoning_content
            response_dict["outputs"]["tool_call_content"] = None
            if tool_call_content:
                tool_name = tool_call_content.get("name", None)
                tool_args = tool_call_content.get("arguments", None)
                if tool_name and not tool_args:
                    delta_tool_call = DeltaToolCall(
                        id=tool_call_content.get("id", None),
                        type="function",
                        index=tool_call_content.get("index", 0),
                        function=DeltaFunctionCall(name=tool_name, arguments=tool_args)
                    )
                else:
                    delta_tool_call = DeltaToolCall(
                        index=tool_call_content.get("index", 0),
                        function=DeltaFunctionCall(name=None,
                                                   arguments=tool_call_content.get("partical_output", ""))
                    )
                response_dict["outputs"]["tool_call_content"] = [delta_tool_call]
        else:
            response_dict["outputs"]["text"] = self.ids2tokens(
                token_ids, req_id)

        if is_end:
            data_processor_logger.debug(
                "Request id: {} has been completed.".format(token_ids))
            if not stream:
                if self.is_thinking:
                    cur_decode_status = self.decode_status.get(req_id, None)
                    response_content, reasoning_content, tool_call_content = \
                        self.reasoning_parser.extract_reasoning_content(
                            cur_decode_status)
                    response_dict["outputs"]["text"] = response_content
                    response_dict["outputs"]["reasoning_content"] = reasoning_content
                    response_dict["outputs"]["tool_call_content"] = None
                    if tool_call_content:
                        for idx, tool_call in enumerate(tool_call_content):
                            tool_name = tool_call.get("name", None)
                            tool_args = tool_call.get("arguments", None)
                            tool_call_id_index_pair = self.reasoning_parser.tool_call_ids_indices_pairs[idx]
                            delta_tool_call = ToolCall(
                                index=tool_call_id_index_pair[0],
                                id=tool_call_id_index_pair[1],
                                function=FunctionCall(name=tool_name, arguments=tool_args)
                            )
                            if not response_dict["outputs"]["tool_call_content"]:
                                response_dict["outputs"]["tool_call_content"] = []
                            response_dict["outputs"]["tool_call_content"].append(delta_tool_call)
                else:
                    full_text, reasoning_content = self.clear_request_status(req_id)
                    data_processor_logger.debug(
                        f"full_text: {full_text}, reasoning_content: {reasoning_content}")
                    response_dict["outputs"]["text"] = reasoning_content
                data_processor_logger.info(f"req_id:{req_id}, decode_status: {self.decode_status[req_id]}")
            del self.decode_status[req_id]

        return response_dict

    def messages2ids(self, request_or_messages):
        """
        Convert multi-turn messages into ID sequences.

        Args:
            request_or_messages: Either a request dict containing 'messages' field,
                                or a list of message dicts directly

        Returns:
            List of token IDs as strings (converted from token objects)
        """
        if self.tokenizer.chat_template is None:
            raise ValueError("This model does not support chat_template.")
        spliced_message = self.tokenizer.apply_chat_template(
            request_or_messages, tokenize=False,
            split_special_tokens=False, add_special_tokens=False
        )

        req_id = None
        if isinstance(request_or_messages, dict):
            req_id = request_or_messages.get("request_id", None)
        tokens = self.tokenizer.tokenize(spliced_message)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        data_processor_logger.info(f"req_id:{req_id}, tokens:{tokens}, token_ids: {token_ids}")
        return token_ids

    def ids2tokens(self, token_id, task_id):
        """
        token ids to strings

        Args:
            token_ids (List[int]): token ids
                        task_id (str): task id

        Returns:
            List[str]: strings
        """

        if task_id not in self.decode_status:
            # prefix offset & read offset & history token ids & history token strings
            self.decode_status[task_id] = [0, 0, [], "", ""]

        prefix_offset = self.decode_status[task_id][0]
        read_offset = self.decode_status[task_id][1]
        previous_token_ids = self.decode_status[task_id][2]
        decode_str, prefix_offset, read_offset = self.tokenizer.decode_token(
            previous_token_ids + token_id, prefix_offset, read_offset)
        self.decode_status[task_id][0] = prefix_offset
        self.decode_status[task_id][1] = read_offset
        self.decode_status[task_id][2] += token_id
        self.decode_status[task_id][3] += decode_str
        return decode_str

    def ids2tokens_thinking(self, token_id, task_id):
        """
        token ids to strings

        Args:
            token_ids (List[int]): token ids
                        task_id (str): task id

        Returns:
            List[str]: strings
        """

        if task_id not in self.decode_status:
            # prefix offset & read offset & history token ids & history token strings
            self.decode_status[task_id] = [0, 0, [], "", ""]

        prefix_offset = self.decode_status[task_id][0]
        read_offset = self.decode_status[task_id][1]
        previous_token_ids = self.decode_status[task_id][2]
        decode_str, prefix_offset, read_offset = self.tokenizer.decode_token(
            previous_token_ids + token_id, prefix_offset, read_offset)
        self.decode_status[task_id][0] = prefix_offset
        self.decode_status[task_id][1] = read_offset
        self.decode_status[task_id][2] += token_id

        data_processor_logger.debug(f"{token_id}, {decode_str}")
        reasoning_content = ""
        content = ""
        if decode_str == "</think>":
            self.decode_status[task_id][4] = decode_str
        elif self.decode_status[task_id][4] == "":
            self.decode_status[task_id][3] += decode_str
        else:
            self.decode_status[task_id][4] += decode_str

        response_cache = self.decode_status[task_id][4]
        if response_cache == "":
            reasoning_content = decode_str
        elif '<response>' in self.decode_status[task_id][4] and \
                decode_str not in ["<reponse>", "</response>", "<|im_end|>"]:
            content = decode_str
        return content, reasoning_content

    def _load_tokenizer(self):
        """
        load tokenizer

        Returns:
            tokenizer (AutoTokenizer)
        """
        vocab_file_names = ["tokenizer.model",
                            "spm.model", "ernie_token_100k.model"]
        for i in range(len(vocab_file_names)):
            if os.path.exists(os.path.join(self.model_name_or_path, vocab_file_names[i])):
                ErnieBotTokenizer.resource_files_names["vocab_file"] = vocab_file_names[i]
                break
        self.tokenizer = ErnieBotTokenizer.from_pretrained(
            self.model_name_or_path)

    def clear_request_status(self, task_id):
        """
        clear request status

        Args:
            task_id (str): task id

        Returns:
            results_all (str): all token strings
        """
        results_all = ""
        reasoning_content = ""

        if task_id in self.decode_status:
            if self.use_hf_tokenizer:
                results_all = self.decode_status[task_id][2]
            else:
                reasoning_content = "".join(self.decode_status[task_id][3])
                results_all = "".join(self.decode_status[task_id][4])
        return results_all, reasoning_content

    def get_pad_id(self):
        """
        get pad_token_id, if not pad_token_id, use eos_token

        Returns:
            int: pad_token_id
        """
        # if isinstance(self.tokenizer, (LlamaTokenizer, Llama3Tokenizer)) and not self.tokenizer.pad_token_id:
        #     return self.tokenizer.eos_token
        return self.tokenizer.pad_token_id

    def pad_batch_data(self, insts, pad_id=0, return_seq_len=False, return_array=True, pad_style="right"):
        """Pad the instances to the max sequence length in batch."""
        if len(insts) == 0:
            padded_insts = np.array(
                [[]], dtype=np.int64) if return_array else [[]]
            if return_seq_len:
                seq_len = np.array([], dtype=np.int64) if return_array else []
                return padded_insts, seq_len
            return padded_insts

        max_len = max(map(len, insts))
        if pad_style == "left":
            padded_insts = [
                [pad_id] * (max_len - len(inst)) + list(inst) for inst in insts]
        else:
            padded_insts = [list(inst) + [pad_id] *
                            (max_len - len(inst)) for inst in insts]
        if return_array:
            padded_insts = np.array(
                padded_insts, dtype=np.int64).reshape([-1, max_len])

        if return_seq_len:
            seq_len = [len(inst) for inst in insts]
            if return_array:
                seq_len = np.array(seq_len, dtype=np.int64).reshape(-1, 1)
            return padded_insts, seq_len
        return padded_insts

    def update_stop_seq(self, stop_sequences):
        """
        Update stop sequences from request.
        """
        stop_seqs = []
        for seq in stop_sequences:
            if seq != self.tokenizer.eos_token_id:
                stop_seqs.append(self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(seq)))
        stop_seqs, stop_seqs_len = self.pad_batch_data(
            stop_seqs,
            pad_id=-1,
            return_seq_len=True,
            return_array=False
        )
        data_processor_logger.debug(
            f"processed stop_seqs: {stop_seqs}, {stop_seqs_len}")
        return stop_seqs, stop_seqs_len


class ErnieX1ReasoningParserStatus:
    """
    Status of the ErnieX1 reasoning parser.
    """
    INIT = "init"
    THINKING = "thinking"
    RESPONSE = "response"
    TOOL_CALL = "tool_call"
    END = "end"


class ErnieX1ReasoningParser:
    """
    Reasoning parser for ErnieX1 model.

    The ErnieX1 model uses <think>...</think> tokens to denote reasoning
    text,  <response>...</response> tokens to denote response text,
    and use <tool_call>...</tool_call> tokens to denote tool call.
    This parser extracts the reasoning content, response content, and tool call content from the model output.
    """
    thinking_start_token: str = "<think>"
    thinking_end_token: str = "</think>"
    response_start_token: str = "<response>"
    response_end_token: str = "</response>"
    tool_call_start_token: str = "<tool_call>"
    tool_call_end_token: str = "</tool_call>"

    reasoning_content: str = ""
    response_content: str = ""
    tool_call_content: str = ""

    thoughts_pattern = re.compile(r"<think>(.*?)(?:</think>|$)",
                                  re.DOTALL)
    response_pattern = re.compile(r"<response>(.*?)(?:</response>|$)",
                                  re.DOTALL)
    function_call_pattern = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)",
                                       re.DOTALL)

    status: str = ErnieX1ReasoningParserStatus.INIT

    def __init__(self, tokenizer: ErnieBotTokenizer):
        self.tokenizer = tokenizer
        self.finish_reason = None
        self.tool_call_index = 0
        self.tool_call_ids_indices_pairs = []
        if not self.tokenizer:
            raise ValueError(
                "The tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")

    def extract_reasoning_content_streaming(
            self,
            decode_status: List[Union[int, List[int], str]],
            token_ids: List[int],
            task_id: str,
    ) -> Tuple[str, str, Dict[str, str], List[Union[int, List[int], str]]]:
        """
        Extract reasoning content from a delta message.
        Handles streaming output where previous + delta = current.
        Uses token IDs for faster processing.

        Args:
            decode_status: [prefix_offset, read_offset, token_ids, all_content, current_buffer, partial_tool_call]
            token_ids: only one token id in the token_ids list
            task_id: task id
        """

        if not decode_status:
            # [prefix_offset, read_offset, token_ids, all_content, current_buffer, partial_tool_call]
            decode_status = [0, 0, [], "", "", {}]
            self.status = ErnieX1ReasoningParserStatus.THINKING

        # 解析当前状态
        prefix_offset = decode_status[0]
        read_offset = decode_status[1]
        previous_token_ids = decode_status[2]
        all_content = decode_status[3]
        current_buffer = decode_status[4]
        partial_tool_call = decode_status[5]

        reasoning_content = ""
        response_content = ""
        tool_call_content = partial_tool_call.copy() if partial_tool_call else {}

        segment_text, new_prefix_offset, new_read_offset = self.tokenizer.decode_token(
            previous_token_ids + token_ids, prefix_offset, read_offset
        )
        all_content += segment_text
        # 检查状态转换
        if segment_text == self.thinking_start_token:
            self.status = ErnieX1ReasoningParserStatus.THINKING
            current_buffer = ""
            partial_tool_call = {}
        elif segment_text == self.response_start_token:
            self.status = ErnieX1ReasoningParserStatus.RESPONSE
            current_buffer = ""
            partial_tool_call = {}
        elif segment_text == self.tool_call_start_token:
            self.status = ErnieX1ReasoningParserStatus.TOOL_CALL
            current_buffer = ""
        elif segment_text in [self.thinking_end_token, self.response_end_token, self.tool_call_end_token]:
            if self.status == ErnieX1ReasoningParserStatus.TOOL_CALL:
                try:
                    tool_call_data = json.loads(current_buffer)
                    tool_call_id = f"call_{self.tool_call_index}-{uuid.uuid4()}"
                    self.tool_call_ids_indices_pairs.append((self.tool_call_index, tool_call_id))
                    tool_call_content = {
                        "name": tool_call_data.get("name", ""),
                        "arguments": tool_call_data.get("arguments", {}),
                        "partical_output": "",
                        "index": self.tool_call_index,
                        "id": tool_call_id
                    }
                    self.finish_reason = "tool_calls"
                    self.tool_call_index += 1
                except json.JSONDecodeError:
                    tool_call_content = partial_tool_call

            self.status = ErnieX1ReasoningParserStatus.INIT
            current_buffer = ""
            partial_tool_call = {}
        elif segment_text:
            current_buffer += segment_text
            if self.status == ErnieX1ReasoningParserStatus.THINKING:
                reasoning_content = segment_text
            elif self.status == ErnieX1ReasoningParserStatus.RESPONSE:
                response_content = segment_text
            elif self.status == ErnieX1ReasoningParserStatus.TOOL_CALL:
                try:
                    name_match = re.search(r'"name"\s*:\s*"([^"]*)"', current_buffer)
                    if name_match:
                        tool_call_content["name"] = name_match.group(1)
                        partial_tool_call["name"] = name_match.group(1)
                    # 尝试解析arguments
                    args_start = current_buffer.find('"arguments"')
                    if args_start != -1:
                        args_text = current_buffer[args_start:]
                        args_content_start = args_text.find('{')
                        if args_content_start != -1:
                            args_content = args_text[args_content_start:]
                            brace_count = 0
                            args_end = -1
                            for i, char in enumerate(args_content):
                                if char == '{':
                                    brace_count += 1
                                elif char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        args_end = i + 1

                            if args_end != -1:
                                try:
                                    args = json.loads(args_content[:args_end])
                                    tool_call_content["arguments"] = args
                                    partial_tool_call["arguments"] = args
                                    tool_call_content["partical_output"] = segment_text
                                    partial_tool_call["partical_output"] = segment_text
                                except json.JSONDecodeError:
                                    tool_call_content["arguments"] = args_content[:args_end]
                                    partial_tool_call["arguments"] = args_content[:args_end]
                            else:
                                tool_call_content["arguments"] = args_content
                                partial_tool_call["arguments"] = args_content
                                tool_call_content["partical_output"] = segment_text
                                partial_tool_call["partical_output"] = segment_text

                            if brace_count == -1:
                                tool_call_content["partical_output"] = ""
                                partial_tool_call["partical_output"] = ""


                except Exception as e:
                    data_processor_logger.warning(f"Parsing partial tool call: {e}")
            elif self.status == ErnieX1ReasoningParserStatus.INIT:
                response_content = segment_text

        decode_status[0] = new_prefix_offset
        decode_status[1] = new_read_offset
        decode_status[2] = previous_token_ids + token_ids
        decode_status[3] = all_content
        decode_status[4] = current_buffer
        decode_status[5] = partial_tool_call
        tool_call_content = None if tool_call_content == {} else tool_call_content

        return response_content, reasoning_content, tool_call_content, decode_status

    def extract_reasoning_content(
            self,
            decode_status: List[Union[int, List[int], str]],
    ) -> Tuple[str, str, Dict[str, str]]:
        """
        Extract reasoning, response and tool call content from complete model output.

        Args:
            model_output: Complete model output string

        Returns:
            Tuple of (response_content, reasoning_content, tool_call_content)
        """
        all_content = self.thinking_start_token + decode_status[3]
        thoughts = self.thoughts_pattern.search(all_content) or ""
        thoughts = thoughts.group(1).strip() if thoughts else ""
        response_match = self.response_pattern.search(all_content) or ""
        response = response_match.group(1).strip() if response_match else ""
        function_calls = self.function_call_pattern.findall(all_content)
        tools = []
        for fc in function_calls:
            try:
                arg_dict = json.loads(fc)
                if not isinstance(arg_dict, dict):
                    raise ValueError("Arguments must be a JSON object")
                tools.append({
                    "name": arg_dict.get("name", ""),
                    "arguments": json.dumps(arg_dict.get("arguments", ""), ensure_ascii=False)
                })
            except Exception as e:
                pass
        tools = None if tools == [] else tools
        return response, thoughts, tools

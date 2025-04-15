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

import os
import numpy as np

from fastdeployllm.utils import data_processor_logger
from paddlenlp.generation import GenerationConfig
from efficientllm.models.tokenizer import ErnieBotTokenizer
from fastdeployllm.input.text_processor import BaseDataProcessor
from fastdeployllm.utils import data_processor_logger

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

        self.decode_status = dict()
        self._load_tokenizer()
        data_processor_logger.info(f"tokenizer information: bos_token is {self.tokenizer.bos_token} \
                                   {self.tokenizer.bos_token_id}, \
                                   eos_token is {self.tokenizer.eos_token}, {self.tokenizer.eos_token_id} ")
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        self.eos_token_id_len = len(self.eos_token_ids)
        self.pad_token_id = self.get_pad_id() 

    def _init_config(self):
        self.use_hf_tokenizer = int(os.getenv("USE_HF_TOKENIZER", 0)) == 1

        # Generation config
        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_name_or_path)
        except:
            data_processor_logger.warning(
                "Can't find generation config, so it will not use generation_config field in the model config"
            )
            self.generation_config = None

        
    def process_request(self, request, max_seq_len=None):
        """
        Preprocess the request

        Args:
            request (Dict): may contain text and messages fields

        Returns:
            bool: Whether preprocessing is successful
            str: error message
        """
        if "eos_token_ids" not in request or request["eos_token_ids"] == [None]:
            request["eos_token_ids"] = []
        request["eos_token_ids"].extend(self.eos_token_ids)

        if "stop_seqs" not in request or (isinstance(
            request["stop_seqs"], (list, tuple)) and len(request["stop_seqs"]) == 0):
            self.update_stop_seq(request)

        if "input_ids" not in request or \
            (isinstance(request["input_ids"], (list, tuple)) and len(request["input_ids"]) == 0):
            if "text" in request:
                request["input_ids"] = self.text2ids(request["text"], max_seq_len)
            elif "messages" in request:
                if self.tokenizer.chat_template is None:
                    raise ValueError(f"This model does not support chat_template.")
                request["input_ids"] = self.messages2ids(request["messages"])
            else:
                raise ValueError(f"The request should have `input_ids`, `text` or `messages`: {request}.")

        if max_seq_len is not None and len(request["input_ids"]) > max_seq_len:
            request["input_ids"] = request["input_ids"][:max_seq_len - 1]
        data_processor_logger.info(f"processed request: {request}")
        return request

    def process_response(self, response_dict, **kwargs):
        """
        Preprocess the response

        Args:
            response_dict (Dict): response for engine, contain ids fields

        Returns:
            Dict: response contain text fields
        """
        is_end = response_dict.get("is_end", 0)
        req_id = response_dict.get("req_id")
        if "choices" in response_dict:
            for i in range(len(response_dict["choices"])):
                response_dict["token"] = self.ids2tokens(response_dict["choices"][i]["token_ids"], req_id)
            return response_dict

        token_ids = response_dict.get("token_ids", [])
        response_dict["token"] = self.ids2tokens(token_ids, response_dict["req_id"])
        response_dict["usage"] = {"completion_tokens" : response_dict["send_idx"] + 1}

        if is_end:
            self.clear_request_status(req_id)
            token_ids = response_dict.get("tokens_all_ids", [])
            response_dict["tokens_all"] = self.ids2tokens(token_ids, response_dict["req_id"])
        return response_dict

    def text2ids(self, text, max_seq_len):
        """
        text to token ids

        Args:
            text (str): text

        Returns:
            List[int]: token ids list
        """

        # TODO: tokenizer 模版匹配
        tokens = self.tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=self.tokenizer.chat_template is None,
        )
        return tokens["input_ids"][0]

    def messages2ids(self, messages):
        """
        Convert multi-turn messages into ID sequences.

        Args:
            messages (List[List[Dict[str, Any]]]): multi-turn messages.

        Returns:
            List[int]: ID sequences
        """
        message_result = self.tokenizer.apply_chat_template(messages, return_tensors="pd")
        return message_result["input_ids"][0]

    def ids2tokens(self, token_id, task_id):
        """
        token ids to strings

        Args:
            token_ids (List[int]): token ids
			task_id (str): task id

        Returns:
            List[str]: strings
        """
        if self.use_hf_tokenizer:
            if task_id not in self.decode_status:
                # history token ids & history token strings & befer decode str
                self.decode_status[task_id] = [[], [], ""]

            previous_token_ids = self.decode_status[task_id][0]
            decode_str = self.tokenizer.batch_decode([previous_token_ids + token_id],
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
            if isinstance(decode_str, list) and len(decode_str):
                new_str = decode_str[0].replace(self.decode_status[task_id][2], "", 1)
                self.decode_status[task_id][1].append(new_str)
                self.decode_status[task_id][2] = decode_str[0]
            else:
                new_str = ""
            self.decode_status[task_id][0] += token_id
            return new_str
        else:
            if task_id not in self.decode_status:
                # prefix offset & read offset & history token ids & history token strings
                self.decode_status[task_id] = [0, 0, [], []]

            prefix_offset = self.decode_status[task_id][0]
            read_offset = self.decode_status[task_id][1]
            previous_token_ids = self.decode_status[task_id][2]
            decode_str, prefix_offset, read_offset = self.tokenizer.decode_token(
                previous_token_ids + token_id, prefix_offset, read_offset)
            self.decode_status[task_id][0] = prefix_offset
            self.decode_status[task_id][1] = read_offset
            self.decode_status[task_id][2] += token_id
            self.decode_status[task_id][3].append(decode_str)
            return decode_str

    def _load_tokenizer(self):
        """
        load tokenizer

        Returns:
            tokenizer (AutoTokenizer)
        """
        self.tokenizer = ErnieBotTokenizer.from_pretrained(self.model_name_or_path)
    def clear_request_status(self, task_id):
        """
        clear request status

        Args:
            task_id (str): task id

        Returns:
            results_all (str): all token strings
        """
        results_all = ""
        if task_id in self.decode_status:
            if self.use_hf_tokenizer:
                results_all = self.decode_status[task_id][2]
            else:
                results_all = "".join(self.decode_status[task_id][3])
            del self.decode_status[task_id]
        return results_all



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
            padded_insts = np.array([[]], dtype=np.int64) if return_array else [[]]
            if return_seq_len:
                seq_len = np.array([], dtype=np.int64) if return_array else []
                return padded_insts, seq_len
            return padded_insts

        max_len = max(map(len, insts))
        if pad_style == "left":
            padded_insts = [[pad_id] * (max_len - len(inst)) + list(inst) for inst in insts]
        else:
            padded_insts = [list(inst) + [pad_id] * (max_len - len(inst)) for inst in insts]
        if return_array:
            padded_insts = np.array(padded_insts, dtype=np.int64).reshape([-1, max_len])

        if return_seq_len:
            seq_len = [len(inst) for inst in insts]
            if return_array:
                seq_len = np.array(seq_len, dtype=np.int64).reshape(-1, 1)
            return padded_insts, seq_len
        return padded_insts

    def update_stop_seq(self, request):
        """
        Update stop sequences from request.
        """
        stop_seqs =  []
        for seq in request.get("stop_sequences", []):
            if seq != self.tokenizer.eos_token_id:
                stop_seqs.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(seq)))
        request["stop_seqs"], request["stop_seqs_len"] = self.pad_batch_data(
            stop_seqs,
            pad_id=-1,
            return_seq_len=True,
            return_array=False
        )
        data_processor_logger.debug(f"processed request: {request['stop_seqs'], request['stop_seqs_len']}")

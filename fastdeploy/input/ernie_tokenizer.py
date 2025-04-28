"""
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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

# cipher_token=WjI1fQOvhN  # do not edit this line

import os
from shutil import copyfile
from typing import Dict, Optional, Tuple, Union

import numpy as np
from paddlenlp.transformers import PretrainedTokenizer
from paddlenlp.transformers.tokenizer_utils_base import PaddingStrategy
from paddlenlp.utils.log import logger
from sentencepiece import SentencePieceProcessor
import paddle

from . import spm_pb2 as spm

__all__ = ["ErnieBotTokenizer"]


class ErnieBotTokenizer(PretrainedTokenizer):
    """
    A basic tokenizer class that will be inherited by all subclasses.
    """

    resource_files_names = {
        "vocab_file": "spm.model",
    }
    pretrained_resource_files_map = {"vocab_file": {"ernie-bot": None}}
    pretrained_init_configuration = {
        "ernie-bot": {},
    }
    model_input_names = [
        "input_ids",
        "position_ids",
        "attention_mask",
        "labels",
    ]
    padding_side = "right"

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        cls_token="<cls>",
        eos_token="</s>",
        mask_token="<mask:0>",
        pad_token="<pad>",
        sep_token="<sep>",
        unk_token="<unk>",
        additional_special_tokens=None,
        split_special_tokens=True,
        alpha=None,  # do not work
        tokenizer_alpha=None,
        **kwargs,
    ):
        """Constructs a ErnieBotTokenizer."""
        if additional_special_tokens is None:
            additional_special_tokens = ["<mask:1>", "<mask:7>"]
        super().__init__(
            bos_token=bos_token,
            cls_token=cls_token,
            eos_token=eos_token,
            mask_token=mask_token,
            pad_token=pad_token,
            sep_token=sep_token,
            unk_token=unk_token,
            additional_special_tokens=additional_special_tokens,
            split_special_tokens=split_special_tokens,
            **kwargs,
        )
        self.verbose = False
        self.vocab_file = vocab_file

        self.model = spm.ModelProto()
        with open(vocab_file, "rb") as fp:
            self.model.ParseFromString(fp.read())
        self.sp_model = SentencePieceProcessor()
        self.sp_model.Load(model_proto=self.model.SerializeToString())
        self.alpha = alpha
        self.pad_id = self._convert_token_to_id(pad_token)
        self.tokenizer_alpha = tokenizer_alpha

    @property
    def vocab_size(self):
        """
        Returns the size of the vocabulary.
        """
        return self.sp_model.vocab_size()

    def get_vocab(self):
        """
        Returns the vocabulary as a dict.
        """
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Tokenize text."""
        if self.tokenizer_alpha is not None:
            return self.sp_model.encode_as_pieces(
                text,
                enable_sampling=True,
                nbest_size=-1,
                alpha=self.tokenizer_alpha,
            )
        else:
            return self.sp_model.encode_as_pieces(text)

    def _convert_token_to_id(self, token):
        """
        Convert a token into its id representation.
        """
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, id):
        """
        Convert an index (integer) into a token (string).
        """

        if id >= self.vocab_size:
            return self.unk_token
        else:
            return self.sp_model.id_to_piece(id)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        """
        output = token_ids_0
        # TODO (huijuan): whether to handle cls and sep token here.
        last_cls_index = -1
        last_sep_index = -1
        if self.cls_token_id in output:
            last_cls_index = len(output) - output[::-1].index(self.cls_token_id) - 1
        if self.sep_token_id in output:
            last_sep_index = len(output) - output[::-1].index(self.sep_token_id) - 1

        if last_cls_index > last_sep_index:
            next_token_id = self.sep_token_id
        elif last_sep_index > last_cls_index:
            next_token_id = self.cls_token_id
        else:
            output = [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
            next_token_id = self.cls_token_id

        output = [self.bos_token_id] + output
        # Assume no markup in text if token_ids_1 is given.
        if token_ids_1 is not None:
            output = output + token_ids_1 + [next_token_id]
        return output

    def get_special_tokens_mask(
        self, token_ids_0, token_ids_1=None, already_has_special_tokens=False
    ):
        """
        Get the special tokens mask for a sequence or a pair of sequences.
        Args:
            token_ids_0 (List[int]): List of IDs to which the special tokens will be added.
            token_ids_1 (List[int], optional): Optional second list of IDs for sequence pairs.
            already_has_special_tokens (bool, optional): Whether the input already contains special
                tokens at the beginning or end. Defaults to `False`.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0, token_ids_1, already_has_special_tokens=True
            )

        # [bos_token, cls_token, tokens_0, sep_token]
        if token_ids_1 is None:
            return [1, 1] + ([0] * len(token_ids_0)) + [1]
        # [bos_token, cls_token, tokens_0, sep_token, tokens_1, cls_token]
        return [1, 1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def save_vocabulary(
        self, save_directory, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.
        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + self.resource_files_names["vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(
            out_vocab_file
        ) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def _pad(
        self,
        encoded_inputs: Union[Dict],
        max_length: Optional[int] = None,
        padding_strategy=PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs to the longest length in the batch.
        """

        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names
        if return_attention_mask:
            required_input = encoded_inputs[self.model_input_names[0]]
            if padding_strategy == PaddingStrategy.LONGEST:
                max_length = len(required_input)
            if (
                max_length is not None
                and pad_to_multiple_of is not None
                and (max_length % pad_to_multiple_of != 0)
            ):
                max_length = (
                    (max_length // pad_to_multiple_of) + 1
                ) * pad_to_multiple_of
            needs_to_be_padded = (
                padding_strategy != PaddingStrategy.DO_NOT_PAD
                and len(required_input) != max_length
            )

            if (
                "attention_mask" in encoded_inputs
                and encoded_inputs["attention_mask"] is not None
            ):
                attention_mask = encoded_inputs.pop("attention_mask")
                if isinstance(attention_mask, paddle.Tensor):
                    attention_mask = attention_mask.numpy()
                elif isinstance(attention_mask, list):
                    attention_mask = np.array(attention_mask)
                elif not isinstance(attention_mask, np.ndarray):
                    raise ValueError(
                        f"Unexpected type {type(attention_mask)} of attention_mask, "
                    )
            else:
                attention_mask = paddle.tril(
                    paddle.ones(
                        [len(required_input), len(required_input)],
                        dtype="int64",
                    )
                )
                attention_mask = attention_mask.unsqueeze(0).numpy()

            if needs_to_be_padded:
                difference = max_length - len(required_input)
                if self.padding_side == "right":
                    if attention_mask.ndim == 1:
                        pad_width = [(0, difference)]
                    else:
                        pad_width = [(0, 0), (0, difference), (0, difference)]
                elif self.padding_side == "left":
                    if attention_mask.ndim == 1:
                        pad_width = [(difference, 0)]
                    else:
                        pad_width = [(0, 0), (difference, 0), (difference, 0)]
                else:
                    raise ValueError(
                        "Invalid padding strategy:" + str(self.padding_side)
                    )
                attention_mask = np.pad(
                    attention_mask,
                    pad_width=pad_width,
                    mode="constant",
                    constant_values=0,
                )

        encoded_inputs = super()._pad(
            encoded_inputs,
            max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=False,
        )
        if return_attention_mask:
            encoded_inputs["attention_mask"] = attention_mask.tolist()
        return encoded_inputs

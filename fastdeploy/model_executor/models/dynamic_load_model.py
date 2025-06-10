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

import json
import os
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import paddle
from paddle import nn
from paddle.distributed import fleet
from paddlenlp.trl import llm_utils
from paddlenlp.utils.log import logger

from fastdeploy.model_executor.models.ernie_vl.configuration import ErnieBotMoEVLConfig
from fastdeploy.model_executor.models.ernie_vl.dfnrope import DFNRopeVisionTransformerConfig
from fastdeploy.model_executor.models.ernie_vl.dfnrope.modeling import DFNRopeVisionTransformerPretrainedModel
from fastdeploy.model_executor.models.ernie_vl.modeling_resampler import ScatterOp
from fastdeploy.model_executor.models.ernie_vl.modeling_resampler import VariableResolutionResamplerModel


class DynamicLoadModel(nn.Layer):
    """EfficientLLM model"""

    def __init__(
        self,
        model_name_or_path: str,
        dtype: str = "bfloat16",
        block_size: int = 64,
        max_len: int = 8192,
        stage_flag: str = "EfficientLLM-Inference",
        model_path: Optional[str] = None,
        ori_vocab_size: Optional[int] = None,
        draft_type: str = "None",
        local_test: bool = False,
        load_model_from_ipc: bool = False,
        nranks: int = 1,
        rank: int = 0,
        use_stop_seqs: bool = False,
        moe_quant_type: str = "weight_only_int8",
        export_model_type: str = "weight_only_int8",
        is_unified_ckpt: bool = False,
        pad_vocab: bool = True,
        output_via_mq: bool = True,
        tokenizer=None,
        model_cfg=None,
        vision_model=None,
        resampler_model=None,
        use_for_train: bool = False,
        use_empty_parameter: bool = True,
        **kwargs,
    ):
        """
        Initialize EfficientLLM model with configuration and build model immediately.

        Args:
            config: Model configuration dictionary
            dtype: Data type for model parameters
            block_size: Block size for attention
            max_seq_len: Maximum sequence length
            stage_flag: Stage flag for EfficientLLM
            model_path: Path to model weights
            ori_vocab_size: Original vocabulary size
            **kwargs: Additional arguments for model configuration
        """
        super(DynamicLoadModel, self).__init__()

        self.dtype = dtype
        paddle.set_default_dtype(self.dtype)
        self.block_size = block_size
        self.max_len = max_len
        self.stage_flag = stage_flag
        self.model_path = model_name_or_path
        self.use_stop_seqs = use_stop_seqs
        self.moe_quant_type = moe_quant_type
        self.export_model_type = export_model_type
        self.is_unified_ckpt = is_unified_ckpt
        self.kwargs = kwargs
        self.pad_vocab = pad_vocab
        self.output_via_mq = output_via_mq
        self.use_empty_parameter = use_empty_parameter

        self.load_model_from_ipc = load_model_from_ipc
        self.first_load = True
        self.nranks = nranks
        self.rank = rank
        self.tokenizer = tokenizer
        self.model_cfg = model_cfg if model_cfg else os.path.join(
            self.model_path, os.getenv("CONFIG_JSON_FILE", "config.json"))

        self.vision_model, self.resampler_model = vision_model, resampler_model
        if use_for_train:
            self.inject_pp_vision_model()

        # build model
        self.model = self._build_model()

        # Create a list of all models to process
        self.models = [self.model]
        if self.vision_model:
            self.models.append(self.vision_model)
        if self.resampler_model:
            self.models.append(self.resampler_model)

        self.local_test = local_test
        # self.get_model_static_info()
        # Build model during initialization

        if self.load_model_from_ipc:
            self.update_parameters()

        logger.info(
            "EfficientLLM model built successfully by DynamicLoadModel")

    def inject_pp_vision_model(self):
        """
        注入vision model参数
        """
        from fastdeploy.input.mm_processor.tokenizer import ErnieVLTokenizer
        tokenizer = ErnieVLTokenizer.from_pretrained(
            os.path.dirname(self.model_path),
            model_max_length=self.max_len,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.ignored_index = -100
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        self.tokenizer = tokenizer

        vision_model_name_or_path = f"{os.path.dirname(self.model_path)}/DFNRopeVisionTransformer"
        context = paddle.LazyGuard()
        with context:
            config = ErnieBotMoEVLConfig.from_pretrained(
                self.model_path,
                tensor_parallel_degree=self.nranks,
                tensor_parallel_rank=self.rank,
                moe_group="dummy",
            )
            vision_config = DFNRopeVisionTransformerConfig.from_pretrained(
                vision_model_name_or_path,
                tensor_parallel_degree=1,
                tensor_parallel_rank=0,
                attn_sep=False,
                dtype="bfloat16",
            )
            config.vision_config = vision_config
            config.pixel_hidden_size = config.vision_config.hidden_size

            config.tensor_parallel_output = False
            config.sequence_parallel = False

            vision_model = DFNRopeVisionTransformerPretrainedModel.from_config(
                config=config.vision_config)

            vision_model = paddle.amp.decorate(models=vision_model,
                                            level="O2",
                                            dtype="bfloat16")

            resampler_model = VariableResolutionResamplerModel(
                config.pixel_hidden_size,
                config.hidden_size,
                config.spatial_conv_size,
                config.temporal_conv_size,
                config=config,
            )
            resampler_model = paddle.amp.decorate(models=resampler_model,
                                                level="O2",
                                                dtype="bfloat16")

            vision_model.eval()
            resampler_model.eval()
            self.vision_model = vision_model
            self.resampler_model = resampler_model
            logger.info("inject vision model successfully")

    def _build_model(self) -> paddle.nn.Layer:
        """Build the EfficientLLM model architecture."""
        from .export_model import build_stream_line_model

        _, _, model, _ = build_stream_line_model(
            self.model_cfg,
            self.model_path,
            self.dtype,
            block_size=self.block_size,
            max_len=self.max_len,
            stage_flag=self.stage_flag,
            output_via_mq=self.output_via_mq,
            export_model_type=self.export_model_type,
            use_fake_parameter=True,
            use_stop_seqs=self.use_stop_seqs,
            use_beam_search=False,
            speculate_method=None,
            speculate_max_draft_token_num=5,
            return_all_hidden_states=False,
            moe_quant_type=self.moe_quant_type,
            use_safetensors=self.is_unified_ckpt,
            embeddings_column_cut=self.kwargs.get("embeddings_column_cut",
                                                  False),
            tokenizer=self.tokenizer,
            pad_vocab=self.pad_vocab,
            use_empty_parameter=self.use_empty_parameter)
        model.eval()

        return model

    @staticmethod
    def load_tensor_from_ipc_meta(
            ipc_state_dict: Dict[str, Any]) -> Dict[str, paddle.Tensor]:
        """
        Convert ipc_meta to tensor while keeping keys unchanged.

        Args:
            state_dict: Dictionary containing ipc_meta objects

        Returns:
            Dictionary with ipc_meta objects converted to tensors
        """
        result = {}
        for k, v in ipc_state_dict.items():
            v[0] = v[0].encode("latin-1")
            tensor = paddle.base.core.LoDTensor._new_shared_cuda(tuple(v))
            result[k] = paddle.to_tensor(tensor)

        return result

    def get_model(self) -> paddle.nn.Layer:
        """Get the underlying model instance."""
        return self.model

    def get_model_static_info(self) -> None:
        """get static info."""
        for k, v in self.state_dict().items():
            logger.info(
                f"efficientl model key name is :{k}, shape : {v.shape}, dtype : {v.dtype}"
            )
    
    def get_name_mappings_to_training(self):
        """Get name mappings to training parameters for all models."""
        all_name_mappings = {}
        for model in self.models:
            all_name_mappings.update(model.get_name_mappings_to_training())
        return all_name_mappings

    def forward(self, **kwargs):
        """generate."""
        self.model(**kwargs)

    def _update_shared_status(self, pid: int, status: int) -> None:
        """Update shared memory status flag."""
        array = np.zeros([1], dtype=np.int32)
        shm = SharedMemory(create=False,
                           size=array.nbytes,
                           name=f"model_weights_status.{pid}")
        value = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        if self.rank == 0:
            value[self.rank] = status

    def update_parameters(self, pid: int = 0) -> None:
        """Update model parameters from IPC state dictionary."""
        self.log_memory_usage("start update parameters")

        if self.vision_model and self.resampler_model:
            for model in [self.resampler_model, self.vision_model]:
                for name, param in model.state_dict().items():
                    logger.info(f"Clearing model parameter: {name}")
                    param._clear_data()


        paddle.device.cuda.empty_cache()
        if not self.first_load:
            paddle.distributed.restart_process_group()
            logger.info("Paddle distributed restart_process_group.")

        self.log_memory_usage("start update parameters")

        if self.local_test:
            model_path = f"{self.model_path}/model_state.tp0{self.rank}.pdparams"
            logger.info(f"Loading model from: {model_path}")

            set_start = time.perf_counter()
            print("使用shared_buf_to_local_test")
            state_dict = paddle.load(model_path)
            model_state_dict = self.state_dict()
            for name, param in state_dict.items():
                if name in model_state_dict:
                    logger.info(f"Updating model parameter: {name}, shape : {param.shape}")
                    update_param = model_state_dict[name]

                    if update_param.dtype != param.dtype:
                        raise TypeError(
                            f"Type mismatch for {name}: {param.dtype} vs {update_param.dtype}"
                        )
                    if update_param.shape != param.shape:
                        raise ValueError(
                            f"Shape mismatch for {name}: {param.shape} vs {update_param.shape}"
                        )

                    param._share_buffer_to(update_param)

            logger.info(
                f"set_state_dict completed in {time.perf_counter()  - set_start:.2f} seconds"
            )

            self.verify_parameters_updated()
            self.log_memory_usage("update parameters end")

            if self.nranks > 1:
                paddle.distributed.barrier()

            if not self.first_load:
                logger.info("send update signal")
                self._update_shared_status(pid, 0)

            self.first_load = False
            return

        logger.info("Starting parameter update process...")

        ipc_state_dict_path = f"/shared_ipc_meta/ipc_metas_{self.rank}"
        logger.info(f"Loading IPC state dict from: {ipc_state_dict_path}")

        convert_start = time.perf_counter()
        state_dict = self.load_tensor_from_ipc_meta(
            paddle.load(ipc_state_dict_path))
        logger.info(
            f"IPC meta converted to tensors in {time.perf_counter()  - convert_start:.2f} seconds"
        )

        logger.info("Updating parameters via shared_buffer_to...")
        share_start = time.perf_counter()

        infer_model_state_dict = self.state_dict()
        for name, param in state_dict.items():
            # name = name.replace("ernie.", "gpt.")
            if name in infer_model_state_dict:  # 在全局 state_dict 中查找匹配项
                logger.info(f"Updating model parameter: train-{name}")
                update_param = infer_model_state_dict[name]

                if update_param.dtype != param.dtype:
                    raise TypeError(
                        f"Type mismatch for {name}: train-{param.dtype} vs infer-{update_param.dtype}"
                    )
                if update_param.shape != param.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: train-{param.shape} vs infer-{update_param.shape}"
                    )
                param._share_buffer_to(update_param)
            else:
                logger.error(f"No matching parameter found for train-{name} in global state_dict")

        logger.info(
            f"Parameter sharing completed in {time.perf_counter() - share_start:.2f} seconds"
        )

        if self.nranks > 1:
            paddle.distributed.barrier()

        if not self.first_load:
            logger.info("send update signal")
            self._update_shared_status(pid, 0)

        self.first_load = False
        self.verify_parameters_updated()
        paddle.device.cuda.empty_cache()
        self.log_memory_usage("update parameters end")

    def clear_parameters(self, pid: int = 0) -> None:
        """Clear all model parameters."""
        self.log_memory_usage("start clear parameters")

        start_time = time.perf_counter()
        paddle.device.cuda.empty_cache()

        # Process all models in a loop
        for model in self.models:
            for name, param in model.state_dict().items():
                logger.info(f"Clearing model parameter: {name}")
                param._clear_data()

        clear_time = time.perf_counter() - start_time
        logger.info(
            f"Parameter clearing completed in {clear_time:.2f} seconds")

        self.verify_parameters_cleared()
        logger.info("Model parameters cleared successfully")

        if self.nranks > 1:
            paddle.distributed.barrier()

        paddle.distributed.shutdown_process_group()
        logger.info("Paddle distributed shutdown_process_group.")

        self._update_shared_status(pid, -2)
        paddle.device.cuda.empty_cache()
        self.log_memory_usage("clear parameters end")
        logger.info("send clear signal done!")

    def verify_parameters_cleared(self, erro_log: bool = True) -> bool:
        """
        Verify that all model parameters have been cleared.

        Returns:
            bool: True if all parameters are cleared, False otherwise
        """
        all_cleared = True
        for name, param in self.model.state_dict().items():
            if param._is_initialized():
                if erro_log:
                    logger.error(f"Parameter {name} was not properly cleared!")
                all_cleared = False

        if all_cleared:
            logger.info("All parameters verified as cleared successfully")
        else:
            if erro_log:
                logger.error("Some parameters were not properly cleared!")

        return all_cleared

    def verify_parameters_updated(self, erro_log: bool = True) -> bool:
        """
        Verify that model parameters match the source state dictionary.

        Args:
            source_state_dict: Dictionary containing the expected parameters

        Returns:
            bool: True if all parameters match, False otherwise
        """
        logger.info("Verifying parameters are cleared...")
        all_update = True

        for model in self.models:
            for name, param in model.state_dict().items():
                if not param._is_initialized():
                    if erro_log:
                        logger.error(
                            f"Parameter {name}-{param} was not properly cleared!")
                    all_update = False

        if all_update:
            logger.info("All parameters verified as updated successfully")
        else:
            if erro_log:
                logger.error("Some parameters were not properly updated!")

        return all_update

    def log_memory_usage(self, context: str = "") -> None:
        """Log current GPU memory usage."""
        max_alloc = paddle.device.cuda.max_memory_allocated() / (1024**3)
        max_reserved = paddle.device.cuda.max_memory_reserved() / (1024**3)
        curr_alloc = paddle.device.cuda.memory_allocated() / (1024**3)
        curr_reserved = paddle.device.cuda.memory_reserved() / (1024**3)

        logger.info(f"GPU memory usage {context}:")
        logger.warning(f"max_allocated: {max_alloc:.2f}GB\n"
                       f"max_reserved: {max_reserved:.2f}GB\n"
                       f"current_allocated: {curr_alloc:.2f}GB\n"
                       f"current_reserved: {curr_reserved:.2f}GB")
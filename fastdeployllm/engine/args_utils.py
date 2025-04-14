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

import argparse
import json
from dataclasses import dataclass, fields as dataclass_fields
from typing import Any, Dict, List, Optional

from fastdeployllm.engine.config import Config, ModelConfig, TaskOption

def nullable_str(x: str) -> Optional[str]:
    """Convert empty string to None while preserving other string values"""
    return x if x else None

@dataclass
class EngineArgs:
    # Model configuration parameters
    model: str = "facebook/opt-125m"
    model_config_path: Optional[str] = None
    tokenizer: Optional[str] = None
    download_dir: Optional[str] = None
    max_model_len: Optional[int] = None
    tensor_parallel_size: int = 1
    block_size: Optional[int] = None
    task: TaskOption = "generate"
    max_num_seqs: Optional[int] = None
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    speculative_config: Optional[Dict[str, Any]] = None

    # Inference configuration parameters

    # TODO block bs memory profiling 计算得出
    block_bs: float = 5
    block_ratio: float = 0.75
    nnode: int = 1
    pod_ips: Optional[List[str]] = None
    max_cache_task_num: int = 128

    # System configuration parameters
    use_warmup: int = 0
    enable_prefix_caching: bool = False
    use_tqdm_on_load: bool = True

    def __post_init__(self):
        """Post-initialization processing"""
        if not self.tokenizer:
            self.tokenizer = self.model

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add command line interface arguments"""
        # Model parameters group
        model_group = parser.add_argument_group("Model Configuration")
        model_group.add_argument(
            "--model",
            type=str,
            default=EngineArgs.model,
            help="Model name or path"
        )
        model_group.add_argument(
            "--model-config-path",
            type=nullable_str,
            default=EngineArgs.model_config_path,
            help="Path to model configuration file"
        )
        model_group.add_argument(
            "--tokenizer",
            type=nullable_str,
            default=EngineArgs.tokenizer,
            help="Tokenizer name or path (defaults to model path)"
        )
        model_group.add_argument(
            "--download-dir",
            type=nullable_str,
            default=EngineArgs.download_dir,
            help="Directory to download weights"
        )
        model_group.add_argument(
            "--max-model-len",
            type=int,
            default=EngineArgs.max_model_len,
            help="Maximum context length for the model"
        )

        # Parallel processing parameters group
        parallel_group = parser.add_argument_group("Parallel Configuration")
        parallel_group.add_argument(
            "--tensor-parallel-size",
            "-tp",
            type=int,
            default=EngineArgs.tensor_parallel_size,
            help="Tensor parallelism degree"
        )

        # Cluster system parameters group
        system_group = parser.add_argument_group("System Configuration")
        system_group.add_argument(
            "--pod-ips",
            type=lambda s: s.split(",") if s else None,
            default=EngineArgs.pod_ips,
            help="Cluster node IP list (comma-separated)"
        )

        # Performance tuning parameters group
        perf_group = parser.add_argument_group("Performance Tuning")
        perf_group.add_argument(
            "--enable-prefix-caching",
            action=argparse.BooleanOptionalAction,
            default=EngineArgs.enable_prefix_caching,
            help="Enable prefix caching"
        )
        perf_group.add_argument(
            "--max-num-seqs",
            type=int,
            default=EngineArgs.max_num_seqs,
            help="Maximum number of sequences per iteration"
        )

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "EngineArgs":
        """Create instance from command line arguments"""
        return cls(**{
            field.name: getattr(args, field.name)
            for field in dataclass_fields(cls)
        })

    def create_model_config(self) -> ModelConfig:
        """Create model configuration object"""
        return ModelConfig(
            model_name_or_path=self.model,
            config_json_file=self.model_config_path
        )

    def create_engine_config(self) -> Config:
        """Create engine configuration object"""
        return Config(
            model_config=self.create_model_config(),
            model=self.model,
            tensor_parallel_size=self.tensor_parallel_size,
            # Add other required parameters according to actual Config class definition
        )

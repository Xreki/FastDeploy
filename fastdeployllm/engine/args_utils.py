import argparse
import json
from dataclasses import dataclass, fields as dataclass_fields
from typing import Any, Dict, List, Optional

from fastdeployllm.engine.config import Config, ModelConfig, TaskOption
from fastdeployllm.utils import FlexibleArgumentParser



def nullable_str(x: str) -> Optional[str]:
    """Convert empty string to None while preserving other string values"""
    return x if x else None

@dataclass
class EngineArgs:
    # Model configuration parameters
    model: str = "facebook/opt-125m"
    model_config_path: Optional[str] = None
    tokenizer: str = None
    download_dir: str = None
    max_model_len: int = 2048
    tensor_parallel_size: int = 1
    block_size: int = 64
    task: TaskOption = "generate"
    max_num_seqs: int = 8
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
    use_tqdm_on_load: bool = False

    def __post_init__(self):
        """Post-initialization processing"""
        if not self.tokenizer:
            self.tokenizer = self.model

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
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

        model_group.add_argument(
            "--block-size",
            type=int,
            default=EngineArgs.block_size,
            help="one block contain token number"
        )

        model_group.add_argument(
            "--task",
            type=str,
            default=EngineArgs.task,
            help="Task to execute"
        )


        model_group.add_argument(
            "--use-warmup",
            type=int,
            default=EngineArgs.use_warmup,
            help="before inference whether to use warm up"
        )

        model_group.add_argument(
            "--use_tqdm_on_load",
            type=int,
            default=EngineArgs.use_tqdm_on_load,
            help="load model weights with tqdm"
        )

        model_group.add_argument(
            "--mm_processor_kwargs",
            default=None,
            help="mm processorkwargs"
        )

        model_group.add_argument(
            "--speculative_config",
            default=None,
            help="speculative config path"
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
        parallel_group.add_argument(
            "--max-num-seqs",
            type=int,
            default=EngineArgs.max_num_seqs,
            help="Maximum number of sequences per iteration"
        )

        parallel_group.add_argument(
            "--block-bs",
            type=float,
            default=EngineArgs.block_bs
        )

        parallel_group.add_argument(
            "--block-ratio",
            type=float,
            default=EngineArgs.block_ratio
        )

        parallel_group.add_argument(
            "--max_cache_task_num",
            type=int,
            default=EngineArgs.max_cache_task_num,
            help="waiting list max task num"
        )

        # Cluster system parameters group
        system_group = parser.add_argument_group("System Configuration")
        system_group.add_argument(
            "--pod-ips",
            type=lambda s: s.split(",") if s else None,
            default=EngineArgs.pod_ips,
            help="Cluster node IP list (comma-separated)"
        )

        system_group.add_argument(
            "--nnode",
            type=int,
            default=EngineArgs.nnode,
            help="number of nodes"
        )

        # Performance tuning parameters group
        perf_group = parser.add_argument_group("Performance Tuning")
        perf_group.add_argument(
            "--enable-prefix-caching",
            action='store_true',
            default=EngineArgs.enable_prefix_caching,
            help="Enable prefix caching"
        )


        return parser

    @classmethod
    def from_cli_args(cls, args: FlexibleArgumentParser) -> "EngineArgs":
        """Create instance from command line arguments"""
        return cls(**{
            field.name: getattr(args, field.name)
            for field in dataclass_fields(cls)
        })

    def create_model_config(self) -> ModelConfig:
        """Create model configuration object"""
        return ModelConfig(
            model_name_or_path=self.model,
            use_tqdm_on_load=self.use_tqdm_on_load
        )

    def create_engine_config(self) -> Config:
        """Create engine configuration object"""
        model_cfg = self.create_model_config()
        tensor_parallel_size = model_cfg.mp_num if hasattr(model_cfg, "mp_num") else self.tensor_parallel_size
        return Config(
            model=self.model,
            model_config=model_cfg,
            download_dir=self.download_dir,
            max_model_len=self.max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=self.max_num_seqs,
            mm_processor_kwargs=self.mm_processor_kwargs,
            speculative_config=self.speculative_config,
            block_bs=self.block_bs,
            block_ratio=self.block_ratio,
            nnode=self.nnode,
            pod_ips=self.pod_ips,
            max_cache_task_num=self.max_cache_task_num,
            use_warmup=self.use_warmup,
            enable_prefix_caching=self.enable_prefix_caching
        )

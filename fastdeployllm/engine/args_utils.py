import argparse
import json
from dataclasses import dataclass, fields as dataclass_fields
from typing import Any, Dict, List, Optional

from fastdeployllm.engine.config import Config, ModelConfig, CacheConfig, TaskOption
from fastdeployllm.utils import FlexibleArgumentParser

def nullable_str(x: str) -> Optional[str]:
    """
    Convert an empty string to None, preserving other string values.
    """
    return x if x else None

@dataclass
class EngineArgs:
    # Model configuration parameters
    model: str = "facebook/opt-125m"
    """
    The name or path of the model to be used.
    """
    model_config_name: Optional[str] = "config.json"
    """
    The name of the model configuration file.
    """
    tokenizer: str = None
    """
    The name or path of the tokenizer (defaults to model path if not provided).
    """
    max_model_len: int = 2048
    """
    Maximum context length supported by the model.
    """
    tensor_parallel_size: int = 1
    """
    Degree of tensor parallelism.
    """
    block_size: int = 64
    """
    Number of tokens in one processing block.
    """
    task: TaskOption = "generate"
    """
    The task to be executed by the model.
    """
    max_num_seqs: int = 8
    """
    Maximum number of sequences per iteration.
    """
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    """
    Additional keyword arguments for the multi-modal processor.
    """
    speculative_config: Optional[Dict[str, Any]] = None
    """
    Configuration for speculative execution.
    """

    # Inference configuration parameters
    gpu_memory_utilization: float = 0.9
    """
    The fraction of GPU memory to be utilized.
    """
    num_gpu_blocks_override: Optional[int] = None
    """
    Override for the number of GPU blocks.
    """
    max_num_batched_tokens: Optional[int] = None
    """
    Maximum number of tokens to batch together.
    """
    block_ratio: float = 0.75
    """
    Ratio of tokens to process in a block.
    """
    nnode: int = 1
    """
    Number of nodes in the cluster.
    """
    pod_ips: Optional[List[str]] = None
    """
    List of IP addresses for nodes in the cluster.
    """
    max_cached_task_num: int = 128
    """
    Maximum number of tasks in the cache waiting list.
    """

    # System configuration parameters
    use_warmup: int = 0
    """
    Flag to indicate whether to use warm-up before inference.
    """
    enable_prefix_caching: bool = False
    """
    Flag to enable prefix caching.
    """
    engine_worker_queue_port: int = 8002

    def __post_init__(self):
        """
        Post-initialization processing to set default tokenizer if not provided.
        """
        if not self.tokenizer:
            self.tokenizer = self.model

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """
        Add command line interface arguments to the parser.
        """
        # Model parameters group
        model_group = parser.add_argument_group("Model Configuration")
        model_group.add_argument(
            "--model",
            type=str,
            default=EngineArgs.model,
            help="Model name or path to be used."
        )
        model_group.add_argument(
            "--model-config-name",
            type=nullable_str,
            default=EngineArgs.model_config_name,
            help="The model configuration file name."
        )
        model_group.add_argument(
            "--tokenizer",
            type=nullable_str,
            default=EngineArgs.tokenizer,
            help="Tokenizer name or path (defaults to model path if not specified)."
        )
        model_group.add_argument(
            "--max-model-len",
            type=int,
            default=EngineArgs.max_model_len,
            help="Maximum context length supported by the model."
        )
        model_group.add_argument(
            "--block-size",
            type=int,
            default=EngineArgs.block_size,
            help="Number of tokens processed in one block."
        )
        model_group.add_argument(
            "--task",
            type=str,
            default=EngineArgs.task,
            help="Task to be executed by the model."
        )
        model_group.add_argument(
            "--use-warmup",
            type=int,
            default=EngineArgs.use_warmup,
            help="Flag to indicate whether to use warm-up before inference."
        )
        model_group.add_argument(
            "--mm_processor_kwargs",
            default=None,
            help="Additional keyword arguments for the multi-modal processor."
        )
        model_group.add_argument(
            "--speculative_config",
            default=None,
            help="Configuration for speculative execution."
        )

        model_group.add_argument(
            "--engine-worker-queue-port",
            type=int,
            default=8002,
            help="port for engine worker queue"
        )

        # Parallel processing parameters group
        parallel_group = parser.add_argument_group("Parallel Configuration")
        parallel_group.add_argument(
            "--tensor-parallel-size",
            "-tp",
            type=int,
            default=EngineArgs.tensor_parallel_size,
            help="Degree of tensor parallelism."
        )
        parallel_group.add_argument(
            "--max-num-seqs",
            type=int,
            default=EngineArgs.max_num_seqs,
            help="Maximum number of sequences per iteration."
        )
        parallel_group.add_argument(
            "--num-gpu-blocks-override",
            type=int,
            default=EngineArgs.num_gpu_blocks_override,
            help="Override for the number of GPU blocks."
        )
        parallel_group.add_argument(
            "--max-num-batched-tokens",
            type=int,
            default=EngineArgs.max_num_batched_tokens,
            help="Maximum number of tokens to batch together."
        )
        parallel_group.add_argument(
            "--gpu-memory-utilization",
            type=float,
            default=EngineArgs.gpu_memory_utilization,
            help="Fraction of GPU memory to be utilized."
        )
        parallel_group.add_argument(
            "--block-ratio",
            type=float,
            default=EngineArgs.block_ratio,
            help="Ratio of tokens to process in a block."
        )
        parallel_group.add_argument(
            "--max_cached_task_num",
            type=int,
            default=EngineArgs.max_cached_task_num,
            help="Maximum number of tasks in the cache waiting list."
        )

        # Cluster system parameters group
        system_group = parser.add_argument_group("System Configuration")
        system_group.add_argument(
            "--pod-ips",
            type=lambda s: s.split(",") if s else None,
            default=EngineArgs.pod_ips,
            help="List of IP addresses for nodes in the cluster (comma-separated)."
        )
        system_group.add_argument(
            "--nnode",
            type=int,
            default=EngineArgs.nnode,
            help="Number of nodes in the cluster."
        )

        # Performance tuning parameters group
        perf_group = parser.add_argument_group("Performance Tuning")
        perf_group.add_argument(
            "--enable-prefix-caching",
            action='store_true',
            default=EngineArgs.enable_prefix_caching,
            help="Flag to enable prefix caching."
        )

        return parser

    @classmethod
    def from_cli_args(cls, args: FlexibleArgumentParser) -> "EngineArgs":
        """
        Create an instance of EngineArgs from command line arguments.
        """
        return cls(**{
            field.name: getattr(args, field.name)
            for field in dataclass_fields(cls)
        })

    def create_model_config(self) -> ModelConfig:
        """
        Create and return a ModelConfig object based on the current settings.
        """
        return ModelConfig(
            model_name_or_path=self.model,
            config_json_file=self.model_config_name
        )

    def create_cache_config(self) -> CacheConfig:
        """
        Create and return a CacheConfig object based on the current settings.
        """
        return CacheConfig(
            block_size=self.block_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            block_ratio=self.block_ratio,
            enable_prefix_caching=self.enable_prefix_caching
        )

    def create_engine_config(self) -> Config:
        """
        Create and return a Config object based on the current settings.
        """
        model_cfg = self.create_model_config()
        if not model_cfg.is_unified_ckpt and hasattr(model_cfg, 'tensor_parallel_size'):
            self.tensor_parallel_size = model_cfg.tensor_parallel_size
        return Config(
            model_name_or_path=self.model,
            model_config=model_cfg,
            tokenizer=self.tokenizer,
            cache_config=self.create_cache_config(),
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            max_num_seqs=self.max_num_seqs,
            mm_processor_kwargs=self.mm_processor_kwargs,
            speculative_config=self.speculative_config,
            max_num_batched_tokens=self.max_num_batched_tokens,
            nnode=self.nnode,
            pod_ips=self.pod_ips,
            max_cached_task_num=self.max_cached_task_num,
            use_warmup=self.use_warmup,
            engine_worker_queue_port=self.engine_worker_queue_port
        )

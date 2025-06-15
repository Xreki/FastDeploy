"""
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import paddle
from paddlenlp.transformers.configuration_utils import PretrainedConfig

from fastdeploy.model_executor.layers.quantization.quant_base import \
    QuantConfigBase
from fastdeploy.utils import get_logger

logger = get_logger("config", "config.log")

__all__ = [
    "ERNIEBOT_PRETRAINED_INIT_CONFIGURATION",
    "ModelConfig",
    "ErnieBotMoEConfig",
]

ERNIEBOT_PRETRAINED_INIT_CONFIGURATION = {
    "ernie-bot": {
        "hidden_act": "swiglu",
        "hidden_dropout_prob": 0.0,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_layers": 48,
        "max_position_embeddings": 4096,
        "initializer_range": 0.01,
        "type_vocab_size": 4,
        "vocab_size": 100224,
        "use_rope": True,
        "weight_sharing": True,
        "weight_sharing_add_bias": False,
        "sequence_parallel": False,
        "use_flash_attention": False,
        "recompute": False,
        "fused_linear": False,
    }
}


class GenerationPhase(Enum):
    """
    The generation phase of the model.
    """

    PREFILL = 1
    DECODER = 2


class ModelConfig(PretrainedConfig):
    """
    The configuration class to store the configuration of a `LLM`.
    """

    model_type = "ernie_bot"
    pretrained_init_configuration = ERNIEBOT_PRETRAINED_INIT_CONFIGURATION

    max_stop_seqs_num = 5  # int(os.getenv("MAX_STOP_SEQS_NUM", "5"))
    stop_seqs_max_len = 8  # int(os.getenv("STOP_SEQS_MAX_LEN", "8"))

    architectures: list[str] = []

    # NOTE(gongshaotain): form _load_model_init_val()
    top_p = 0.0
    temperature = 1.0
    rope_theta = 10000.0
    rope_scaling = None
    penalty_score = 1.0
    frequency_score = 0.0
    presence_score = 0.0
    min_length = 1

    def __init__(
        self,
        vocab_size: int = 100224,
        hidden_size: int = 4096,
        intermediate_size: Optional[int] = None,
        num_layers: int = 48,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        hidden_act: str = "swiglu",
        hidden_dropout_prob: float = 0.0,
        max_position_embeddings: int = 512,
        max_seq_len: int = 512,
        initializer_range: float = 0.02,
        type_vocab_size: int = 4,
        use_rope=True,
        use_rmsnorm=True,
        weight_sharing=True,
        weight_sharing_add_bias=False,
        sequence_parallel=False,
        use_flash_attention=False,
        use_fast_ffn: bool = False,
        tensor_parallel_output: bool = True,
        fused_linear=False,
        compression_ratio: float = 1.0,
        rope_theta: int = 10000,
        rope_3d: bool = False,
        ori_vocab_size: int | None = None,
        smooth: bool = False,
        group_size: int = -1,
        tools_version="4.10.0.dev",
        system_prompt_version="V1",
        moe_layer_start_index: int | None = None,
        moe_layer_end_index: int | None = None,
        moe_use_gate_correction_bias: bool | None = None,
        num_hidden_layers: int | None = None,
        prefix_name="",
        freeze_embedding=False,
        rope_head_dim=None,
        base_model_prefix=None,
        ffn_hidden_size: Optional[int] = None,
        dtype="bfloat16",
        export_model_type: str = "weight_only_int8",
        use_stop_seqs: bool = False,
        return_all_hidden_states: bool = False,
        start_layer_index: int = 0,
        output_via_mq: bool = True,
        generation_phase: GenerationPhase = GenerationPhase.PREFILL,
        head_dim: Optional[int] = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        if num_hidden_layers is not None:
            self.num_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.weight_sharing = weight_sharing

        self.weight_sharing_add_bias = weight_sharing_add_bias
        self.use_flash_attention = use_flash_attention
        self.use_fast_ffn = use_fast_ffn
        self.tensor_parallel_output = tensor_parallel_output
        self.skip_recompute_ops = dict()
        self.fused_linear = fused_linear
        self.compression_ratio = compression_ratio
        self.rope_theta = rope_theta
        self.ori_vocab_size = ori_vocab_size or vocab_size
        self.smooth = smooth
        self.group_size = group_size
        self.max_seq_len = max_seq_len
        self.tools_version = tools_version
        self.system_prompt_version = system_prompt_version
        self.prefix_name = prefix_name
        self.freeze_embedding = freeze_embedding
        self.rope_head_dim = rope_head_dim
        self.base_model_prefix = base_model_prefix
        if moe_layer_start_index is not None:
            self.moe_layer_start_index = moe_layer_start_index
        if moe_layer_end_index is not None:
            self.moe_layer_end_index = moe_layer_end_index
        elif moe_use_gate_correction_bias is not None:
            self.moe_use_gate_correction_bias = moe_use_gate_correction_bias
        self.ffn_hidden_size = ffn_hidden_size
        self.rope_3d = rope_3d
        self.export_model_type = export_model_type
        self.use_stop_seqs = use_stop_seqs
        self.return_all_hidden_states = return_all_hidden_states
        self.start_layer_index = start_layer_index
        self.output_via_mq = output_via_mq
        self.dtype = dtype
        self.tie_word_embeddings = tie_word_embeddings


# This class will be removed in future and replaced by MoEConfig
class ErnieBotMoEConfig(ModelConfig):
    """ErnieBotMoEConfig Class"""

    model_type = "ernie_bot"
    pretrained_init_configuration = ERNIEBOT_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        moe_num_experts=0,
        use_recompute_moe=False,
        moe_capacity=(),
        moe_layer_interval=2,
        moe_aux_loss_lambda=1e-2,
        moe_z_loss_lambda=1e-4,
        moe_orthogonal_loss_lambda=1e-2,
        sinkhorn_2gate=True,
        sinkhorn_temp=3e-2,
        global_aux_loss=False,
        moe_dropout_prob=0.0,
        moe_group="dummy",
        moe_gate="top2",
        moe_gate_detach: bool = True,
        moe_all_to_all_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # moe
        self.moe_num_experts = moe_num_experts
        self.use_recompute_moe = use_recompute_moe
        self.moe_capacity = moe_capacity
        self.moe_layer_interval = moe_layer_interval
        self.moe_aux_loss_lambda = moe_aux_loss_lambda
        self.moe_z_loss_lambda = moe_z_loss_lambda
        self.moe_orthogonal_loss_lambda = moe_orthogonal_loss_lambda
        self.sinkhorn_2gate = sinkhorn_2gate
        self.sinkhorn_temp = sinkhorn_temp
        self.global_aux_loss = global_aux_loss
        self.moe_dropout_prob = moe_dropout_prob
        self.moe_group = moe_group
        self.moe_gate = moe_gate
        self.moe_gate_detach = moe_gate_detach
        self.moe_all_to_all_dropout = moe_all_to_all_dropout

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        moe config 中还有一些不能被序列化的对象，例如 paddle.distributed.communication.group.Group,
        为此，重写json序列化方法。
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()

        def _serializer(obj):
            if isinstance(obj, paddle.distributed.communication.group.Group):
                return repr(obj)
            raise TypeError(f"Type {type(obj)} is not serializable")

        return (json.dumps(
            config_dict,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_serializer,
        ) + "\n")


@dataclass
class MoEConfig:
    """
    Configuration for MoE.
    """

    use_moe: bool = False
    num_experts: int = -1
    top_k: int = 8
    moe_intermediate_size: int = -1
    num_experts_per_rank: int = -1
    num_experts_start_offset: int = -1
    activation = "swiglu"

    moe_use_gate_correction_bias = False
    moe_every2 = (False, )
    moe_num_shared_experts = (0, )
    moe_layer_start_index = 0
    moe_layer_end_index = None
    moe_use_ffn_shared_weight_and_bias = (False, )
    moe_group = (False, )
    moe_quant_type = "weight_only_int4"
    num_max_dispatch_tokens_per_rank = 256

    has_multimodality: bool = False
    im_patch_id = (
        100295  # multimodality, TODO(liuyuanle): read from config.json
    )


@dataclass
class ParallelConfig:
    """Configuration for the distributed execution."""
    block_size = 16  # The block size for processing.
    sequence_parallel = False  # Whether to enable sequence parallelism.
    use_ep = False  # Whether to enable Expert Parallelism
    moe_group = False  # Whether to enable moe group
    msg_queue_id = 1  # mesage queue id
    use_micro_batch = False  # Whether to enable micro batch
    tensor_parallel_rank = None  # TP rank ID
    tensor_parallel_degree = None  # TP degree
    mp_size = 1  # mp size
    ep_size = 1  # ep size
    column_cut = False  # (bool, optional): The embedding weight distributed on your gpu cards is divided by row or column. Defaults to False means divide by row. When vocab_size can not be divided by world_size but hidden_size can, we can consider split embedding weight by column.
    lm_head_column_cut = False
    """
    From old wersion worker args
    TODO(gongshaotian): Reclassify
    """
    model_name_or_path: str = "./output"
    max_num_seqs: int = 34
    # Set default block num for profile run
    max_block_num: int = 2000
    # block size
    block_size: int = 64
    # Engine worker queue port
    engine_worker_queue_port: int = 9923
    # Max model len
    max_model_len: int = 3072  # max_seq_len
    # cuda visible devices
    device_ids: str = "0"
    # Input dtype
    dtype: str = "bfloat16"
    # Encoder's decoder num
    enc_dec_block_num: int = 1
    # KV cache ratio for input
    kv_cache_ratio: float = 0.7
    # First token id
    first_token_id: int = 1
    # Gpu memory utilization
    gpu_memory_utilization: float = 0.9
    # Process ID of engine
    engine_pid: Optional[int] = None
    # Do profile or not
    do_profile: bool = False
    # Dynamic load weight or not
    dynamic_load_weight: bool = False
    #
    pad_token_id: int = -1
    #
    eos_tokens_lens: int = 2
    # Enable chunked prefill
    enable_chunked_prefill: str = "store_true"
    """

    - autoregressive:
    - inference_with_reference:
    - draft_model:
    - hydra:
    - eagle:
    """
    speculate_method: str = None
    """
    - APPEND_ATTN:
    """
    attention_backend: str = "APPEND_ATTN"

    # speculate_max_draft_tokens
    speculate_max_draft_tokens: int = 1
    max_num_batched_tokens: int = 2048
    # enable prefix cache
    enable_prefix_caching = None
    # splitwise role
    splitwise_role: str = "mixed"


@dataclass
class SpeculativeConfig:
    """
    Configuration for speculative decoding.
    """
    speculate_method = None  # speculate method
    # the max length of draft tokens for speculate method
    speculate_max_draft_token_num = 1  # speculate_max_draft_tokens
    draft_type = "None"  # draft type
    is_mtp = False  # is mtp
    speculate_max_candidate_len = 5  # the max length of candidate tokens for speculate method
    speculate_verify_window = 2  # the max length of verify window for speculate method


@dataclass
class DeviceConfig:
    """
    Configuration for device settings.
    """
    device_type = "cuda"


@dataclass
class AdditionalConfig:
    """
    Configuration for testing, debugging or others
    """

    use_fake_parameter = False  # use fake parameter for test
    ep_just_for_test = True  # whether to use ep just for test
    fake_server_p = False  # whether to use fake server


class WeightKeys:
    """
    The parameter keys stored in your model_state.padarams.
    """

    def __init__(self, num_layers):
        """
        Initialization keys retrive weight from model_state.padarams.

        Args:
        num_layers (int): Number of layers in the Transformer model.
        Returns:
        None
        """
        self.norm_before_qkv_weight_keys = [None for i in range(num_layers)]
        self.norm_before_qkv_bias_keys = [None for i in range(num_layers)]
        self.qkv_linear_weight_keys = [None for i in range(num_layers)]
        self.qkv_linear_bias_keys = [None for i in range(num_layers)]
        self.out_linear_weight_keys = [None for i in range(num_layers)]
        self.out_linear_bias_keys = [None for i in range(num_layers)]

        self.ffn_layernorm_weight_keys = [None for i in range(num_layers)]
        self.ffn_layernorm_bias_keys = [None for i in range(num_layers)]
        self.ffn1_weight_keys = [None for i in range(num_layers)]
        self.ffn1_bias_keys = [None for i in range(num_layers)]
        self.ffn2_weight_keys = [None for i in range(num_layers)]
        self.ffn2_bias_keys = [None for i in range(num_layers)]

        self.moe_gate_weight_keys = None
        self.moe_gate_correction_bias_keys = None
        self.moe_ffn1_weight_keys = None
        self.moe_ffn2_weight_keys = None
        self.moe_ffn1_bias_keys = None
        self.moe_ffn2_bias_keys = None

        self.moe_ffn1_weight_scale_key = None
        self.moe_ffn2_weight_scale_key = None
        self.moe_ffn1_expert_in_scale_key = None
        self.moe_ffn2_expert_in_scale_key = None


class GraphOptimizationConfig:
    """The Top-level graph optimization contral corresponds to different backends.
    - 0: dyncmic graph
    - 1: static graph
    - 2: static graph + cinn compilation backend
    """
    graph_opt_level: int = 0

    # CUDA Graph Config
    """ Whether to use cudagraph.
    - Fasle: cudagraph is not used.
    - True: cudagraph is used.
        It requires that all input buffers have fixed addresses, and all
        splitting ops write their outputs to input buffers.
        - With dyncmic graph backend: ...
        - With static grpah backend: WIP
    """
    use_cudagraph: bool = False
    """Sizes to capture cudagraph.
    - None (default): capture sizes are inferred from llm config.
    - list[int]: capture sizes are specified as given."""
    cudagraph_capture_sizes: Optional[list[int]] = None
    """ Number of warmup runs for cudagraph. """
    cudagraph_num_of_warmups: int = 2
    """Whether to copy input tensors for cudagraph.
    If the caller can guarantee that the same input buffers
    are always used, it can set this to False. Otherwise, it should
    set this to True."""
    cudagraph_copy_inputs: bool = False
    """ In static graph, this is an operation list that does not need to be captured by the CUDA graph.
    CudaGraphBackend will split these operations from the static graph.
    Example usage:
        cudagraph_splitting_ops = ["paddle.unified_attention"]

    Note: If want to use subgraph capture functionality in a dynamic graph,
    can manually split the model into multiple layers and apply the @support_cuda_graph decorator
    only to the layer where CUDA graph functionality is required.
    """
    cudagraph_splitting_ops = Optional[list[str]]
    """"whether to use a full cuda graph for the entire forward pass rather than
    splitting certain operations such as attention into subgraphs.
    Thus this flag cannot be used together with splitting_ops."""
    full_cuda_graph: bool = False

    max_capture_size: int = field(default=None, init=False)  # type: ignore
    batch_size_to_captured_size: dict[int,
                                      int] = field(default=None,
                                                   init=False)  # type: ignore

    # CINN Config ...

    def init_with_cudagrpah_size(self,
                                 cudagraph_capture_sizes: list[int]) -> None:
        """To complete the initialization of config,
        we need to know the cudagraph sizes"""
        if self.cudagraph_capture_sizes is None:
            self.cudagraph_capture_sizes = cudagraph_capture_sizes
        else:
            dedup_sizes = list(set(self.cudagraph_capture_sizes))
            if len(dedup_sizes) < len(self.cudagraph_capture_sizes):
                logger.info(("cudagraph sizes specified by model runner"
                             " %s is overridden by config %s"),
                            cudagraph_capture_sizes, dedup_sizes)
            self.cudagraph_capture_sizes = dedup_sizes

        # sort to make sure cudagraph capture sizes are in descending order
        self.cudagraph_capture_sizes.sort(reverse=True)
        self.max_capture_size = self.cudagraph_capture_sizes[
            0] if self.cudagraph_capture_sizes else 0

        # pre-compute the mapping from batch size to padded graph size
        self.batch_size_to_captured_size = [
            0 for i in range(self.max_capture_size + 1)
        ]
        for end, start in zip(self.cudagraph_capture_sizes,
                              self.cudagraph_capture_sizes[1:] + [0]):
            for bs in range(start, end):
                if bs == start:
                    self.batch_size_to_captured_size[bs] = start
                else:
                    self.batch_size_to_captured_size[bs] = end
        self.batch_size_to_captured_size[
            self.max_capture_size] = self.max_capture_size


@dataclass
class LoadConfig:
    """
    Configuration for loading parameter
    """
    model_path: str = None  # The path to the model file.
    weight_keys: Optional[
        WeightKeys] = None  # Keys stored in your model, which is used to retrieve weights from the state dict.
    scale_dir: str = None  # The directory where the scale file is located.

    act_scales = None
    bias_keys = None

    def _post_init(self, model_config):
        if self.weight_keys:
            self.weight_keys_mapping = self._create_weight_key_by_layer_name(
                model_config)
        else:
            self.weight_keys_mapping = {}
        self.quant_scale_mapping = self._create_quant_scale_mapping(
            model_config)

    def _create_weight_key_by_layer_name(self, model_config) -> dict:
        mapping = {}
        weight_keys = self.weight_keys

        num_layers = model_config.num_layers
        for i in range(num_layers):
            if i == 0:
                layer_name = f"{model_config.base_model_prefix}.decoder.layers.0.norm1"
                mapping[layer_name] = weight_keys.norm_before_qkv_weight_keys[
                    0]
            if i < num_layers:
                layer_name = f"{model_config.base_model_prefix}.decoder.layers.{i}.norm2"
                mapping[layer_name] = weight_keys.ffn_layernorm_weight_keys[i]

        for i in range(num_layers - 1):
            layer_name = f"{model_config.base_model_prefix}.decoder.layers.{i+1}.norm1"
            mapping[layer_name] = weight_keys.norm_before_qkv_weight_keys[i +
                                                                          1]

        layer_name = f"{model_config.base_model_prefix}.decoder.norm"
        mapping[layer_name] = "ernie.norm.weight"

        layer_name = f"{model_config.base_model_prefix}.e_norm"
        mapping[layer_name] = f"{model_config.base_model_prefix}.e_norm.weight"
        layer_name = f"{model_config.base_model_prefix}.h_norm"
        mapping[layer_name] = f"{model_config.base_model_prefix}.h_norm.weight"

        return mapping

    def _create_quant_scale_mapping(self, model_config) -> dict:
        mapping = {}
        act_scales = self.act_scales
        num_layers = model_config.num_layers
        for i in range(num_layers):
            if i == 0:
                layer_name = f"{model_config.base_model_prefix}.decoder.layers.0.norm1"
                mapping[layer_name] = act_scales.get(
                    f"{model_config.base_model_prefix}.decoder.layers.0.self_attn.qkv_proj.activation_quanter",
                    -1)
            if i < num_layers:
                layer_name = f"{model_config.base_model_prefix}.decoder.layers.{i}.norm2"
                mapping[layer_name] = act_scales.get(
                    f"{model_config.base_model_prefix}.decoder.layers.{i}.linear1.activation_quanter",
                    -1)

        for i in range(num_layers - 1):
            layer_name = f"{model_config.base_model_prefix}.decoder.layers.{i+1}.norm1"
            mapping[layer_name] = act_scales.get(
                f"{model_config.base_model_prefix}.decoder.layers.{i + 1}.self_attn.qkv_proj.activation_quanter",
                -1)

        return mapping

    def get_weight_key_by_layer_name(self, layer_name: str) -> Optional[str]:
        return self.weight_keys_mapping.get(layer_name)

    def get_quant_scale_by_layer_name(self, layer_name: str) -> Optional[int]:
        return self.quant_scale_mapping.get(layer_name)


@dataclass
class LoRAConfig:
    """ LoRA Config """
    pass


@dataclass
class SchedulerConfig:
    """ Scheduler Config """
    pass


@dataclass
class KVCacheConfig:
    """ KV Cache Config """
    block_size: int = 0
    enc_dec_block_num: int = 2
    kv_cache_ratio: float = 0.75
    dtype: str = 'bfloat16'
    kvcache_quant_config: Optional[QuantConfigBase] = None
    cache_quant_dtype: str = "none"


class TmpConfig:
    """
    TODO(yuanrisheng):TmpConfig will be moved to other config class when refactor work is relatively complete.
    """
    cache_quant_dtype: str = "default"
    has_zero_point: bool = False
    is_channel_wise: bool = False
    weight_block_size: int = 16
    use_offline_quant: bool = False


@dataclass
class DecodingConfig:
    """
    Configuration for decoding
    """
    max_dec_len = 20
    min_dec_len = 0
    decode_strategy = "sampling"
    bos_token_id = None
    pad_token_id = None
    num_return_sequences: int = 1


@dataclass
class FDConfig:
    """
    The configuration class which contains all fastdeploy-related configuration. This
    simplifies passing around the distinct configurations in the codebase.
    """
    model_config: ModelConfig = field(default=None, init=True)  # type: ignore

    parallel_config: ParallelConfig = field(default=None, init=True)
    speculative_config: SpeculativeConfig = field(default=None,
                                                  init=True)  # type: ignore
    device_config: DeviceConfig = field(default=None,
                                        init=True)  # type: ignore
    additional_config: AdditionalConfig = field(default=None,
                                                init=True)  # type: ignore
    load_config: LoadConfig = field(default=None, init=True)  # type: ignore
    quant_config: Optional[QuantConfigBase] = None
    graph_opt_config: Optional[GraphOptimizationConfig] = None
    tmp_config: TmpConfig = field(default=None, init=True)
    moe_config: MoEConfig = field(default=None, init=True)  # type: ignore
    decoding_config: DecodingConfig = field(default=None,
                                            init=True)  # type: ignore
    kv_cache_config: KVCacheConfig = field(default=None,
                                           init=True)  # type: ignore

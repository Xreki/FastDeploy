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

# cipher_token=WjI1fQOvhN  # do not edit this line


from __future__ import annotations

import json
from typing import Optional

from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.utils.log import logger

import paddle
import os

__all__ = [
    "ERNIEBOT_PRETRAINED_INIT_CONFIGURATION",
    "ErnieBotConfig",
    "ErnieBotMoEConfig",
    "ERNIEBOT_PRETRAINED_RESOURCE_FILES_MAP",
]

ERNIEBOT_PRETRAINED_INIT_CONFIGURATION = {
    "ernie-bot": {
        "attention_probs_dropout_prob": 0.0,
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
        "recompute_granularity": "core_attn",
        "fuse_attn_qkv": True,
        "fused_linear": False,
        "scale_qk_coeff": 1.0,
        "fused_softmax_with_triangular": True,
        "fused_rotary": False,
        "fused_softmax_mask": False,
    }
}


ERNIEBOT_PRETRAINED_RESOURCE_FILES_MAP = {"model_state": {"ernie-bot": ""}}


class ErnieBotConfig(PretrainedConfig):
    """
    The configuration class to store the configuration of a `ErnieBot`.
    """

    model_type = "ernie_bot"
    pretrained_init_configuration = ERNIEBOT_PRETRAINED_INIT_CONFIGURATION

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
        attention_probs_dropout_prob: float = 0.0,
        max_position_embeddings: int = 512,
        max_sequence_length: int = 512,
        initializer_range: float = 0.01,
        type_vocab_size: int = 4,
        use_rope=True,
        use_rmsnorm=False,
        weight_sharing=True,
        weight_sharing_add_bias=False,
        sequence_parallel=False,
        use_flash_attention=False,
        use_fast_ln=False,
        use_fast_ffn: bool = False,
        tensor_parallel_output: bool = True,
        recompute=False,
        recompute_granularity="core_attn",
        pp_recompute_interval=1,
        no_recompute_layers=None,
        recompute_use_reentrant=False,
        refined_recompute=dict(),
        virtual_pp_degree=1,
        pp_seg_method="layer:TransformerDecoderLayer|EmptyLayer",
        fuse_attn_qkv=True,
        fused_linear=False,
        use_sparse_flash_attn=True,
        use_sparse_head_and_loss_fn=False,
        use_fused_head_and_loss_fn=False,
        scale_qk_coeff=1.0,
        fused_softmax_with_triangular=True,
        fused_rotary=False,
        fused_softmax_mask=False,
        fused_mt=False,
        compression_ratio: float = 1.0,
        rope_theta: int = 10000,
        ori_vocab_size: int | None = None,
        cachekv_quant: bool = False,
        smooth: bool = False,
        group_size: int = -1,
        dpo_config=None,
        kto_config=None,
        tools_version="4.10.0.dev",
        only_hidden_states=False,
        add_tail_layer=False,
        use_var_len_flash_attn=False,
        system_prompt_version="V1",
        moe_layer_start_index: int | None = None,
        moe_intermediate_sizes: int | None = None,
        moe_use_gate_correction_bias: bool | None = None,
        moe_gate_correct_bias: bool | None = None,
        num_hidden_layers: int | None = None,
        weight_block_size=[-1, -1],
        moe_use_ffn_shared_weight_and_bias=False,
        moe_intermediate_size: int | None = None,
        moe_use_aux_free: bool | None = None,
        is_quantized: bool  = False,
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
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.weight_sharing = weight_sharing
        self.weight_sharing_add_bias = weight_sharing_add_bias
        self.sequence_parallel = sequence_parallel
        self.use_flash_attention = use_flash_attention
        self.use_fast_ln = use_fast_ln
        self.use_fast_ffn = use_fast_ffn
        self.tensor_parallel_output = tensor_parallel_output
        self.recompute = recompute
        self.recompute_granularity = recompute_granularity
        self.pp_recompute_interval = pp_recompute_interval
        self.no_recompute_layers = no_recompute_layers
        self.recompute_use_reentrant = recompute_use_reentrant
        self.refined_recompute = refined_recompute
        """
        `refined_recompute` 内容为一个dict:[op_name, skip_num], 目前只在PP模式下才生效。
            在PP中会根据`refined_recompute` 填充`self.skip_recompute_ops`。
            `op_name` 选择范围是："mlp_row_ln", "attention_row_ln", "attention_column_ln",
                                   "mlp_column_ln", "flash_attn"
            `skip_num` 表示选择不进行重计算的次数。
            0表示 0次不重计算，也就是全部都重计算，显存最少。
            -1表示全部不重计算，显存最多。
            还可以填 【0，1，。。。，12】中的任意值，进行调整次数。
            大于等于12。相当于 -1取值
        """
        self.skip_recompute_ops = dict()
        self.virtual_pp_degree = virtual_pp_degree
        self.pp_seg_method = pp_seg_method
        self.fuse_attn_qkv = fuse_attn_qkv
        self.fused_linear = fused_linear
        self.use_sparse_flash_attn = use_sparse_flash_attn
        self.use_sparse_head_and_loss_fn = use_sparse_head_and_loss_fn
        self.use_fused_head_and_loss_fn = use_fused_head_and_loss_fn
        self.scale_qk_coeff = scale_qk_coeff
        self.fused_softmax_with_triangular = fused_softmax_with_triangular
        self.fused_mt = fused_mt
        self.compression_ratio = compression_ratio
        self.rope_theta = rope_theta
        self.ori_vocab_size = ori_vocab_size or vocab_size
        self.fused_rotary = fused_rotary
        self.fused_softmax_mask = fused_softmax_mask
        self.cachekv_quant = cachekv_quant
        self.smooth = smooth
        self.group_size = group_size
        self.max_sequence_length = max_sequence_length
        self.dpo_config = dpo_config
        self.kto_config = kto_config
        self.tools_version = tools_version
        self.only_hidden_states = only_hidden_states
        self.add_tail_layer = add_tail_layer
        self.use_var_len_flash_attn = use_var_len_flash_attn
        self.system_prompt_version = system_prompt_version
        self.moe_use_ffn_shared_weight_and_bias = moe_use_ffn_shared_weight_and_bias
        if moe_layer_start_index is not None:
            self.moe_layer_start_index = moe_layer_start_index
        if moe_intermediate_sizes is not None:
            self.moe_intermediate_size = moe_intermediate_sizes
        if moe_intermediate_size is not None:
            self.moe_intermediate_size = moe_intermediate_size
        if moe_gate_correct_bias is not None:
            self.moe_use_gate_correction_bias = moe_gate_correct_bias
        elif moe_use_gate_correction_bias is not None:
            self.moe_use_gate_correction_bias = moe_use_gate_correction_bias
        elif moe_use_aux_free is not None:
            self.moe_use_gate_correction_bias = moe_use_aux_free
        self.weight_block_size = weight_block_size
        self.is_quantized = is_quantized
        self.register_unsavable_keys(
            [
                "refined_recompute",
                "skip_recompute_ops",
                "dpo_config",
                "kto_config",
                "use_var_len_flash_attn",
            ]
        )
        if not hasattr(self, "head_dim"):
            self.head_dim = self.hidden_size // self.num_attention_heads


class ErnieBotMoEConfig(ErnieBotConfig):
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
        if use_recompute_moe:
            logger.warning(
                "set `use_recompute_moe`=True, disabling `recompute_granularity=full`, change to full_attn."
            )
            if kwargs["recompute_granularity"] == "full":
                kwargs["recompute_granularity"] = "full_attn"
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

        return (
            json.dumps(
                config_dict,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                default=_serializer,
            )
            + "\n"
        )


class QuantizationConfig:
    """
    The quantization class to use for offline quantization.
    """

    def __init__(
        self, quantization_type, groupsize=-1, scale_dtype="float16", arch=None
    ):
        """
            Initializes the Quantizer class with the given parameters.
        
        Args:
            quantization_type (str): Type of quantization to be used. Supported values are
                'uniform', 'tensor_abs_max', and 'channel_wise_abs_max'.
            groupsize (int, optional): Number of channels in each group for channel-wise
                quantization. Defaults to -1, which means all channels will be grouped together.
            scale_dtype (str, optional): Data type of the scales used in quantization. Defaults to
                "float16".
            arch (int, optional): Architecture of the model. If not specified, it will be set based
                on the environment variable FLAGS_weight_only_linear_arch. Defaults to None.
        
        Raises:
            ValueError: If the value of `quantization_type` is not one of 'uniform',
                'tensor_abs_max', or 'channel_wise_abs_max'.
        """
        self.quantization_type = quantization_type
        self.groupsize = groupsize
        self.scale_dtype = scale_dtype
        if arch is not None:
            self.arch = arch
        elif os.getenv("FLAGS_weight_only_linear_arch") is not None:
            self.arch = int(os.getenv("FLAGS_weight_only_linear_arch"))
        else:
            self.arch = None

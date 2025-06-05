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

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet
from paddle.framework import in_dynamic_mode
from paddle.incubate.nn.functional import blha_get_max_len
from paddlenlp.utils.log import logger

import fastdeploy

try:
    from paddle.base.core import EventHandle
    from paddle.distributed.communication import deep_ep
except ImportError:
    logger.warning("import EventHandle and deep_ep Failed!")

from fastdeploy.inference_args import GenerationPhase

from ..layers.activation import SiluAndMul
from ..layers.linear import (MergedColumnParallelLinear, QKVParallelLinear,
                             RowParallelLinear)
from ..layers.normalization import LayerNorm, RMSNorm
from .micro_batch_control import MicroBatchControl
from typing import Optional
from fastdeploy.model_executor.eplb.experts_manager import RedundantExpertManger

EP_MICRO_BATCH_NUM = 2  # DeepEP can only support


class FusedTransformer(nn.Layer):
    """
    FusedTransformer
    """

    def __init__(
        self,
        inference_args,
        fmt_keys,
        act_method="gelu",
        epsilon=1e-5,
        ffn1_concat=True,
        use_smooth_quant=True,
        rope_theta=10000.0,
        rope_3d=False,
        use_neox_rotary_style=False,
        fuse_ffn_act=False,
        ring_id=-1,
        return_all_hidden_states=False,
        base_model_prefix="gpt",
        draft_type="",
        llm_config=None,
        max_len=32768,
        redundant_table_manger: Optional[RedundantExpertManger] = None,
    ):
        """
        Initialize the fused transformer model.

        Args:
            inference_args (InferenceArgs): Configuration arguments for model inference.
            fmt_keys (FMTKeys): Keys stored in your model, which is used to retrieve weights from the state dict.
            act_method (str, optional): Activation method, defaults to "gelu".
            epsilon (float, optional): Epsilon value for normalization layers, defaults to 1e-5.
            ffn1_concat (bool, optional): Whether to concatenate FFN1 layer outputs, defaults to True.
            use_smooth_quant (bool, optional): Whether to use smooth quantization, defaults to True.
            use_neox_rotary_style (bool, optional): Whether to use NeoX rotary position encoding, defaults to False.
            fuse_ffn_act (bool, optional): Whether to fuse FFN and activation layers, defaults to False.
            ring_id (int, optional): Ring ID for multi-process parallel training, defaults to -1.
        """
        super().__init__()
        self.inference_args = inference_args

        if self.inference_args.use_ep and self.inference_args.use_micro_batch:
            self.use_micro_batch = True
            self.micro_batch_control = MicroBatchControl()
            logger.info("use_micro_batch is enable")
        else:
            self.use_micro_batch = False

        self.num_layers = inference_args.num_layers
        self.act_scales = inference_args.act_scale_dict
        self.fuse_ffn_act = fuse_ffn_act
        self.rank = inference_args.mp_rank
        self.nranks = inference_args.mp_size
        self.num_heads = inference_args.num_attention_heads // self.nranks
        self.kv_num_heads = inference_args.num_key_value_heads // self.nranks
        self.return_all_hidden_states = return_all_hidden_states
        self.use_pd_disaggregation = int(
            os.getenv("FLAGS_use_pd_disaggregation", 0))
        self.use_fa3 = int(os.getenv("FLAGS_use_fa3", 0))
        self.keep_pd_step_flag = draft_type in ["mtp", "eagle"]
        self.num_dense_layers = min(
            self.inference_args.moe_config.moe_layer_start_index, self.num_layers
        )
        self.splitwise_role = os.getenv("SPLITWISE_ROLE", "mixed")
        assert self.splitwise_role in [
            "prefill", "decode", "mixed"
        ], (f"Invalid role: {self.splitwise_role}. " +
            "Expected one of ['prefill', 'decode', 'mixed'].")

        if self.nranks > 1:
            assert ring_id != -1

        self.norm_before_qkv = RMSNorm(
            llm_config,
            hidden_size=llm_config.model_config.hidden_size,
            eps=epsilon,
            prefix=llm_config.load_config.get_weight_key_by_layer_name(
            f"{base_model_prefix}.decoder.layers.0.norm1").rpartition('.')[0],
            quant_scale=llm_config.load_config.get_quant_scale_by_layer_name(
            f"{base_model_prefix}.decoder.layers.0.norm1")
        )

        self.qkv_linear_layers = nn.LayerList([
            QKVParallelLinear(
                llm_config=llm_config,
                prefix=fmt_keys.qkv_linear_weight_keys[i].rpartition('.')[0],
                with_bias=fmt_keys.qkv_linear_bias_keys[i] is not None,
            ) for i in range(self.num_layers)
        ])
        self.out_linear_layers = nn.LayerList([
            RowParallelLinear(
                llm_config=llm_config,
                prefix=fmt_keys.out_linear_weight_keys[i].rpartition('.')[0],
                with_bias=fmt_keys.out_linear_bias_keys[i] is not None,
                input_size=self.num_heads *
                (llm_config.model_config.hidden_size //
                 llm_config.model_config.num_attention_heads),
                output_size=llm_config.model_config.hidden_size,
            ) for i in range(self.num_layers)
        ])
        if not self.use_micro_batch:
            from fastdeploy.model_executor.layers.attention import Attention

            self.attn_layers = nn.LayerList([
                Attention(
                    llm_config=llm_config,
                    layer_id=i,
                    qkv_bias=(None if not (inference_args.weight_dtype == "int8" and inference_args.act_dtype == "int8")
                              else getattr(self.qkv_linear_layers[i], "qkv_bias", None)),
                    qkv_scale=(getattr(self.qkv_linear_layers[i], "qkv_out_scale", None)
                               if inference_args.weight_dtype == "int8" and inference_args.act_dtype == "int8"
                               else None),
                    layer_name=(
                        f"ernie.mtp_block.{i}.self_attn"
                        if draft_type in ["mtp", "eagle"]
                        else f"ernie.layers.{i}.self_attn"
                    ),
                )
                for i in range(self.num_layers)
            ])
        else:
            from ..layers.attention.base import Attention
            self.attn_layers = nn.LayerList([
                Attention(
                    inference_args=inference_args,
                    layer_name=(
                        f"ernie.layers.{i}.self_attn"
                        if self.inference_args.moe_config.use_moe and
                        self.inference_args.moe_config.moe_layer_start_index > 0
                        else f"{base_model_prefix}.decoder.layers.{i}.self_attn"),
                    rope_theta=rope_theta,
                    rope_3d=rope_3d,
                    use_neox_rotary_style=use_neox_rotary_style,
                    out_scale=self.act_scales.get(
                        f"{base_model_prefix}.decoder.layers.{i}.self_attn.out_proj.activation_quanter",
                        -1,
                    ),
                    qkv_scale=getattr(self.qkv_linear_layers[i], "qkv_out_scale",
                                      None),
                    qkv_bias=getattr(
                        self.qkv_linear_layers[i], "qkv_bias", None),
                    linear_shift=getattr(self.out_linear_layers[i], "linear_shift",
                                         None),
                    linear_smooth=getattr(self.out_linear_layers[i],
                                          "linear_smooth", None),
                ) for i in range(self.num_layers)
            ])

        self.ffn_layernorm_layers = nn.LayerList([
            RMSNorm(
                llm_config,
                hidden_size=llm_config.model_config.hidden_size,
                eps=epsilon,
                prefix=llm_config.load_config.get_weight_key_by_layer_name(
                f"{base_model_prefix}.decoder.layers.{i}.norm2").rpartition('.')[0],
                quant_scale=llm_config.load_config.get_quant_scale_by_layer_name(
                f"{base_model_prefix}.decoder.layers.{i}.norm2"),
                linear_bias=getattr(self.out_linear_layers[i], "linear_bias",
                                    None),
            ) for i in range(self.num_layers)
        ])

        self.ffn1_layers = nn.LayerList([
            MergedColumnParallelLinear(
                llm_config=llm_config,
                prefix=fmt_keys.ffn1_weight_keys[i].rpartition('.')[0],
                with_bias=fmt_keys.ffn1_bias_keys[i] is not None,
                activation=act_method,
                use_fast_ffn=(
                            False
                            if not (
                                self.inference_args.moe_config.use_moe
                                and not self.inference_args.moe_config.moe_use_ffn_shared_weight_and_bias
                                or not self.inference_args.moe_config.use_moe
                            )
                            else True
                        ),
            ) for i in range(self.num_layers if not (
                self.inference_args.moe_config.use_moe and not self.
                inference_args.moe_config.moe_use_ffn_shared_weight_and_bias
            ) else self.num_dense_layers)
        ])

        self.ffn2_layers = nn.LayerList([
            RowParallelLinear(
                llm_config=llm_config,
                prefix=fmt_keys.ffn2_weight_keys[i].rpartition('.')[0],
                with_bias=fmt_keys.ffn2_bias_keys[i] is not None,
                input_size=(llm_config.model_config.ffn_hidden_size //
                            self.nranks),
                output_size=llm_config.model_config.hidden_size,
            ) for i in range(self.num_layers if not (
                self.inference_args.moe_config.use_moe and not self.
                inference_args.moe_config.moe_use_ffn_shared_weight_and_bias
            ) else self.num_dense_layers)
        ])
        if not self.fuse_ffn_act:
            self.bias_act_layers = nn.LayerList([
                SiluAndMul(
                    inference_args=inference_args,
                    bias=getattr(self.ffn1_layers[i], "ffn1_bias", None),
                    act_method=act_method,
                    dequant_scales=getattr(self.ffn1_layers[i],
                                           "ffn1_out_scale", None),
                    shift=getattr(self.ffn2_layers[i], "linear_shift", None),
                    smooth=getattr(self.ffn2_layers[i], "linear_smooth", None),
                    quant_scale=self.act_scales.get(
                        f"{base_model_prefix}.decoder.layers.{i}.linear2.activation_quanter",
                        -1,
                    ),
                ) for i in range(self.num_layers if not (
                    self.inference_args.moe_config.use_moe
                    and not self.inference_args.moe_config.
                    moe_use_ffn_shared_weight_and_bias
                ) else self.num_dense_layers)
            ])
        self.moe_layers = None
        self.decoder_ep_runner = []
        if inference_args.moe_config.use_moe:
            if self.inference_args.moe_config.moe_use_ffn_shared_weight_and_bias:
                from ..layers.moe.moe import MoELayer

                self.moe_layers = nn.LayerList([
                    MoELayer(
                        inference_args=inference_args,
                        moe_config=inference_args.moe_config,
                        layer_name=f"moe_layers.{i}",
                        gate_weight_key=f"ernie.decoder.moe_layers.{i}.gate_weight",
                        ffn1_expert_weight_key=f"ernie.decoder.moe_layers.{i}.moe_ffn1_weight",
                        ffn2_expert_weight_key=f"ernie.decoder.moe_layers.{i}.moe_ffn2_weight",
                        ffn1_bias_key=f"ernie.decoder.moe_layers.{i}.moe_ffn1_bias",
                        ffn2_bias_key=f"ernie.decoder.moe_layers.{i}.moe_ffn2_bias",
                        layer_idx=i,
                    ) for i in range(self.num_layers)
                ])
            elif self.inference_args.use_ep:
                if self.inference_args.generation_phase == GenerationPhase.PREFILL:
                    logger.debug("Initializing Prefill EP Layer")
                    from ..layers.moe.ep import PrefillMoeEPLayer as MoELayer
                elif self.inference_args.generation_phase == GenerationPhase.DECODER:
                    logger.debug("Initializing Decoder EP Layer")
                    from ..layers.moe.ep import DecoderMoeEPLayer as MoELayer
                    from ..layers.moe.ep import DecoderEPMicroBatchRunner
                from ..layers.moe.ep import DeepEPEngine

                # For EP
                # new group for ep
                world_size = dist.get_world_size()
                ep_group = paddle.distributed.new_group(range(world_size))
                ep_size = ep_group.nranks
                ep_rank = ep_group.rank
                num_experts = inference_args.moe_config.num_experts
                num_max_dispatch_tokens_per_rank = (
                    inference_args.moe_config.num_max_dispatch_tokens_per_rank)
                num_local_experts = (
                    num_experts + inference_args.redundant_experts_num
                ) // ep_size
                if draft_type not in ["mtp", "eagle"]:
                    ep_engine = DeepEPEngine(
                        ep_group,
                        ep_size,
                        ep_rank,
                        num_max_dispatch_tokens_per_rank,
                        inference_args.hidden_size,
                        num_experts + inference_args.redundant_experts_num,
                        self.inference_args.generation_phase,
                        async_finish=self.inference_args.use_micro_batch,
                    )
                else:
                    ep_engine = None
                self.moe_layers = nn.LayerList([
                    None for i in range(self.num_dense_layers)
                ] + [
                    MoELayer(
                        ep_engine=ep_engine,
                        num_local_experts=num_local_experts,
                        inference_args=inference_args,
                        moe_config=inference_args.moe_config,
                        layer_name=f"moe_layers.{i}",
                        gate_weight_key=f"ernie.layers.{i}.mlp.gate.weight",
                        ffn1_expert_weight_key=f"ernie.layers.{i}.mlp.experts"
                        + ".{}.up_gate_proj.weight",
                        ffn2_expert_weight_key=f"ernie.layers.{i}.mlp.experts"
                        + ".{}.down_proj.weight",
                        ffn1_expert_weight_scale_key=f"ernie.layers.{i}.mlp.experts" +
                        ".{}.up_gate_proj.weight_quanter",
                        ffn2_expert_weight_scale_key=f"ernie.layers.{i}.mlp.experts" +
                        ".{}.down_proj.weight_quanter",
                        ffn1_expert_in_scale_key=f"ernie.layers.{i}.mlp.experts"
                        + ".{}.up_gate_proj.activation_quanter",
                        ffn2_expert_in_scale_key=f"ernie.layers.{i}.mlp.experts"
                        + ".{}.down_proj.activation_quanter",
                        gate_correction_bias_key=f"ernie.layers.{i}.mlp.moe_statics.e_score_correction_bias",
                        ffn1_bias_key=None,
                        ffn2_bias_key=None,
                        ffn1_shared_weight_key=None,
                        ffn1_shared_bias_key=None,
                        ffn2_shared_weight_key=None,
                        ffn2_shared_bias_key=None,
                        layer_idx=i,
                        redundant_table_manger=redundant_table_manger,
                    ) for i in range(
                        self.num_dense_layers,
                        self.num_layers,
                    )
                ])

                if (self.use_micro_batch
                        and self.inference_args.generation_phase
                        == GenerationPhase.DECODER):
                    for _ in range(EP_MICRO_BATCH_NUM):
                        self.decoder_ep_runner.append(
                            DecoderEPMicroBatchRunner(self.moe_layers,
                                                      ep_engine))
            elif self.inference_args.moe_config.has_multimodality:
                from ..layers.moe.mm import MultimodalityMoeLayer as MoELayer

                self.moe_layers = nn.LayerList([
                    None for i in range(
                        self.num_dense_layers)
                ] + [
                    MoELayer(
                        inference_args=inference_args,
                        layer_name=f"moe_layers.{i}",
                        layer_idx=i,
                    ) for i in range(
                        self.num_dense_layers,
                        self.num_layers,
                    )
                ])
            else:
                from ..layers.moe.moe import FusedMoE

                self.moe_layers = nn.LayerList([
                    None for i in range(
                        self.num_dense_layers)
                ] + [
                    FusedMoE(
                        llm_config=llm_config,
                        layer_name=f"moe_layers.{i}",
                        layer_idx=i,
                    ) for i in range(
                        self.num_dense_layers,
                        self.num_layers,
                    )
                ])

        self.bias_residual_layernorm_layers = nn.LayerList([
            RMSNorm(
                llm_config,
                hidden_size=llm_config.model_config.hidden_size,
                eps=epsilon,
                prefix=llm_config.load_config.get_weight_key_by_layer_name(
                f"{base_model_prefix}.decoder.layers.{i + 1}.norm1").rpartition('.')[0],
                quant_scale=llm_config.load_config.get_quant_scale_by_layer_name(
                f"{base_model_prefix}.decoder.layers.{i + 1}.norm1"),
                linear_bias=(getattr(self.ffn2_layers[i], "linear_bias", None)
                             if not inference_args.moe_config.use_moe else
                             None),
            ) for i in range(self.num_layers - 1)
        ])

        self.last_layernorm = LayerNorm(
            llm_config,
            prefix="",
            hidden_size=llm_config.model_config.hidden_size,
            eps=epsilon,
            linear_bias=(getattr(self.ffn2_layers[self.num_layers -
                                                  1], "linear_bias", None)
                         if not inference_args.moe_config.use_moe else None),
        )
        logger.info(
            f"FusedTransformer inference_args {self.inference_args.__dict__}")

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        import threading

        enable_efficientllm_load_model_concurrency = int(
            os.getenv("ENABLE_EFFICIENTLLM_LOAD_MODEL_CONCURRENCY", "1"))

        self.norm_before_qkv.load_state_dict(state_dict)

        def load_layer_state_dict(i):
            self.qkv_linear_layers[i].load_state_dict(state_dict)
            self.out_linear_layers[i].load_state_dict(state_dict)
            self.ffn_layernorm_layers[i].load_state_dict(state_dict)
            if not self.inference_args.moe_config.use_moe:
                self.ffn1_layers[i].load_state_dict(state_dict)
                self.ffn2_layers[i].load_state_dict(state_dict)
            else:
                if self.inference_args.moe_config.moe_use_ffn_shared_weight_and_bias:
                    self.moe_layers[i].load_state_dict(state_dict)
                else:
                    if i >= self.num_dense_layers:
                        self.moe_layers[i].load_state_dict(state_dict)
                    else:
                        self.ffn1_layers[i].load_state_dict(state_dict)
                        self.ffn2_layers[i].load_state_dict(state_dict)
            if i < self.num_layers - 1:
                self.bias_residual_layernorm_layers[i].load_state_dict(
                    state_dict)

            paddle.device.cuda.empty_cache()

        num_wave = 8  # 4 will oom for mp4
        wave_size = max((self.num_layers + num_wave - 1) // num_wave, 1)
        for wave in range(num_wave):
            threads = []
            current_start_layer = wave * wave_size
            current_end_layer = min((wave + 1) * wave_size, self.num_layers)
            for i in range(current_start_layer, current_end_layer):
                logger.info(f"Start load layer {i}")
                if enable_efficientllm_load_model_concurrency:
                    thread = threading.Thread(target=load_layer_state_dict,
                                              args=(i, ))
                    threads.append(thread)
                    thread.start()
                else:
                    load_layer_state_dict(i)

            for t in threads:
                t.join()

            logger.debug(
                f"memory {paddle.device.cuda.memory_allocated() / 1024 / 1024 / 1024} GB"
            )

    def update_state_dict(self, state_dict):
        """
        Update the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        if not self.inference_args.moe_config.use_moe:
            return
        if self.inference_args.moe_config.moe_use_ffn_shared_weight_and_bias:
            return

        import threading

        enable_efficientllm_load_model_concurrency = int(
            os.getenv("ENABLE_EFFICIENTLLM_LOAD_MODEL_CONCURRENCY", "1")
        )

        def load_layer_state_dict(i):
            if i < self.inference_args.moe_config.moe_layer_start_index:
                return
            self.moe_layers[i].load_state_dict(state_dict, True)
            paddle.device.cuda.empty_cache()

        num_wave = 8  # 4 will oom for mp4
        wave_size = max(self.num_layers // num_wave, 1)
        for wave in range(num_wave + 1):
            threads = []
            current_start_layer = wave * wave_size
            if current_start_layer >= self.num_layers:
                break
            current_end_layer = (
                self.num_layers
                if (wave + 1) * wave_size > self.num_layers
                else (wave + 1) * wave_size
            )
            for i in range(current_start_layer, current_end_layer):
                logger.info(f"Start update layer {i}")
                if enable_efficientllm_load_model_concurrency:
                    thread = threading.Thread(target=load_layer_state_dict, args=(i,))
                    threads.append(thread)
                    thread.start()
                else:
                    load_layer_state_dict(i)

            for t in threads:
                t.join()

            logger.debug(
                f"update memory {paddle.device.cuda.memory_allocated() / 1024 / 1024 / 1024} GB"
            )

    def pre_process(self, **kwargs):
        """
        pre_process
        """
        pass

    def post_process(self, **kwargs):
        """
            Rebuild padding for the output of EfficientLLM.

        Args:
            multi_block_output (Tensor, optional): Output from EfficientLLM. Defaults to None.
            cum_offsets (Tensor, optional): Cumulative offsets for each block in the input sequence. Defaults to None.
            seq_lens_encoder (Tensor, optional): Sequence lengths of encoder inputs. Defaults to None.
            seq_lens_decoder (Tensor, optional): Sequence lengths of decoder inputs. Defaults to None.
            max_input_length (int, optional): Maximum length of the input sequence. Defaults to -1.

        Returns:
            Tensor: The rebuilt padding output.
        """
        multi_block_output = kwargs.get("multi_block_output", None)
        cum_offsets = kwargs.get("cum_offsets", None)
        seq_lens_this_time = kwargs.get("seq_lens_this_time", None)
        seq_lens_encoder = kwargs.get("seq_lens_encoder", None)
        seq_lens_decoder = kwargs.get("seq_lens_decoder", None)
        max_input_length = kwargs.get("max_input_length", -1)
        output_padding_offset = kwargs.get("output_padding_offset", None)

        out = fastdeploy.model_executor.ops.gpu.rebuild_padding(
            multi_block_output,
            cum_offsets,
            seq_lens_this_time,
            seq_lens_decoder,
            seq_lens_encoder,
            output_padding_offset,
            max_input_length,
        )

        return out

    def micro_batch_attention(
        self,
        micro_batch_id,
        layer_id,
        moe_layer_start_index,
        input_ids,
        forward_meta,
        rotary_embs=None,
        rotary_emb_dims=0,
        caches=None,
        pre_caches=None,
        pre_caches_length=0,
        attn_mask=None,
        kv_signal_data=None,
    ):
        r"""
        Compute attention stage in micro-batch.

        Parameters:
            micro_batch_id (int): The index of micro-batch.
            layer_id (int): The index of layer in fused transformer block.
            moe_layer_start_index (int): The index of first moe layer.
            input_ids (Tensor, optional): The input ids of the batch. Used for computing the
                attention mask. Default: None. Shape: [batch_size, max_sequence_length].
            rotary_embs (Tensor optional): The RoPE embs for the rotary computation.
                The shape is `[2, bsz, 1, seq\_len, head\_dim]`. Default None.
            rotary_emb_dims (int, optional): The rotary_emb_dims of rotary computation,
                and it is 0 when rotary_embs is None,
                1 when rotary_embs is not None and pos_extra_ids is None,
                2 when rotary_embs and pos_extra_ids are both not None. Default 0.
            caches (list(Tensor)|tuple(Tensor), optional): The cache structure
                tensors for the inference generation model. It is only used for
                inference and should be None for training. The shape is
                `[2, batch_size, num_head, max_seq_len, head_dim]`. Default None.
            pre_caches (list(Tensor)|tuple(Tensor), optional): The prefix caches
                for the generation model. The shape is `[2, bsz, num\_head, cache\_len, head\_dim]`.
                Default None.
            pre_caches_length (int, optional): The length of the pre-computed cache tensors.
                Default: None.
            attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                `[batch_size, 1, sequence_length, sequence_length]`. It can be
                None when nothing wanted or needed to be prevented attention to.
                Default None.
            kv_signal_data (Tensor, optional): A tensor used in multi-head attention
                to mark kv_cache's share.

        """
        ffn2_out = None
        if layer_id > moe_layer_start_index:
            ffn2_out = self.micro_batch_control.micro_batches[
                micro_batch_id].before_norm_fused_moe_out
        else:
            ffn2_out = self.micro_batch_control.micro_batches[
                micro_batch_id].ffn2_out
        (ln_out,
         residual_input) = self.bias_residual_layernorm_layers[layer_id - 1](
             ffn2_out,
             self.micro_batch_control.micro_batches[micro_batch_id].
             residual_input,
        )
        # qkv matmul
        qkv_out = self.qkv_linear_layers[layer_id](ln_out)

        # attention
        args = self.micro_batch_control.micro_batches[micro_batch_id].args
        # breakpoint()
        attn_out = self.attn_layers[layer_id](
            q=None,
            k=None,
            v=None,
            forward_meta=forward_meta,
            qkv=qkv_out,
            kv_signal_data=kv_signal_data,
        )
        # out_linear
        out_linear_out = self.out_linear_layers[layer_id](attn_out)
        # ffn layernorm
        (
            self.micro_batch_control.micro_batches[micro_batch_id].tmp_out,
            self.micro_batch_control.micro_batches[micro_batch_id].
            residual_input,
        ) = self.ffn_layernorm_layers[layer_id](out_linear_out, residual_input)

        (
            self.micro_batch_control.micro_batches[micro_batch_id].topk_idx,
            self.micro_batch_control.micro_batches[micro_batch_id].
            topk_weights,
        ) = self.moe_layers[layer_id].micro_batch_gate(
            self.micro_batch_control.micro_batches[micro_batch_id].tmp_out)
        # record event
        self.micro_batch_control.micro_batches[
            micro_batch_id].compute_event = (deep_ep.utils.EventOverlap(
                EventHandle()))

    def micro_batch_dispatch(self, micro_batch_id, layer_id):
        r"""
        Run dispatch stage in micro-batch.

        Parameters:
            micro_batch_id (int): The index of micro-batch.
            layer_id (int): The index of layer in fused transformer block.
            moe_layer_start_index (int): The index of first moe layer.

        Returns:
            None
        """
        if self.inference_args.weight_block_size[0] != -1:
            x, x_scale_tensor = fastdeploy.model_executor.ops.gpu.per_token_quant(
                self.micro_batch_control.micro_batches[micro_batch_id].tmp_out,
                self.inference_args.weight_block_size[0],
            )
            (
                self.micro_batch_control.micro_batches[micro_batch_id].recv_x,
                self.micro_batch_control.micro_batches[micro_batch_id].
                recv_topk_idx,
                self.micro_batch_control.micro_batches[micro_batch_id].
                recv_topk_weights,
                self.micro_batch_control.micro_batches[micro_batch_id].
                recv_num_tokens_per_expert_list,
                self.micro_batch_control.micro_batches[micro_batch_id].handle,
                self.micro_batch_control.micro_batches[micro_batch_id].
                comunication_event,
            ) = self.moe_layers[layer_id].micro_batch_dispatch(
                (x, x_scale_tensor),
                self.micro_batch_control.micro_batches[micro_batch_id].
                topk_idx,
                self.micro_batch_control.micro_batches[micro_batch_id].
                topk_weights,
                self.micro_batch_control.micro_batches[micro_batch_id].
                compute_event,
            )
        else:
            (
                self.micro_batch_control.micro_batches[micro_batch_id].recv_x,
                self.micro_batch_control.micro_batches[micro_batch_id].
                recv_topk_idx,
                self.micro_batch_control.micro_batches[micro_batch_id].
                recv_topk_weights,
                self.micro_batch_control.micro_batches[micro_batch_id].
                recv_num_tokens_per_expert_list,
                self.micro_batch_control.micro_batches[micro_batch_id].handle,
                self.micro_batch_control.micro_batches[micro_batch_id].
                comunication_event,
            ) = self.moe_layers[layer_id].micro_batch_dispatch(
                self.micro_batch_control.micro_batches[micro_batch_id].tmp_out,
                self.micro_batch_control.micro_batches[micro_batch_id].
                topk_idx,
                self.micro_batch_control.micro_batches[micro_batch_id].
                topk_weights,
                self.micro_batch_control.micro_batches[micro_batch_id].
                compute_event,
            )

    def micro_batch_moe_ffn(self, micro_batch_id, layer_id):
        r"""
        Compute moe ffn stage in micro-batch.

        Parameters:
            micro_batch_id (int): The index of micro-batch.
            layer_id (int): The index of layer in fused transformer block.
            moe_layer_start_index (int): The index of first moe layer.

        Returns:
            None
        """
        self.micro_batch_control.micro_batches[
            micro_batch_id].tmp_ffn_out = self.moe_layers[
                layer_id].micro_batch_ffn(
                    self.micro_batch_control.micro_batches[micro_batch_id].
                    recv_x,
                    self.micro_batch_control.micro_batches[micro_batch_id].
                    recv_topk_idx,
                    self.micro_batch_control.micro_batches[micro_batch_id].
                    recv_topk_weights,
                    self.micro_batch_control.micro_batches[micro_batch_id].
                    recv_num_tokens_per_expert_list,
                    self.micro_batch_control.micro_batches[micro_batch_id].
                    handle,
        )
        # record event
        self.micro_batch_control.micro_batches[
            micro_batch_id].compute_event = (deep_ep.utils.EventOverlap(
                EventHandle()))

    def micro_batch_combine(self, micro_batch_id, layer_id):
        r"""
        Run combine stage in micro-batch.

        Parameters:
            micro_batch_id (int): The index of micro-batch.
            layer_id (int): The index of layer in fused transformer block.
            moe_layer_start_index (int): The index of first moe layer.

        Returns:
            None
        """
        (
            self.micro_batch_control.micro_batches[micro_batch_id].
            before_norm_fused_moe_out,
            self.micro_batch_control.micro_batches[micro_batch_id].
            combined_topk_weights,
            self.micro_batch_control.micro_batches[micro_batch_id].
            comunication_event,
        ) = self.moe_layers[layer_id].micro_batch_combine(
            self.micro_batch_control.micro_batches[micro_batch_id].tmp_ffn_out,
            self.micro_batch_control.micro_batches[micro_batch_id].
            recv_topk_weights,
            self.micro_batch_control.micro_batches[micro_batch_id].handle,
            self.micro_batch_control.micro_batches[micro_batch_id].
            compute_event,
        )

    def forward(
        self,
        input_ids,
        src,
        forward_meta,
        cum_offsets=None,
        attn_mask=None,
        caches=None,
        pre_caches=None,
        pre_caches_length=0,
        rotary_embs=None,
        rotary_emb_dims=0,
        **kwargs,
    ):
        r"""
        Applies multi transformer layers on the input.

        Parameters:
            src (Tensor): The input of Transformer layers. It is
                a tensor with shape `[batch_size, sequence_length, d_model]`.
                The data type should be float16 or float32.
            attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                `[batch_size, 1, sequence_length, sequence_length]`. It can be
                None when nothing wanted or needed to be prevented attention to.
                Default None.
            caches (list(Tensor)|tuple(Tensor), optional): The cache structure
                tensors for the inference generation model. It is only used for
                inference and should be None for training. The shape is
                `[2, batch_size, num_head, max_seq_len, head_dim]`. Default None.
            pre_caches (list(Tensor)|tuple(Tensor), optional): The prefix caches
                for the generation model. The shape is `[2, bsz, num\_head, cache\_len, head\_dim]`.
                Default None.
            rotary_embs (Tensor optional): The RoPE embs for the rotary computation.
                The shape is `[2, bsz, 1, seq\_len, head\_dim]`. Default None.
            rotary_emb_dims (int, optional): The rotary_emb_dims of rotary computation,
                and it is 0 when rotary_embs is None,
                1 when rotary_embs is not None and pos_extra_ids is None,
                2 when rotary_embs and pos_extra_ids are both not None. Default 0.

        Returns:
            Tensor|tuple: If `caches` is None, return a tensor that has
            the same shape and data type with `src`, representing the output
            of Transformer layers. If `caches` is not None, return the
            tuple (output, caches), which output is the output of
            Transformer layers, caches is inplace with input `caches`.
        """
        self.pre_process(**kwargs)
        kwargs["cum_offsets"] = cum_offsets

        bsz = cum_offsets.shape[0]
        token_num = src.shape[0]
        cu_seqlens_q = kwargs["cu_seqlens_q"]
        cu_seqlens_k = kwargs["cu_seqlens_k"]

        if self.use_micro_batch and bsz >= EP_MICRO_BATCH_NUM:
            self.micro_batch_control.micro_batch_num = EP_MICRO_BATCH_NUM
            self.micro_batch_control.split_micro_batch(EP_MICRO_BATCH_NUM,
                                                       **kwargs)

        if caches is not None:
            assert len(caches) == self.num_layers or len(
                caches) == 2 * self.num_layers

        residual_input = src

        if self.inference_args.use_append_attn:
            if self.use_micro_batch:
                kwargs["encoder_block_shape_q"] = 64
                kwargs["decoder_block_shape_q"] = 16
                kwargs["max_partition_size"] = 32768
                kwargs["encoder_max_partition_size"] = self.max_len

                (
                    kwargs["encoder_batch_ids"],
                    kwargs["encoder_tile_ids_per_batch"],
                    kwargs["encoder_num_blocks"],
                    kwargs["kv_batch_ids"],
                    kwargs["kv_tile_ids_per_batch"],
                    kwargs["kv_num_blocks"],
                    kwargs["decoder_batch_ids"],
                    kwargs["decoder_tile_ids_per_batch"],
                    kwargs["decoder_num_blocks"],
                    kwargs["max_len_kv"],
                    set_max_lengths,
                ) = fastdeploy.model_executor.ops.gpu.get_block_shape_and_split_kv_block(
                    kwargs.get("seq_lens_encoder", None),
                    kwargs.get("seq_lens_decoder", None),
                    kwargs.get("seq_lens_this_time", None),
                    kwargs.get("cum_offsets", None),
                    kwargs.get("encoder_block_shape_q", 64),
                    kwargs.get("decoder_block_shape_q", 16),
                    self.num_heads // self.kv_num_heads,
                    kwargs.get("block_size", 64),
                    self.inference_args.speculate_max_draft_token_num + 1,
                )
            else:
                attntion_meta = forward_meta.attn_backend.get_attntion_meta()
                set_max_lengths = forward_meta.attn_backend.get_attntion_meta().set_max_lengths
            if self.use_fa3 == 1 and attntion_meta.set_max_lengths[1] > 0:
                (
                    cu_seqlens_k,
                    pre_cache_batch_ids,
                    pre_cache_tile_ids_per_batch,
                    pre_cache_num_blocks_cpu,
                    kv_token_num_cpu,
                ) = fastdeploy.model_executor.ops.gpu.pre_cache_len_concat(
                    kwargs.get("seq_lens_decoder", None),
                    kwargs.get("seq_lens_this_time", None),
                    set_max_lengths[2],
                    kwargs.get("block_size", 64),
                )
            """set_max_lengths: max_len_this_time, max_enc_len_this_time, max_dec_len_this_time,
            max_enc_dec_len_this_time, max_just_dec_len_this_time, max_just_dec_merged_len_this_time,
            max_system_len, max_just_dec_len_without_system"""
            kwargs["set_max_lengths"] = set_max_lengths

            if self.use_micro_batch and bsz > 1:
                self.micro_batch_control.get_block_shape_and_split_kv_block(
                    kwargs.get("encoder_block_shape_q", 64),
                    kwargs.get("decoder_block_shape_q", 16),
                    kwargs.get("encoder_max_partition_size", 32768),
                    kwargs.get("max_partition_size", 32768),
                    self.num_heads // self.kv_num_heads,
                    self.inference_args.speculate_max_draft_token_num,
                )
        else:
            max_enc_len_this_time, max_dec_len_this_time = blha_get_max_len(
                kwargs["seq_lens_encoder"],
                kwargs["seq_lens_decoder"],
                cum_offsets,
            )
            kwargs["max_enc_len_this_time"] = max_enc_len_this_time
            kwargs["max_dec_len_this_time"] = max_dec_len_this_time

            if self.use_micro_batch and bsz > 1:
                self.micro_batch_control.blha_get_max_len()

        if self.use_pd_disaggregation:
            kv_signal_metadata = fastdeploy.model_executor.ops.gpu.open_shm_and_get_meta_signal(
                self.rank,
                self.keep_pd_step_flag,
            )
            self.kv_signal_datas = (
                [])  # 用于保持住tensor的引用，防止回收写别的内容，导致c++层的异步访问出现报错

        def _compute_attn(x: paddle.Tensor, residual_input: paddle.Tensor,
                          i: int, **attn_args):
            # qkv_layernorm
            if i == 0:
                ln_out = self.norm_before_qkv(x)
            else:
                ln_out = x
            # qkv matmul
            qkv_out = self.qkv_linear_layers[i](ln_out)
            # attention
            if self.use_fa3 == 1 and set_max_lengths[1] > 0:
                seq_lens_this_time = attn_args.get("seq_lens_this_time", None)
                seq_lens_encoder = attn_args.get("seq_lens_encoder", None)
                seq_lens_decoder = attn_args.get("seq_lens_decoder", None)
                padding_offsets = attn_args.get("padding_offsets", None)
                cum_offsets = attn_args.get("cum_offsets", None)
                block_tables = attn_args.get("block_tables", None)
                if self.use_micro_batch:
                    kv_batch_ids = attn_args.get("kv_batch_ids", None)
                    kv_tile_ids_per_batch = attn_args.get("kv_tile_ids_per_batch",
                                                          None)
                else:

                    kv_batch_ids = attntion_meta.kv_batch_ids
                    kv_tile_ids_per_batch = attntion_meta.kv_tile_ids_per_batch
                kv_num_blocks = attn_args.get("kv_num_blocks", None)
                max_input_length = attn_args.get("max_input_length", -1)
                k_quant_scale = getattr(self.attn_layers[i], "cache_k_scale",
                                        None)
                v_quant_scale = getattr(self.attn_layers[i], "cache_v_scale",
                                        None)
                k_dequant_scale = getattr(self.attn_layers[i],
                                          "cache_k_out_scale", None)
                v_dequant_scale = getattr(self.attn_layers[i],
                                          "cache_v_out_scale", None)
                cache_k_zp = getattr(self.attn_layers[i], "cache_k_zp", None)
                cache_v_zp = getattr(self.attn_layers[i], "cache_v_zp", None)
                qkv_out_gqa_rope = fastdeploy.model_executor.ops.gpu.gqa_rope_write_cache(
                    qkv_out,
                    caches[2 * i],  # key_cache
                    caches[2 * i + 1],  # value_cache
                    cu_seqlens_q,
                    cu_seqlens_k,
                    rotary_embs,
                    seq_lens_this_time,
                    seq_lens_encoder,
                    seq_lens_decoder,
                    padding_offsets,
                    cum_offsets,
                    block_tables,
                    kv_batch_ids,
                    kv_tile_ids_per_batch,
                    kv_num_blocks,
                    pre_cache_batch_ids,
                    pre_cache_tile_ids_per_batch,
                    pre_cache_num_blocks_cpu,
                    k_quant_scale,  # cache_k_quant_scales
                    v_quant_scale,
                    k_dequant_scale,
                    v_dequant_scale,
                    cache_k_zp,
                    cache_v_zp,  # cache_v_zp
                    kv_signal_data,
                    kv_token_num_cpu[0],
                    max_input_length,
                    self.inference_args.cache_quant_type,
                )
                q, k, v = qkv_out_gqa_rope[0], qkv_out_gqa_rope[
                    1], qkv_out_gqa_rope[2]

                atten_out = (paddle.nn.functional.flash_attention.
                             flash_attention_v3_varlen(
                                 q,
                                 k,
                                 v,
                                 cu_seqlens_q,
                                 cu_seqlens_k,
                                 max_seqlen_q=set_max_lengths[0],
                                 max_seqlen_k=set_max_lengths[3],
                                 causal=True,
                             )[0].reshape([token_num, -1]))
            else:
                if self.use_micro_batch:

                    atten_out = self.attn_layers[i](
                        qkv_out,
                        padding_offset,
                        input_ids,
                        rotary_embs,
                        rotary_emb_dims,
                        caches[2 * i],  # key_cache
                        caches[2 * i + 1],  # value_cache
                        (pre_caches[2 * i]
                         if pre_caches is not None else None),  # pre_key_cache
                        (pre_caches[2 * i + 1]
                         if pre_caches is not None else None),  # pre_value_cache
                        pre_caches_length,
                        attn_mask,
                        kv_signal_data,
                        **attn_args,
                    )

                else:
                    atten_out = self.attn_layers[i](
                        q=None,
                        k=None,
                        v=None,
                        forward_meta=forward_meta,
                        qkv=qkv_out,
                        kv_signal_data=kv_signal_data,
                    )

            # out_linear
            out_linear_out = self.out_linear_layers[i](atten_out)

            # all_reduce
            if self.nranks > 1 and (not self.inference_args.use_ep):
                # if in_dynamic_or_pir_mode():
                if in_dynamic_mode():
                    hcg = fleet.get_hybrid_communicate_group()
                    mp_group = hcg.get_model_parallel_group()
                    dist.all_reduce(out_linear_out, group=mp_group)
                else:
                    dist.all_reduce(out_linear_out)

            # ffn layernorm
            tmp_out, residual_input = self.ffn_layernorm_layers[i](
                out_linear_out, residual_input)

            return tmp_out, residual_input

        for i in range(self.num_layers):
            if self.use_pd_disaggregation:
                kv_signal_data = fastdeploy.model_executor.ops.gpu.init_signal_layerwise(
                    kv_signal_metadata,
                    i + self.inference_args.start_layer_index)
                self.kv_signal_datas.append(kv_signal_data)
            else:
                kv_signal_data = None

            if (
                not self.use_micro_batch
                or i < self.num_dense_layers
                or bsz < self.micro_batch_control.micro_batch_num
            ):
                tmp_out, residual_input = _compute_attn(
                    src, residual_input, i, **kwargs)

                if (
                    self.inference_args.moe_config.use_moe
                    and not self.inference_args.moe_config.moe_use_ffn_shared_weight_and_bias
                ):
                    if i >= self.num_dense_layers:
                        ffn2_out = self.moe_layers[i](tmp_out, **kwargs)
                    else:
                        ffn1_out = self.ffn1_layers[i](tmp_out)
                        if not self.fuse_ffn_act:
                            ffn1_out = self.bias_act_layers[i](ffn1_out)
                        ffn2_out = self.ffn2_layers[i](ffn1_out)
                else:
                    # ffn1 matmul
                    ffn1_out = self.ffn1_layers[i](tmp_out)
                    if not self.fuse_ffn_act:
                        ffn1_out = self.bias_act_layers[i](ffn1_out)
                    # ffn2 matmul
                    ffn2_out = self.ffn2_layers[i](ffn1_out)
                # all_reduce
                if self.nranks > 1 and (not self.inference_args.use_ep):
                    # if in_dynamic_or_pir_mode():
                    if in_dynamic_mode():
                        hcg = fleet.get_hybrid_communicate_group()
                        mp_group = hcg.get_model_parallel_group()
                        dist.all_reduce(ffn2_out, group=mp_group)
                    else:
                        dist.all_reduce(ffn2_out)

                if (self.inference_args.moe_config.use_moe
                        and self.inference_args.moe_config.
                        moe_use_ffn_shared_weight_and_bias):
                    moe_out = self.moe_layers[i](tmp_out, **kwargs)
                    ffn2_out = moe_out + ffn2_out
                if (
                    not self.use_micro_batch
                    or bsz < self.micro_batch_control.micro_batch_num
                    or (
                        self.inference_args.generation_phase == GenerationPhase.PREFILL
                        and i < self.num_dense_layers - 1
                    )
                    or (
                        self.inference_args.generation_phase == GenerationPhase.DECODER
                        and i < self.num_dense_layers
                    )
                ):
                    # norm + residual_add_bias
                    if i != self.num_layers - 1:
                        tmp_out, residual_input = self.bias_residual_layernorm_layers[
                            i](ffn2_out, residual_input)
                    else:
                        tmp_out, _ = self.last_layernorm(
                            ffn2_out, residual_input)
            elif self.inference_args.generation_phase == GenerationPhase.PREFILL:
                # slice
                if i == self.num_dense_layers:
                    for mbid in range(self.micro_batch_control.micro_batch_num):
                        self.micro_batch_control.micro_batches[mbid].ffn2_out = (
                            ffn2_out[
                                self.micro_batch_control.micro_batches[mbid]
                                .start_token_id : self.micro_batch_control.micro_batches[
                                    mbid
                                ]
                                .end_token_id
                            ]
                        )
                        self.micro_batch_control.micro_batches[mbid].residual_input = (
                            residual_input[
                                self.micro_batch_control.micro_batches[mbid]
                                .start_token_id : self.micro_batch_control.micro_batches[
                                    mbid
                                ]
                                .end_token_id
                            ]
                        )
                for mbid in range(self.micro_batch_control.micro_batch_num):
                    if i > self.num_dense_layers:
                        # 第一层不wait
                        self.micro_batch_control.wait_combine(mbid)
                    self.micro_batch_attention(
                        mbid,
                        i,
                        self.num_dense_layers,
                        input_ids,
                        rotary_embs,
                        rotary_emb_dims,
                        caches,
                        pre_caches,
                        pre_caches_length,
                        attn_mask,
                        kv_signal_data,
                        forward_meta,
                    )
                # dispatch and moe
                for mbid in range(self.micro_batch_control.micro_batch_num):
                    # wait in dispatch
                    self.micro_batch_dispatch(mbid, i)
                    self.micro_batch_control.wait_dispatch(mbid)
                    self.micro_batch_moe_ffn(mbid, i)

                # combine
                for mbid in range(self.micro_batch_control.micro_batch_num):
                    # wait in combine
                    self.micro_batch_combine(mbid, i)

                if i == self.num_layers - 1:
                    for mbid in range(
                            self.micro_batch_control.micro_batch_num):
                        self.micro_batch_control.wait_combine(mbid)
                        fused_out = self.micro_batch_control.micro_batches[
                            mbid].before_norm_fused_moe_out
                        (self.micro_batch_control.micro_batches[mbid].tmp_out,
                         _) = (self.last_layernorm(
                             fused_out,
                             self.micro_batch_control.micro_batches[mbid].
                             residual_input,
                         ))
                    tmp_out = paddle.concat([
                        self.micro_batch_control.micro_batches[mbid].tmp_out
                        for mbid in range(
                            self.micro_batch_control.micro_batch_num)
                    ])
            else:

                def _tail_layernorm(output, residual_inputs, srcs, mb_id,
                                    layer_id, outputs):
                    if layer_id != self.num_layers - 1:
                        output, residual_inputs[mb_id] = (
                            self.bias_residual_layernorm_layers[layer_id](
                                output, residual_inputs[mb_id]))
                        srcs[mb_id] = output
                    else:
                        output, _ = self.last_layernorm(
                            output, residual_inputs[mb_id])
                        outputs.append(output)

                if i == self.num_dense_layers:
                    # Split
                    srcs = [
                        src[self.micro_batch_control.micro_batches[mbid].
                            start_token_id:self.micro_batch_control.
                            micro_batches[mbid].end_token_id] for mbid in
                        range(self.micro_batch_control.micro_batch_num)
                    ]
                    residual_inputs = [
                        residual_input[self.micro_batch_control.
                                       micro_batches[mbid].start_token_id:self.
                                       micro_batch_control.micro_batches[mbid].
                                       end_token_id] for mbid in
                        range(self.micro_batch_control.micro_batch_num)
                    ]
                outputs = []
                # Stage 1
                for mb_id in range(self.micro_batch_control.micro_batch_num):
                    micro_batch_src = srcs[mb_id]
                    micro_batch_residual_input = residual_inputs[mb_id]

                    attn_args = self.micro_batch_control.micro_batches[
                        mb_id].args
                    tmp_out, residual_inputs[mb_id] = _compute_attn(
                        micro_batch_src, micro_batch_residual_input, i,
                        **attn_args)

                    topk_idx, topk_weights = self.moe_layers[i].gate(tmp_out)

                    if mb_id == 0 and i > self.num_dense_layers:
                        output = self.decoder_ep_runner[
                            self.micro_batch_control.micro_batch_num -
                            1].combine_hook_wrap()

                        _tail_layernorm(
                            output,
                            residual_inputs,
                            srcs,
                            self.micro_batch_control.micro_batch_num - 1,
                            i - 1,
                            outputs,
                        )
                    elif mb_id > 0:
                        self.decoder_ep_runner[mb_id - 1].dispatch_hook_wrap()
                    self.decoder_ep_runner[mb_id].dispatch_issue(
                        tmp_out, topk_idx, topk_weights, i)

                # Stage 2
                for mb_id in range(self.micro_batch_control.micro_batch_num):
                    self.decoder_ep_runner[mb_id].ffn(i)
                    if mb_id == 0:
                        self.decoder_ep_runner[
                            self.micro_batch_control.micro_batch_num -
                            1].dispatch_hook_wrap()
                    else:
                        output = self.decoder_ep_runner[mb_id -
                                                        1].combine_hook_wrap()
                        _tail_layernorm(output, residual_inputs, srcs,
                                        mb_id - 1, i, outputs)

                    self.decoder_ep_runner[mb_id].combine_issue()

                if i == self.num_layers - 1:
                    output = self.decoder_ep_runner[
                        self.micro_batch_control.micro_batch_num -
                        1].combine_hook_wrap()
                    output, _ = self.last_layernorm(output,
                                                    residual_inputs[mb_id])
                    outputs.append(output)
                    tmp_out = paddle.concat(outputs, axis=0)

            src = tmp_out

        kwargs["multi_block_output"] = tmp_out
        kwargs["input_ids"] = input_ids
        if (self.return_all_hidden_states
                or self.inference_args.moe_config.has_multimodality):
            return kwargs["multi_block_output"], kwargs["cum_offsets"], caches
        else:
            out = self.post_process(**kwargs)
            return out, caches

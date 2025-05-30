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

import contextlib
import os
import sys

import paddle
import paddle.distributed as dist
from paddle.common_ops_import import convert_dtype
from paddle.distributed import fleet
from paddlenlp.trainer import RuntimeTimer
from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.transformers.model_utils import load_tp_checkpoint
from paddlenlp.trl import llm_utils
from paddlenlp.utils.log import logger

from fastdeploy.config import (AdditionalConfig, DeviceConfig, LLMConfig,
                               LoadConfig, ModelConfig, ParallelConfig,
                               SpeculativeConfig)
from fastdeploy.inference_args import GenerationPhase

from .modeling_ernie_bot import ErnieBotForGeneration, ErnieBotFusedModel
from .tokenizer import ErnieBotTokenizer
from .utils import _vocab_size_with_padding, convert_ndarray_dtype

current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(
    os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(grandparent_dir)


def offload_model(model):
    """
    Offload the model to CUDAPinnedPlace.
    """
    device = paddle.CUDAPinnedPlace()
    for name, src in model.named_parameters():
        if src._is_initialized() and not isinstance(src.place,
                                                    paddle.CUDAPinnedPlace):
            dst = src._copy_to(device, True)
            dst_tensor = dst.value().get_tensor()
            src_tensor = src.value().get_tensor()
            src_tensor._clear()
            src_tensor._share_data_with(dst_tensor)


def reload_model(model):
    """
    Reload the model from CUDAPinnedPlace to GPU.
    """
    model.to(paddle.device.get_device())


def reconstruct_memory(model):
    """
    reconstruct_memory to avoid memory chunks
    """
    offload_model(model)
    paddle.device.cuda.empty_cache()
    reload_model(model)


def load_tensor_from_ipc_meta(state_dict):
    """
    convert ipc_meta to tensor, but keep keys unchanged
    { 'key': ipc_meta } --> { 'key': tensor }
    example:
    state_dict = load_tensor_from_ipc_meta(state_dict)
    """
    for k, v in state_dict.items():
        # for pickling, we have to convert bytes object before save
        v[0] = v[0].encode("latin-1")
        state_dict[k] = paddle.to_tensor(
            paddle.base.core.LoDTensor._new_shared_cuda(tuple(v)))
    return state_dict


def build_stream_line_model(
    config_path,
    model_path,
    dtype,
    block_size,
    max_len,
    stage_flag,
    min_dec_len=1,
    max_dec_len=128,
    temperature=1,
    top_k=0,
    top_p=0.8,
    pre_caches_length=0,
    export_model_type="default",
    use_stop_seqs=False,
    use_fake_parameter=False,
    show_topk: int = 0,
    msg_queue_id=None,
    pad_vocab=True,
    tokenizer=None,
    cache_quant_dtype="default",
    use_beam_search: bool = False,
    enf_gen: bool = False,
    speculate_method=None,
    speculate_max_draft_token_num: int = 1,
    speculate_max_candidate_len: int = 5,
    speculate_verify_window: int = 2,
    return_all_hidden_states: bool = False,
    draft_type: str = "None",
    start_layer_index: int = 0,
    moe_quant_type: str = "default",
    use_ep: bool = False,
    ep_just_for_test: bool = False,
    generation_phase: GenerationPhase = GenerationPhase.PREFILL,
    use_micro_batch: bool = False,
    fake_server_p: bool = False,
    scale_dir: str = "None",
    output_via_mq: bool = True,
    use_safetensors: bool = False,
):
    """
    Build a fused inference model

    Args:
        config_path (str): Path to the configuration file
        model_path (str): Path to the model file
        dtype (str): Data type of the model
        block_size (int): Block size
        max_len (int): Maximum sequence length
        stage_flag (str): Qianfan requirement, stage flag, used to identify different stages in \
            time-consuming statistics logs, such as prediction ("msgid-1 predict") or export ("convert").
        min_dec_len (int, optional): Minimum decoding length. Default is 1.
        max_dec_len (int, optional): Maximum decoding length. Default is 128.
        temperature (float, optional): Temperature coefficient. Default is 1.
        top_k (int, optional): k value in top-k sampling. Default is 0.
        top_p (float, optional): p value in top-p sampling. Default is 0.8.
        pre_caches_length (int, optional): Pre-cache length. Default is 0.
        export_model_type (str, optional): Type of model to export. Default is "default".
        use_stop_seqs (bool, optional): Whether to use stop sequences. Default is False.
        use_fake_parameter (bool, optional): Whether to use fake parameters. Default is False.
        show_topk (int, optional): Whether to show top-k results. Default is 0.
        msg_queue_id (int, optional): Message queue ID. Default is None.
        pad_vocab (bool, optional): Whether to pad the vocabulary. Default is True.
        cache_quant_dtype (str, optional): Cache quantization data type. Default is "default".
        use_beam_search (bool, optional): Whether to use beam search . Defaults is False.
        enf_gen (bool, optional): Whether to use enforce generation. Defaults is False.
    Returns:
        tuple[dict, ErnieBotTokenizer, ErnieBotForGeneration]:
        A tuple containing the configuration, tokenizer, and model.
    """
    runtime_timer = RuntimeTimer("build_model")
    runtime_timer.start(f"{stage_flag} stage model loading time")

    if tokenizer is None:
        tokenizer = ErnieBotTokenizer.from_pretrained(model_path)

    config, _ = PretrainedConfig.get_config_dict(model_path)
    model_config = ModelConfig.from_dict(config)

    parallel_config = ParallelConfig()
    speculative_config = SpeculativeConfig()
    device_config = DeviceConfig()
    additional_config = AdditionalConfig()
    load_config = LoadConfig()

    tensor_parallel_rank, tensor_parallel_degree = llm_utils.init_dist_env()
    parallel_config.tensor_parallel_rank = tensor_parallel_rank
    parallel_config.tensor_parallel_degree = tensor_parallel_degree

    speculative_config.is_mtp = draft_type in ["eagle", "mtp"]
    speculative_config.draft_type = draft_type

    additional_config.use_fake_parameter = use_fake_parameter
    additional_config.ep_just_for_test = ep_just_for_test

    # use the length of tokenizer as the origin vocab size
    ori_vocab_size = len(tokenizer)

    if pad_vocab:
        config["vocab_size"] = _vocab_size_with_padding(
            config.get("vocab_size", tokenizer.vocab_size),
            config.pop("vocab_size_divisible_unit", 128),
            paddle.distributed.get_world_size(),
        )

    group_size = config.get("group_size", -1)
    wint4_smooth = config.get("smooth", False)
    num_key_value_heads = config.get("num_key_value_heads", -1)
    if num_key_value_heads is None:
        num_key_value_heads = -1

    if config.get("ffn_hidden_size", None) is not None:
        ffn_hidden_size = config["ffn_hidden_size"]
    elif config.get("intermediate_size", None) is not None:
        ffn_hidden_size = config["intermediate_size"]
    else:
        ffn_hidden_size = 4 * config["hidden_size"]
        if config["hidden_act"].lower() == "swiglu":
            if paddle.distributed.get_world_size() > 1:
                multiple_of = 8 * config["num_attention_heads"]
            else:
                multiple_of = 4 * config["num_attention_heads"]
            ffn_hidden_size = multiple_of * (
                (int(2 * ffn_hidden_size / 3) + multiple_of - 1) //
                multiple_of)

    num_layers = config.get("num_layers", None) or config.get(
        "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError(f"num_layers<{num_layers}> is invalid")

    use_moe = config.get("moe_layer_start_index", num_layers) < num_layers

    if use_fake_parameter:
        context = contextlib.nullcontext()
    elif use_safetensors:
        context = paddle.LazyGuard()
        state_dict = load_tp_checkpoint(model_path,
                                        ErnieBotFusedModel,
                                        model_config,
                                        return_numpy=False)
    elif use_moe:
        tensor_parallel_degree = dist.get_world_size()
        if tensor_parallel_degree > 1:
            hcg = fleet.get_hybrid_communicate_group()
            mp_id = hcg.get_model_parallel_rank()
            # 统计文件子目录数量
            subdir_count = 0
            for entry in os.listdir(model_path):
                if "pp" in entry:
                    full_path = os.path.join(model_path, entry)
                    if os.path.isdir(full_path):
                        subdir_count += 1

            pp_num = subdir_count
            rank_model_paths = [
                os.path.join(model_path,
                             f"pp{i}/model_state.tp0{mp_id}.pdparams")
                for i in range(pp_num)
            ]

        context = paddle.LazyGuard()
        if not use_ep:
            logger.info(f"start to loading weight: {rank_model_paths}")
            state_dicts = [
                paddle.load(path, return_numpy=True)
                for path in rank_model_paths
            ]

        else:
            # for EP loading state_dicts
            import glob

            state_dicts = []
            files = glob.glob(model_path + "/merged_tp1_state_split/*")
            for file_name in files:
                try:
                    state_dicts += [{
                        file_name.split("/")[-1]: file_name
                    }]  # save {layer_name: weight_file_name}
                except Exception:
                    pass

        new_state_dict = {}
        for state_dict in state_dicts:
            for key, value in state_dict.items():
                new_state_dict[key] = value

        state_dict = new_state_dict
    elif config.get("quant_type", None) is not None:
        # TODO(@wangbojun) currently, we use paddle.load for ptq model.
        tensor_parallel_degree = dist.get_world_size()
        if tensor_parallel_degree > 1:
            hcg = fleet.get_hybrid_communicate_group()
            mp_id = hcg.get_model_parallel_rank()
            rank_model_path = os.path.join(model_path,
                                           f"model_state.tp0{mp_id}.pdparams")
            if not os.path.exists(rank_model_path):
                full_model_path = os.path.join(model_path,
                                               "model_state.pdparams")
                if not os.path.exists(full_model_path):
                    raise ValueError(
                        f"can not find <model_state.tp0{mp_id}.pdparams> " +
                        f"and model_state.pdparams under dir<{model_path}>")
                raise ValueError(
                    "please run `split_weights.py` to gen weights for multi-gpu inference."
                )
            model_state_path = rank_model_path
            if num_key_value_heads > 0:
                assert (
                    num_key_value_heads % tensor_parallel_degree == 0
                ), "num_key_value_heads must be an integer multiple of tensor_parallel_degree"
        else:
            model_state_path = os.path.join(model_path, "model_state.pdparams")
        context = paddle.LazyGuard()
        logger.info(f"start to loading weight: {model_state_path}")
        if os.path.exists(model_state_path):
            state_dict = paddle.load(model_state_path, return_numpy=True)
    else:
        context = paddle.LazyGuard()
        state_dict = load_tp_checkpoint(
            model_path,
            ErnieBotFusedModel,
            model_config,
            return_numpy=True,
        )
    use_rmsnorm = config.get("use_rmsnorm", False)
    logger.info(f"{runtime_timer.log()}")
    runtime_timer.start(f"{stage_flag} stage set parameters time")

    model_config.ffn_hidden_size = ffn_hidden_size
    llm_config = LLMConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        speculative_config=speculative_config,
        device_config=device_config,
        additional_config=additional_config,
        load_config=load_config,
    )
    with context:
        model = ErnieBotFusedModel(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            max_len=max_len,
            block_size=block_size,
            num_layers=num_layers,
            num_attention_heads=config["num_attention_heads"],
            ffn_hidden_size=ffn_hidden_size,
            activation="swiglu",
            hidden_dropout_prob=0,
            # hidden_dropout_prob=config["hidden_dropout_prob"],
            max_position_embeddings=config["max_position_embeddings"],
            type_vocab_size=1,
            dtype=dtype,
            sequence_parallel=False,
            use_rope=True,
            rope_theta=config.get("rope_theta", 10000.0),
            rope_3d=config.get("rope_3d", False),
            weight_sharing=False,
            inv_compression_ratio=1.0 / config.get("compression_ratio", 1.0),
            export_model_type=export_model_type,  # export model type.
            wint4_smooth=wint4_smooth,  # Whether to use smooth for wint4.
            group_size=group_size,
            model_path=model_path,  # The path of Inference model.
            use_rmsnorm=use_rmsnorm,
            msg_queue_id=msg_queue_id,
            use_fake_parameter=use_fake_parameter,
            num_key_value_heads=num_key_value_heads,
            use_stop_seqs=use_stop_seqs,
            cache_quant_dtype=cache_quant_dtype,
            has_zero_point=config.get("has_zero_point", False),
            is_channel_wise=config.get("is_channel_wise", False),
            use_fast_ffn=config.get("use_fast_ffn", False),
            speculate_method=speculate_method,
            speculate_max_draft_token_num=speculate_max_draft_token_num,
            return_all_hidden_states=return_all_hidden_states,
            draft_type=draft_type,
            start_layer_index=start_layer_index,
            use_moe=use_moe,
            moe_num_experts=config.get("moe_num_experts", None),
            moe_intermediate_size=config.get("moe_intermediate_size", None),
            moe_use_gate_correction_bias=config.get(
                "moe_use_gate_correction_bias", True),
            moe_every2=config.get("moe_every2", False),
            moe_topk=config.get("moe_topk", 8),
            moe_num_shared_experts=config.get("moe_num_shared_experts", 0),
            moe_layer_start_index=config.get("moe_layer_start_index", 0),
            moe_use_ffn_shared_weight_and_bias=config.get(
                "moe_use_ffn_shared_weight_and_bias", False),
            moe_group=config.get("moe_group", False),
            moe_quant_type=moe_quant_type,
            use_ep=use_ep,
            ep_just_for_test=ep_just_for_test,
            generation_phase=generation_phase,
            use_micro_batch=use_micro_batch,
            weight_block_size=config.get("weight_block_size", [-1, -1]),
            scale_dir=scale_dir,
            output_via_mq=output_via_mq,
            llm_config=llm_config,
        )
    if use_beam_search:
        decode_strategy = "beam_search"
    elif speculate_method is not None:
        if draft_type in ["draft_model", "eagle", "mtp"]:
            decode_strategy = "draft_model_sampling"
        else:
            decode_strategy = "speculate_decoding"
    else:
        decode_strategy = "sampling"
    configs = {
        "model_path": model_path,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "hidden_size": config["hidden_size"],
        "num_attention_heads": config["num_attention_heads"],
        "vocab_size": config["vocab_size"],
        "ori_vocab_size": ori_vocab_size,
        "hidden_act": config["hidden_act"],
        "weight_sharing": config.get("weight_sharing", False),
        "weight_sharing_add_bias": config.get("weight_sharing_add_bias",
                                              False),
        "initializer_range": 0.02,
        "fused_linear": False,
        "min_dec_len": min_dec_len,
        "max_dec_len": max_dec_len,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "use_top_k": top_k > 0,
        "show_topk": show_topk,
        "use_topp_sampling": True,
        "inference": False,
        "export_model_type": export_model_type,
        "wint4_smooth": wint4_smooth,  # Whether to use smooth for wint4.
        "group_size": group_size,
        "use_rmsnorm": use_rmsnorm,
        "decode_strategy": decode_strategy,
        # "outputs_op": outputs_op,
        "cache_quant_dtype": cache_quant_dtype,
        "use_fake_parameter": use_fake_parameter,
        "enf_gen": enf_gen,
        "speculate_max_candidate_len": speculate_max_candidate_len,
        "speculate_verify_window": speculate_verify_window,
        "return_all_hidden_states": return_all_hidden_states,
        "fake_server_p": fake_server_p,
    }
    with context:
        model = ErnieBotForGeneration(model, configs)

    model.eval()

    if use_fake_parameter:
        return config, tokenizer, model
    elif not use_moe:
        for k, v in state_dict.items():
            if convert_dtype(v.dtype) == dtype:
                continue
            elif convert_dtype(v.dtype) == "float32":
                continue
            state_dict[k] = convert_ndarray_dtype(v, dtype)

    paddle.device.cuda.empty_cache()
    assert state_dict is not None
    model.set_state_dict(state_dict)
    if generation_phase == GenerationPhase.DECODER:
        reconstruct_memory(model)
    logger.info(f"{runtime_timer.log()}")

    return config, tokenizer, model

"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import random
import paddle
import paddle.distributed.fleet as fleet

from fastdeploy.model_executor.model_runner.model_runner_base import ModelRunnerBase
from fastdeploy.input.mm_processor import DataProcessor
from fastdeploy.input.mm_processor.tokenizer import ErnieVLTokenizer
from fastdeploy.model_executor.models.ernie_vl.configuration import ErnieBotMoEVLConfig
from fastdeploy.model_executor.models.ernie_vl.dfnrope import DFNRopeVisionTransformerConfig
from fastdeploy.model_executor.models.ernie_vl.dfnrope.modeling import DFNRopeVisionTransformerPretrainedModel
from fastdeploy.model_executor.models.ernie_vl.modeling_resampler import VariableResolutionResamplerModel, ScatterOp


class ModelRunner(ModelRunnerBase):
    def __init__(self, config, args, nranks, rank):
        self.nranks = nranks
        self.rank = rank

        hcg = fleet.get_hybrid_communicate_group()
        self.tensor_parallel_degree = max(hcg.get_model_parallel_world_size(), 1)
        self.tensor_parallel_rank = hcg.get_model_parallel_rank()
        self.mp_src_rank = hcg.get_model_parallel_group_src_rank()
        self.mp_group = hcg.get_model_parallel_group()

        model_path = os.path.dirname(args.model_name_or_path)
        args.llm_model_name_or_path = args.model_name_or_path
        args.tokenizer = model_path
        args.vision_model_name_or_path = f"{model_path}/DFNRopeVisionTransformer"
        args.image_preprocessor = model_path

        self.amp_black = [
            "reduce_sum",
            "c_softmax_with_cross_entropy",
            "elementwise_div",
            "sin",
            "cos",
            "sort",
            "multinomial",
        ]
        self.amp_white = [
            "lookup_table",
            "lookup_table_v2",
            "flash_attn",
            "matmul",
            "matmul_v2",
            "fused_gemm_epilogue",
        ]
        
        super().__init__(config, args)
        self.init_extra_input(config, args)

        self._reset_paddle_env()


    def _reset_paddle_env(self):
        #FLAGS_gqa_use_tensorcore
        #FLAGS_ffn2_use_hardamard
        # gqa .etc paddle Flags set
        pass

    def _load_model(self, model_name, dynamic_load_weight):
        if dynamic_load_weight == True:
            raise Exception("EB45T-VL Not Support Dynamic Load Weight For Now")

        tokenizer = ErnieVLTokenizer.from_pretrained(
            self.args.tokenizer,
            model_max_length=self.args.max_model_len,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.ignored_index = -100
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token

        config = ErnieBotMoEVLConfig.from_pretrained(
            self.args.llm_model_name_or_path,
            tensor_parallel_degree=self.tensor_parallel_degree,
            tensor_parallel_rank=self.tensor_parallel_rank,
            moe_group="dummy",  
        )
        self.model_cfg = config

        vision_config = DFNRopeVisionTransformerConfig.from_pretrained(
            self.args.vision_model_name_or_path,
            tensor_parallel_degree=1,
            tensor_parallel_rank=0,
            attn_sep=False,
            dtype="bfloat16",
        )
        config.vision_config = vision_config
        config.pixel_hidden_size = config.vision_config.hidden_size
        config.im_patch_id = tokenizer.get_vocab()["<|IMAGE_PLACEHOLDER|>"]
        config.max_text_id = config.im_patch_id

        config.tensor_parallel_output = False
        config.sequence_parallel = False

        self.dtype = self.args.dtype
        paddle.set_default_dtype(self.dtype)

        self.vision_model, self.resampler_model = self.inject_pp_vision_model(self.args, config)

        processor = DataProcessor(
            tokenizer_name=self.args.tokenizer,
            image_preprocessor_name=str(self.args.image_preprocessor),
        )
        processor.eval()
        image_preprocess = processor.image_preprocessor
        image_preprocess.image_mean_tensor = paddle.to_tensor(image_preprocess.image_mean, dtype="float32").reshape(
            [1, 3, 1, 1]
        )
        image_preprocess.image_std_tensor = paddle.to_tensor(image_preprocess.image_std, dtype="float32").reshape(
            [1, 3, 1, 1]
        )
        image_preprocess.rescale_factor = paddle.to_tensor(image_preprocess.rescale_factor, dtype="float32")
        image_preprocess.image_mean_tensor = image_preprocess.image_mean_tensor.squeeze([-2, -1]).repeat_interleave(
            config.vision_config.patch_size**2 * 1, -1
        )
        image_preprocess.image_std_tensor = image_preprocess.image_std_tensor.squeeze([-2, -1]).repeat_interleave(
            config.vision_config.patch_size**2 * 1, -1
        )
        self.image_preprocess = image_preprocess

        from ..models.export_model import build_stream_line_model
        _, _, self.model = build_stream_line_model(
            self.model_cfg,
            self.args.model_name_or_path,
            self.args.dtype,
            self.args.block_size,
            max_len=self.args.max_model_len,
            stage_flag=None,
            use_fake_parameter=True,
            pad_vocab=False,
            tokenizer=tokenizer,
            output_via_mq=True,
            export_model_type="W8A16C16",
            moe_quant_type="weight_only_int8",
        )
        self.model.eval()

        self.set_state_dict(self.args)
        print("load model finished")

    def init_extra_input(self, config, args):
        head_dim = self.model_cfg.hidden_size // self.model_cfg.num_attention_heads
        self.share_inputs.update({
            "rope_emb": paddle.full(shape=[args.max_num_seqs, 2, 1, self.max_length, 1, head_dim//2], fill_value=0, dtype="float32")
        })

    def init_rotary_position_embedding(self, max_model_len):
        pass

    def _init_kvcache(self):
        """
        分享不拷贝数据
        """
        cache_kvs = {}
        max_block_num = self.num_gpu_blocks
        num_layers = self.model_cfg.get("num_layers", None) or self.model_cfg.get("num_hidden_layers", None)

        kv_num_head = self.model_cfg.get(
            "num_key_value_heads",
            self.model_cfg.num_attention_heads,
        )
        kv_num_head = kv_num_head // self.tensor_parallel_degree
        self.model_cfg.kv_num_head = kv_num_head

        for i in range(num_layers):
            cache_type = self.args.dtype
            cache_kvs["key_caches_{}".format(i)] = paddle.full(
                shape=[
                    max_block_num,
                    kv_num_head,
                    self.args.block_size,
                    self.model_cfg.hidden_size // self.model_cfg.num_attention_heads,
                ],
                fill_value=0,
                dtype=cache_type,
            )
            cache_kvs["value_caches_{}".format(i)] = paddle.full(
                shape=[
                    max_block_num,
                    kv_num_head,
                    self.args.block_size,
                    self.model_cfg.hidden_size // self.model_cfg.num_attention_heads,
                ],
                fill_value=0,
                dtype=cache_type,
            )

        self.share_inputs["caches"] = list(cache_kvs.values())
        for value in cache_kvs.values():
            del value
        paddle.device.cuda.empty_cache()

    @paddle.no_grad()
    def set_state_dict(self, args):
        """set_state_dict"""
        rank_model_paths = []
        for root, dirs, files in os.walk(self.args.llm_model_name_or_path):
            for file in files:
                if file == f"model_state.tp0{self.tensor_parallel_rank}.pdparams":
                    rank_model_paths.append(os.path.join(root, file))
        print(rank_model_paths)
        state_dict = {}
        for path in rank_model_paths:
            loaded_dict = paddle.load(path, return_numpy=True)
            state_dict.update(loaded_dict)

        resampler_state = {}
        for key in list(state_dict.keys()):
            if "vision" in key:
                state_dict.pop(key)
            if key.startswith("ernie.resampler_model."):
                value = state_dict.pop(key)
                value = paddle.to_tensor(value).cast("bfloat16")
                value = value.numpy()
                resampler_state[key[len("ernie.resampler_model.") :]] = value
        self.model.set_state_dict(state_dict)
        self.resampler_model.set_state_dict(resampler_state)

    @paddle.no_grad()
    def inject_pp_vision_model(self, args, cfg):
        """
        注入vision model参数
        """
        vision_model = DFNRopeVisionTransformerPretrainedModel.from_pretrained(args.vision_model_name_or_path, config=cfg.vision_config)
        vision_model = paddle.amp.decorate(models=vision_model, level="O2", dtype="bfloat16")

        resampler_model = VariableResolutionResamplerModel(
            cfg.pixel_hidden_size,
            cfg.hidden_size,
            cfg.spatial_conv_size,
            cfg.temporal_conv_size,
            config=cfg,
        )
        resampler_model = paddle.amp.decorate(models=resampler_model, level="O2", dtype="bfloat16")
        vision_model.eval()
        resampler_model.eval()

        return vision_model, resampler_model
    
    @paddle.no_grad()
    def extract_vision_features(self, inputs):
        """extract_vision_features"""
        assert inputs["images"] is not None
        grid_thw = inputs["grid_thw"]

        images = inputs["images"].cast("float32")
        images = self.image_preprocess.rescale_factor * images - self.image_preprocess.image_mean_tensor
        images = images / self.image_preprocess.image_std_tensor
        images = images.cast("bfloat16")

        token_type_ids = inputs["token_type_ids"]
        token_type_ids_w_video = token_type_ids
        input_ids = inputs["input_ids"]
        # convert to img patch id
        image_mask = input_ids == self.model_cfg.im_patch_id
        image_type_ids = inputs["image_type_ids"]
        with paddle.amp.auto_cast(
            True,
            custom_black_list=self.amp_black,
            custom_white_list=self.amp_white,
            level="O2",
            dtype=self.dtype,
        ):
            image_features = self.vision_model.extract_feature(images, grid_thw)
            if self.tensor_parallel_degree > 1:
                S, C = image_features.shape
                image_features = image_features.reshape([-1, C * self.model_cfg.spatial_conv_size**2])
                image_features = ScatterOp.apply(image_features, axis=-1)  # mp 切 Fea
                image_features = image_features.reshape([S, -1])
            image_features = self.resampler_model(
                image_features,
                image_mask,
                token_type_ids_w_video,
                image_type_ids,
                grid_thw,
            )
        return image_features
    
    @paddle.no_grad()
    def prepare_rope3d(self, inputs, **kwargs):
        """prepare_rope3d"""
        position_ids = inputs["position_ids"]

        prefix_max_position_ids = paddle.max(position_ids) + 1
        dec_pos_ids = paddle.tile(
            paddle.arange(kwargs["max_length"], dtype="int64").unsqueeze(0).unsqueeze(-1), [1, 1, 3]
        )
        dec_pos_ids = dec_pos_ids + prefix_max_position_ids
        position_ids_3d_real = paddle.concat([position_ids, dec_pos_ids], axis=1)

        from ..models.utils import get_rotary_position_embedding_3d

        rope_emb = get_rotary_position_embedding_3d(
            position_ids_3d_real,
            head_dim=self.model_cfg.hidden_size // self.model_cfg.num_attention_heads,
            compression_ratio=1.0,
            rope_theta=self.model_cfg.rope_theta,
            seq_len=self.args.max_model_len,
            freq_allocation=self.model_cfg.freq_allocation,
        )
        return rope_emb

    def dy_input_preprocess(self, tasks):
        """
        dynamic insertion
        """
        for i in range(len(tasks)):
            task = tasks[i]
            idx = task.idx
            
            kwargs = {
                "max_length": task.get("max_tokens", 2048),
                "top_p": task.get("top_p", 0.8),
                "temperature": task.get("temperature", 0.2),
                "top_k": task.get("top_k", 0),
                "penalty_score": task.get("repetition_penalty", 1.0),
                "frequency_score": task.get("frequency_penalty", 0.0),
                "presence_score": task.get("presence_penalty", 0.0),
                "decode_strategy": "sampling",
                "pad_token_id": self.args.pad_token_id,
            }

            inputs = self._preprocess(task)
            if inputs.get("images") is not None:
                self.share_inputs["image_features"] = self.extract_vision_features(inputs)
            else:
                # 兼容没有图片和视频的情况
                self.share_inputs["image_features"] = None
            print("extract vision features done")
            self.share_inputs["rope_emb"][idx : idx + 1, :] = self.prepare_rope3d(inputs, **kwargs)
            print("prepare rope3d done")
            length = inputs["input_ids"].shape[1]
            self.share_inputs["input_ids"][idx : idx + 1, :length] = inputs["input_ids"]
            self.share_inputs["top_p"][idx : idx + 1] = kwargs["top_p"]
            self.share_inputs["temperature"][idx : idx + 1] = kwargs["temperature"]
            self.share_inputs["eos_token_id"][:] = np.array(task.eos_token_ids).astype("int64").reshape(-1, 1)
            self.share_inputs["penalty_score"][idx : idx + 1] = kwargs["penalty_score"]
            self.share_inputs["frequency_score"][idx : idx + 1] = kwargs["frequency_score"]
            self.share_inputs["presence_score"][idx : idx + 1] = kwargs["presence_score"]
            self.share_inputs["seq_lens_this_time"][idx : idx + 1] = length
            self.share_inputs["seq_lens_encoder"][idx : idx + 1] = length
            self.share_inputs["seq_lens_decoder"][idx : idx + 1] = 0
            self.share_inputs["step_idx"][idx : idx + 1] = 0
            self.share_inputs["min_dec_len"][idx : idx + 1] = 1
            self.share_inputs["max_dec_len"][idx : idx + 1] = kwargs["max_length"]
            self.share_inputs["stop_flags"][idx :idx + 1] = False
            self.share_inputs["pre_ids"][idx : idx + 1] = -1
            encoder_block_num = len(task.get("block_tables"))
            self.share_inputs["encoder_block_lens"][idx : idx + 1] = encoder_block_num
            self.share_inputs["block_tables"][idx : idx + 1, :] = -1
            self.share_inputs["block_tables"][idx : idx + 1, :encoder_block_num] = np.array(
                task.block_tables, dtype="int32"
            )

            from ..ops.gpu import reset_stop_value
            reset_stop_value(self.share_inputs["not_need_stop"])

    def generate(self):
        self.model(**self.share_inputs)

    def _cal_theortical_kvcache(self):
        """
        计算理论的kvcache大小
        """
        num_layers = self.model_cfg.get("num_layers", None) or self.model_cfg.get("num_hidden_layers", None)
        byte_of_cache = 2
        #TODO
        # 支持c8 c4

        hidden_size = self.model_cfg.hidden_size
        attention_heads = self.model_cfg.num_attention_heads
        hidden_dim = hidden_size / attention_heads * self.model_cfg.kv_num_head
        theoretical_kv_cache_memory = (2 * byte_of_cache * self.args.block_size * num_layers * hidden_dim)
        return theoretical_kv_cache_memory


    def _update_share_input_block_num(self):
        num_gpu_blocks = self.num_gpu_blocks

        del self.share_inputs["caches"]
        self._init_kvcache()

        del self.share_inputs["block_tables"]
        self.share_inputs["block_tables"] = paddle.full(
            [self.args.max_num_seqs, num_gpu_blocks], -1, dtype="int32"
        )

        # 初始化free list
        free_list = list(
            range(num_gpu_blocks - 1, int(num_gpu_blocks * self.args.kv_cache_ratio) - 1, -1)
        )
        self.free_list_len = len(free_list)
        self.share_inputs.update({
            "free_list": paddle.to_tensor(free_list, dtype="int32"),
            "free_list_len": paddle.full([1], self.free_list_len, dtype="int32"),
        })


    def dummy_input(self, num_total_tokens, number_of_tasks):
        """
        fake input to profile
        """
        full_length = num_total_tokens // number_of_tasks
        input_length = int(full_length * self.args.kv_cache_ratio)
        block_num = (input_length + self.args.block_size - 1 + self.args.enc_dec_block_num) // self.args.block_size

        for i in range(number_of_tasks):
            idx = i
            self.share_inputs["input_ids"][idx : idx + 1, :input_length] = np.array([5] * input_length)
            self.share_inputs["eos_token_id"][:] = np.array([2], dtype="int64").reshape(-1, 1)
            self.share_inputs["seq_lens_this_time"][idx : idx + 1] = input_length
            self.share_inputs["step_seq_lens_encoder"][idx : idx + 1] = input_length
            self.share_inputs["seq_lens_encoder"][idx : idx + 1] = input_length
            self.share_inputs["seq_lens_decoder"][idx : idx + 1] = 0
            self.share_inputs["step_idx"][idx : idx + 1] = 0
            self.share_inputs["max_dec_len"][idx : idx + 1] = 10
            self.share_inputs["stop_flags"][idx : idx + 1] = False

            self.share_inputs["first_token_ids"][idx : idx + 1] = self.share_inputs["input_ids"][idx : idx + 1, :1]
            self.share_inputs["ori_seq_lens_encoder"][idx : idx + 1] = input_length

            self.share_inputs["infer_seed"][idx : idx + 1] = random.randint(0, 922337203685477580)
            self.share_inputs["encoder_block_lens"][idx : idx + 1] = block_num
            self.share_inputs["block_tables"][idx : idx + 1, :block_num] = np.arange(idx * block_num, \
                                                                                (idx + 1) * block_num, 1)


    def _preprocess(self, task):
        """process batch"""
        one = task.multimodal_inputs
        print(one)

        input_ids = one["input_ids"][np.newaxis, :]
        input_ids = paddle.to_tensor(input_ids, dtype=paddle.int64)
        token_type_ids = one["token_type_ids"][np.newaxis, :]
        token_type_ids = paddle.to_tensor(token_type_ids, dtype=paddle.int64)
        print(f"token_type_ids {token_type_ids.shape} {token_type_ids.dtype}")

        if one["images"] is not None:
            image_type_ids = one["image_type_ids"][np.newaxis, :]
            images = one["images"]
            image_type_ids = paddle.to_tensor(image_type_ids, dtype=paddle.int64)
            images = paddle.to_tensor(images, dtype="uint8")
            grid_thw = paddle.to_tensor(one["grid_thw"], dtype="int64")
        else:
            image_type_ids = None
            images = None
            grid_thw = None

        if one["position_ids"] is not None:
            position_ids = paddle.to_tensor(one["position_ids"], dtype="int64").unsqueeze([0])
        else:
            position_ids = None

        result = dict(
                input_ids=input_ids,
                image_type_ids=image_type_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                grid_thw=grid_thw,
                images=images,
        )
        return result

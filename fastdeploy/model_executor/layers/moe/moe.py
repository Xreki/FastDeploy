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

# cipher_token=WjI1fQOvhN  # do not edit this line

import paddle
from paddle import nn
from paddle.distributed import fleet
from paddle.framework import in_dynamic_or_pir_mode
from paddle.nn.quant import weight_quantize
from paddlenlp.utils.log import logger

from fastdeploy.model_executor.layers.utils import (_set_var_distributed,
                                                    get_tensor,
                                                    per_block_cast_to_fp8)
from fastdeploy.model_executor.ops.gpu import (moe_expert_dispatch,
                                               moe_expert_ffn,
                                               moe_expert_reduce)


class FusedMoE(nn.Layer):
    """
    FusedMoE is a layer that performs MoE (Mixture of Experts) computation.
    """

    def __init__(
        self,
        llm_config,
        moe_config,
        layer_name,
        layer_idx=-1,
    ):
        """
        Initialize the Moe layer with given parameters.
        Args:
            llm_config (LLMConfig): Arguments related to inference, containing
                attributes such as weight_dtype, act_dtype, mp_size, hidden_size, head_dim,
                num_attention_heads, and ffn_hidden_size.

            layer_name (str): Unique name of the layer.
        """
        super().__init__()

        self.llm_config = llm_config
        self.layer_name = layer_name
        self.layer_idx = layer_idx

        self.weight_only_linear_arch = llm_config.quant_config.weight_only_linear_arch

        self.weight_dtype = llm_config.model_config.weight_dtype

        self.tp_size = llm_config.parallel_config.mp_size
        self.ep_size = llm_config.parallel_config.ep_size

        self.hidden_size = llm_config.model_config.hidden_size
        self.skip_quant = False
        self.moe_config = moe_config
        self.activation = self.moe_config.activation

        self.top_k = self.moe_config.top_k

        self.moe_quant_type = self.moe_config.moe_quant_type
        logger.info(f"MoE is running in {self.moe_quant_type} mode")

        self.num_experts = self.moe_config.num_experts
        self.num_local_experts = self.num_experts // self.ep_size

        if self.ep_size >= 2:
            logger.debug("MoE is running in ep mode")
            self.moe_intermediate_size = self.moe_config.moe_intermediate_size
        else:
            logger.debug(f"MoE is running in tp{self.tp_size} mode")
            self.moe_intermediate_size = (
                self.moe_config.moe_intermediate_size // self.tp_size)

        self.num_experts_start_offset = self.moe_config.num_experts_start_offset

        weight_keys = llm_config.load_config.weight_keys
        self.gate_weight_key = weight_keys.moe_gate_weight_keys.format(
            layer_idx)
        self.gate_correction_bias_key = weight_keys.moe_gate_correction_bias_keys.format(
            layer_idx)

        self.ffn1_expert_weight_key = weight_keys.moe_ffn1_weight_keys
        self.ffn2_expert_weight_key = weight_keys.moe_ffn2_weight_keys
        self.ffn1_bias_key = weight_keys.moe_ffn1_bias_keys
        self.ffn2_bias_key = weight_keys.moe_ffn2_bias_keys

        self.ffn1_expert_weight_scale_key = weight_keys.moe_ffn1_weight_scale_key
        self.ffn2_expert_weight_scale_key = weight_keys.moe_ffn2_weight_scale_key
        self.ffn1_expert_in_scale_key = weight_keys.moe_ffn1_expert_in_scale_key
        self.ffn2_expert_in_scale_key = weight_keys.moe_ffn2_expert_in_scale_key

        self.with_moe_ffn1_bias = self.ffn1_bias_key is not None
        self.with_moe_ffn2_bias = self.ffn2_bias_key is not None

        self.gate_weight_name = self.layer_name + ".gate.weight"
        self.gate_correction_bias_name = self.layer_name + ".gate.correction_bias"
        self.ffn1_weight_name = self.layer_name + ".ffn1.weight"
        self.ffn2_weight_name = self.layer_name + ".ffn2.weight"
        self.ffn1_bias_name = self.layer_name + ".ffn1.bias"
        self.ffn2_bias_name = self.layer_name + ".ffn2.bias"

        self.ffn1_shared_weight_name = self.layer_name + ".ffn1_shared.weight"
        self.ffn1_shared_bias_name = self.layer_name + ".ffn1_shared.bias"
        self.ffn2_shared_weight_name = self.layer_name + ".ffn2_shared.weight"
        self.ffn2_shared_bias_name = self.layer_name + ".ffn2_shared.bias"

        self._dtype = paddle.get_default_dtype()

        if (self.moe_quant_type == "weight_only_int8"
                or self.moe_quant_type == "weight_only_int4"
                or self.moe_quant_type == "w4a8"):
            self.init_weight_only_scale()
        elif self.moe_quant_type == "fp8":
            self.init_weight_block_scale()

        self.init_weight()

    def init_weight_block_scale(self):
        """init_weight_block_scale for fp8"""
        self.moe_ffn1_weight_scale = self.create_parameter(
            shape=[
                self.num_local_experts,
                self.moe_intermediate_size * 2 // 128,
                self.hidden_size // 128,
            ],  # [g, n, k]
            attr=paddle.ParamAttr(
                name=f"{self.layer_name}.linear1.weight_block_scale"),
            dtype="float32",
            is_bias=False,
        )
        self.moe_ffn2_weight_scale = self.create_parameter(
            shape=[
                self.num_local_experts,
                self.hidden_size // 128,
                self.moe_intermediate_size // 128,
            ],  # [g, n, k]
            attr=paddle.ParamAttr(
                name=f"{self.layer_name}.linear2.weight_block_scale"),
            dtype="float32",
            is_bias=False,
        )

    def get_weight_create_dtype(self):
        """
        Get the data type for creating weights based on quantization settings.

        Args:
            self (object): The instance of the class where this method is defined.

        Returns:
            str: The data type for creating weights. It depends on the quantization settings:
                - If `self.skip_quant` is True, returns the original data type `self._dtype`.
                - If `self.inference_args.use_weight_only` is True and `self.weight_dtype` is "int4",
                  returns "int8" to ensure compatibility or optimization.
                - Otherwise, returns the specified weight data type `self.weight_dtype`.
        """
        if self.skip_quant:
            return self._dtype
        if (self.moe_quant_type == "weight_only_int4"
                or self.moe_quant_type == "weight_only_int8"
                or self.moe_quant_type == "w4a8"):
            return "int8"
        return self.weight_dtype

    def init_weight_only_scale(self):
        """
        Initialize the weight scale.
        """

        assert self.layer_idx >= self.moe_config.moe_layer_start_index
        self.moe_ffn1_weight_scale = self.create_parameter(
            shape=[self.num_local_experts, self.moe_intermediate_size * 2],
            attr=paddle.ParamAttr(
                name=f"{self.layer_name}.linear1.weight_scale"),
            dtype=self._dtype,
            is_bias=False,
        )
        self.moe_ffn2_weight_scale = self.create_parameter(
            shape=[self.num_local_experts, self.hidden_size],
            attr=paddle.ParamAttr(
                name=f"{self.layer_name}.linear2.weight_scale"),
            dtype=self._dtype,
            is_bias=False,
        )
        if self.moe_quant_type == "w4a8":
            self.moe_ffn1_in_scale = self.create_parameter(
                shape=[self.num_local_experts],
                attr=paddle.ParamAttr(
                    name=f"{self.layer_name}.linear1.in_scale"),
                dtype="float32",
                is_bias=False,
            )
            self.moe_ffn2_in_scale = self.create_parameter(
                shape=[self.num_local_experts],
                attr=paddle.ParamAttr(
                    name=f"{self.layer_name}.linear2.in_scale"),
                dtype="float32",
                is_bias=False,
            )

        if self.moe_config.moe_use_ffn_shared_weight_and_bias:
            self.moe_ffn1_shared_weight_scale = self.create_parameter(
                shape=[1],
                attr=paddle.ParamAttr(
                    name=f"{self.layer_name}.linear1_shared.weight_scale"),
                dtype=self._dtype,
                is_bias=False,
            )
            self.moe_ffn2_shared_weight_scale = self.create_parameter(
                shape=[1],
                attr=paddle.ParamAttr(
                    name=f"{self.layer_name}.linear2_shared.weight_scale"),
                dtype=self._dtype,
                is_bias=False,
            )

    def load_scale_state_dict(self):
        """
        load_scale_state_dict function.
        """
        up_gate_proj_weight_scale = []
        down_proj_weight_scale = []
        up_gate_proj_in_scale = []
        down_proj_in_scale = []

        for j in range(self.num_experts):
            up_gate_proj_weight_scale.append(
                self.inference_args.weight_scale_dict.pop(
                    self.ffn1_expert_weight_scale_key.format(j)))
            down_proj_weight_scale.append(
                self.inference_args.weight_scale_dict.pop(
                    self.ffn2_expert_weight_scale_key.format(j)))
            up_gate_proj_in_scale.append(
                self.inference_args.act_scale_dict.pop(
                    self.ffn1_expert_in_scale_key.format(j)))
            down_proj_in_scale.append(
                self.inference_args.act_scale_dict.pop(
                    self.ffn2_expert_in_scale_key.format(j)))
        return (
            up_gate_proj_weight_scale,
            down_proj_weight_scale,
            up_gate_proj_in_scale,
            down_proj_in_scale,
        )

    def init_weight_shape(self):
        """
        Initialize the weight shape for the moe layer.
        """
        # gate shape
        self.gate_weight_shape = [self.hidden_size, self.num_experts]
        self.gate_correction_bias_shape = [1, self.num_experts]

        # ffn1 shape
        if self.moe_quant_type == "fp8":
            self.ffn1_weight_shape = [
                self.num_local_experts,
                self.moe_intermediate_size * 2,
                self.hidden_size,
            ]
        elif self.moe_quant_type == "weight_only_int4":
            self.ffn1_weight_shape = [
                self.num_local_experts,
                self.hidden_size,
                self.moe_intermediate_size,
            ]
        elif self.moe_quant_type == "w4a8":
            self.ffn1_weight_shape = [
                self.num_local_experts,
                self.moe_intermediate_size * 2,
                self.hidden_size // 2,
            ]
        else:
            self.ffn1_weight_shape = [
                self.num_local_experts,
                self.hidden_size,
                self.moe_intermediate_size * 2,
            ]
        if not self.activation.endswith("glu"):
            self.ffn1_weight_shape = [
                self.num_local_experts,
                self.hidden_size,
                self.moe_intermediate_size,
            ]
        self.ffn1_bias_shape = (
            [self.num_local_experts, self.moe_intermediate_size *
             2] if self.activation.endswith("glu") else
            [self.num_local_experts, self.moe_intermediate_size])
        # ffn2 shape
        if self.moe_quant_type == "fp8":
            self.ffn2_weight_shape = [
                self.num_local_experts,
                self.hidden_size,
                self.moe_intermediate_size,
            ]
        elif self.moe_quant_type == "weight_only_int4":
            self.ffn2_weight_shape = [
                self.num_local_experts,
                self.moe_intermediate_size,
                self.hidden_size // 2,
            ]
        elif self.moe_quant_type == "w4a8":
            self.ffn2_weight_shape = [
                self.num_local_experts,
                self.hidden_size,
                self.moe_intermediate_size // 2,
            ]
        else:
            self.ffn2_weight_shape = [
                self.num_local_experts,
                self.moe_intermediate_size,
                self.hidden_size,
            ]
        self.ffn2_bias_shape = [self.num_local_experts, self.hidden_size]

    def init_weight(self):
        """
        Initialize the weights and biases.
        """

        self.init_weight_shape()
        # gate
        self.gate_weight = self.create_parameter(
            shape=self.gate_weight_shape,
            attr=paddle.ParamAttr(name=self.gate_weight_name),
            dtype="float32",
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        if self.moe_config.moe_use_gate_correction_bias:
            self.gate_correction_bias = self.create_parameter(
                shape=self.gate_correction_bias_shape,
                attr=paddle.ParamAttr(name=self.gate_correction_bias_name),
                dtype="float32",
                is_bias=True,
                default_initializer=paddle.nn.initializer.Constant(0),
            )

        # ffn1
        self.moe_ffn1_weight = self.create_parameter(
            shape=self.ffn1_weight_shape,
            attr=paddle.ParamAttr(name=self.ffn1_weight_name),
            dtype=self.get_weight_create_dtype(),
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        self.moe_ffn1_bias = None
        if self.with_moe_ffn1_bias:
            self.moe_ffn1_bias = self.create_parameter(
                shape=self.ffn1_bias_shape,
                attr=paddle.ParamAttr(name=self.ffn1_bias_name),
                dtype=self._dtype,
                is_bias=True,
            )

        # ffn2
        self.moe_ffn2_weight = self.create_parameter(
            shape=self.ffn2_weight_shape,
            attr=paddle.ParamAttr(name=self.ffn2_weight_name),
            dtype=self.get_weight_create_dtype(),
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        self.moe_ffn2_bias = None
        if self.with_moe_ffn2_bias:
            self.moe_ffn2_bias = self.create_parameter(
                shape=self.ffn2_bias_shape,
                attr=paddle.ParamAttr(name=self.ffn2_bias_name),
                dtype=self._dtype,
                is_bias=True,
            )

        if self.tp_size > 0:
            # column parallel
            _set_var_distributed(self.moe_ffn1_weight, split_axis=1)
            _set_var_distributed(self.moe_ffn1_bias, split_axis=0)
            # row parallel
            _set_var_distributed(self.moe_ffn2_weight, split_axis=0)

    def load_gate_state_dict(self, state_dict):
        """
        load_gate_state_dict function.
        """
        logger.info("Load TP FFN1")
        up_gate_proj_weight = []
        down_proj_weight = []
        for j in range(self.num_experts):
            up_gate_proj_weight.append(
                get_tensor(
                    state_dict.pop(
                        self.ffn1_expert_weight_key.format(self.layer_idx,
                                                           j))))
            down_proj_weight.append(
                get_tensor(
                    state_dict.pop(
                        self.ffn2_expert_weight_key.format(self.layer_idx,
                                                           j))))
        return up_gate_proj_weight, down_proj_weight

    def load_gate_correction_bias(self, state_dict):
        """
        load_gate_correction_bias function.
        """
        if self.moe_config.moe_use_gate_correction_bias:
            gate_correction_bias_tensor = get_tensor(
                state_dict.pop(self.gate_correction_bias_key))
            self.gate_correction_bias.set_value(gate_correction_bias_tensor)

    def load_state_dict(self, state_dict):
        """
        load_state_dict function.
        """
        # gate
        gate_weight_tensor = get_tensor(state_dict.pop(self.gate_weight_key))
        self.gate_weight.set_value(gate_weight_tensor)

        self.load_gate_correction_bias(state_dict)

        up_gate_proj_weight, down_proj_weight = self.load_gate_state_dict(
            state_dict)

        if self.moe_quant_type == "w4a8":
            (
                up_gate_proj_weight_scale_list,
                down_proj_weight_scale_list,
                up_gate_proj_in_scale_list,
                down_proj_in_scale_list,
            ) = self.load_scale_state_dict()
            ffn1_in_scale_tensor = paddle.to_tensor(up_gate_proj_in_scale_list,
                                                    dtype="float32").reshape_([
                                                        self.num_local_experts
                                                    ])
            self.moe_ffn1_in_scale.set_value(ffn1_in_scale_tensor)
            ffn2_in_scale_tensor = paddle.to_tensor(down_proj_in_scale_list,
                                                    dtype="float32").reshape_([
                                                        self.num_local_experts
                                                    ])
            self.moe_ffn2_in_scale.set_value(ffn2_in_scale_tensor)

        # ffn1
        ffn1_weight_tensor = paddle.concat(
            up_gate_proj_weight,
            axis=0).reshape_([self.num_local_experts, self.hidden_size, -1])
        ffn1_weight_tensor_list = []
        ffn1_weight_scale_tensor_list = []
        if self.moe_quant_type == "fp8":
            ffn1_weight_tensor = ffn1_weight_tensor.transpose([0, 2, 1])
            ffn1_fp8 = (
                paddle.empty_like(ffn1_weight_tensor,
                                  dtype=paddle.float8_e4m3fn),
                paddle.empty(
                    (
                        self.num_local_experts,
                        (self.ffn1_weight_shape[1] + 127) // 128,
                        self.ffn1_weight_shape[2] // 128,
                    ),
                    dtype=paddle.float32,
                ),
            )

            for i in range(self.num_local_experts):
                quanted_weight_tensor, weight_block_scale_tensor = (
                    per_block_cast_to_fp8(ffn1_weight_tensor[i]))
                paddle.assign(quanted_weight_tensor, ffn1_fp8[0][i])
                paddle.assign(weight_block_scale_tensor, ffn1_fp8[1][i])
            self.moe_ffn1_weight.copy_(ffn1_fp8[0], False)
            self.moe_ffn1_weight_scale.set_value(ffn1_fp8[1])
        elif self.moe_quant_type == "w4a8":
            if paddle.is_compiled_with_cuda():
                for i in range(self.num_local_experts):
                    ffn1_weight_tensor_i, _ = weight_quantize(
                        ffn1_weight_tensor[i].cast("int8"),
                        algo="w4a8",
                        arch=80,
                    )
                    ffn1_weight_tensor_list.append(
                        ffn1_weight_tensor_i.reshape(
                            [-1, self.hidden_size // 2]))
                ffn1_weight_scale_tensor_list = up_gate_proj_weight_scale_list
                ffn1_weight_tensor = paddle.concat(ffn1_weight_tensor_list,
                                                   axis=0)
            ffn1_weight_scale_tensor = paddle.concat(
                ffn1_weight_scale_tensor_list, axis=0)
            self.moe_ffn1_weight_scale.set_value(
                ffn1_weight_scale_tensor.cast(
                    paddle.get_default_dtype()).reshape(
                        [self.num_local_experts, -1]))
            self.moe_ffn1_weight.set_value(
                ffn1_weight_tensor.reshape(
                    [self.num_local_experts, -1,
                     ffn1_weight_tensor.shape[-1]]))
        elif self.moe_quant_type in ["weight_only_int4", "weight_only_int8"]:
            for i in range(self.num_local_experts):
                quant_weight, scale = weight_quantize(
                    ffn1_weight_tensor[i],
                    algo=self.moe_quant_type,
                    arch=self.weight_only_linear_arch,
                )
                ffn1_weight_tensor_list.append(quant_weight)
                ffn1_weight_scale_tensor_list.append(scale)
            ffn1_weight_tensor = paddle.concat(ffn1_weight_tensor_list, axis=0)
            self.moe_ffn1_weight.set_value(
                ffn1_weight_tensor.reshape(self.ffn1_weight_shape))
            ffn1_weight_scale_tensor = paddle.stack(
                ffn1_weight_scale_tensor_list, axis=0)
            self.moe_ffn1_weight_scale.set_value(ffn1_weight_scale_tensor)

        else:  # default
            self.moe_ffn1_weight.set_value(
                ffn1_weight_tensor.reshape(
                    [self.num_experts, self.hidden_size, -1]))
            if self.with_moe_ffn1_bias:
                moe_ffn1_bias_tensor = get_tensor(
                    state_dict.pop(self.ffn1_bias_key))
                self.moe_ffn1_bias.set_value(moe_ffn1_bias_tensor)

        # ffn2
        ffn2_weight_tensor = paddle.concat(down_proj_weight, axis=0).reshape_(
            [self.num_local_experts, -1, self.hidden_size])
        ffn2_weight_tensor_list = []
        ffn2_weight_scale_tensor_list = []
        if self.moe_quant_type == "fp8":
            ffn2_weight_tensor = ffn2_weight_tensor.transpose([0, 2, 1])
            ffn2_fp8 = (
                paddle.empty_like(ffn2_weight_tensor,
                                  dtype=paddle.float8_e4m3fn),
                paddle.empty(
                    (
                        self.num_local_experts,
                        (self.ffn2_weight_shape[1] + 127) // 128,
                        self.ffn2_weight_shape[2] // 128,
                    ),
                    dtype=paddle.float32,
                ),
            )
            for i in range(self.num_local_experts):
                quanted_weight_tensor, weight_block_scale_tensor = (
                    per_block_cast_to_fp8(ffn2_weight_tensor[i]))
                paddle.assign(quanted_weight_tensor, ffn2_fp8[0][i])
                paddle.assign(weight_block_scale_tensor, ffn2_fp8[1][i])
            self.moe_ffn2_weight.copy_(ffn2_fp8[0], False)
            self.moe_ffn2_weight_scale.set_value(ffn2_fp8[1])
        elif self.moe_quant_type == "w4a8":
            if paddle.is_compiled_with_cuda():
                for i in range(self.num_local_experts):
                    ffn2_weight_tensor_i, _ = weight_quantize(
                        ffn2_weight_tensor[i].cast("int8"),
                        algo="w4a8",
                        arch=80,
                    )
                    ffn2_weight_tensor_list.append(
                        ffn2_weight_tensor_i.reshape([self.hidden_size, -1]))
            ffn2_weight_scale_tensor_list = down_proj_weight_scale_list
            ffn2_weight_tensor = paddle.concat(ffn2_weight_tensor_list, axis=0)
            ffn2_weight_scale_tensor = paddle.concat(
                ffn2_weight_scale_tensor_list, axis=0)
            self.moe_ffn2_weight_scale.set_value(
                ffn2_weight_scale_tensor.cast(
                    paddle.get_default_dtype()).reshape(
                        [self.num_local_experts, -1]))
            self.moe_ffn2_weight.set_value(
                ffn2_weight_tensor.reshape(
                    [self.num_local_experts, -1,
                     ffn2_weight_tensor.shape[-1]]))
        elif self.moe_quant_type in ["weight_only_int4", "weight_only_int8"]:
            for i in range(self.num_local_experts):
                quant_weight, scale = weight_quantize(
                    ffn2_weight_tensor[i],
                    algo=self.moe_quant_type,
                    arch=self.weight_only_linear_arch,
                )
                ffn2_weight_tensor_list.append(quant_weight)
                ffn2_weight_scale_tensor_list.append(scale)
            ffn2_weight_tensor = paddle.concat(ffn2_weight_tensor_list, axis=0)
            ffn2_weight_scale_tensor = paddle.stack(
                ffn2_weight_scale_tensor_list, axis=0)
            self.moe_ffn2_weight_scale.set_value(ffn2_weight_scale_tensor)
            self.moe_ffn2_weight.set_value(
                ffn2_weight_tensor.reshape(self.ffn2_weight_shape))
        else:
            self.moe_ffn2_weight.set_value(
                ffn2_weight_tensor.reshape([
                    self.num_experts,
                    self.moe_intermediate_size,
                    -1,
                ]))
        if self.with_moe_ffn2_bias:
            moe_ffn2_bias_tensor = get_tensor(
                state_dict.pop(self.ffn2_bias_key))
            self.moe_ffn2_bias.set_value(moe_ffn2_bias_tensor)

    def forward(self, x, **kwargs):
        """
        Defines the forward computation of the moe layer.

        Args:
            x (Tensor): Input tensor to the moe layer.

        Returns:
            Tensor: Output tensor.

        """
        gate_out = paddle.matmul(x.cast("float32"), self.gate_weight)

        (
            permute_input,
            token_nums_per_expert,
            permute_indices_per_token,
            top_k_weights,
            top_k_indices,
        ) = moe_expert_dispatch(
            x,
            gate_out,
            (self.gate_correction_bias
             if self.moe_config.moe_use_gate_correction_bias else None),
            self.top_k,
            self.moe_config.moe_group,
            topk_only_mode=False,
        )

        ffn_out = moe_expert_ffn(
            permute_input,
            token_nums_per_expert,
            self.moe_ffn1_weight,
            self.moe_ffn2_weight,
            self.moe_ffn1_bias,
            (self.moe_ffn1_weight_scale
             if hasattr(self, "moe_ffn1_weight_scale") else None),
            (self.moe_ffn2_weight_scale
             if hasattr(self, "moe_ffn2_weight_scale") else None),
            (self.moe_ffn2_in_scale
             if hasattr(self, "moe_ffn2_in_scale") else None),
            None,  # expert_idx_per_token
            self.moe_quant_type,
            False,  # used_in_ep_low_latency
        )

        if self.with_moe_ffn1_bias and self.tp_size > 1:
            if in_dynamic_or_pir_mode():
                hcg = fleet.get_hybrid_communicate_group()
                mp_group = hcg.get_model_parallel_group()
                paddle.distributed.all_reduce(ffn_out, group=mp_group)
            else:
                paddle.distributed.all_reduce(ffn_out, group=mp_group)

        # reduce 中会做 topk 个 weight 的 norm 和 routed_scaling_factor
        fused_moe_out = moe_expert_reduce(
            ffn_out,
            top_k_weights,
            permute_indices_per_token,
            top_k_indices,
            self.moe_ffn2_bias if self.with_moe_ffn2_bias else None,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
        )
        return fused_moe_out

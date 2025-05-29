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

import paddle
from paddle import nn
from paddle.nn.quant import weight_only_linear, weight_quantize
from paddlenlp.utils.log import logger

import fastdeploy
import fastdeploy.model_executor.ops.gpu.deep_gemm as deep_gemm

from .utils import _set_var_distributed, get_tensor, per_block_cast_to_fp8


class QKVLinear(nn.Layer):
    """
    QKVLinear Layer.
    """

    def __init__(self,
                 inference_args,
                 layer_name,
                 weight_key,
                 bias_key=None,
                 skip_quant=False):
        """
        Initialize the QKV Linear layer with given parameters.

        Args:
            inference_args (dict-like): Contains inference-related parameters, including
                - weight_dtype (str): Data type of weights, e.g., 'float32', 'int8'.
                - act_dtype (str): Data type of activations, e.g., 'float32', 'int8'.
                - mp_size (int): Model parallelism size, used for distributed training.
                - num_attention_heads (int): Total number of attention heads.
                - num_key_value_heads (int): Number of key-value heads, often equal to num_attention_heads.
                - hidden_size (int): Hidden dimension size.
                - ffn_hidden_size (int): Feedforward network hidden dimension size.
                - head_dim (int): Dimension per attention head.
                - use_weight_only (bool): Whether to use weight-only quantization.

            layer_name (str): Unique name of the layer, used for naming weights and biases.
            weight_key (str): Key name of weight in the pdparams state dict.
            bias_key (str): Key name of bias in the pdparams state dict. Defaults to None, means no bias.
            with_bias (bool, optional): Whether to include bias term. Defaults to True.
            skip_quant (bool, optional): Whether to skip quantization steps. Defaults to False.
        """
        super().__init__()
        self.inference_args = inference_args
        self.with_bias = bias_key is not None
        self.skip_quant = skip_quant
        self.weight_dtype = inference_args.weight_dtype
        self.act_dtype = inference_args.act_dtype
        self.nranks = inference_args.mp_size
        self.num_heads = inference_args.num_attention_heads // self.nranks
        self.kv_num_heads = inference_args.num_key_value_heads // self.nranks
        self.embed_dim = inference_args.hidden_size
        self.dim_feedforward = inference_args.dim_feedforward // self.nranks
        self.head_dim = inference_args.head_dim

        self.weight_key = weight_key
        self.bias_key = bias_key

        self.layer_name = layer_name
        self.weight_name = self.layer_name + ".weight"
        self.bias_name = self.layer_name + ".bias"
        self.weight_scale_layer_name = self.layer_name + ".weight_scale"
        self.out_scale_layer_name = self.layer_name + ".out_scale"
        self._dtype = self._helper.get_default_dtype()

        if inference_args.use_weight_only:
            self.init_weight_only_scale()
        if self.inference_args.weight_block_size[0] != -1:
            logger.debug("qkv use_fp8_blockwise")
            self.init_weight_block_scale()
        if (inference_args.weight_dtype == "int8" and inference_args.act_dtype
                == "int8") or ("float8" in inference_args.weight_dtype
                               and "float8" in inference_args.act_dtype):
            self.set_ptq_scale()  # init and load scale
        self.init_weight()

    def init_weight_shape(self, trans_qkvw=True):
        """
        Initialize the weight shape for the first feedforward network layer.

        Args:
            trans (bool, optional): Whether to transpose the weight shape.
                Defaults to False. If True, the shape will be reversed.

        Returns:
            None.
        """

        self.qkv_weight_shape = ([
            (self.num_heads + 2 * self.kv_num_heads) * self.head_dim,
            self.embed_dim,
        ] if trans_qkvw else [
            self.embed_dim,
            (self.num_heads + 2 * self.kv_num_heads) * self.head_dim,
        ])
        if self.weight_dtype == "int4":
            self.qkv_weight_shape[0] //= 2

    def get_weight_create_dtype(self):
        """
        Get the data type for creating weights based on quantization settings.

        Args:
            self (object): The instance of the class where this method is defined.

        Returns:
            str: The data type for creating weights. It depends on the quantization settings:
                - If `self.skip_quant` is True, returns the original data type `self._dtype`.
                - If `self.weight_dtype` is "int4", returns "int8" to ensure compatibility or optimization.
                - Otherwise, returns the specified weight data type `self.weight_dtype`.
        """
        if self.skip_quant:
            return self._dtype
        if self.weight_dtype == "int4":
            return "int8"
        # TODO(wangzhe24) create_parameter not support FP8
        if "float8" in self.weight_dtype:
            return self._dtype
        return self.weight_dtype

    def init_weight_only_scale(self):
        """
        Initialize the weight scale.
        """
        self.qkv_weight_scale = self.create_parameter(
            shape=[(self.num_heads + 2 * self.kv_num_heads) * self.head_dim],
            attr=paddle.ParamAttr(name=self.weight_scale_layer_name),
            dtype=self._dtype,
            is_bias=False,
        )

    def init_weight_block_scale(self):
        """init_weight_block_scale for fp8"""
        self.qkv_weight_scale = self.create_parameter(
            shape=[
                ((self.num_heads + 2 * self.kv_num_heads) * self.head_dim +
                 127) // 128,
                (self.embed_dim + 127) // 128,
            ],
            attr=paddle.ParamAttr(name=self.layer_name +
                                  ".weight_block_scale"),
            dtype="float32",
            is_bias=False,
        )

    def init_weight(self):
        """
        Initialize the weights and biases.
        """
        self.init_weight_shape()
        self.qkv_weight = self.create_parameter(
            shape=self.qkv_weight_shape,
            attr=paddle.ParamAttr(name=self.weight_name),
            dtype=self.get_weight_create_dtype(),
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        self.qkv_bias = None
        if self.with_bias:
            self.qkv_bias = self.create_parameter(
                shape=[(self.num_heads + 2 * self.kv_num_heads) * self.head_dim
                       ],
                attr=paddle.ParamAttr(name=self.bias_name),
                dtype=self._dtype,
                is_bias=True,
            )
        if self.nranks > 0:
            # column parallel
            _set_var_distributed(self.qkv_weight, split_axis=1)
            _set_var_distributed(self.qkv_bias, split_axis=0)

    def set_ptq_scale(self):
        """
        Set the post-training quantization (PTQ) scale for the layer.

        This method fetches weight and input activation scales from the inference arguments,
        and computes the output scale for the layer.
        It also handles skipping quantization for missing scales.

        Args:
            None (Method operates on the instance's attributes and arguments.)

        Returns:
            None (Modifies the instance's attributes.)

        Raises:
            None
        """
        if self.inference_args.weight_block_size[0] != -1:
            return

        weight_scale = self.inference_args.weight_scale_dict.get(
            self.layer_name + ".weight_quanter")
        in_scale = self.inference_args.act_scale_dict.get(
            self.layer_name + ".activation_quanter")

        if weight_scale is None or in_scale is None:
            logger.debug(f"{self.layer_name} skip quant")
            self.skip_quant = True
            return

        if "float8" in self.weight_dtype:
            max_range = 448.0
            self.scalar_scale_name = self.layer_name + ".scalar_weight_quanter"
            self.scalar_scale = self.create_parameter(
                shape=([1]),
                attr=paddle.ParamAttr(name=self.scalar_scale_name),
                dtype="float32",
            )
            self.scalar_scale.set_value(
                paddle.to_tensor([1.0 / (max_range * in_scale)],
                                 dtype="float32"))
            qkv_scale = weight_scale / max_range
        else:
            max_range = 127.0
            qkv_scale = weight_scale / (max_range * max_range * in_scale)
        self.qkv_out_scale = self.create_parameter(
            shape=[self.head_dim * (2 * self.kv_num_heads + self.num_heads)],
            attr=paddle.ParamAttr(name=self.out_scale_layer_name),
            dtype="float32",
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        if self.inference_args.num_key_value_heads <= 0:
            qkv_weight_scale = (paddle.to_tensor(qkv_scale).reshape([
                self.num_heads,
                3,
                self.embed_dim // self.inference_args.num_attention_heads,
            ]).transpose((1, 0, 2)).reshape([-1]).astype("float32"))
        else:
            # GQA
            qkv_weight_scale = (paddle.to_tensor(qkv_scale).reshape([
                self.num_heads + 2 * self.kv_num_heads,
                self.embed_dim // self.inference_args.num_attention_heads,
            ]).astype("float32"))
            single_qkv_weight_scales = paddle.split(qkv_weight_scale,
                                                    self.kv_num_heads,
                                                    axis=0)
            q_weight_scales, k_weight_scales, v_weight_scales = [], [], []
            for single_qkv_weight_scale in single_qkv_weight_scales:
                q_weight_scale, k_weight_scale, v_weight_scale = paddle.split(
                    single_qkv_weight_scale,
                    [
                        self.inference_args.num_attention_heads //
                        self.inference_args.num_key_value_heads,
                        1,
                        1,
                    ],
                    axis=0,
                )
                q_weight_scales.append(q_weight_scale)
                k_weight_scales.append(k_weight_scale)
                v_weight_scales.append(v_weight_scale)

            q_weight_scale = paddle.concat(q_weight_scales, axis=0)
            k_weight_scale = paddle.concat(k_weight_scales, axis=0)
            v_weight_scale = paddle.concat(v_weight_scales, axis=0)

            qkv_weight_scale = paddle.concat(
                [q_weight_scale, k_weight_scale, v_weight_scale],
                axis=0).reshape([-1])
        self.qkv_out_scale.set_value(qkv_weight_scale)

    def load_state_dict_wint8(self, qkv_proj_weight):
        """
        Load the quantized weight for QKV projection in INT8 precision.

        Args:
            qkv_proj_weight (Tensor): The weight tensor for QKV projection before quantization.

        """
        if not self.inference_args.moe_config.use_moe:
            # Transpose Back to RowMajor.
            qkv_proj_weight = qkv_proj_weight.reshape([-1, self.embed_dim
                                                       ]).transpose([1, 0])
            qkv_quanted_weight_tensor, qkv_weight_scale_tensor = weight_quantize(
                qkv_proj_weight,
                algo="weight_only_int8",
                arch=self.inference_args.weight_only_linear_arch,
            )
        else:
            gqa_hidden_size = (
                self.inference_args.num_attention_heads // self.nranks +
                2 * self.inference_args.num_key_value_heads // self.nranks) * (
                    self.embed_dim // self.inference_args.num_attention_heads)
            qkv_proj_weight = qkv_proj_weight.reshape_(
                [gqa_hidden_size, self.embed_dim])
            qkv_proj_weight = paddle.transpose(
                qkv_proj_weight,
                perm=[1, 0])  # ConvertBack to RowMajor Weight and to CPU.
            qkv_quanted_weight_tensor, qkv_weight_scale_tensor = weight_quantize(
                qkv_proj_weight,
                algo="weight_only_int8",
                arch=self.inference_args.weight_only_linear_arch,
            )
            qkv_quanted_weight_tensor.reshape_([
                gqa_hidden_size,
                self.embed_dim,
            ])
        self.qkv_weight.set_value(qkv_quanted_weight_tensor)
        self.qkv_weight_scale.set_value(
            qkv_weight_scale_tensor.astype(paddle.get_default_dtype()))

    def load_state_dict_wint4(self, qkv_proj_weight):
        """
        Load and quantize the QKV projection weight tensor to int4 format.

        Args:
            qkv_proj_weight (paddle.Tensor): The original QKV projection weight tensor,
                expected to be in ColumnMajor format.

        Returns:
            None. The quantized weight tensor and scale are set to the internal `qkv_weight`
            and `qkv_weight_scale` parameters respectively.
        """
        if not self.inference_args.moe_config.use_moe:
            # Transpose Back to RowMajor.
            qkv_proj_weight = qkv_proj_weight.reshape([-1, self.embed_dim
                                                       ]).transpose([1, 0])
            qkv_proj_weight = paddle.to_tensor(qkv_proj_weight).cpu()
            qkv_quanted_weight_tensor, qkv_weight_scale_tensor = weight_quantize(
                qkv_proj_weight,
                algo="weight_only_int4",
                arch=self.inference_args.weight_only_linear_arch,
            )
        else:
            gqa_hidden_size = (
                self.inference_args.num_attention_heads // self.nranks +
                2 * self.inference_args.num_key_value_heads // self.nranks) * (
                    self.embed_dim // self.inference_args.num_attention_heads)
            qkv_proj_weight = qkv_proj_weight.reshape_(
                [gqa_hidden_size, self.embed_dim])
            qkv_proj_weight = paddle.transpose(
                qkv_proj_weight,
                perm=[1, 0])  # ConvertBack to RowMajor Weight and to CPU.
            qkv_quanted_weight_tensor, qkv_weight_scale_tensor = weight_quantize(
                qkv_proj_weight,
                algo="weight_only_int4",
                arch=self.inference_args.weight_only_linear_arch,
            )
            qkv_quanted_weight_tensor.reshape_([
                gqa_hidden_size // 2,
                self.embed_dim,
            ])
        self.qkv_weight.set_value(qkv_quanted_weight_tensor)
        self.qkv_weight_scale.set_value(qkv_weight_scale_tensor)

    def load_state_dict_wint4_fp8(self, qkv_proj_weight):
        """
        Load and quantize the QKV projection weight tensor to int4 format.

        Args:
            qkv_proj_weight (paddle.Tensor): The original QKV projection weight tensor,
                expected to be in ColumnMajor format.

        Returns:
            None. The quantized weight tensor and scale are set to the internal `qkv_weight`
            and `qkv_weight_scale` parameters respectively.
        """
        # Transpose Back to RowMajor.
        qkv_proj_weight = qkv_proj_weight.reshape([-1, self.embed_dim
                                                   ]).transpose([1, 0])
        qkv_proj_weight = paddle.to_tensor(qkv_proj_weight).cpu()
        qkv_quanted_weight_tensor, qkv_weight_scale_tensor = (
            fastdeploy.model_executor.ops.gpu.
            scaled_gemm_f8_i4_f16_weight_quantize(
                paddle.cast(qkv_proj_weight, "float32"),
                groupsize=-1,
                scale_dtype="float16",
            ))
        qkv_weight_scale_tensor = paddle.view(qkv_weight_scale_tensor,
                                              self._dtype)
        self.qkv_weight.set_value(qkv_quanted_weight_tensor)
        self.qkv_weight_scale.set_value(qkv_weight_scale_tensor)

    def load_state_dict_block_fp8(self, qkv_proj_weight):
        """
        Load the quantized weight for QKV projection in INT8 precision.

        Args:
            qkv_proj_weight (Tensor): The weight tensor for QKV projection before quantization.
        """
        qkv_quanted_weight_tensor, qkv_weight_scale_tensor = per_block_cast_to_fp8(
            qkv_proj_weight)
        self.qkv_weight.copy_(qkv_quanted_weight_tensor, False)
        self.qkv_weight_scale.set_value(qkv_weight_scale_tensor)

    def load_state_dict(self, state_dict):
        """
        Load the checkpoint state dictionary into the layer.

        Args:
            state_dict (dict): A dictionary containing the checkpoint weights and biases.
        """
        # weight
        if self.inference_args.num_key_value_heads <= 0:
            qkv_proj_weight = (get_tensor(state_dict.pop(
                self.weight_key)).reshape([
                    self.embed_dim,
                    self.num_heads,
                    3,
                    self.embed_dim // self.inference_args.num_attention_heads,
                ]).transpose([2, 1, 3, 0]))
        else:
            # qkv_weight [hidden_size, num_head + 2 * num_key_value_head, dim_head]
            # layout [q q q q k v] * num_key_value_head
            qkv_proj_weight = (get_tensor(state_dict.pop(
                self.weight_key)).reshape([
                    self.embed_dim,
                    self.num_heads + 2 * self.kv_num_heads,
                    self.embed_dim // self.inference_args.num_attention_heads,
                ]).transpose([1, 2, 0])).reshape([-1, self.embed_dim])

        # set weight
        if self.skip_quant:
            qkv_proj_weight = qkv_proj_weight.cast(self._dtype)
            self.qkv_weight.set_value(qkv_proj_weight)
        else:
            if self.inference_args.weight_block_size[0] != -1:
                self.load_state_dict_block_fp8(qkv_proj_weight)
            elif self.weight_dtype == "int8" and self.act_dtype in [
                    "bfloat16",
                    "float16",
                    "float32",
            ]:  # WINT8
                self.load_state_dict_wint8(qkv_proj_weight)
            elif self.weight_dtype == "int4" and self.act_dtype in [
                    "bfloat16",
                    "float16",
                    "float32",
            ]:  # WINT4
                self.load_state_dict_wint4(qkv_proj_weight)
            elif (self.weight_dtype == "int4"
                  and self.act_dtype == "float8_e4m3fn"):  # W4Afp8
                self.load_state_dict_wint4_fp8(qkv_proj_weight)
            else:  # bf16/fp16/fp32, A8W8, FP8
                qkv_proj_weight = qkv_proj_weight.cast(self.weight_dtype)
                if ("float8" in self.weight_dtype
                    ):  # TODO(wangzhe24) FP8 cannot use set_value now
                    self.qkv_weight.copy_(qkv_proj_weight, False)
                else:
                    self.qkv_weight.set_value(qkv_proj_weight)

        # bias
        if self.with_bias:
            if self.inference_args.num_key_value_heads <= 0:
                qkv_bias = (get_tensor(state_dict.pop(self.bias_key)).reshape([
                    self.num_heads,
                    3,
                    self.embed_dim // self.inference_args.num_attention_heads,
                ]).transpose([1, 0, 2]))
            else:
                # GQA
                qkv_bias = get_tensor(state_dict.pop(self.bias_key)).reshape([
                    self.num_heads + 2 * self.kv_num_heads,
                    self.embed_dim // self.inference_args.num_attention_heads,
                ])
                single_qkv_biases = paddle.split(qkv_bias,
                                                 self.kv_num_heads,
                                                 axis=0)
                q_biases, k_biases, v_biases = [], [], []
                for single_qkv_bias in single_qkv_biases:
                    q_bias, k_bias, v_bias = paddle.split(
                        single_qkv_bias,
                        [
                            self.inference_args.num_attention_heads //
                            self.inference_args.num_key_value_heads,
                            1,
                            1,
                        ],
                        axis=0,
                    )
                    q_biases.append(q_bias)
                    k_biases.append(k_bias)
                    v_biases.append(v_bias)
                q_bias = paddle.concat(q_biases, axis=0)
                k_bias = paddle.concat(k_biases, axis=0)
                v_bias = paddle.concat(v_biases, axis=0)
                qkv_bias = paddle.concat([q_bias, k_bias, v_bias], axis=0)
            qkv_bias = qkv_bias.reshape([-1])
            self.qkv_bias.set_value(qkv_bias)

    def forward(self, x):
        """
        Defines the forward computation of the layer.

        Args:
            x (Tensor): Input tensor to the QKVLinear layer.

        Returns:
            Tensor: Output tensor.

        Raises:
            NotImplementedError: If the weight dtype is not float8 or act dtype is not equal to weight dtype.
        """
        if self.skip_quant:
            qkv_out = paddle.matmul(x, self.qkv_weight, False, True)
            if self.qkv_bias is not None:
                qkv_out = paddle.add(qkv_out, self.qkv_bias)
            return qkv_out
        if self.inference_args.weight_block_size[0] != -1:
            x, x_scale_tensor = fastdeploy.model_executor.ops.gpu.per_token_quant_padding(
                x, self.inference_args.weight_block_size[0])
            qkv_out = paddle.empty(
                (x.shape[0], self.inference_args.qkv_hidden_size),
                dtype=paddle.bfloat16)
            deep_gemm.gemm_fp8_fp8_bf16_nt(
                (x, x_scale_tensor), (self.qkv_weight, self.qkv_weight_scale),
                qkv_out)
        elif self.inference_args.use_weight_only and self.act_dtype in [
                "bfloat16",
                "float16",
                "float32",
        ]:
            # print('===== qkv_weight',self.qkv_weight)
            # print('====== self.qkv_bias ', self.qkv_bias)
            # print('====== self.qkv_weight_scale', self.qkv_weight_scale)
            qkv_out = weight_only_linear(
                x,
                weight=self.qkv_weight,
                bias=self.qkv_bias,
                weight_scale=self.qkv_weight_scale,
                weight_dtype=self.weight_dtype,
                arch=self.inference_args.weight_only_linear_arch,
            )
        elif self.weight_dtype == "int8" and self.act_dtype == self.weight_dtype:
            qkv_out = paddle.matmul(x, self.qkv_weight, False, True)
        elif self.weight_dtype == "int4" and self.act_dtype == "float8_e4m3fn":
            qkv_out = fastdeploy.model_executor.ops.gpu.scaled_gemm_f8_i4_f16(
                x,
                self.qkv_weight,
                self.qkv_weight_scale,
                zero_points=None,
                bias=self.qkv_bias,
                out_scale=self.inference_args.weight_scale_dict.get(
                    self.layer_name + ".weight_quanter") /
                (self.inference_args.act_scale_dict.get(
                    self.layer_name + ".activation_quanter") * 448 * 448),
                groupsize=0,
                out_dtype=self._dtype,
            )
        elif "float8" in self.weight_dtype and self.act_dtype == self.weight_dtype:
            qkv_out = fastdeploy.model_executor.ops.gpu.per_channel_fp8_fp8_half_gemm_fused(
                x,
                self.qkv_weight,
                bias=self.qkv_bias,
                scalar_scale=self.scalar_scale,
                channel_scale=self.qkv_out_scale,
                transpose_x=False,
                transpose_y=True,
                output_dtype=self._dtype,
            )
        elif (self.weight_dtype in ["bfloat16", "float16", "float32"]
              and self.act_dtype == self.weight_dtype):
            qkv_out = paddle.matmul(x, self.qkv_weight, False, True)
            if self.qkv_bias is not None:
                qkv_out = paddle.add(qkv_out, self.qkv_bias)
        else:
            raise ValueError(
                f"QKVLinear is not implemented for W[{self.weight_dtype}]A[{self.act_dtype}] yet."
            )
        return qkv_out

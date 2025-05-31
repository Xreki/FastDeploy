from abc import abstractmethod

import paddle
from paddle import nn

from fastdeploy.model_executor.layers.quantization.quant_base import \
    QuantMethodBase


class FusedMoEMethodBase(QuantMethodBase):
    """
    Use Cutlass Group Gemm to compute Fused MoE.
    """

    @abstractmethod
    def create_weights(self,
                       layer: nn.Layer,
                       moe_compute_params,
                       ffn1_tensor,
                       ffn2_tensor,
                       ffn1_bias=None,
                       ffn2_bias=None):
        """
        How to create weights, you must implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: nn.Layer,
        moe_compute_params,
        x: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Compute methods, you must implement this method.
        """

        raise NotImplementedError

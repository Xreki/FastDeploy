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

class ModelRegistry:
    """Model 类的注册管理器"""
    _registry: Dict[str, Type['Model']] = {}  # 存储 {类名: 类对象}

    @classmethod
    def register(cls, Model_cls: Type['Model']):
        """注册 Model 子类"""
        if not issubclass(Model_cls, Model):
            raise TypeError(f"{Model_cls.__name__} 不是 Model 的子类")
        cls._registry[Model_cls.__name__] = Model_cls
        return Model_cls

    @classmethod
    def get_registered(cls) -> Dict[str, Type['Model']]:
        """获取所有已注册的 Model 子类"""
        return cls._registry.copy()

# 定义 Model 基类（使用元类控制子类注册）
class ModelMeta(type):
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if name != 'Model' and issubclass(cls, Model):
            ModelRegistry.register(cls)

class ModelForCasualLM(nn.Layer, ABC, metaclass = ModelMeta):
    """
    Base class for LM
    """

    def __init__(self, gpt, configs):
        """
        Args:
            gpt (ErnieBotFusedModel): ErnieBotFusedModel model used for generation.
            configs (dict): Configurations including parameters such as max_dec_len, min_dec_len, decode_strategy,
                ori_vocab_size, use_topp_sampling, use_top_k, etc.
        """
        super(ModelForCasualLM, self).__init__()
        
    @abstractmethod    
    def set_state_dict(self, state_dict: dict[str, np.ndarray | paddle.Tensor]):
        """
        Load model parameters from a given state dictionary.

        Args:
            state_dict (dict[str, np.ndarray | paddle.Tensor]):
                A dictionary containing model parameters, where keys are parameter names
                and values are NumPy arrays or PaddlePaddle tensors.
        """
        raise NotImplementedError 

    def forward(
        self,
        input_ids=None, 
        pos_emb=None,
        **model_kwargs,
    ):
        """
        Defines the forward pass of the model for generating text.

        Args:
            input_ids (Tensor, optional): The input token ids to the model.
            pos_emb (Tensor, optional): position Embeddings for model.
            **model_kwargs: Additional keyword arguments for the model.

        Returns:
            Tensor or list of Tensors: Generated tokens or decoded outputs.
        """
        raise NotImplementedError 

    @abstractmethod  
    def compute_logits(self, hidden_state, **logits_prosessor_kwargs):
        raise NotImplementedError 

    @abstractmethod  
    def sample(
        self,
        logits,
        **sample_kwargs,
    ):
        """Sample from GPT using beam search and post process the generated sequence.

        Args:
            logits (Tensor): The id of the token indicating the end of a sentence.
            sample_kwargs (dict): Number of highest probability vocabulary tokens to keep for top-k-filtering.

        Returns:
            Tensor: The sampled tokens. The shape is [batch_size].
        """
        raise NotImplementedError 


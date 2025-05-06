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


class InputPreprocessor:
    """
        Args:
        model_name_or_path (str):
            Model name or path to the pretrained model. If a model name is provided, it should be a
            key in the Hugging Face Transformers' model registry (https://huggingface.co/models).
            The model will be downloaded from the Hugging Face model hub if necessary.
            If a path is provided, the model will be loaded from that path.
        enable_mm_registry (bool, optional):
            Whether to use the MMSelfAttention registry for loading pre-trained models. Defaults to False.
            If True, the model will be loaded using the MMSelfAttention registry. Otherwise, the model will be
            loaded directly from the specified path.

        Raises:
            ValueError:
                If the model name is not found in the Hugging Face Transformers' model registry and the path does not
                exist.
    """
    def __init__(
        self,
        model_name_or_path: str,
        enable_mm_registry: bool = False,
    ) -> None:

        self.model_name_or_path = model_name_or_path
        self.enable_mm_registry = enable_mm_registry


    def create_processor(self):
        """
            创建数据处理器。如果启用了多模态注册表，则使用该表中的模型；否则，使用传递给构造函数的模型名称或路径。
        返回值：DataProcessor（如果不启用多模态注册表）或MultiModalRegistry.Processor（如果启用多模态注册表）。

        Args:
            无参数。

        Returns:
            DataProcessor or MultiModalRegistry.Processor (Union[DataProcessor, MultiModalRegistry.Processor]): 数据处理器。
        """
        if not self.enable_mm_registry:
            if int(os.getenv("OPEN_SOURCE", "0")) == 1:
                from fastdeploy.input.text_processor import DataProcessor
                self.processor = DataProcessor(model_name_or_path=self.model_name_or_path)
            else:
                from fastdeploy.input.ernie_processor import ErnieProcessor
                self.processor = ErnieProcessor(model_name_or_path=self.model_name_or_path)
        else:
            from fastdeploy.input.mm_register import MultiModalRegistry
            self.processor = MultiModalRegistry.create_processor(self.model_name_or_path)
        return self.processor

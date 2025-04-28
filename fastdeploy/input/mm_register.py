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

import importlib
PROCESSOR_MODELS = {
    "qwen2_5_vl": {
        "tokenizer": "MIXQwen2_5_Tokenizer",
        "image_processor": "Qwen2_5_VLImageProcessor",
        "processor": "Qwen2_5_VLProcessor",
    },
}

class MultiModalRegistry:
    @classmethod
    def create_processor(cls, model_name: str):
        """根据模型名动态创建处理器"""

        if model_name not in PROCESSOR_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")


        config = PROCESSOR_MODELS[model_name]


        tokenizer_cls = cls._import_class(
            module_path=f"paddlemix.models.{model_name}",
            class_name=config["tokenizer"]
        )
        tokenizer = tokenizer_cls.from_pretrained(model_name)

        image_processor_cls = cls._import_class(
            module_path=f"paddlemix.processors.{model_name}_processing",
            class_name=config["image_processor"]
        )
        image_processor = image_processor_cls()


        processor_cls = cls._import_class(
            module_path=f"paddlemix.processors.{model_name}_processing",
            class_name=config["processor"]
        )
        return processor_cls(image_processor, tokenizer)

    @staticmethod
    def _import_class(module_path: str, class_name: str):
        """动态导入类"""
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except ImportError:
            raise ImportError(f"Module '{module_path}' not found")
        except AttributeError:
            raise AttributeError(f"Class '{class_name}' not found in {module_path}")


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

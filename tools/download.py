import sys
from paddlenlp.transformers import (
    AutoConfig,
    AutoInferenceModelForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    ChatGLMTokenizer,
    ChatGLMv2Tokenizer,
    Llama3Tokenizer,
    LlamaTokenizer,
    PretrainedConfig,
    PretrainedModel,
    PretrainedTokenizer,
)
model_name_or_path = sys.argv[1]
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
print(model)
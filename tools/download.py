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

input_features = tokenizer("北京天安门在哪里", return_tensors="pd")
outputs = model.generate(**input_features, max_new_tokens=128)
print(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))
# print(model)
## 下载模型到本地

```
python3 tools/download.py Qwen/Qwen3-0.6B/
```

## 运行脚本

```python
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.entrypoints.llm import LLM

model_name_or_path = "./Qwen/Qwen3-0.6B/"

prompts = [
    "北京天安门在哪里?",
    "解释温故而知新",
]

sampling_params = SamplingParams(temperature=0.1, max_tokens=500)
llm = LLM(model=model_name_or_path, tensor_parallel_size=1)
output = llm.generate(prompts=prompts,
                      use_tqdm=True,
                      sampling_params=sampling_params)
print(output)
```

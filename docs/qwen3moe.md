## 下载模型到本地
```
python3 tools/download.py Qwen/Qwen3-30B-A3B/
```

## 运行脚本

```python
from fastdeploy.engine.sampling_params import SamplingParams
from fastdeploy.entrypoints.llm import LLM

model_name_or_path = "./Qwen/Qwen3-30B-A3B/"


sampling_params = SamplingParams(temperature=0.1, max_tokens=100)
llm = LLM(model=model_name_or_path, tensor_parallel_size=1, engine_worker_queue_port=8005, max_num_seqs=2, max_model_len=200)
output = llm.generate(prompts=[
    "北京天安门在哪里", 
    "解释一下温故而知新"
    ], use_tqdm=True)
print(output)
```

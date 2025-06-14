# FastDeploy千问模型离线推理

## 1. 使用方式
通过FastDeploy离线推理，可支持本地加载Qwen2模型，并处理用户数据，使用方式如下，

```python
from fastdeploy import LLM

prompts = [
    "where is Beijing?"
]

llm = LLM(model="Qwen/Qwen2-7B-Instruct",tensor_parallel_size=1,max_model_len=8192)

# 批量进行推理（llm内部基于资源情况进行请求排队、动态插入处理）
output = llm.generate(prompts,use_tqdm=True)

# 输出结果
print(output)
```

本示例中 `SamplingParams` ， `LLM` ，`LLM.generate` 以及输出output对应的结构体 `RequestOutput` ,接口说明可以参考[Offline Inference](offline_inference.md)

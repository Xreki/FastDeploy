FastDeploy LLM部署
===

4.5T 快速部署

## 环境安装
安装efficient llm (https://console.cloud.baidu-int.com/devops/icode/repos/baidu/paddle_internal/EfficientLLM/tree/master)

拉取fastdeploy llm 到本地，并将路径添加至PYTHONPATH


```
cd FastDeploy
pip install -r requirements.txt
export PYTHONPATH=${PWD}:$PYTHONPATH
```

## LLM API使用
```
from fastdeploy import LLM, SamplingParams

# Define a list of input prompts
prompts = [
    "Hello, my name is",
    "The largest ocean is",
]

# Define sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Initialize the LLM engine with the ERNIE 4.5 Turbo model
llm = LLM(model="ERNIE/4.5-Turbo")

# Generate outputs for the input prompts
outputs = llm.generate(prompts, sampling_params)

# Print the generated outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
```

## LLM api server

启动api server
```
python api_server.py --config test.yaml --port 9904
```

测试api server
```
curl 0.0.0.0:9904/generate    -H 'Content-Type: application/json'    -d '{"prompt": "hello, llm","stream":1}'
```


# 开源协议

FastDeploy遵循[Apache-2.0开源协议](./LICENSE)。 在本项目的开发中，为了对齐[vLLM](https://github.com/vllm-project/vllm)使用接口，参考或直接使用了部分vLLM代码，在此表示感谢。

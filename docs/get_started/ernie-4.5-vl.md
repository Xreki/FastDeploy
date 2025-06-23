# ERNIE-4.5-VL多模态模型
@ 王心禹

本文档讲解如何部署ERNIE-4.5-VL多模态模型，支持用户使用多模态数据与模型进行对话交互，在准备开始部署前，请确保你的硬件环境满足如下条件，

- GPU驱动 >= 535
- CUDA >= 12.3
- CUDNN >= 9.5
- Linux X86_64
- Python >= 3.10
- 80G A/H 8卡

安装FastDeploy方式参考[安装文档](./installation.md)。

## 准备模型
以下两种模式均可获取 ERNIE-4.5-VL 模型，

- (TODO待完善) 自行在HuggingFace下载
- 部署时指定```--model ERNIE-4.5-VL-300B-A47B-BF16```[自动下载模型](../supported_models.md)

## 启动服务

执行如下命令，启动服务,其中启动命令配置方式参考[参数说明](../parameters.md)

```shell
python -m fastdeploy.entrypoints.openai.api_server \
       --model ERNIE-4.5-300B-A47B-BF16 \
       --port 8180 --engine-worker-queue-port 8181 \
       --cache-queue-port 8182 --metrics-port 8182 \
       --max-model-len 32768 \
       --enable-mm \
       --mm-processor-kwargs xxxx \
       --mm--xxxxx \
       --max-num-seqs 32
```


## 请求服务
在服务启动后，当打印如下信息后，说明服务已经启动成功。
```shell
api_server.py[line:72] Launching metrics service at http://0.0.0.0:8171/metrics
api_server.py[line:73] Launching chat completion service at http://0.0.0.0:8170/v1/chat/completions
api_server.py[line:74] Launching completion service at http://0.0.0.0:8170/v1/completions

INFO:     Started server process [13909]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8171 (Press CTRL+C to quit)
INFO:     Started server process [13909]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8170 (Press CTRL+C to quit)
```

可以通过服务探活接口判断服务的启动状态是否成功，执行如下命令返回200即表示服务启动成功
```shell
curl -i http://0.0.0.0:${port}/health
```

通过如下命令进行服务请求
```shell
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "你好，你的名字是什么？"}
  ]
}'
```

因为FastDeploy服务提供的接口兼容OpenAI协议，你也可以通过如下Python代码调用服务,

```python
import openai
host = "0.0.0.0"
port = "8170"
client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="null")

response = client.completions.create(
    model="null",
    prompt="深圳福田区有什么好玩的地方?",
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].text, end='')
print('\n')

response = client.chat.completions.create(
    model="null",
    messages=[
        {"role": "system", "content": "I'm a helpful AI assistant."},
        {"role": "user", "content": "深圳南山区有什么好玩的地方?"},
    ],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta:
        print(chunk.choices[0].delta.content, end='')
print('\n')
```

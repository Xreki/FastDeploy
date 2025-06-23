# ERNIE-X1思考模型

本文档讲解如何部署ERNIE-X1思考模型，在准备开始部署前，请确保你的硬件环境满足如下条件，

- GPU驱动 >= 535
- CUDA >= 12.3
- CUDNN >= 9.5
- Linux X86_64
- Python >= 3.10
- 80G A/H 4卡

安装FastDeploy方式参考[安装文档](./installation.md)。

## 准备模型
以下两种模式均可获取 ERNIE-X1 模型，

- (TODO待完善) 自行在HuggingFace下载
- 部署时指定```--model ERNIE-X1-300B-A47B-BF16```[自动下载模型](../supported_models.md)

## 启动服务

执行如下命令，启动服务，其中启动命令配置方式参考[参数说明](../parameters.md)
```shell
python -m fastdeploy.entrypoints.openai.api_server \
       --model ERNIE-X1-300B-A47B-BF16 \
       --port 8180 --engine-worker-queue-port 8181 \
       --cache-queue-port 8182 --metrics-port 8182 \
       --max-model-len 32768 \
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

X1模型返回包含『思考内容』和『回复内容』，如下示例结果所示，其中无论在流式返回或非流式返回中，前者内容在```reasoning_content```中，后者内容在```content```中。
```
{
   "choices" : [
      {
         "finish_reason" : "stop",
         "index" : 0,
         "message" : {
            "content" : "你好！我是百度研发的AI对话产品，你可以叫我“文心X1”。我的任务是通过自然对话的方式，帮助你解答问题、提供信息或完成各种任务。如果你有任何需要，随时告诉我哦~ 😊",
            "reasoning_content" : "用户问我的名字是什么。首先，我需要确认用户的需求。他们可能刚接触我，想了解我的身份。我需要友好地回答，同时保持专业。\n\n接下来，我应该介绍自己作为百度研发的模型，强调我的功能是帮助用户解决问题。可能用户想了解我的背景，所以需要简明扼要地说明。\n\n然后，要避免使用技术术语，保持口语化，让用户容易理解。同时，确保回答准确，不误导用户。比如，提到我的知识截止日期，但也要说明如果有新信息可能需要查证。\n\n还要注意用户可能的后续问题，比如如何工作、能做什么，所以回答中可以稍微铺垫，但主要先回答名字的问题。保持简洁，不过于冗长。\n\n最后，检查有没有错误，比如名字是否正确，功能描述是否准确。确保回答符合友好和帮助的基调。",
            "role" : "assistant"
         }
      }
   ],
   "created" : 1750238398,
   "id" : "chatcmpl-1f009a67-3dd9-4ca8-b481-782da599b0a2",
   "model" : "default",
   "object" : "chat.completion",
   "usage" : {
      "completion_tokens" : 232,
      "prompt_tokens" : 90,
      "prompt_tokens_details" : null,
      "total_tokens" : 322
   }
}
```

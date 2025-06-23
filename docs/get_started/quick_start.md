# 10分钟完成ERNIE-4.5-0.3B纯文部署

部署前请检查运行环境是否满足需求

- GPU驱动 >= 535
- CUDA >= 12.3
- CUDNN >= 9.5
- Linux X86_64
- Python >= 3.10
- 运行模型满足最低硬件配置要求，参考[支持模型列表文档](../supported_models.md)

为了快速在各类硬件部署，本文档采用ERNIE-4.5-0.3B模型作为示例，可在大部分硬件上完成部署。

推荐使用Docker镜像方式进行部署。

## 1. 拉取镜像，创建容器
```shell
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy:2.0.0.0-alpha
```

## 2. 启动服务
在容器内执行如下命令，启动服务，其中启动命令配置方式参考[参数说明](../parameters.md)
```shell
python -m fastdeploy.entrypoints.openai.api_server \
       --model ERNIE-45-0.3B \
       --port 8180 \
       --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --max-model-len 32768 \ # 最长支持Token数
       --max-num-seqs 32 # 最大并发处理数
```
注意：在```--model```指定的路径中，如路径在命令执行当前目录不存在，则会尝试通过指定的值查询AIStudio是否存在预置模型下载，例如当前指定```ERNIE-xxxx```且当前目录如无同名子目录，则会自动开始下载，下载的默认路径在```~/xx```。关于模型自动下载的说明和配置参阅[模型下载](../supported_models.md)

**相关文档**

- [服务部署配置](../serving/serving.md)
- [服务监控metrics](../serving/metrics.md)

## 3. 请求服务

在服务启动后，当打印如下信息后，说明服务已经启动成功。
```
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

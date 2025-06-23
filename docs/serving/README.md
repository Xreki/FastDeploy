# 服务化部署

FastDeploy基于底层高性能推理引擎，提供与OpenAI协议兼容的服务化部署方式。用户通过如下命令快速进行部署

```bash
python -m fastdeploy.entrypoints.openai.api_server \
       --model ernie-45-turbo \
       --port 8188 --tensor-parallel-size 8 \
       --max-model-len 32768
```

服务部署时的命令行更多使用方式参考[参数说明](../parameters.md)。

## 请求服务

因为接口兼容OpenAI协议，可以直接使用openai的请求方式请求服务，如下分别提供curl和python示例,

```bash
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}'
```

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

关于OpenAI协议的说明可参考文档 [OpenAI Chat Compeltion API](https://platform.openai.com/docs/api-reference/chat/create)，需要说明的是，FastDeploy提供的服务在参数上存在如下差异，

1. 仅支持OpenAI如下参数（其余参数配置会被服务忽略）
- prompt (仅```v1/completions```接口)
- messages (仅```v1/chat/completions```接口)
- frequency_penalty: Optional[float] = 0.0
- max_tokens: Optional[int] = 16
- presence_penalty: Optional[float] = 0.0
- seed: Optional[int] = None
- stream: Optional[bool] = False
- stream_options: Optional[StreamOptions] = None
- temperature: Optional[float] = None
- top_p: Optional[float] = None
- metadata: Optional[dict] = None (仅在v1/chat/compeltions中支持，用于配置min_tokens即最小输出长度，例如metadata={"min_tokens": 20})

> 注:若为X1 模型 由于思考链默认打开导致输出过长，max tokens 可以设置为模型最长输出，或无需设置(默认会最大)。

2. 在返回的信息

新增返回参数：
arrival_time ：每个token 的返回的累计耗时
reasoning_content: 思考链返回结果

```python
ChatCompletionStreamResponse:
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
 ChatCompletionResponseStreamChoice:
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None
    arrival_time: Optional[float] = None
DeltaMessage:
    role: Optional[str] = None
    content: Optional[str] = None
    token_ids: Optional[List[int]] = None
    reasoning_content: Optional[str] = None
```

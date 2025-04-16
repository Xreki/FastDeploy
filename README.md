FastDeploy LLM部署
===

## 安装

```
python setup.py install
```

## LLM API使用
```
from fastdeployllm import LLM, SamplingParams

# Define a list of input prompts
prompts = [
    "Hello, my name is",
    "The largest ocean is",
]

# Define sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Initialize the LLM engine with the OPT-125M model
llm = LLM(model="ERNIE/4.5-Turbo")

# Generate outputs for the input prompts
outputs = llm.generate(prompts, sampling_params)

# Print the generated outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
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

## 代码提交规范

- 本仓库已添加pre-commit hook，代码clone至本地后，进入主目录，执行```pre-commit install```，即可安装代码提交自动格式化工具（git commit时会自动执行)。
- Python代码遵循PEP-8风格
- 所有新增函数鼓励增加注释（英文），核心函数/用户API接口强制增加注释
- 配置参数均需注释说明作用
- 开源代码遵循Apache 2.0开源协议，如存在拷贝或借鉴自外部开源代码，请务必在文件头说明引用来源

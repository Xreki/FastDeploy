# FastDeploy: Large Language Model Serving

## 环境依赖
- A800/H800/H100
- Python>=3.10
- CUDA>=12.3
- CUDNN>=9.5
- Linux X64

## 安装

### Docker安装(推荐)
```
docker pull xxxx
```

### 源码安装
1. 安装PaddlePaddle GPU，详见[PaddlePaddle安装](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html)，指定安装CUDA 12.6 develop(Nightly build)版本，如下命令可完成安装
```
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu126/
```

2. 源码安装FastDeploy
```
cd FastDeploy
python setup.py install
```

## 快速使用

在使用Docker或源码安装后，执行如下命令进行服务部署, 更多参数的配置与含义参考[参数说明](docs/parameters.md).

```
python -m fastdeploy.entrypoints.openai.api_server --model ernie-45-turbo --port 8188 --tensor-parallel-size 8
```

使用如下命令请求服务
```
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}'
```
FastDeploy提供与OpenAI完全兼容的服务API(字段`model`与`api_key`目前不支持，设定会被忽略)，用户也可基于openai python api请求服务。

## 部署文档

- [项目架构](docs/architecture.md)
- [本地部署](docs/offline_inference.md)
- [服务部署](docs/serving.md)
- [参数说明](docs/parameters.md)
- [日志说明](docs/log.md)

# 代码说明
- 当前仅包含服务层代码，模型组网代码依赖requirements.txt中的EfficentLLM模块，此模块将在5月底前完成代码合入。
- FastDeploy的使用中存在任何建议和问题，可随时如流反馈。

# 开源说明

FastDeploy遵循[Apache-2.0开源协议](./LICENSE)。 在本项目的开发中，为了对齐[vLLM](https://github.com/vllm-project/vllm)使用接口，参考或直接使用了部分vLLM代码，在此表示感谢。


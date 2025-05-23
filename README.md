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
1. 安装PaddlePaddle GPU(nightly build，代码版本需新于2025.05.08)，详见[PaddlePaddle安装](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html)，指定安装CUDA 12.6 develop(Nightly build)版本，如下命令可完成安装
```
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu126/
```

2. 安装FastDeploy

pip安装
```
pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/fastdeploy/8677bde0dd60c3ac6f416b823b0e3093ac74025a/fastdeploy-0.1.0-py3-none-any.whl
```

源码安装
```
# git clone FastDeploy仓库
cd FastDeploy
# 一键编译+安装本机可用的sm架构，whl包产物在dist/
bash tools/build_wheel.sh

# 只编译不打包成whl包(推荐开发时配合PYTHONPATH使用)
bash tools/build_wheel.sh 0
```

## 快速使用

在使用Docker或源码安装后，执行如下命令进行服务部署, 更多参数的配置与含义参考[参数说明](docs/serving.md).

```
python -m fastdeploy.entrypoints.openai.api_server --model ernie-45-turbo --port 8188 --tensor-parallel-size 8
```

使用如下命令请求纯文模型服务
```
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}'
```
使用如下命令请求多模模型服务
```
curl -X POST "http://0.0.0.0:8188/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://ku.baidu-int.com/vk-assets-ltd/space/2024/09/13/933d1e0a0760498e94ec0f2ccee865e0", "detail": "high"}},
      {"type": "text", "text": "请描述图片内容"}
    ]}
  ]
}'
```
FastDeploy提供与OpenAI完全兼容的服务API(字段`model`与`api_key`目前不支持，设定会被忽略)，用户也可基于openai python api请求服务。

## 部署文档

- [本地部署](docs/offline_inference.md)
- [服务部署](docs/serving.md)

# 代码说明
- [代码目录说明](docs/code_guide.md)
- FastDeploy的使用中存在任何建议和问题，可随时如流反馈。

# 开源说明

FastDeploy遵循[Apache-2.0开源协议](./LICENSE)。 在本项目的开发中，为了对齐[vLLM](https://github.com/vllm-project/vllm)使用接口，参考和直接使用了部分vLLM代码，在此表示感谢。

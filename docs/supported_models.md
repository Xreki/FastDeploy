# 支持模型列表

FastDeploy目前支持模型列表如下，以下模型提供如下3种下载方式，

- 1. 在FastDeploy部署时，指定```model```参数为如下表格中的模型名，即可自动从AIStudio下载模型权重（支持断点续传）
- 2. 前往HuggingFace官网下载，@TODO 进一步说明
- 3. 前往ModelScope官网下载，@TODO 进一步说明

其中第一种方式自动下载时，默认下载路径为```~/aistudio/data```，用户可以通过配置环境变量```FD_MODEL_CACHE```修改默认下载的路径，例如  
```
export FD_MODEL_CACHE=/ssd1/download_models
```

| 模型名 | 支持无损量化方式 | 最小资源要求说明 |
| :----- | :--------------  | :----------- |
| ERNIE-X1-300B-A47B | xxxx 待填写 | A/H卡，最小显存24GB |
| ERNIE-4.5-300B-A47B-Base | xxxx 待填写 | A/H卡，最小显存24GB |
| ERNIE-4.5-300B-A47B | xxxx 待填写 | A/H卡，最小显存24GB |
| ERNIE-4.5-VL-424B-A47B-Base | xxxx 待填写 | A/H卡，最小显存24GB |
| ERNIE-4.5-VL-424B-A47B | xxxx 待填写 | A/H卡，最小显存24GB |
| ERNIE-4.5-VL-28B-A3B-Base | xxxx 待填写 | A/H卡，最小显存24GB |
| ERNIE-4.5-VL-28B-A3B | xxxx 待填写 | A/H卡，最小显存24GB |
| ERNIE-4.5-21B-A3B-Base | xxxx 待填写 | A/H卡，最小显存24GB |
| ERNIE-4.5-21B-A3B | xxxx 待填写 | A/H卡，最小显存24GB |
| ERNIE-4.5-0.3B-Base | xxxx 待填写 | A/H卡，最小显存24GB |
| ERNIE-4.5-0.3B | xxxx 待填写 | A/H卡，最小显存24GB |
| Qwen2-7B-Instruct | xxxx 待填写 | A/H卡，最小显存24GB |

更多模型同步支持中，你可以通过[Github ISSUE](https://github.com/PaddlePaddle/FastDeploy/issues)向我们提交新模型的支持需求。

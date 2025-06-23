# FastDeploy 2.0

FastDeploy 2.0 是飞桨（PaddlePaddle）开源的面向大模型的推理部署工具。当前，FastDeploy 支持[多种大模型结构](supported_models.md) ，其推理部署功能涵盖：

- 一行命令即可快速实现模型的服务化部署，并支持流式生成
- 利用张量并行技术加速模型推理
- 支持 PagedAttention 与 continuous batching（动态批处理）
- 兼容 OpenAI 的 HTTP 协议
- 提供 Weight only int8/int4 无损压缩方案
- 支持 Prometheus Metrics 指标

了解更多FastDeploy使用方式，阅读相关文档，

- [10分钟上手部署全流程](./get_started/quick_start.md)
- [ERNIE-X1思考模型部署](./get_started/ernie-x1.md)
- [ERNIE-4.5-VL多模模型部署](./get_started/ernie-4.5-vl.md)
- [ERNIE-4.5模型部署](./get_started/ernie-4.5.md)
- [支持模型列表](./supported_models.md)
- [代码模块说明](./design/code_guide.md)

## 文档说明

本项目文档基于mkdocs支持编译可视化查看，参考如下命令进行编译预览，
```
pip install mkdocs

cd FastDeploy
mkdocs build

mkdocs serve
```
根据提示打开相应地址即可。

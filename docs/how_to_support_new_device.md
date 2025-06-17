# 如何接入新的硬件（贡献代码）

## 步骤1 准备硬件兼容的算子
如图所示，需要准备好红色、灰色和蓝色部分的自定义算子，以供后续开发调用。其中灰色算子paddle提供了基础版本（CPU）实现，开发者调试阶段可以调用。

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=9697ae90492d4a04a7cdcee739c017b1&docGuid=2U0kBiPiqYuQcP "")
## 步骤2 注册新的Worker
参考代码路径：FastDeploy/fastdeploy/worker/gpu_worker.py

* Worker负责初始化设备，对应一个加速卡和一个进程，核心作用是启动一个ModelRunner负责模型推理。
* 开发者需要：继承WorkerBase类并实现相应的接口，部分关键需要重点关注的子函数如下：

```
def init_dist_env
def determine_num_available_blocks
def step_cuda
def run_profile
```
如果改动逻辑过多，不可避免会出现if/else判断，硬件开发者亦可实现新的worker_xxxpu.py的实现。

## 步骤3 注册新的ModelRunner
参考代码路径：FastDeploy/fastdeploy/worker/gpu_model_runner.py

* ModelRunner负责推理的全流程，包括模型加载、模型预热、前后处理等。
* 开发者需要：继承ModelRunnerBase类并实现相应的接口，部分关键需要重点关注的函数如下：

```
initialize_attn_backend：负责选择Attention后端
initialize_kv_cache：负责构造KV Cache
_prepare_inputs：核心是remove_padding算子（目前有GPU/CPU算子实现）需要兼容（参考步骤1）

execute_model：包括前后处理，需要关注是否包含了自定义算子的调用，需要改为调用步骤1中提到的硬件兼容的算子。
_dummy_run：流程和execute_model一致，需要处理算子兼容问题。
```


## 步骤4 注册新的AttentionBackend
参考代码路径：FastDeploy/fastdeploy/model_executor/layers/attention/append_attn_backend.py

* 一个完整的Transformer模型必定需要包含Attention计算，每个Attention算子都需要考虑到框架设计和硬件兼容等多方面。
* 开发者需要：继承AttentionBackend类并实现相应的接口，关键需要重点关注的部分包括：

```
forward_decode:负责Decoding
forward_extend:负责Prefill
```
GPU的AppendAttentionBackend实现仅作参考和精度对齐使用，AttentionBackend有很高的自由度，开发者可以自行发挥硬件的优势。

## 步骤5 兼容其他的Layer
目前的FastDeploy的Layers设计是硬件无关的，但分两种情况

1. Layer本身足够简单，可以实现多套硬件共享一个类，实现逻辑类似如下：

```
if current_platform.is_cuda():
    self.forward = self.forward_cuda
elif current_platform.is_xxxpu()
    self.forward = self.forward_xxxpu() # 实现新的前向
```
2. Layer的分支逻辑比较复杂，必须每个硬件实现一个独立的类（类的实现放在FastDeploy/fastdeploy/model_executor/layers/backends/中）

```
class WeightOnlyLinearMethod(QuantMethodBase):
""""""

class GPUWeightOnlyLinearMethod(WeightOnlyLinearMethod):

class XXXPUWeightOnlyLinearMethod(WeightOnlyLinearMethod):
```
新实现的类/方法需要保证所有平台能够兼容，硬件相关的调用需要包含在各自的代码逻辑内部，不互相污染。

## 步骤6 添加相应的单测
* AttentionBackend的单测添加到test/layers路径下
* ModelRunner/Worker的单测添加到test/worker路径下



## 步骤7 跑通模型并验证精度
* 本地测试参考[本地跑通](./offline_inference.md)
* 测试完整数据集精度参考[服务验证](./serving.md) 并推理一个完整的GSM8K/MMLU-PRO数据集。

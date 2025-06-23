# 单机分离式部署

本文档说明在单机上进行分离式部署的流程。单机实现方式上，Prefill和Decode分为不同的服务进程，并通过进程间通信共同完成请求的处理。

## 在线推理服务
使用如下命令进行服务部署

**prefill 实例**

```bash
export FD_LOG_DIR="log_prefill"
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m fastdeploy.entrypoints.openai.api_server \
       --model ERNIE-4.5-300B-A47B-BF16 \
       --port 8180 --metrics-port 8181 \
       --engine-worker-queue-port 8182 \
       --cache-queue-port 8183 \
       --tensor-parallel-size 4 \
       --quantization wint4 \
       --splitwise-role "prefill"
```

**decode 实例**

```bash
export FD_LOG_DIR="log_decode"
export CUDA_VISIBLE_DEVICES=4,5,6,7
# 注意innode-prefill-ports指定为Prefill服务的engine-worker-queue-port
python -m fastdeploy.entrypoints.openai.api_server \
       --model ERNIE-4.5-300B-A47B-BF16 \
       --port 8184 --metrics-port 8185 \
       --engine-worker-queue-port 8186 \
       --cache-queue-port 8187 \
       --tensor-parallel-size 4 \
       --quantization wint4 \
       --innode-prefill-ports 8182 \ 
       --splitwise-role "decode"
```

注意在请求单机PD分离服务时，**用户需请求Decode服务的端口**。

## 离线推理服务

参考`demo` 目录下 `offline_disaggregated_demo.py` 示例代码，进行离线推理服务部署


### 环境变量说明

* FLAGS_use_pd_disaggregation: 指定是否进行分离式部署，1为开启，0为关闭

* FLAGS_fmt_write_cache_completed_signal: 指定是否开启cache 写入，Prefill 实例开启，Decode 实例关闭

* INFERENCE_MSG_QUEUE_ID: 指定当前服务的消息队列id，用于区分不同服务的队列

* FD_LOG_DIR: 指定当前服务的日志目录

### 参数说明

* --splitwise-role: 指定当前服务为prefill还是decode

* --innode-prefill-ports: decode服务需要指定prefill服务的engine-worker-queue-port，可以指定多个P实例，用逗号隔开

* --cache-queue-port: 指定cache服务的端口，用于prefill和decode服务通信

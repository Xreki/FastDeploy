# 分离式服务化部署

## 单机分离式部署

### 在线推理服务
使用如下命令进行服务部署

**prefill 实例**

```bash
export FD_LOG_DIR="log_prefill"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m fastdeploy.entrypoints.openai.api_server --model ernie-45-turbo --port 8188 --tensor-parallel-size 4 --splitwise-role "prefill"  --engine-worker-queue-port 6677  --cache-queue-port 55663
```

**decode 实例**

```bash
export FD_LOG_DIR="log_decode"
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m fastdeploy.entrypoints.openai.api_server --model ernie-45-turbo --port 8189 --tensor-parallel-size 4 --splitwise-role "decode"  --engine-worker-queue-port 6678 --cache-queue-port 55664 --innode-prefill-ports 6677
```

### 离线推理服务

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


### 请求方式

数据请求方式与非分离式部署相同，端口为**decode**实例的端口
具体请求方式，可直接参考[服务化部署](./serving.md)中请求服务部分
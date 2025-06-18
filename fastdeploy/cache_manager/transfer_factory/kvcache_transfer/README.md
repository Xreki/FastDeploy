# KVCacheMessager
简要说明

## 准备环境

可以使用docker或者自行安装paddlepaddle。

### 使用docker

拉取docker镜像```iregistry.baidu-int.com/fastdeploy/ernie-serving-gpu-cu118:20241203_1733207622010```

执行创建docker容器并启动。

### 安装PaddylePaddle

参考[飞桨官网](https://www.paddlepaddle.org.cn/)安装GPU版本的PaddlePaddle。

## 编译安装

安装rdma依赖库

```
apt update
apt install libibverbs-dev librdmacm-dev
```

安装pybind11

```
pip install "pybind11[global]"
```

拉取KVCacheMessager库，编译安装

```
python setup.py bdist_wheel
python -m pip install dist/rdma_comm-0.0.1-cp38-cp38-linux_x86_64.whl
```

## 测试

### 准备

安装测试依赖

```
pip install pyzmq
```

RDMA 传输库通过 KV_CACHE_SOCKET_IFNAME 来设置进行 verbs 建立连接的网卡。设置KV_CACHE_SOCKET_IFNAME（如下）；如果不设置KV_CACHE_SOCKET_IFNAME，默认就会选择第一张网卡。

```
设置 eth0 来进行 TCP 建立连接交换 RDMA 连接信息:
export KV_CACHE_SOCKET_IFNAME=eth0
、、、

RDMA 传输库通过 KVCACHE_RDMA_NICS 环境变量中给定的网卡名称查找IP地址。设置KVCACHE_RDMA_NICS（如下）；如果不设置KVCACHE_RDMA_NICS，默认就会选择主网卡。不同 GPU 根据 idx 选取对应网卡。
```

8张RDMA网卡的机器:
export KVCACHE_RDMA_NICS=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
、、、

RDMA 超时时间通过 KVCACHE_IB_TIMEOUT 来进行设置，如果不设置KVCACHE_IB_TIMEOUT，默认就会设置 18。
、、、
超时时间设置 22:
export KVCACHE_IB_TIMEOUT=22
、、、

### 单进程测试

### 多进程测试

## 如何贡献
贡献patch流程、质量要求

## 讨论
百度Hi讨论群：XXXX

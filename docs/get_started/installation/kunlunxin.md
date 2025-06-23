# 昆仑芯P800安装
TODO @ 洪明

在环境满足如下条件前提下

- GPU驱动 >= 535
- CUDA >= 12.3
- CUDNN >= 9.5
- Python >= 3.10
- Linux X86_64

可通过如下3种方式进行安装

## 1. Docker安装(推荐)
``` shell
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy:${fastdeploy_latest_version}
```
其中```${fastdeploy_latest_version}```是FastDeploy发布的release版本号，[点击查看最新release版本](https://github.com/PaddlePaddle/FastDeploy/releases),例如
``` shell
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy:2.0.0.0-alpha
```

## 2. Pip安装

首先安装paddlepaddle-gpu，详细安装方式参考[PaddlePaddle安装](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html)
``` shell
python -m pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

再安装fastdeploy，**注意不要通过pypi源安装**，需要通过如下方式安装
```
# 安装稳定版本fastdeploy
python -m pip install fastdeploy -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ 

# 安装Nightly Build的最新版本fastdeploy
# python -m pip install --pre fastdeploy -i https://www.paddlepaddle.org.cn/packages/nightly/cu126/
```

## 3. 源码编译安装

首先安装paddlepaddle-gpu，详细安装方式参考[PaddlePaddle安装](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html)
``` shell
python -m pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

接着克隆源代码，编译安装
``` shell
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy

bash build.sh
```
编译后的产物在```FastDeploy/dist```目录下。

## 环境检查

在安装FastDeploy后，通过如下Python代码检查环境的可用性
``` python
import paddle
from paddle.jit.marker import unified
# 检查GPU卡的可用性
paddle.utils.run_check()
# 检查FastDeploy自定义算子编译成功与否
from fastdeploy.model_executor.ops.gpu import beam_search_softmax
```
如上代码执行成功，则认为环境可用。

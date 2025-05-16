#!/bin/bash

# 参数检查
if [ $# -ne 1 ]; then
    echo "用法: $0 <FastDeploy代码部署目录>"
    exit 1
fi
fastdeploy_code_path=$1
fastdeploy_exec_root_path=/root/paddlejob/workspace/env_run/gaoziyuan
if [ ! -d $fastdeploy_exec_root_path ];then
    echo "no fastdeploy env found, start building..."
    mkdir -p ${fastdeploy_exec_root_path}
    cd  ${fastdeploy_exec_root_path}
    wget http://10.95.247.14:8879/miniconda3.tar
    tar -xvf miniconda3.tar
    mkdir -p develop_nlp
    cd develop_nlp
    wget http://10.95.247.14:8879/PaddleNLP.tar
    tar -xvf PaddleNLP.tar
    cp -r $fastdeploy_code_path ./
    cd PaddleNLP
    git config --global --add safe.directory ${fastdeploy_exec_root_path}/develop_nlp/PaddleNLP
else
    echo "fastdeploy env found, building will be skipped."
fi

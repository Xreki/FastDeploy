#!/bin/bash

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

#export XDNN_PATH=/path/to/xdnn # <path_to_xdnn>
#export XRE_PATH=/path/to/xre # <path_to_xre>
#export XFT_PATH=/path/to/xft # <path_to_xft>
#export BKCL_PATH=/path/to/bkcl # <path_to_bkcl>
#export CLANG_PATH=/path/to/xtdk # <path_to_xtdk>
# export XVLLM_PATH=/path/to/xvllm # <path_to_xvllm>
#export HOST_SYSROOT=/usr/local/gcc-8.2 # <path_to_gcc>
#export BUILD_XPU_VERSION=3
#export FLAGS_enable_pir_api=1
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$XVLLM_PATH/infer_ops/so:$XVLLM_PATH/xft_blocks/so # link xvllm libpath

rm -rf dist
rm -rf fastdeploy_ops.egg-info
rm -rf build
cd plugin
./build.sh
cd -

python setup_ops.py install

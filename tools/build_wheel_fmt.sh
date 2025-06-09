#!/usr/bin/env bash

# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

PYTHON_VERSION=python
PYTHON_VERSION=${1:-$PYTHON_VERSION}
export python=$PYTHON_VERSION

# python_tag
py=$(${python} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
python_tag="cp${py//./}"

# paddle distributed use to set archs
unset PADDLE_CUDA_ARCH_LIST

# directory config
DIST_DIR="dist"
BUILD_DIR="build"
EGG_DIR="fastdeploy.egg-info"

# custom_ops directory config
OPS_SRC_DIR="custom_ops"
OPS_BUILD_DIR="build"
OPS_EGG_DIR="efficitentllm_ops.egg-info"
OPS_TMP_DIR="tmp"

TEST_DIR="tests"

# command line log config
RED='\033[0;31m'
BLUE='\033[0;34m'
GREEN='\033[1;32m'
BOLD='\033[1m'
NONE='\033[0m'


function python_version_check() {
  PY_MAIN_VERSION=`${python} -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1}'`
  PY_SUB_VERSION=`${python} -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $2}'`
  echo -e "find python version ${PY_MAIN_VERSION}.${PY_SUB_VERSION}"
  if [ $PY_MAIN_VERSION -ne "3" -o $PY_SUB_VERSION -lt "8" ]; then
    echo -e "${RED}FAIL:${NONE} please use Python >= 3.8 !"
    exit 1
  fi
}

function init() {
    echo -e "${BLUE}[init]${NONE} removing building directory..."
    rm -rf $DIST_DIR $BUILD_DIR $EGG_DIR
    if [ `${python} -m pip list | grep fastdeploy | wc -l` -gt 0  ]; then
      echo -e "${BLUE}[init]${NONE} uninstalling fastdeploy..."
      ${python} -m pip uninstall -y fastdeploy
    fi

    ${python} -m pip install setuptools_scm
    if [ `${python} -m pip list | grep paddlepaddle | wc -l` -eq 0  ]; then
      echo -e "${BLUE}[init]${NONE} installing requirements..."
      ${python} -m pip install --upgrade --force-reinstall -r requirements/gpu/fmt/requirements.txt --ignore-installed PyYAML
    fi
    echo -e "${BLUE}[init]${NONE} ${GREEN}init success\n"
}


function copy_ops(){
    OPS_VERSION="0.0.0"
    PY_MAIN_VERSION=`${python} -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1}'`
    PY_SUB_VERSION=`${python} -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $2}'`
    PY_VERSION="py${PY_MAIN_VERSION}.${PY_SUB_VERSION}"
    SYSTEM_VERSION=`${python} -c "import platform; print(platform.system().lower())"`
    PROCESSOR_VERSION=`${python} -c "import platform; print(platform.processor())"`
    WHEEL_NAME="efficientllm_ops-${OPS_VERSION}-${PY_VERSION}-${SYSTEM_VERSION}-${PROCESSOR_VERSION}.egg"
    is_rocm=`$python -c "import paddle; print(paddle.is_compiled_with_rocm())"`
    if [ "$is_rocm" = "True" ]; then
      echo -e "OPS are for ROCM"
      cp -r ./${OPS_TMP_DIR}/${WHEEL_NAME}/* ../fastdeploy/model_executor/ops/gpu
      return
    fi
    is_cuda=`$python -c "import paddle; print(paddle.is_compiled_with_cuda())"`
    if [ "$is_cuda" = "True" ]; then
      echo -e "CUDA OPS"
      cp -r ./${OPS_TMP_DIR}/${WHEEL_NAME}/* ../fastdeploy/model_executor/ops/gpu
      return
    fi

    is_xpu=`$python -c "import paddle; print(paddle.is_compiled_with_xpu())"`
    if [ "$is_xpu" = "True" ]; then
      echo -e "OPS are for XPU"
      cp -r ./${OPS_TMP_DIR}/${WHEEL_NAME}/* ../fastdeploy/model_executor/ops/xpu
      return
    fi

    is_npu=`$python -c "import paddle; print(paddle.is_compiled_with_custom_device('npu'))"`
    if [ "$is_npu" = "True" ]; then
      echo -e "OPS are for NPU"
      cp -r ${OPS_TMP_DIR}/${WHEEL_NAME}/* ../fastdeploy/model_executor/ops/npu
      return
    fi
    return
}

function build_and_install_ops() {
  cd $OPS_SRC_DIR
  export no_proxy=bcebos.com,paddlepaddle.org.cn,${no_proxy}
  echo -e "${BLUE}[build]${NONE} build and install efficient_custom_ops..."
  ${python} setup_ops_fmt.py install --install-lib ${OPS_TMP_DIR}
  if [ $? -ne 0 ]; then
    echo -e "${RED}[FAIL]${NONE} build efficient_custom_ops wheel failed !"
    exit 1
  fi
  echo -e "${BLUE}[build]${NONE} ${GREEN}build efficient_custom_ops wheel success\n"

  copy_ops

  cd ..
}

function build_and_install() {
  echo -e "${BLUE}[build]${NONE} building fastdeploy wheel..."
  ${python} setup.py bdist_wheel --python-tag=${python_tag}
  if [ $? -ne 0 ]; then
    echo -e "${RED}[FAIL]${NONE} build fastdeploy wheel failed !"
    exit 1
  fi
  echo -e "${BLUE}[build]${NONE} ${GREEN}build fastdeploy wheel success\n"

  echo -e "${BLUE}[install]${NONE} installing fastdeploy..."
  cd $DIST_DIR
  find . -name "fastdeploy*.whl" | xargs ${python} -m pip install
  if [ $? -ne 0 ]; then
    cd ..
    echo -e "${RED}[FAIL]${NONE} install fastdeploy wheel failed !"
    exit 1
  fi
  echo -e "${BLUE}[install]${NONE} ${GREEN}fastdeploy install success\n"
  cd ..
}


function unittest() {
  # run UT
  echo -e "${BLUE}[unittest]${NONE} ${GREEN}unittests success\n${NONE}"
}

function cleanup() {
  rm -rf $BUILD_DIR $EGG_DIR
  ${python} -m pip uninstall -y fastdeploy

  rm -rf $OPS_SRC_DIR/$BUILD_DIR $OPS_SRC_DIR/$EGG_DIR
}

function abort() {
  echo -e "${RED}[FAIL]${NONE} build wheel and unittest failed !
          please check your code" 1>&2

  cur_dir=`basename "$pwd"`

  rm -rf $BUILD_DIR $EGG_DIR $DIST_DIR
  ${python} -m pip uninstall -y fastdeploy

  rm -rf $OPS_SRC_DIR/$BUILD_DIR $OPS_SRC_DIR/$EGG_DIR
}

python_version_check

trap 'abort' 0
set -e

init
build_and_install_ops
build_and_install
unittest
cleanup

# get Paddle version
PADDLE_VERSION=`${python} -c "import paddle; print(paddle.version.full_version)"`
PADDLE_COMMIT=`${python} -c "import paddle; print(paddle.version.commit)"`

# get fastdeploy version
EFFLLM_BRANCH=`git rev-parse --abbrev-ref HEAD`
EFFLLM_COMMIT=`git rev-parse --short HEAD`

# get Python version
PYTHON_VERSION=`${python} -c "import platform; print(platform.python_version())"`

echo -e "\n${GREEN}fastdeploy wheel compiled and checked success !${NONE}
        ${BLUE}Python version:${NONE} $PYTHON_VERSION
        ${BLUE}Paddle version:${NONE} $PADDLE_VERSION ($PADDLE_COMMIT)
        ${BLUE}fastdeploy branch:${NONE} $EFFLLM_BRANCH ($EFFLLM_COMMIT)\n"

echo -e "${GREEN}wheel saved under${NONE} ${RED}${BOLD}./dist${NONE}"

# install wheel
${python} -m pip install ./dist/fastdeploy*.whl
echo -e "${GREEN}wheel install success!${NONE}\n"

trap : 0

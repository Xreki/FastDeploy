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

set -ex

xvllm_path=/home/zhupengyang/PaddleInternal/xpu_sdk/xvllm_output/output
xvllm_infer_so=${xvllm_path}/infer_ops/so
xvllm_xft_blocks_so=${xvllm_path}/xft_blocks/so
export LD_LIBRARY_PATH=${xvllm_infer_so}:${xvllm_xft_blocks_so}:${LD_LIBRARY_PATH}

export XPU_VISIBLE_DEVICES=0

# export XPUAPI_DEBUG=0xA1

# python test_moe_topk_select.py
# python test_moe_expert_ffn.py
# python test_moe_ep_dispatch.py
python test_moe_ep_combine.py

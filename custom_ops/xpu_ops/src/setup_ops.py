#!/usr/bin/env python3

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
"""
Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved.

Build and setup XPU custom ops for ERNIE Bot.
"""
import os

import paddle
from paddle.utils.cpp_extension import CppExtension, setup


def xpu_setup_ops():
    """
    setup xpu ops
    """
    PADDLE_PATH = os.path.dirname(paddle.__file__)
    PADDLE_INCLUDE_PATH = os.path.join(PADDLE_PATH, "include")
    PADDLE_LIB_PATH = os.path.join(PADDLE_PATH, "libs")

    BKCL_PATH = os.getenv("BKCL_PATH")
    if BKCL_PATH is None:
        BKCL_INC_PATH = os.path.join(PADDLE_INCLUDE_PATH, "xpu")
        BKCL_LIB_PATH = os.path.join(PADDLE_LIB_PATH, "libbkcl.so")
    else:
        BKCL_INC_PATH = os.path.join(BKCL_PATH, "include")
        BKCL_LIB_PATH = os.path.join(BKCL_PATH, "so", "libbkcl.so")

    XRE_PATH = os.getenv("XRE_PATH")
    if XRE_PATH is None:
        XRE_INC_PATH = os.path.join(PADDLE_INCLUDE_PATH, "xre")
        XRE_LIB_PATH = os.path.join(PADDLE_LIB_PATH, "libxpucuda.so")
    else:
        XRE_INC_PATH = os.path.join(XRE_PATH, "include")
        XRE_LIB_PATH = os.path.join(XRE_PATH, "so", "libxpucuda.so")
    print(XRE_PATH)

    XVLLM_PATH = os.getenv("XVLLM_PATH")
    if XVLLM_PATH is None:
        XVLLM_KERNEL_INC_PATH = os.path.join(PADDLE_INCLUDE_PATH, "xvllm")
        XVLLM_KERNEL_LIB_PATH = os.path.join(PADDLE_LIB_PATH, "libapiinfer.so")
        XVLLM_OP_INC_PATH = os.path.join(PADDLE_INCLUDE_PATH, "xvllm")
        XVLLM_OP_LIB_PATH = os.path.join(PADDLE_LIB_PATH, "libxft_blocks.so")
    else:
        XVLLM_KERNEL_INC_PATH = os.path.join(XVLLM_PATH, "infer_ops",
                                             "include")
        XVLLM_KERNEL_LIB_PATH = os.path.join(XVLLM_PATH, "infer_ops", "so",
                                             "libapiinfer.so")
        XVLLM_OP_INC_PATH = os.path.join(XVLLM_PATH, "xft_blocks", "include")
        XVLLM_OP_LIB_PATH = os.path.join(XVLLM_PATH, "xft_blocks", "so",
                                         "libxft_blocks.so")

    ops = [
        # custom ops
        "./ops/save_with_output_msg.cc",
        "./ops/stop_generation_multi_ends.cc",
        "./ops/set_value_by_flags_and_idx.cc",
        "./ops/get_token_penalty_multi_scores.cc",
        "./ops/get_padding_offset.cc",
        "./ops/update_inputs.cc",
        "./ops/get_output.cc",
        "./ops/step.cc",
        "./ops/get_infer_param.cc",
        "./ops/adjust_batch.cc",
        "./ops/gather_next_token.cc",
        "./ops/block_attn.cc",
        "./ops/moe_layer.cc",
        "./ops/weight_quantize_xpu.cc",

        # device manage ops
        "./ops/device/get_context_gm_max_mem_demand.cc",
        "./ops/device/get_free_global_memory.cc",
        "./ops/device/get_total_global_memory.cc",
        "./ops/device/get_used_global_memory.cc",
    ]

    include_dirs = [
        ".",
        "./plugin/include",
        BKCL_INC_PATH,
        XRE_INC_PATH,
        XVLLM_KERNEL_INC_PATH,
        XVLLM_OP_INC_PATH,
    ]
    extra_objects = [
        "./plugin/build/libxpuplugin.a",
        BKCL_LIB_PATH,
        XRE_LIB_PATH,
        XVLLM_KERNEL_LIB_PATH,
        XVLLM_OP_LIB_PATH,
    ]

    print(f"include_dirs: {include_dirs}")
    print(f"extra_objects: {extra_objects}")

    setup(
        name="fastdeploy_ops",
        ext_modules=[
            CppExtension(
                sources=ops,
                include_dirs=include_dirs,
                extra_objects=extra_objects,
                extra_compile_args={
                    "cxx": [
                        "-D_GLIBCXX_USE_CXX11_ABI=1",
                        "-DPADDLE_WITH_XPU",
                        "-DBUILD_MULTI_XPU",
                        # "-DUSE_XFT_FORWARD_GPT_DYBATCH",
                        # "-DDEBUG_BEAM_SEARCH"
                    ]
                },
            )
        ],
    )


if __name__ == "__main__":
    xpu_setup_ops()

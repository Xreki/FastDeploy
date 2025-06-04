# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

""" setup for EfficientLLM custom ops """

import os
from paddle.utils.cpp_extension import CUDAExtension, setup
import subprocess


def clone_git_repo(version, repo_url, destination_path):
    """
    Clone git repo to destination path.
    """
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "-b",
                version,
                "--single-branch",
                repo_url,
                destination_path,
            ],
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


cuda_version = float(
    subprocess.check_output("nvcc -V | grep -o cuda.....", shell=True, text=True).split(
        "_"
    )[1]
)
print(f"Compiling, cuda_version:{cuda_version}")
if cuda_version >= 11.8:
    print("for cuda version >= 11.8, we compile both sm80 and sm90")
    gencode_flags = [
        "-gencode",
        "arch=compute_80,code=sm_80",
        "-gencode",
        "arch=compute_90,code=sm_90",
    ]
else:
    print("for cuda version <= 11.8, we compile sm80 only")
    gencode_flags = ["-gencode", "arch=compute_80,code=sm_80"]

gencode_flags += ["-Igpu_ops", "-Ithird_party/nlohmann_json/include"]

json_dir = "third_party/nlohmann_json"
if not os.path.exists(json_dir) or not os.listdir(json_dir):
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    clone_git_repo("v3.11.3", "https://github.com/nlohmann/json.git", json_dir)

setup(
    name="efficientllm_ops",
    ext_modules=CUDAExtension(
        sources=[
            "gpu_ops/helper.cu",
            "gpu_ops/save_with_output.cc",
            "gpu_ops/set_mask_value.cu",
            "gpu_ops/set_value_by_flags.cu",
            "gpu_ops/ngram_mask.cu",
            "gpu_ops/gather_idx.cu",
            "gpu_ops/token_penalty_multi_scores.cu",
            "gpu_ops/token_penalty_only_once.cu",
            "gpu_ops/stop_generation.cu",
            "gpu_ops/stop_generation_multi_ends.cu",
            "gpu_ops/stop_generation_multi_stop_seqs.cu",
            "gpu_ops/set_flags.cu",
            "gpu_ops/fused_get_rope.cu",
            "gpu_ops/transfer_output.cc",
            "gpu_ops/get_padding_offset.cu",
            "gpu_ops/get_padding_offset_system.cu",
            "gpu_ops/update_inputs.cu",
            "gpu_ops/rebuild_padding.cu",
            "gpu_ops/save_with_output_msg.cc",
            "gpu_ops/get_output.cc",
            "gpu_ops/reset_need_stop_value.cc",
            "gpu_ops/step.cu",
            "gpu_ops/step_reschedule.cu",
            "gpu_ops/step_system_cache.cu",
            "gpu_ops/set_data_ipc.cu",
            "gpu_ops/read_data_ipc.cu",
            "gpu_ops/enforce_generation.cu",
            "gpu_ops/update_inputs_beam.cu",
            "gpu_ops/save_output_msg_with_topk.cc",
            "gpu_ops/get_output_msg_with_topk.cc",
            "gpu_ops/get_mm_split_fuse.cc",
            "gpu_ops/speculate_decoding/speculate_get_padding_offset.cu",
            "gpu_ops/speculate_decoding/speculate_verify.cu",
            "gpu_ops/speculate_decoding/speculate_set_value_by_flags.cu",
            "gpu_ops/speculate_decoding/speculate_get_seq_lens_output.cu",
            "gpu_ops/speculate_decoding/speculate_save_output.cc",
            "gpu_ops/speculate_decoding/speculate_get_output.cc",
            "gpu_ops/speculate_decoding/speculate_clear_accept_nums.cu",
            "gpu_ops/speculate_decoding/speculate_update_input_ids_cpu.cc",
            "gpu_ops/speculate_decoding/speculate_update_seq_lens_this_time.cu",
            "gpu_ops/speculate_decoding/speculate_get_output_padding_offset.cu",
            "gpu_ops/speculate_decoding/speculate_step.cu",
            "gpu_ops/speculate_decoding/speculate_update_v2.cu",
            "gpu_ops/speculate_decoding/speculate_token_penalty_multi_scores.cu",
            "gpu_ops/speculate_decoding/ngram_match.cc",
            "gpu_ops/swap_cache.cu",
            "gpu_ops/swap_cache_batch.cu",
            "gpu_ops/seqs2seqs.cu",
            "gpu_ops/system2group.cu",
            "gpu_ops/hydra_fetch_hidden_states.cu",
            "gpu_ops/draft_model_update.cu",
            "gpu_ops/draft_model_preprocess.cu",
            "gpu_ops/draft_model_postprocess.cu",
            "gpu_ops/speculate_decoding/speculate_calcu_accept_ratio.cu",
            "gpu_ops/updata_split_fuse_input.cu",
            "gpu_ops/speculate_decoding/speculate_hydra_update_seqlens_this_time.cu",
            "gpu_ops/speculate_decoding/speculate_hydra_set_score_threshold.cu",
            "gpu_ops/speculate_decoding/speculate_update_v3.cu",
            "gpu_ops/speculate_decoding/speculate_stop_generation_multi_stop_seqs.cu",
            "gpu_ops/get_data_ptr_ipc.cu",
            "gpu_ops/ipc_sent_key_value_cache_by_remote_ptr.cu",
            "gpu_ops/extract_text_token_output.cu",
            "gpu_ops/text_image_index_out.cu",
            "gpu_ops/text_image_gather_scatter.cu",
        ],
        extra_compile_args={"nvcc": gencode_flags},
    ),
)

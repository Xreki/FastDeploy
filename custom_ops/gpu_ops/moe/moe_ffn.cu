// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "cutlass/numeric_conversion.h"
#include "cutlass_kernels/w4a8_moe/cutlass_extensions/epilogue/epilogue_quant_helper.h"
#include "cutlass_kernels/w4a8_moe/w4a8_moe_gemm_kernel.h"
#include "group_swiglu_with_masked.h"
#include "helper.h"
#include "moe/fast_hardamard_kernel.h"
#include "moe/fused_moe_helper.h"

template <typename DataT, typename NvType, typename WeightSavedT, cutlass::WintQuantMethod QuantMethod>
void WeightOnlyMoeFFNKernel(const paddle::Tensor& permute_input,
                  const paddle::Tensor& tokens_expert_prefix_sum,
                  const paddle::Tensor& ffn1_weight,
                  const paddle::Tensor& ffn2_weight,
                  const paddle::Tensor* ffn1_bias,
                  const paddle::Tensor* ffn1_super_scale,
                  const paddle::Tensor* ffn2_super_scale,
                  const paddle::Tensor* ffn1_local_scale,
                  const paddle::Tensor* ffn1_code_scale,
                  const paddle::Tensor* ffn1_code_zp,
                  const paddle::Tensor* ffn2_local_scale,
                  const paddle::Tensor* ffn2_code_scale,
                  const paddle::Tensor* ffn2_code_zp,
                  paddle::Tensor fc1_out,
                  paddle::Tensor ffn_out,
                  const int64_t total_rows_in_ll_else_minus1,
                  const int64_t actual_total_rows,
                  const int64_t inter_size,
                  const int64_t hidden_size,
                  const int num_experts,
                  bool used_in_ep_low_latency) {
    using namespace phi;
    using WeightOnlyTraits = cutlass::WintQuantTraits<NvType, QuantMethod>;
    using WeightType = typename WeightOnlyTraits::WeightType;

    typename WeightOnlyTraits::Arguments ffn1_quant_args;
    typename WeightOnlyTraits::Arguments ffn2_quant_args;
    if constexpr (QuantMethod == cutlass::WintQuantMethod::kWeightOnlyInt2) {
        ffn1_quant_args.local_scale_ptr = ffn1_local_scale->data<uint8_t>();
        ffn1_quant_args.code_scale_ptr = ffn1_code_scale->data<float>();
        ffn1_quant_args.code_zp_ptr = ffn1_code_zp->data<float>();
        ffn2_quant_args.local_scale_ptr = ffn2_local_scale->data<uint8_t>();
        ffn2_quant_args.code_scale_ptr = ffn2_code_scale->data<float>();
        ffn2_quant_args.code_zp_ptr = ffn2_code_zp->data<float>();
    }

    auto moe_gemm_runner = MoeGemmRunner<NvType, WeightOnlyTraits>();
    auto stream = permute_input.stream();

    moe_gemm_runner.moe_gemm_bias_act(
        reinterpret_cast<const NvType*>(permute_input.data<DataT>()),
        reinterpret_cast<const WeightType*>(ffn1_weight.data<WeightSavedT>()),
        reinterpret_cast<const NvType*>(ffn1_super_scale ? ffn1_super_scale->data<DataT>() : nullptr),
        reinterpret_cast<const NvType*>(ffn1_bias ? ffn1_bias->data<DataT>() : nullptr),
        reinterpret_cast<NvType*>(fc1_out.data<DataT>()),
        const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
        total_rows_in_ll_else_minus1,
        actual_total_rows,
        inter_size,
        hidden_size,
        num_experts,
        ffn1_quant_args,
        "none",
        stream);

    paddle::Tensor act_out;
    if (used_in_ep_low_latency) {
        act_out = GroupSwigluWithMasked(fc1_out, tokens_expert_prefix_sum);
    } else {
        act_out = paddle::experimental::swiglu(fc1_out, nullptr);
    }

    moe_gemm_runner.moe_gemm(
        reinterpret_cast<const NvType*>(act_out.data<DataT>()),
        reinterpret_cast<const WeightType*>(ffn2_weight.data<WeightSavedT>()),
        reinterpret_cast<const NvType*>(ffn2_super_scale ? ffn2_super_scale->data<DataT>() : nullptr),
        reinterpret_cast<NvType*>(ffn_out.data<DataT>()),
        const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
        total_rows_in_ll_else_minus1,
        actual_total_rows,
        hidden_size,
        inter_size / 2,
        num_experts,
        ffn2_quant_args,
        stream);
}

template <typename DataT, typename NvType>
void W4A8MoeFFNKernel(const paddle::Tensor& permute_input,
                  const paddle::Tensor& tokens_expert_prefix_sum,
                  const paddle::Tensor& ffn1_weight,
                  const paddle::Tensor& ffn2_weight,
                  const paddle::Tensor* ffn1_scale,
                  const paddle::Tensor* ffn2_scale,
                  const paddle::Tensor* ffn2_in_scale,
                  const paddle::Tensor* expert_idx_per_token,
                  paddle::Tensor fc1_out,
                  paddle::Tensor ffn_out,
                  const int64_t total_rows_in_ll_else_minus1,
                  const int64_t actual_total_rows,
                  const int64_t inter_size,
                  const int64_t hidden_size,
                  const int num_experts,
                  const int expanded_active_expert_rows,
                  bool used_in_ep_low_latency) {
    using namespace phi;
    auto w4a8_moe_gemm_runner = W4A8MoeGemmRunner<NvType, int8_t, cutlass::uint4b_t>();
    auto quant_mode = cutlass::epilogue::QuantMode::PerChannelQuant;
   
    auto place = permute_input.place();
    auto stream = permute_input.stream();

    constexpr size_t workspace_size = 1 * 1024 * 1024 * 1024; // for nf4 stream-k
    Allocator* allocator = paddle::GetAllocator(place);
    Allocator::AllocationPtr workspace;
    workspace = allocator->Allocate(SizeOf(paddle::DataType::INT8) * workspace_size);

    w4a8_moe_gemm_runner.moe_gemm(
        reinterpret_cast<const int8_t *>(permute_input.data<int8_t>()),
        reinterpret_cast<const cutlass::uint4b_t *>(ffn1_weight.data<int8_t>()),
        quant_mode,
        reinterpret_cast<const NvType*>(ffn1_scale->data<DataT>()),
        nullptr, // ffn1_scale_dyquant
        nullptr, // nf4_look_up_table
        reinterpret_cast<NvType *>(fc1_out.data<DataT>()),
        const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
        total_rows_in_ll_else_minus1,
        actual_total_rows,
        inter_size,
        hidden_size,
        reinterpret_cast<char*>(workspace->ptr()),
        workspace_size,
        num_experts,
        stream);

    paddle::Tensor act_out;
    if (used_in_ep_low_latency) {
        act_out = GroupSwigluWithMasked(fc1_out, tokens_expert_prefix_sum);
    } else {
        act_out = paddle::experimental::swiglu(fc1_out, nullptr);
    }

    DataT *ffn2_shift = nullptr;
    DataT *ffn2_smooth = nullptr;
    Allocator::AllocationPtr int8_act_out;
    int8_act_out = allocator->Allocate(SizeOf(paddle::DataType::INT8) * act_out.numel());
    MoeFastHardamardWrapper<DataT, int8_t>(
        act_out.data<DataT>(),
        expert_idx_per_token ? expert_idx_per_token->data<int64_t>() : nullptr,
        ffn2_shift, // ffn2_shift->data<T>(),
        ffn2_smooth, // ffn2_smooth->data<T>(),
        ffn2_in_scale ? ffn2_in_scale->data<float>() : nullptr,
        1,
        127.0,
        -127.0,
        expanded_active_expert_rows,
        inter_size / 2,
        reinterpret_cast<int8_t *>(int8_act_out->ptr()),
        stream
    );
    w4a8_moe_gemm_runner.moe_gemm(
        reinterpret_cast<int8_t *>(int8_act_out->ptr()),
        reinterpret_cast<const cutlass::uint4b_t *>(ffn2_weight.data<int8_t>()),
        quant_mode,
        reinterpret_cast<const NvType*>(ffn2_scale->data<DataT>()),
        nullptr, // ffn2_scale_dyquant
        nullptr, // reinterpret_cast<const int32_t*>(d_nf4_look_up_table), // nf4_look_up_table
        reinterpret_cast<NvType *>(ffn_out.data<DataT>()),
        const_cast<int64_t*>(tokens_expert_prefix_sum.data<int64_t>()),
        total_rows_in_ll_else_minus1,
        actual_total_rows,
        hidden_size,
        inter_size / 2,
        reinterpret_cast<char*>(workspace->ptr()),
        workspace_size,
        num_experts,
        stream);
}

template <paddle::DataType T>
void MoeFFNKernel(const paddle::Tensor& permute_input,
                  const paddle::Tensor& tokens_expert_prefix_sum,
                  const paddle::Tensor& ffn1_weight,
                  const paddle::Tensor& ffn2_weight,
                  const paddle::optional<paddle::Tensor>& ffn1_bias,
                  const paddle::optional<paddle::Tensor>& ffn1_scale,
                  const paddle::optional<paddle::Tensor>& ffn2_scale,
                  const paddle::optional<paddle::Tensor>& ffn2_in_scale,
                  const paddle::optional<paddle::Tensor>& expert_idx_per_token,
                  const paddle::optional<paddle::Tensor>& ffn1_local_scale,
                  const paddle::optional<paddle::Tensor>& ffn1_code_scale,
                  const paddle::optional<paddle::Tensor>& ffn1_code_zp,
                  const paddle::optional<paddle::Tensor>& ffn2_local_scale,
                  const paddle::optional<paddle::Tensor>& ffn2_code_scale,
                  const paddle::optional<paddle::Tensor>& ffn2_code_zp,
                  const std::string& quant_method,
                  paddle::Tensor ffn_out,
                  bool used_in_ep_low_latency) {
    using namespace phi;
    using data_t = typename PDTraits<T>::data_t;
    using NvType = typename PDTraits<T>::DataType;

    auto place = permute_input.place();

    assert(permute_input.dims().size() == 3 || permute_input.dims().size() == 2);
    assert(ffn1_weight.dims().size() == 3);

    const int num_experts = ffn1_weight.dims()[0];
    const int hidden_size = permute_input.dims()[permute_input.dims().size() - 1];

    int inter_dim = ffn1_weight.dims()[1] * ffn1_weight.dims()[2] / hidden_size;
    if (quant_method == "weight_only_int4" || quant_method == "w4a8") {
        inter_dim = inter_dim * 2;
    } else if (quant_method == "weight_only_int2") {
        inter_dim = inter_dim * 4;
    }
    const int64_t inter_size = inter_dim;

    int num_experts_ = num_experts;
    int num_max_tokens_per_expert = 0;
    int expanded_active_expert_rows = 0;

    paddle::Tensor fc1_out_tensor;
    if (permute_input.dims().size() == 3) {
        num_experts_ = permute_input.dims()[0];
        assert(num_experts == num_experts_);

        num_max_tokens_per_expert = permute_input.dims()[1];
        expanded_active_expert_rows = num_experts_ * num_max_tokens_per_expert;
        fc1_out_tensor = GetEmptyTensor(
            {num_experts_, num_max_tokens_per_expert, inter_size}, T, place);
    } else {
        expanded_active_expert_rows = permute_input.dims()[0];
        fc1_out_tensor = GetEmptyTensor(
            {expanded_active_expert_rows, inter_size}, T, place);
    }

    // This is a trick.
    // expanded_active_expert_rows is not needed in variable group gemm.
    // but is needed in accommodating deepep low latency mode
    const int64_t total_rows_in_ll_else_minus1 = used_in_ep_low_latency ? expanded_active_expert_rows : -1;

    // When we tune the optimal configuration, we need the actual total_rows.
    const int64_t actual_total_rows = expanded_active_expert_rows;

    if (quant_method == "weight_only_int8") {
        WeightOnlyMoeFFNKernel<data_t, NvType, int8_t, cutlass::WintQuantMethod::kWeightOnlyInt8>(
            permute_input,
            tokens_expert_prefix_sum,
            ffn1_weight,
            ffn2_weight,
            const_cast<paddle::Tensor*>(ffn1_bias.get_ptr()),
            const_cast<paddle::Tensor*>(ffn1_scale.get_ptr()),
            const_cast<paddle::Tensor*>(ffn2_scale.get_ptr()),
            nullptr, // ffn1_local_scale
            nullptr, // ffn1_code_scale
            nullptr, // ffn1_code_zp
            nullptr, // ffn2_local_scale
            nullptr, // ffn2_code_scale
            nullptr, // ffn2_code_zp
            fc1_out_tensor,
            ffn_out,
            total_rows_in_ll_else_minus1,
            actual_total_rows,
            inter_size,
            hidden_size,
            num_experts,
            used_in_ep_low_latency);
    } else if (quant_method == "weight_only_int4") {
        WeightOnlyMoeFFNKernel<data_t, NvType, int8_t, cutlass::WintQuantMethod::kWeightOnlyInt4>(
            permute_input,
            tokens_expert_prefix_sum,
            ffn1_weight,
            ffn2_weight,
            const_cast<paddle::Tensor*>(ffn1_bias.get_ptr()),
            const_cast<paddle::Tensor*>(ffn1_scale.get_ptr()),
            const_cast<paddle::Tensor*>(ffn2_scale.get_ptr()),
            nullptr, // ffn1_local_scale
            nullptr, // ffn1_code_scale
            nullptr, // ffn1_code_zp
            nullptr, // ffn2_local_scale
            nullptr, // ffn2_code_scale
            nullptr, // ffn2_code_zp
            fc1_out_tensor,
            ffn_out,
            total_rows_in_ll_else_minus1,
            actual_total_rows,
            inter_size,
            hidden_size,
            num_experts,
            used_in_ep_low_latency);
    } else if (quant_method == "weight_only_int2") {
        WeightOnlyMoeFFNKernel<data_t, NvType, uint8_t, cutlass::WintQuantMethod::kWeightOnlyInt2>(
            permute_input,
            tokens_expert_prefix_sum,
            ffn1_weight,
            ffn2_weight,
            const_cast<paddle::Tensor*>(ffn1_bias.get_ptr()),
            const_cast<paddle::Tensor*>(ffn1_scale.get_ptr()),
            const_cast<paddle::Tensor*>(ffn2_scale.get_ptr()),
            const_cast<paddle::Tensor*>(ffn1_local_scale.get_ptr()),
            const_cast<paddle::Tensor*>(ffn1_code_scale.get_ptr()),
            const_cast<paddle::Tensor*>(ffn1_code_zp.get_ptr()),
            const_cast<paddle::Tensor*>(ffn2_local_scale.get_ptr()),
            const_cast<paddle::Tensor*>(ffn2_code_scale.get_ptr()),
            const_cast<paddle::Tensor*>(ffn2_code_zp.get_ptr()),
            fc1_out_tensor,
            ffn_out,
            total_rows_in_ll_else_minus1,
            actual_total_rows,
            inter_size,
            hidden_size,
            num_experts,
            used_in_ep_low_latency);
    } else if (quant_method == "w4a8") {
        W4A8MoeFFNKernel<data_t, NvType>(
            permute_input,
            tokens_expert_prefix_sum,
            ffn1_weight,
            ffn2_weight,
            const_cast<paddle::Tensor*>(ffn1_scale.get_ptr()),
            const_cast<paddle::Tensor*>(ffn2_scale.get_ptr()),
            const_cast<paddle::Tensor*>(ffn2_in_scale.get_ptr()),
            const_cast<paddle::Tensor*>(expert_idx_per_token.get_ptr()),
            fc1_out_tensor,
            ffn_out,
            total_rows_in_ll_else_minus1,
            actual_total_rows,
            inter_size,
            hidden_size,
            num_experts,
            expanded_active_expert_rows,
            used_in_ep_low_latency);
    } else {
        WeightOnlyMoeFFNKernel<data_t, NvType, data_t, cutlass::WintQuantMethod::kNone>(
            permute_input,
            tokens_expert_prefix_sum,
            ffn1_weight,
            ffn2_weight,
            const_cast<paddle::Tensor*>(ffn1_bias.get_ptr()),
            nullptr, // ffn1_super_scale
            nullptr, // ffn2_super_scale
            nullptr, // ffn1_local_scale
            nullptr, // ffn1_code_scale
            nullptr, // ffn1_code_zp
            nullptr, // ffn2_local_scale
            nullptr, // ffn2_code_scale
            nullptr, // ffn2_code_zp
            fc1_out_tensor,
            ffn_out,
            total_rows_in_ll_else_minus1,
            actual_total_rows,
            inter_size,
            hidden_size,
            num_experts,
            used_in_ep_low_latency);
    }
}

paddle::Tensor MoeExpertFFNFunc(
    const paddle::Tensor& permute_input,
    const paddle::Tensor& tokens_expert_prefix_sum,
    const paddle::Tensor& ffn1_weight,
    const paddle::Tensor& ffn2_weight,
    const paddle::optional<paddle::Tensor>& ffn1_bias,
    const paddle::optional<paddle::Tensor>& ffn1_scale,
    const paddle::optional<paddle::Tensor>& ffn2_scale,
    const paddle::optional<paddle::Tensor>& ffn2_in_scale,
    const paddle::optional<paddle::Tensor>& expert_idx_per_token,
    const paddle::optional<paddle::Tensor>& ffn1_local_scale,
    const paddle::optional<paddle::Tensor>& ffn1_code_scale,
    const paddle::optional<paddle::Tensor>& ffn1_code_zp,
    const paddle::optional<paddle::Tensor>& ffn2_local_scale,
    const paddle::optional<paddle::Tensor>& ffn2_code_scale,
    const paddle::optional<paddle::Tensor>& ffn2_code_zp,
    const std::string& quant_method, const bool used_in_ep_low_latency) {
    
    const auto dtype = quant_method == "w4a8" ? ffn1_scale.get().dtype() : permute_input.dtype();
    auto ffn_out = paddle::empty_like(permute_input, dtype);

    switch (dtype) {
        case paddle::DataType::BFLOAT16:
            MoeFFNKernel<paddle::DataType::BFLOAT16>(permute_input,
                                                     tokens_expert_prefix_sum,
                                                     ffn1_weight,
                                                     ffn2_weight,
                                                     ffn1_bias,
                                                     ffn1_scale,
                                                     ffn2_scale,
                                                     ffn2_in_scale,
                                                     expert_idx_per_token,
                                                     ffn1_local_scale,
                                                     ffn1_code_scale,
                                                     ffn1_code_zp,
                                                     ffn2_local_scale,
                                                     ffn2_code_scale,
                                                     ffn2_code_zp,
                                                     quant_method,
                                                     ffn_out, used_in_ep_low_latency);
            break;
        case paddle::DataType::FLOAT16:
            MoeFFNKernel<paddle::DataType::FLOAT16>(permute_input,
                                                    tokens_expert_prefix_sum,
                                                    ffn1_weight,
                                                    ffn2_weight,
                                                    ffn1_bias,
                                                    ffn1_scale,
                                                    ffn2_scale,
                                                    ffn2_in_scale,
                                                    expert_idx_per_token,
                                                    ffn1_local_scale,
                                                    ffn1_code_scale,
                                                    ffn1_code_zp,
                                                    ffn2_local_scale,
                                                    ffn2_code_scale,
                                                    ffn2_code_zp,
                                                    quant_method,
                                                    ffn_out, used_in_ep_low_latency);
            break;
        default:
            PD_THROW("Unsupported data type for MoeExpertFFN");
    }
    return ffn_out;
}

std::vector<paddle::Tensor> MoeExpertFFN(
    const paddle::Tensor& permute_input,
    const paddle::Tensor& tokens_expert_prefix_sum,
    const paddle::Tensor& ffn1_weight,
    const paddle::Tensor& ffn2_weight,
    const paddle::optional<paddle::Tensor>& ffn1_bias,
    const paddle::optional<paddle::Tensor>& ffn1_scale,
    const paddle::optional<paddle::Tensor>& ffn2_scale,
    const paddle::optional<paddle::Tensor>& ffn2_in_scale,
    const paddle::optional<paddle::Tensor>& expert_idx_per_token,
    const std::string& quant_method, const bool used_in_ep_low_latency) {

    PD_CHECK(quant_method != "weight_only_int2",
             "weight_only_int2 is not supported in moe_expert_ffn, use moe_expert_ffn_wint2 instead!");

    paddle::optional<paddle::Tensor> ffn1_local_scale;
    paddle::optional<paddle::Tensor> ffn1_code_scale;
    paddle::optional<paddle::Tensor> ffn1_code_zp;
    paddle::optional<paddle::Tensor> ffn2_local_scale;
    paddle::optional<paddle::Tensor> ffn2_code_scale;
    paddle::optional<paddle::Tensor> ffn2_code_zp;

    return {MoeExpertFFNFunc(permute_input,
                             tokens_expert_prefix_sum,
                             ffn1_weight,
                             ffn2_weight,
                             ffn1_bias,
                             ffn1_scale,
                             ffn2_scale,
                             ffn2_in_scale,
                             expert_idx_per_token,
                             ffn1_local_scale,
                             ffn1_code_scale,
                             ffn1_code_zp,
                             ffn2_local_scale,
                             ffn2_code_scale,
                             ffn2_code_zp,
                             quant_method, used_in_ep_low_latency)};
}

std::vector<std::vector<int64_t>> MoeExpertFFNInferShape(
    const std::vector<int64_t>& permute_input_shape,
    const std::vector<int64_t>& tokens_expert_prefix_sum_shape,
    const std::vector<int64_t>& ffn1_weight_shape,
    const std::vector<int64_t>& ffn2_weight_shape,
    const paddle::optional<std::vector<int64_t>>& ffn1_bias_shape,
    const paddle::optional<std::vector<int64_t>>& ffn1_scale_shape,
    const paddle::optional<std::vector<int64_t>>& ffn2_scale_shape,
    const paddle::optional<std::vector<int64_t>>& ffn2_in_scale_shape,
    const paddle::optional<std::vector<int64_t>>& expert_idx_per_token_shape,
    const std::string& quant_method,
    const bool used_in_ep_low_latency) {

    return {permute_input_shape};
}

std::vector<paddle::DataType> MoeExpertFFNInferDtype(
    const paddle::DataType &permute_input_dtype,
    const paddle::DataType &tokens_expert_prefix_sum_dtype,
    const paddle::DataType &ffn1_weight_dtype,
    const paddle::DataType &ffn2_weight_dtype,
    const paddle::optional<paddle::DataType> &ffn1_bias_dtype,
    const paddle::optional<paddle::DataType> &ffn1_scale_dtype,
    const paddle::optional<paddle::DataType> &ffn2_scale_dtype,
    const paddle::optional<paddle::DataType> &ffn2_in_scale_dtype,
    const paddle::optional<paddle::DataType> &expert_idx_per_token_dtype,
    const std::string &quant_method, const bool used_in_ep_low_latency) {
  if (quant_method == "w4a8") {
    return {ffn1_scale_dtype.get()};
  } else {
    return {permute_input_dtype};
  }
}

std::vector<paddle::Tensor> MoeExpertFFNWint2(
    const paddle::Tensor& permute_input,
    const paddle::Tensor& tokens_expert_prefix_sum,
    const paddle::Tensor& ffn1_weight,
    const paddle::Tensor& ffn2_weight,
    const paddle::optional<paddle::Tensor>& ffn1_bias,
    const paddle::optional<paddle::Tensor>& ffn1_scale,
    const paddle::optional<paddle::Tensor>& ffn2_scale,
    const paddle::optional<paddle::Tensor>& ffn1_local_scale,
    const paddle::optional<paddle::Tensor>& ffn1_code_scale,
    const paddle::optional<paddle::Tensor>& ffn1_code_zp,
    const paddle::optional<paddle::Tensor>& ffn2_local_scale,
    const paddle::optional<paddle::Tensor>& ffn2_code_scale,
    const paddle::optional<paddle::Tensor>& ffn2_code_zp,
    const bool used_in_ep_low_latency) {

    paddle::optional<paddle::Tensor> ffn2_in_scale;
    paddle::optional<paddle::Tensor> expert_idx_per_token;

    return {MoeExpertFFNFunc(permute_input,
                             tokens_expert_prefix_sum,
                             ffn1_weight,
                             ffn2_weight,
                             ffn1_bias,
                             ffn1_scale,
                             ffn2_scale,
                             ffn2_in_scale,
                             expert_idx_per_token,
                             ffn1_local_scale,
                             ffn1_code_scale,
                             ffn1_code_zp,
                             ffn2_local_scale,
                             ffn2_code_scale,
                             ffn2_code_zp,
                             "weight_only_int2", used_in_ep_low_latency)};
}

std::vector<std::vector<int64_t>> MoeExpertFFNWint2InferShape(
    const std::vector<int64_t>& permute_input_shape,
    const std::vector<int64_t>& tokens_expert_prefix_sum_shape,
    const std::vector<int64_t>& ffn1_weight_shape,
    const std::vector<int64_t>& ffn2_weight_shape,
    const paddle::optional<std::vector<int64_t>>& ffn1_bias_shape,
    const paddle::optional<std::vector<int64_t>>& ffn1_scale_shape,
    const paddle::optional<std::vector<int64_t>>& ffn2_scale_shape,
    const paddle::optional<std::vector<int64_t>>& ffn1_local_scale_shape,
    const paddle::optional<std::vector<int64_t>>& ffn1_code_scale_shape,
    const paddle::optional<std::vector<int64_t>>& ffn1_code_zp_shape,
    const paddle::optional<std::vector<int64_t>>& ffn2_local_scale_shape,
    const paddle::optional<std::vector<int64_t>>& ffn2_code_scale_shape,
    const paddle::optional<std::vector<int64_t>>& ffn2_code_zp_shape,
    const bool used_in_ep_low_latency) {
    
    return {permute_input_shape};
}

std::vector<paddle::DataType> MoeExpertFFNWint2InferDtype(
    const paddle::DataType &permute_input_dtype,
    const paddle::DataType &tokens_expert_prefix_sum_dtype,
    const paddle::DataType &ffn1_weight_dtype,
    const paddle::DataType &ffn2_weight_dtype,
    const paddle::optional<paddle::DataType> &ffn1_bias_dtype,
    const paddle::optional<paddle::DataType> &ffn1_scale_dtype,
    const paddle::optional<paddle::DataType> &ffn2_scale_dtype,
    const paddle::optional<paddle::DataType> &ffn1_local_scale_dtype,
    const paddle::optional<paddle::DataType> &ffn1_code_scale_dtype,
    const paddle::optional<paddle::DataType> &ffn1_code_zp_dtype,
    const paddle::optional<paddle::DataType> &ffn2_local_scale_dtype,
    const paddle::optional<paddle::DataType> &ffn2_code_scale_dtype,
    const paddle::optional<paddle::DataType> &ffn2_code_zp_dtype,
    const bool used_in_ep_low_latency) {

    return {permute_input_dtype};
}

/**
 * @brief Mixture of Experts (MoE) Feed-Forward Network Operator
 * 
 * This operator performs the expert computation in MoE architecture, including:
 * 1. First linear transformation (FFN1) with optional quantization
 * 2. SwiGLU activation function
 * 3. Second linear transformation (FFN2) with optional quantization
 * 
 * Supports multiple quantization methods including weight-only int4/int8 and w4a8 quantization.
 * 
 * Inputs:
 *   - permute_input: Permuted input tensor organized by expert
 *                   Shape: [total_tokens * top_k, hidden_size]
 *                   dtype: bfloat16/float16 (or int8 for w4a8)
 *   - tokens_expert_prefix_sum: Prefix sum array of token counts per expert for group_gemm
 *                              Shape: [num_experts]
 *                              dtype: int64
 *   - ffn1_weight: First FFN layer weights
 *                 Shape: [num_experts, inter_size * 2, hidden_size]
 *                 dtype: Same as input (unquantized) or int8 (quantized)
 *   - ffn2_weight: Second FFN layer weights
 *                 Shape: [num_experts, hidden_size, inter_size]
 *                 dtype: Same as input (unquantized) or int8 (quantized)
 *   - ffn1_bias: Optional bias for first FFN layer
 *               Shape: [num_experts, inter_size * 2]
 *               dtype: Same as input
 *   - ffn1_scale: Quantization scales for first FFN layer
 *                Shape: [num_experts, inter_size * 2]
 *                dtype: Same as input
 *   - ffn2_scale: Quantization scales for second FFN layer
 *                Shape: [num_experts, hidden_size]
 *                dtype: Same as input
 *   - ffn2_in_scale: Optional input scales for second FFN layer (w4a8 only)
 *                   dtype: float32
 *   - expert_idx_per_token: Optional expert indices per token (w4a8 only)
 *                         Shape: [total_tokens]
 *                         dtype: int64
 * 
 * Outputs:
 *   - output_tensor: Output tensor after MoE FFN computation
 *                   Shape: Same as permute_input
 *                   dtype: Same as input (or ffn1_scale dtype for w4a8)
 * 
 * Attributes:
 *   - quant_method: Quantization method to use
 *                 Options: "none", "weight_only_int4", "weight_only_int8", "w4a8"
 *   - used_in_ep_low_latency: Whether running in low latency mode
 *                            Affects activation function implementation
 * 
 * Note:
 * - w4a8 mode requires additional workspace memory allocation
 * - Low latency mode uses specialized grouped SwiGLU implementation
 */
PD_BUILD_STATIC_OP(moe_expert_ffn)
    .Inputs({"permute_input",
             "tokens_expert_prefix_sum",
             "ffn1_weight",
             "ffn2_weight",
             paddle::Optional("ffn1_bias"),
             paddle::Optional("ffn1_scale"),
             paddle::Optional("ffn2_scale"),
             paddle::Optional("ffn2_in_scale"),
             paddle::Optional("expert_idx_per_token")})
    .Outputs({"output_tensor"})
    .Attrs({"quant_method:std::string", "used_in_ep_low_latency:bool"})
    .SetKernelFn(PD_KERNEL(MoeExpertFFN))
    .SetInferShapeFn(PD_INFER_SHAPE(MoeExpertFFNInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoeExpertFFNInferDtype));

PD_BUILD_STATIC_OP(moe_expert_ffn_wint2)
    .Inputs({"permute_input",
             "tokens_expert_prefix_sum",
             "ffn1_weight",
             "ffn2_weight",
             paddle::Optional("ffn1_bias"),
             paddle::Optional("ffn1_scale"),
             paddle::Optional("ffn2_scale"),
             paddle::Optional("ffn1_local_scale"),
             paddle::Optional("ffn1_code_scale"),
             paddle::Optional("ffn1_code_zp"),
             paddle::Optional("ffn2_local_scale"),
             paddle::Optional("ffn2_code_scale"),
             paddle::Optional("ffn2_code_zp")})
    .Outputs({"output_tensor"})
    .Attrs({"used_in_ep_low_latency:bool"})
    .SetKernelFn(PD_KERNEL(MoeExpertFFNWint2))
    .SetInferShapeFn(PD_INFER_SHAPE(MoeExpertFFNWint2InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoeExpertFFNWint2InferDtype));
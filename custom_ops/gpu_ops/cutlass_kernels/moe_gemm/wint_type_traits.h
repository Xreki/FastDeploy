// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <string>

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"

namespace wintx {

enum WintQuantMethod {
  kNone = 0,
  kWeightOnlyInt8 = 1,
  kWeightOnlyInt4 = 2,
  kWeightOnlyInt25 = 3,
  kWeightOnlyInt2 = 4
};

template <int32_t GroupSize, int32_t NumPackedValues>
constexpr int32_t CalcPackedSize() {
  return (GroupSize + NumPackedValues - 1) / NumPackedValues;
}

template <WintQuantMethod Method> struct WintTypeTraits {
  using WeightType = cutlass::bfloat16_t;

  CUTLASS_DEVICE
  static int64_t CaclPackedDim(int64_t dim) { return dim; }
};

template <> struct WintTypeTraits<WintQuantMethod::kWeightOnlyInt8> {
  using WeightType = uint8_t;

  CUTLASS_DEVICE
  static int64_t CaclPackedDim(int64_t dim) { return dim; }
};

template <> struct WintTypeTraits<WintQuantMethod::kWeightOnlyInt4> {
  using WeightType = cutlass::uint4b_t;

  CUTLASS_DEVICE
  static int64_t CaclPackedDim(int64_t dim) { return dim; }
};

template <> struct WintTypeTraits<WintQuantMethod::kWeightOnlyInt25> {
  using WeightType = uint16_t;

  static constexpr int32_t kGroupSize = 64;
  static constexpr int32_t kNumPackedValues = 7;
  static constexpr int32_t kPackedSize =
      CalcPackedSize<kGroupSize, kNumPackedValues>(); // 10

  CUTLASS_DEVICE
  static int64_t CaclPackedDim(int64_t dim) {
    return dim * kPackedSize / kGroupSize;
  }
};

// Convert CUDA data type to cutlass data type
template <typename T> struct CutlassDataType {
  using Type = T;
};

template <> struct CutlassDataType<half> {
  using Type = cutlass::half_t;
};

template <> struct CutlassDataType<__nv_bfloat16> {
  using Type = cutlass::bfloat16_t;
};

template <typename ElementT, typename WeightT> struct CutlassMmaTraits {
  using MmaWeightType = typename CutlassDataType<ElementT>::Type;
};

template <typename ElementT> struct CutlassMmaTraits<ElementT, uint8_t> {
  using MmaWeightType = uint8_t;
};

template <typename ElementT>
struct CutlassMmaTraits<ElementT, cutlass::uint4b_t> {
  using MmaWeightType = cutlass::uint4b_t;
};

template <typename T> std::string GetCutlassDataTypeString() {
  if (std::is_same<T, float>::value) {
    return "float";
  } else if (std::is_same<T, cutlass::half_t>::value) {
    return "cutlass::half_t";
  } else if (std::is_same<T, cutlass::bfloat16_t>::value) {
    return "cutlass::bfloat16_t";
  } else if (std::is_same<T, uint16_t>::value) {
    return "uint16_t";
  } else if (std::is_same<T, uint8_t>::value) {
    return "uint8_t";
  } else if (std::is_same<T, cutlass::uint4b_t>::value) {
    return "cutlass::uint4b_t";
  }
  return "unknown";
}

template <typename Layout> std::string GetCutlassLayoutString() {
  if (std::is_same<Layout, cutlass::layout::RowMajor>::value) {
    return "cutlass::layout::RowMajor";
  } else if (std::is_same<Layout, cutlass::layout::ColumnMajor>::value) {
    return "cutlass::layout::ColumnMajor";
  }
  return "unknown";
}

} // namespace wintx

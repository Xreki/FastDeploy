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

#include "cutlass/gemm_coord.h"
#include "cutlass/trace.h"
#include "cutlass_kernels/moe_gemm/wintx_unzip_impl.h"

template <WintQuantMethod Method>
struct UseSharedMemory : std::false_type {};

template <>
struct UseSharedMemory<WintQuantMethod::kWeightOnlyInt25> : std::true_type {};

template <>
struct UseSharedMemory<WintQuantMethod::kWeightOnlyInt2> : std::true_type {};

template <typename ElementT, typename ScaleElementT, int Rows, int Columns, int NumThreads, WintQuantMethod Method, typename = void>
struct TileDequanter {
  using MmaElementT = typename WintQuantTraits<ElementT, Method>::MmaWeightType;

  static constexpr bool kUseSharedMemory = false;

  static constexpr int kRows = Rows;
  static constexpr int kColumns = Columns;

  struct SharedStorage {};

  char* pointer{nullptr};

  CUTLASS_DEVICE
  TileDequanter(SharedStorage& storage,
                char *pointer,
                int64_t ldm,
                const cutlass::MatrixCoord &extent,
                const cutlass::MatrixCoord &tb_offset,
                ScaleElementT* super_scale_ptr,
                const cutlass::MatrixCoord &extent_scale,
                const cutlass::MatrixCoord &tb_offset_scale) : pointer(pointer) {}

  CUTLASS_DEVICE
  MmaElementT* GetOutPtr() { return reinterpret_cast<MmaElementT*>(pointer); }

  CUTLASS_DEVICE
  void AddTileOffset(const cutlass::MatrixCoord &tile_offset) {}

  CUTLASS_DEVICE
  void Apply() {}
};

template <typename ElementT, typename ScaleElementT, int Rows, int Columns, int NumThreads, WintQuantMethod Method>
struct TileDequanter<ElementT, ScaleElementT, Rows, Columns, NumThreads, Method, std::enable_if_t<UseSharedMemory<Method>::value>> {  
  using MmaElementT = typename WintQuantTraits<ElementT, Method>::MmaWeightType;
  using UnzipAndDequantFunctor = UnzipAndDequantFunctor<MmaElementT, Method, Rows, Columns, NumThreads>;

  static constexpr bool kUseSharedMemory = true;

  static constexpr int kRows = Rows;
  static constexpr int kColumns = Columns;

  struct SharedStorage {
    MmaElementT smem[kRows * kColumns];
  };

  MmaElementT* smem_ptr{nullptr};

  char* pointer{nullptr};
  int64_t ldm{0};
  cutlass::MatrixCoord tb_offset;
  cutlass::MatrixCoord extent;

  ScaleElementT* super_scale_ptr{nullptr};
  cutlass::MatrixCoord tb_offset_scale;
  cutlass::MatrixCoord extent_scale;

  CUTLASS_DEVICE
  TileDequanter(SharedStorage& storage,
                char *pointer,
                int64_t ldm,
                const cutlass::MatrixCoord &extent,
                const cutlass::MatrixCoord &tb_offset,
                ScaleElementT* super_scale_ptr,
                const cutlass::MatrixCoord &extent_scale,
                const cutlass::MatrixCoord &tb_offset_scale)
    : smem_ptr(storage.smem),
      pointer(pointer),
      ldm(ldm),
      extent(extent),
      tb_offset(tb_offset),
      super_scale_ptr(super_scale_ptr),
      extent_scale(extent_scale),
      tb_offset_scale(tb_offset_scale) {
    //CUTLASS_TRACE_DEVICE(" TileDequanter::SharedStorage: {%d, %d} * %d = %d bytes",
    //    kRows, kColumns, static_cast<int>(sizeof(MmaElementT)), static_cast<int>(sizeof(SharedStorage)));
  }

  CUTLASS_DEVICE
  MmaElementT* GetOutPtr() { return smem_ptr; }

  CUTLASS_DEVICE
  void AddTileOffset(const cutlass::MatrixCoord &tile_offset) {
    //CUTLASS_TRACE_DEVICE(" [TileDequanter] tile_offset={%d, %d}", static_cast<int>(tile_offset.row()), static_cast<int>(tile_offset.column()));
    tb_offset.row() += tile_offset.row() * kRows;
    tb_offset.column() += tile_offset.column() * kColumns;
    tb_offset_scale.column() += tile_offset.column() * kColumns;
  }

  CUTLASS_DEVICE
  void Apply() {
    int fake_value = static_cast<int>(tb_offset.row()) / kRows;
    if (tb_offset.row() >= extent.row() || tb_offset.column() >= extent.column()) {
      //CUTLASS_TRACE_DEVICE(" TileDequanter::Apply, tb_offset={%d, %d}, skipped!!!",
      //    static_cast<int>(tb_offset.row()), static_cast<int>(tb_offset.column()));
      return;
    } else {
      //CUTLASS_TRACE_DEVICE(" TileDequanter::Apply, tb_offset={%d, %d}, fake_value={%d}",
      //    static_cast<int>(tb_offset.row()), static_cast<int>(tb_offset.column()), fake_value);
    }

    using ZippedT = typename WintQuantTraits<ElementT, Method>::WeightType;

    MmaElementT* out_ptr = smem_ptr;

    int zipped_row = tb_offset.row() * 10 / 64;
    ZippedT* in_ptr = reinterpret_cast<ZippedT*>(pointer) + zipped_row * ldm + tb_offset.column();
    ScaleElementT* scale_ptr = super_scale_ptr + tb_offset_scale.column();

    UnzipAndDequantFunctor unzip_and_dequant_functor;
    unzip_and_dequant_functor(in_ptr, scale_ptr, out_ptr, ldm);
  }
};
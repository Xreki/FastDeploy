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

template <WintQuantMethod Method> struct UseSharedMemory : std::false_type {};

template <>
struct UseSharedMemory<WintQuantMethod::kWeightOnlyInt25> : std::true_type {};

template <>
struct UseSharedMemory<WintQuantMethod::kWeightOnlyInt2> : std::true_type {};

template <typename ElementT, typename ScaleElementT, int Rows, int Columns,
          int NumThreads, WintQuantMethod Method, typename = void>
struct TileDequanter {
  using WeightQuantTraits = WintQuantTraits<ElementT, Method>;
  using MmaElementT = typename WeightQuantTraits::MmaWeightType;
  using Arguments = typename WeightQuantTraits::Arguments;

  static constexpr bool kUseSharedMemory = false;

  static constexpr int kRows = Rows;
  static constexpr int kColumns = Columns;

  struct SharedStorage {};

  char *pointer{nullptr};

  CUTLASS_DEVICE
  TileDequanter(SharedStorage &storage, char *pointer, int64_t ldm,
                const cutlass::MatrixCoord &extent,
                const cutlass::MatrixCoord &tb_offset,
                ScaleElementT *super_scale_ptr,
                const cutlass::MatrixCoord &tb_offset_scale,
                const Arguments &quant_args)
      : pointer(pointer) {}

  CUTLASS_DEVICE
  MmaElementT *GetOutPtr() { return reinterpret_cast<MmaElementT *>(pointer); }

  CUTLASS_DEVICE
  void AddTileOffset(const cutlass::MatrixCoord &tile_offset) {}

  CUTLASS_DEVICE
  void Apply() {}
};

template <typename ElementT, typename ScaleElementT, int Rows, int Columns,
          int NumThreads, WintQuantMethod Method>
struct TileDequanter<ElementT, ScaleElementT, Rows, Columns, NumThreads, Method,
                     std::enable_if_t<UseSharedMemory<Method>::value>> {
  using WeightQuantTraits = WintQuantTraits<ElementT, Method>;
  using MmaElementT = typename WeightQuantTraits::MmaWeightType;
  using Arguments = typename WeightQuantTraits::Arguments;

  using UnzipAndDequantFunctor =
      UnzipAndDequantFunctor<MmaElementT, Method, Rows, Columns, NumThreads>;

  static constexpr bool kUseSharedMemory = true;

  static constexpr int kRows = Rows;
  static constexpr int kColumns = Columns;

  struct SharedStorage {
    MmaElementT smem[kRows * kColumns];
  };

  MmaElementT *smem_ptr{nullptr};

  char *pointer{nullptr};
  int64_t ldm{0};
  cutlass::MatrixCoord tb_offset;
  cutlass::MatrixCoord extent;

  ScaleElementT *super_scale_ptr{nullptr};
  cutlass::MatrixCoord tb_offset_scale;

  Arguments quant_args;

  CUTLASS_DEVICE
  TileDequanter(SharedStorage &storage, char *pointer, int64_t ldm,
                const cutlass::MatrixCoord &extent,
                const cutlass::MatrixCoord &tb_offset,
                ScaleElementT *super_scale_ptr,
                const cutlass::MatrixCoord &tb_offset_scale,
                const Arguments &quant_args)
      : smem_ptr(storage.smem), pointer(pointer), ldm(ldm), extent(extent),
        tb_offset(tb_offset), super_scale_ptr(super_scale_ptr),
        tb_offset_scale(tb_offset_scale), quant_args(quant_args) {
    // CUTLASS_TRACE_DEVICE(" TileDequanter::SharedStorage: {%d, %d} * %d = %d
    // bytes",
    //     kRows, kColumns, static_cast<int>(sizeof(MmaElementT)),
    //     static_cast<int>(sizeof(SharedStorage)));
  }

  CUTLASS_DEVICE
  MmaElementT *GetOutPtr() { return smem_ptr; }

  CUTLASS_DEVICE
  void AddTileOffset(const cutlass::MatrixCoord &tile_offset) {
    // CUTLASS_TRACE_DEVICE(" [TileDequanter] tile_offset={%d, %d}",
    // static_cast<int>(tile_offset.row()),
    // static_cast<int>(tile_offset.column()));
    tb_offset.row() += tile_offset.row() * kRows;
    tb_offset.column() += tile_offset.column() * kColumns;
    tb_offset_scale.column() += tile_offset.column() * kColumns;
  }

  CUTLASS_DEVICE
  void Apply() {
    int fake_value = static_cast<int>(tb_offset.row()) / kRows;
    int zipped_row = WeightQuantTraits::CaclPackedDim(tb_offset.row());
    if (tb_offset.row() >= extent.row() ||
        tb_offset.column() >= extent.column()) {
      // CUTLASS_TRACE_DEVICE(" TileDequanter::Apply, tb_offset={%d, %d},
      // zipped_row=%d, skipped!!!",
      //     static_cast<int>(tb_offset.row()),
      //     static_cast<int>(tb_offset.column()), zipped_row);
      return;
    } else {
      // CUTLASS_TRACE_DEVICE(" TileDequanter::Apply, tb_offset={%d, %d},
      // zipped_row=%d, fake_value={%d}",
      //     static_cast<int>(tb_offset.row()),
      //     static_cast<int>(tb_offset.column()), zipped_row, fake_value);
    }

    using ZippedT = typename WeightQuantTraits::WeightType;

    MmaElementT *out_ptr = smem_ptr;

    ZippedT *in_ptr = reinterpret_cast<ZippedT *>(pointer) + zipped_row * ldm +
                      tb_offset.column();
    ScaleElementT *scale_ptr = super_scale_ptr + tb_offset_scale.column();

    UnzipAndDequantFunctor unzip_and_dequant_functor;
    if constexpr (Method == WintQuantMethod::kWeightOnlyInt2) {
      int64_t block_start_row = tb_offset.row();
      const uint8_t *local_scale_ptr = quant_args.local_scale_ptr +
                                       (block_start_row / 128) * ldm +
                                       tb_offset_scale.column();
      const float *code_scale_ptr =
          quant_args.code_scale_ptr + tb_offset_scale.column();
      const float *code_zp_ptr =
          quant_args.code_zp_ptr + tb_offset_scale.column();
      unzip_and_dequant_functor(in_ptr, local_scale_ptr, code_scale_ptr,
                                code_zp_ptr, scale_ptr, out_ptr,
                                block_start_row, ldm);
    } else {
      unzip_and_dequant_functor(in_ptr, scale_ptr, out_ptr, ldm);
    }
  }
};

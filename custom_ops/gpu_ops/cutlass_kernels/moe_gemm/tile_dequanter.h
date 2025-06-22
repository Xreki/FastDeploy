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
          int Stages, int NumThreads, WintQuantMethod Method, typename = void>
struct TileDequanter {
  using WeightQuantTraits = WintQuantTraits<ElementT, Method>;
  using MmaElementT = typename WeightQuantTraits::MmaWeightType;
  using QuantArguments = typename WeightQuantTraits::Arguments;

  static constexpr bool kUseSharedMemory = false;

  static constexpr int kRows = Rows;
  static constexpr int kColumns = Columns;

  char *pointer{nullptr};

  CUTLASS_DEVICE
  TileDequanter(MmaElementT *smem_ptr, char *pointer, int64_t ldm,
                const cutlass::MatrixCoord &extent,
                const cutlass::MatrixCoord &tb_offset,
                ScaleElementT *super_scale_ptr,
                const cutlass::MatrixCoord &tb_offset_scale,
                const QuantArguments &quant_args)
      : pointer(pointer) {}

  CUTLASS_DEVICE
  MmaElementT *GetOutPtr() { return reinterpret_cast<MmaElementT *>(pointer); }

  CUTLASS_DEVICE
  void AddTileOffset(const cutlass::MatrixCoord &tile_offset) {}

  CUTLASS_DEVICE
  void Apply() {}
};

template <typename ElementT, typename ScaleElementT, int Rows, int Columns,
          int Stages, int NumThreads, WintQuantMethod Method>
struct TileDequanter<ElementT, ScaleElementT, Rows, Columns, Stages, NumThreads, Method,
                     std::enable_if_t<UseSharedMemory<Method>::value>> {
  using WeightQuantTraits = WintQuantTraits<ElementT, Method>;
  using MmaElementT = typename WeightQuantTraits::MmaWeightType;
  using QuantArguments = typename WeightQuantTraits::Arguments;

  using UnzipAndDequantFunctor =
      UnzipAndDequantFunctor<MmaElementT, Method, Rows, Columns, NumThreads>;

  static constexpr bool kUseSharedMemory = true;

  static constexpr int kRows = Rows;
  static constexpr int kColumns = Columns;
  static constexpr int kStages = Stages;

  MmaElementT *smem_ptr{nullptr};

  char *pointer{nullptr};
  int64_t ldm{0};
  cutlass::MatrixCoord tb_offset;
  cutlass::MatrixCoord extent;

  ScaleElementT *super_scale_ptr{nullptr};
  cutlass::MatrixCoord tb_offset_scale;

  QuantArguments quant_args;

  int64_t block_start_rows[kStages];
  bool need_preload{true};

  CUTLASS_DEVICE
  TileDequanter(MmaElementT *smem_ptr, char *pointer, int64_t ldm,
                const cutlass::MatrixCoord &extent,
                const cutlass::MatrixCoord &tb_offset,
                ScaleElementT *super_scale_ptr,
                const cutlass::MatrixCoord &tb_offset_scale,
                const QuantArguments &quant_args)
      : smem_ptr(smem_ptr), pointer(pointer), ldm(ldm), extent(extent),
        tb_offset(tb_offset), super_scale_ptr(super_scale_ptr),
        tb_offset_scale(tb_offset_scale), quant_args(quant_args) {}

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
  void Load(uint8_t *zipped_smem_ptr, uint8_t *column_wise_smem_ptr, int stage) {
    int zipped_row = WeightQuantTraits::CaclPackedDim(tb_offset.row());
    if (tb_offset.row() >= extent.row() ||
        tb_offset.column() >= extent.column()) {
      CUTLASS_TRACE_DEVICE(" zipped_smem_ptr=%p, stage=%d, tb_offset={%d, %d}, "
                           "zipped_row=%d, skipped!!!",
                           reinterpret_cast<void *>(zipped_smem_ptr), stage,
                           static_cast<int>(tb_offset.row()),
                           static_cast<int>(tb_offset.column()), zipped_row);
      return;
    } else {
      CUTLASS_TRACE_DEVICE(" zipped_smem_ptr=%p, stage=%d, tb_offset={%d, %d}, zipped_row=%d",
          reinterpret_cast<void*>(zipped_smem_ptr), stage,
          static_cast<int>(tb_offset.row()),
          static_cast<int>(tb_offset.column()), zipped_row);
    }

    block_start_rows[stage % kStages] = tb_offset.row();

    using ZippedT = typename WeightQuantTraits::WeightType;
    ZippedT *in_ptr = reinterpret_cast<ZippedT *>(pointer) + zipped_row * ldm +
                      tb_offset.column();
    ScaleElementT *scale_ptr = super_scale_ptr + tb_offset_scale.column();

    UnzipAndDequantFunctor functor;
    if constexpr (Method == WintQuantMethod::kWeightOnlyInt2) {
      const uint8_t *local_scale_ptr = quant_args.local_scale_ptr +
                                       (tb_offset.row() / 128) * ldm +
                                       tb_offset_scale.column();
      const float *code_scale_ptr =
          quant_args.code_scale_ptr + tb_offset_scale.column();
      const float *code_zp_ptr =
          quant_args.code_zp_ptr + tb_offset_scale.column();
      
      typename UnzipAndDequantFunctor::Arguments args(zipped_smem_ptr, column_wise_smem_ptr);
      functor.LoadAsync(in_ptr, local_scale_ptr, code_scale_ptr, code_zp_ptr,
                        scale_ptr, &args, ldm, need_preload);
      need_preload = false;
    } else {
      CUTLASS_TRACE_DEVICE("Not Supported!");
    }
  }

  CUTLASS_DEVICE
  void UnpackAndDequant(uint8_t *zipped_smem_ptr, uint8_t *column_wise_smem_ptr, int stage) {
    MmaElementT *out_ptr = smem_ptr;

    int64_t block_start_row = block_start_rows[stage % kStages];
    int fake_value = (stage < 128) ? (block_start_row / Rows + 1) : 0;
    if (block_start_row >= extent.row()) {
      CUTLASS_TRACE_DEVICE(" zipped_smem_ptr=%p, out_ptr=%p, stage=%d, "
                           "block_start_row=%d, skipped!!!",
                           reinterpret_cast<void *>(zipped_smem_ptr), out_ptr,
                           stage, static_cast<int>(block_start_row));
      return;
    } else {
      CUTLASS_TRACE_DEVICE(" zipped_smem_ptr=%p, out_ptr=%p, stage=%d, "
                           "block_start_row=%d, fake_value=%d",
                           reinterpret_cast<void *>(zipped_smem_ptr), out_ptr,
                           stage, static_cast<int>(block_start_row),
                           fake_value);
    }

    UnzipAndDequantFunctor functor;
    if constexpr (Method == WintQuantMethod::kWeightOnlyInt2) {
      typename UnzipAndDequantFunctor::Arguments args(zipped_smem_ptr, column_wise_smem_ptr);
      functor.Compute(args, out_ptr, block_start_row);
#if 0
      for (int col = threadIdx.x; col < Columns; ++col) {
        for (int row = 0; row < Rows; ++row) {
          out_ptr[row * Columns + col] = static_cast<MmaElementT>(fake_value);
        }
      }
      __syncthreads();
#endif
    } else {
      CUTLASS_TRACE_DEVICE("Not Supported!");
    }
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

#if 0
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
#endif
  }
};

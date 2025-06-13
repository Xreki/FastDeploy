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
#include <cuda_fp16.h>
#include <stdio.h>

#include "cutlass_kernels/moe_gemm/wint_type_traits.h"
#include "helper.h"

#define UNZIP_ENABLE_VECTORIZE 0

template <typename T, WintQuantMethod QuantMethod, int TileRows,
          int TileColumns, int NumThreads = 128>
struct UnzipAndDequantFunctor {
  __device__ void operator()(const T *in_ptr, const T *supper_scale_ptr,
                             T *out_ptr, const int64_t in_stride) {}
};

#if UNZIP_ENABLE_VECTORIZE

template <typename T, int N> using Array = AlignedVector<T, N>;

template <typename T, int TileRows, int TileColumns, int NumThreads>
struct UnzipAndDequantFunctor<T, WintQuantMethod::kWeightOnlyInt25, TileRows,
                              TileColumns, NumThreads> {
  using ZippedT = uint16_t;
  using ScaleComputeT = float;

  static constexpr int32_t kGroupSize = 64;
  static constexpr int32_t kZippedGroupSize = 10;
  static constexpr int32_t kNumPackedValues = 7;

  static constexpr int32_t kWeightMask = 0x7;
  static constexpr int32_t kLocalScaleMask = 0x1FFF;
  static constexpr int32_t kBZP = 4;

  static constexpr int N =
      (TileColumns > NumThreads) ? (TileColumns / NumThreads) : 2;
  static constexpr int RowStride =
      (TileColumns > NumThreads) ? 1 : (NumThreads * N / TileColumns);
  // using AccessType = cutlass::AlignedArray<ZippedT, ElementsPerAccess,
  // (ElementsPerAccess * cutlass::sizeof_bits<ZippedT>::value / 8)>;

  static_assert((TileRows > 0) && (TileRows % 64 == 0),
                "TileRows must be a multiple of 64.");

  __device__ inline Array<T, N> Compute(const Array<uint16_t, N> &zipped_values,
                                        int32_t shift_bit,
                                        const Array<ScaleComputeT, N> &scales) {
    Array<T, N> values;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      int32_t shifted_value =
          (static_cast<int32_t>(zipped_values[i]) >> shift_bit) &
          WeightOnlyTraits::kWeightMask;
      int32_t value = shifted_value - WeightOnlyTraits::kBZP;

      ScaleComputeT scaled_value =
          static_cast<ScaleComputeT>(value) * scales[i];
      values[i] = static_cast<T>(scaled_value);
    }
    return values;
  }

  __device__ inline Array<ScaleComputeT, N>
  ComputeScale(const uint16_t *group_in_ptr, const T *supper_scale_ptr,
               const int begin_col_id, const int64_t in_stride) {
    int zipped_offset_last = 9 * in_stride + begin_col_id;
    Array<uint16_t, N> zipped_value_lasts =
        *reinterpret_cast<const Array<uint16_t, N> *>(group_in_ptr +
                                                      zipped_offset_last);

    Array<T, N> super_scales =
        *reinterpret_cast<const Array<T, N> *>(supper_scale_ptr + begin_col_id);

    Array<ScaleComputeT, N> scales;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      int32_t zipped_value = static_cast<int32_t>(zipped_value_lasts[i]);

      ScaleComputeT local_scale = static_cast<ScaleComputeT>(
          zipped_value & WeightOnlyTraits::kLocalScaleMask);
      scales[i] = local_scale * static_cast<ScaleComputeT>(super_scales[i]);
    }
    return scales;
  }

  __device__ inline void ApplySingleGroup(const uint16_t *group_in_ptr,
                                          const T *supper_scale_ptr,
                                          T *group_out_ptr,
                                          const int64_t in_stride) {
    int32_t shift_bits[7] = {13, 11, 9, 6, 4, 2, 0};

    int tid = threadIdx.x;
    int begin_col_id = (tid * N) % TileColumns;
    int begin_row_id = (tid * N) / TileColumns;

    Array<ScaleComputeT, N> scales =
        ComputeScale(group_in_ptr, supper_scale_ptr, begin_col_id, in_stride);

    int zipped_row = begin_row_id;

#pragma unroll
    for (; zipped_row < 9; zipped_row += RowStride) {
      int zipped_offset = zipped_row * in_stride + begin_col_id;

      Array<uint16_t, N> zipped_values =
          *reinterpret_cast<const Array<uint16_t, N> *>(group_in_ptr +
                                                        zipped_offset);

      int row = zipped_row * 7;

#pragma unroll
      for (int shift_bit_id = 0; shift_bit_id < 7; ++shift_bit_id) {
        int32_t shift_bit = shift_bits[shift_bit_id];
        Array<T, N> values = Compute(zipped_values, shift_bit, scales);
        Array<T, N> *tmp_out_ptr = reinterpret_cast<Array<T, N> *>(
            group_out_ptr + (row + shift_bit_id) * TileColumns + begin_col_id);
        *tmp_out_ptr = values;
      }
    }

    if (zipped_row == 9) {
      int zipped_offset = 9 * in_stride + begin_col_id;
      Array<uint16_t, N> zipped_values =
          *reinterpret_cast<const Array<uint16_t, N> *>(group_in_ptr +
                                                        zipped_offset);
      Array<T, N> values_last = Compute(zipped_values, shift_bits[0], scales);
      Array<T, N> *tmp_out_ptr = reinterpret_cast<Array<T, N> *>(
          group_out_ptr + 63 * TileColumns + begin_col_id);
      *tmp_out_ptr = values_last;
    }
  }

  __device__ void operator()(const uint16_t *in_ptr, const T *supper_scale_ptr,
                             T *out_ptr, const int64_t in_stride) {
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //   printf("N=%d\n", N);
    // }

#pragma unroll
    for (int group_id = 0; group_id < TileRows / 64; ++group_id) {
      const uint16_t *group_in_ptr = in_ptr + group_id * 10 * in_stride;
      T *group_out_ptr = out_ptr + group_id * 64 * TileColumns;

      ApplySingleGroup(group_in_ptr, supper_scale_ptr, group_out_ptr,
                       in_stride);
    }
    __syncthreads();
  }
};

#else

template <typename T, int TileRows, int TileColumns, int NumThreads>
struct UnzipAndDequantFunctor<T, WintQuantMethod::kWeightOnlyInt25, TileRows,
                              TileColumns, NumThreads> {
  using ZippedT = uint16_t;
  using ScaleComputeT = float;

  static constexpr int32_t kGroupSize = 64;
  static constexpr int32_t kZippedGroupSize = 10;
  static constexpr int32_t kNumPackedValues = 7;

  static constexpr int32_t kWeightMask = 0x7;
  static constexpr int32_t kLocalScaleMask = 0x1FFF;
  static constexpr int32_t kBZP = 4;

  __device__ inline T Compute(int32_t zipped_value, int32_t shift_bit,
                              ScaleComputeT scale) {
    int32_t shifted_value = (zipped_value >> shift_bit) & kWeightMask;
    int32_t value = shifted_value - kBZP;

    ScaleComputeT scaled_value = static_cast<ScaleComputeT>(value) * scale;
    return static_cast<T>(scaled_value);
  }

  __device__ void operator()(const uint16_t *in_ptr, const T *super_scale_ptr,
                             T *out_ptr, const int64_t in_stride) {
    int32_t shift_bits[7] = {13, 11, 9, 6, 4, 2, 0};

    int tid = threadIdx.x;

#pragma unroll
    for (int col = tid; col < TileColumns; col += NumThreads) {
      ScaleComputeT super_scale =
          static_cast<ScaleComputeT>(super_scale_ptr[col]);

#pragma unroll
      for (int group_id = 0; group_id < TileRows / 64; ++group_id) {
        // the last row in group
        int zipped_row_last = group_id * 10 + 9;
        int zipped_offset_last = zipped_row_last * in_stride + col;
        int32_t zipped_value_last =
            static_cast<int32_t>(in_ptr[zipped_offset_last]);

        ScaleComputeT local_scale =
            static_cast<ScaleComputeT>(zipped_value_last & kLocalScaleMask);
        ScaleComputeT scale = local_scale * super_scale;

#pragma unroll
        for (int zipped_row_in_group = 0; zipped_row_in_group < 9;
             ++zipped_row_in_group) {
          int zipped_row = group_id * 10 + zipped_row_in_group;
          int zipped_offset = zipped_row * in_stride + col;
          int32_t zipped_value = static_cast<int32_t>(in_ptr[zipped_offset]);

          int row_in_group = group_id * 64 + zipped_row_in_group * 7;

#pragma unroll
          for (int shift_bit_id = 0; shift_bit_id < 7; ++shift_bit_id) {
            int32_t shift_bit = shift_bits[shift_bit_id];
            T value = Compute(zipped_value, shift_bit, scale);
            out_ptr[(row_in_group + shift_bit_id) * TileColumns + col] = value;
          }
        }

        int row_in_group_last = group_id * 64 + 63;
        T value_last = Compute(zipped_value_last, shift_bits[0], scale);
        out_ptr[row_in_group_last * TileColumns + col] = value_last;
      }
    }
    __syncthreads();
  }
};

template <typename T, int TileRows, int TileColumns, int NumThreads>
struct UnzipAndDequantFunctor<T, WintQuantMethod::kWeightOnlyInt2, TileRows,
                              TileColumns, NumThreads> {
  using ZippedT = uint8_t;
  using ScaleComputeT = float;

  static constexpr int32_t kGroupSize = 64;
  static constexpr int32_t kPackNum = 4;
  static constexpr int32_t kWeightMask = 0x3F;
  static constexpr int32_t kLocalScaleMask = 0xF;
  static constexpr int32_t kBZP = 32;

  __device__ void operator()(const uint8_t *in_ptr,
                             const uint8_t *local_scale_ptr,
                             const float *code_scale_ptr,
                             const float *code_zp_ptr, const T *super_scale_ptr,
                             T *out_ptr, const int64_t block_start_row,
                             const int64_t in_stride) {
    int32_t shift_bits[4] = {9, 6, 3, 0};

    int tid = threadIdx.x;

#pragma unroll
    for (int col = tid; col < TileColumns; col += NumThreads) {
      ScaleComputeT super_scale =
          super_scale_ptr ? static_cast<ScaleComputeT>(super_scale_ptr[col])
                          : static_cast<ScaleComputeT>(1);

#pragma unroll
      for (int group_id = 0; group_id < TileRows / 64; ++group_id) {
        int local_scale_offset = (group_id / 2) * in_stride + col;
        int local_scale_shift =
            ((block_start_row / kGroupSize + group_id) % 2) * 4;
        int32_t local_scale =
            static_cast<int32_t>(local_scale_ptr[local_scale_offset]);
        int32_t shifted_local_scale =
            (local_scale >> local_scale_shift) & kLocalScaleMask;
        ScaleComputeT scale =
            static_cast<ScaleComputeT>(shifted_local_scale) * super_scale;

        ScaleComputeT code_scale =
            static_cast<ScaleComputeT>(code_scale_ptr[col]);
        ScaleComputeT code_zp = static_cast<ScaleComputeT>(code_zp_ptr[col]);

#pragma unroll
        for (int zipped_row = 0; zipped_row < 16; ++zipped_row) {
          int zipped_offset = zipped_row * in_stride + col;
          ScaleComputeT zipped_value =
              static_cast<ScaleComputeT>(in_ptr[zipped_offset]);
          int32_t decode_value =
              static_cast<int32_t>(floor(zipped_value * code_scale + code_zp +
                                         static_cast<ScaleComputeT>(0.5)));

          int row = group_id * 64 + zipped_row * 4;

#pragma unroll
          for (int shift_bit_id = 0; shift_bit_id < 4; ++shift_bit_id) {
            int32_t shift_bit = shift_bits[shift_bit_id];
            int32_t shifted_value = (decode_value >> shift_bit) & kWeightMask;

            ScaleComputeT value =
                static_cast<ScaleComputeT>(shifted_value - kBZP);
            out_ptr[(row + shift_bit_id) * TileColumns + col] =
                static_cast<T>(scale * value);
          }
        }
      }
    }
    __syncthreads();
  }
};

#endif

template <typename T, int TileRows, int TileColumns, int NumThreads>
__global__ void Wint25UnzipKernel(const uint16_t *zipped_weight_ptr,
                                  const T *super_scale_ptr, T *weight_ptr,
                                  const int64_t batch, const int64_t num_rows,
                                  const int64_t num_columns) {
  __shared__ T smem[TileRows * TileColumns];

  int64_t block_start_column = blockIdx.x * TileColumns;

  int64_t block_start_row = blockIdx.z * num_rows + blockIdx.y * TileRows;
  int64_t block_start_zipped_row = block_start_row * 10 / 64;

  int64_t block_zipped_offset =
      block_start_zipped_row * num_columns + block_start_column;
  const uint16_t *block_zipped_weight_ptr =
      zipped_weight_ptr + block_zipped_offset;

  const T *block_super_scale_ptr =
      super_scale_ptr + blockIdx.z * num_columns + block_start_column;

  // unzip to shared memory
  UnzipAndDequantFunctor<T, WintQuantMethod::kWeightOnlyInt25, TileRows,
                         TileColumns, NumThreads>
      unzip_functor;
  unzip_functor(block_zipped_weight_ptr, block_super_scale_ptr, smem,
                num_columns);

  // write back to global memory
  for (int row = 0; row < TileRows; ++row) {
    for (int col = 0; col < TileColumns; ++col) {
      int64_t global_row = block_start_row + row;
      int64_t global_col = block_start_column + col;
      weight_ptr[global_row * num_columns + global_col] =
          smem[row * TileColumns + col];
    }
  }
}

template <typename T, int64_t TileRows, int64_t TileColumns, int NumThreads>
__global__ void
Wint2UnzipKernel(const uint8_t *zipped_weight_ptr,
                 const uint8_t *local_scale_ptr, const float *code_scale_ptr,
                 const float *code_zp_ptr, const T *super_scale_ptr,
                 T *weight_ptr, const int64_t batch, const int64_t num_rows,
                 const int64_t num_columns) {
  __shared__ T smem[TileRows * TileColumns];

  int64_t block_start_column = blockIdx.x * TileColumns;

  int64_t block_start_row = blockIdx.z * num_rows + blockIdx.y * TileRows;

  int64_t block_start_zipped_row = block_start_row / 4;
  int64_t block_zipped_offset =
      block_start_zipped_row * num_columns + block_start_column;
  const uint8_t *block_zipped_weight_ptr =
      zipped_weight_ptr + block_zipped_offset;

  // local_scale is uint4
  int64_t block_start_local_scale_row = block_start_row / (64 * 2);
  int64_t block_local_scale_offset =
      block_start_local_scale_row * num_columns + block_start_column;
  const uint8_t *block_local_scale_ptr =
      local_scale_ptr + block_local_scale_offset;

  const float *block_code_scale_ptr =
      code_scale_ptr + blockIdx.z * num_columns + block_start_column;
  const float *block_code_zp_ptr =
      code_zp_ptr + blockIdx.z * num_columns + block_start_column;
  const T *block_super_scale_ptr =
      super_scale_ptr
          ? super_scale_ptr + blockIdx.z * num_columns + block_start_column
          : nullptr;

  // unzip to shared memory
  UnzipAndDequantFunctor<T, WintQuantMethod::kWeightOnlyInt2, TileRows,
                         TileColumns, NumThreads>
      unzip_functor;
  unzip_functor(block_zipped_weight_ptr, block_local_scale_ptr,
                block_code_scale_ptr, block_code_zp_ptr, block_super_scale_ptr,
                smem, block_start_row, num_columns);

  // write back to global memory
  for (int row = 0; row < TileRows; ++row) {
    for (int col = 0; col < TileColumns; ++col) {
      int64_t global_row = block_start_row + row;
      int64_t global_col = block_start_column + col;
      weight_ptr[global_row * num_columns + global_col] =
          smem[row * TileColumns + col];
    }
  }
}

template <typename T>
void Wint25UnzipKernelLauncher(const uint16_t *zipped_weight,
                               const T *supper_scale, T *weight,
                               const int64_t batch, const int64_t num_rows,
                               const int64_t num_columns) {
  constexpr int kTileRows = 64;
  constexpr int kTileColumns = 128;

  constexpr int kNumThreads = 128;
  const int block_dim_x = (num_columns + kTileColumns - 1) / kTileColumns;
  const int block_dim_y = (num_rows + kTileRows - 1) / kTileRows;

  dim3 block_dim(kNumThreads, 1, 1);
  dim3 grid_dim(block_dim_x, block_dim_y, batch);

  Wint25UnzipKernel<T, kTileRows, kTileColumns, kNumThreads>
      <<<grid_dim, block_dim>>>(zipped_weight, supper_scale, weight, batch,
                                num_rows, num_columns);
}

template <typename T>
void Wint2UnzipKernelLauncher(const uint8_t *zipped_weight,
                              const uint8_t *local_scale,
                              const float *code_scale, const float *code_zp,
                              const T *supper_scale, T *weight,
                              const int64_t batch, const int64_t num_rows,
                              const int64_t num_columns) {
  constexpr int kTileRows = 64;
  constexpr int kTileColumns = 128;

  constexpr int kNumThreads = 128;
  const int block_dim_x = (num_columns + kTileColumns - 1) / kTileColumns;
  const int block_dim_y = (num_rows + kTileRows - 1) / kTileRows;

  dim3 block_dim(kNumThreads, 1, 1);
  dim3 grid_dim(block_dim_x, block_dim_y, batch);

  Wint2UnzipKernel<T, kTileRows, kTileColumns, kNumThreads>
      <<<grid_dim, block_dim>>>(zipped_weight, local_scale, code_scale, code_zp,
                                supper_scale, weight, batch, num_rows,
                                num_columns);
}

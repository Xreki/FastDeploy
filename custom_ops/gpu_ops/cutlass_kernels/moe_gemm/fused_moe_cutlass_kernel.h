/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *notice, this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 *contributors may be used to endorse or promote products derived from this
 *software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief
*/

#pragma once

#include "cutlass/complex.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"

#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/trace.h"

#include "cutlass_extensions/gemm/kernel/gemm_moe_problem_visitor.h"
#include "cutlass_kernels/moe_gemm/tile_dequanter.h"
#include "cutlass_kernels/moe_gemm/wint_type_traits.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/tile_interleaved_layout.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
// This section exists to that we can use the same kernel code for regular gemm
// and dequantizing gemms. It will dispatch to the dequantizing gemm if the Mma
// type has an Iterator for scales in global.
template <typename...>
using void_t = void;

template <typename Mma, typename = void>
struct use_dq_gemm : platform::false_type {};

template <typename Mma>
struct use_dq_gemm<Mma, void_t<typename Mma::IteratorScale>>
    : platform::true_type {};

template <typename KernelArch>
constexpr bool NeedCompile() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) && (__CUDA_ARCH__ < 750)
  return platform::is_same<KernelArch, arch::Sm70>::value;
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750) && (__CUDA_ARCH__ < 800)
  return platform::is_same<KernelArch, arch::Sm75>::value;
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDA_ARCH__ < 900)
  return platform::is_same<KernelArch, arch::Sm80>::value;
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 910)
  return platform::is_same<KernelArch, arch::Sm80>::value;
#else
  CUTLASS_NOT_IMPLEMENTED();
  return false;
#endif
}

// SFINAE overload for dequantizing gemm
template <
    typename Mma,
    typename TileDequanterB,
    WintQuantMethod QuantMethod,
    typename ElementScale,
    typename platform::enable_if<use_dq_gemm<Mma>::value, bool>::type = true>
CUTLASS_DEVICE static void run_mma(Mma mma,
                                   int gemm_k_iterations,
                                   typename Mma::FragmentC& accum,  // NOLINT
                                   typename Mma::IteratorA iterator_A,
                                   typename Mma::IteratorB iterator_B,
                                   TileDequanterB tile_dequanter_B,
                                   typename Mma::FragmentC const& src_accum,
                                   ElementScale* weight_scale_ptr,
                                   MatrixCoord scale_extent,
                                   const int thread_idx,
                                   MatrixCoord tb_offset_scale) {
  //CUTLASS_TRACE_DEVICE(" iterator_Scale, weight_scale_ptr=%p", weight_scale_ptr);                                  
  typename Mma::IteratorScale iterator_scale(
      Mma::IteratorScale::Layout(scale_extent.column()),
      weight_scale_ptr,
      scale_extent,
      thread_idx,
      tb_offset_scale);

  mma(gemm_k_iterations,
      accum,
      iterator_A,
      iterator_B,
      iterator_scale,
      src_accum);
}

// SFINAE overload for normal gemm. This completely ignores the scale parameters
template <
    typename Mma,
    typename TileDequanterB,
    WintQuantMethod QuantMethod,
    typename ElementScale,
    typename platform::enable_if<!use_dq_gemm<Mma>::value, bool>::type = true>
CUTLASS_DEVICE static void run_mma(Mma mma,
                                   int gemm_k_iterations,
                                   typename Mma::FragmentC& accum,  // NOLINT
                                   typename Mma::IteratorA iterator_A,
                                   typename Mma::IteratorB iterator_B,
                                   TileDequanterB tile_dequanter_B,
                                   typename Mma::FragmentC const& src_accum,
                                   ElementScale* weight_scale_ptr,
                                   MatrixCoord scale_extent,
                                   const int thread_idx,
                                   MatrixCoord tb_offset_scale) {
  //CUTLASS_TRACE_DEVICE(" use_dq_gemm is false");
  //if constexpr (QuantMethod == WintQuantMethod::kWeightOnlyInt25) {
    mma(gemm_k_iterations, accum, iterator_A, iterator_B, tile_dequanter_B, src_accum);
  //} else {
  //  mma(gemm_k_iterations, accum, iterator_A, iterator_B, src_accum);
  //}
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Mma_,  ///! Threadblock-scoped matrix multiply-accumulate
          typename Epilogue_,            ///! Epilogue
          typename ThreadblockSwizzle_,  ///! Threadblock swizzling function
          typename KernelArch,  ///! The Architecture this kernel is compiled
                                /// for. Used since SIMT kernels lose top-level
                                /// arch.
          GroupScheduleMode GroupScheduleMode_  ///! Type of scheduling to //
                                                /// NOLINT perform
          >
struct MoeFCGemm {
 public:
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static GroupScheduleMode const kGroupScheduleMode = GroupScheduleMode_;
  static bool const kTransposed = false;

  // Optional transpose
  using MapArguments =
      kernel::detail::MapArguments<typename Mma::IteratorA::Element,
                                   typename Mma::IteratorA::Layout,
                                   Mma::kTransformA,
                                   Mma::IteratorA::AccessType::kElements,
                                   typename Mma::IteratorB::Element,
                                   typename Mma::IteratorB::Layout,
                                   Mma::kTransformB,
                                   Mma::IteratorB::AccessType::kElements,
                                   typename Mma::LayoutC,
                                   kTransposed>;

  // Public-facing type definitions related to operand element type, layout, and
  // complex conjugate operation. Must interact with the 'kTransposed' notion.
  static_assert(!kTransposed, "Transpose problem not supported");
  using ElementA = typename MapArguments::ElementA;
  using LayoutA = typename MapArguments::LayoutA;
  using ElementB = typename MapArguments::ElementB;
  using LayoutB = typename MapArguments::LayoutB;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename MapArguments::LayoutC;
  using ElementScale = ElementC;

  static ComplexTransform const kTransformA = MapArguments::kTransformA;
  static ComplexTransform const kTransformB = MapArguments::kTransformB;

  // Type definitions about the mainloop.
  using Operator = typename Mma::Operator;
  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = MapArguments::kAlignmentA;
  static int const kAlignmentB = MapArguments::kAlignmentB;
  static int const kAlignmentC =
      Epilogue::OutputTileIterator::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using ProblemVisitor = GemmMoeProblemVisitor<ThreadblockShape,
                                               kGroupScheduleMode,
                                               kThreadCount,
                                               kThreadCount,
                                               kTransposed>;

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {
    //
    // Data members
    //

    int problem_count;
    int threadblock_count;

    typename EpilogueOutputOp::Params output_op;

    ElementA* ptr_A;
    ElementB* ptr_B;
    ElementScale* weight_scales;
    ElementC* ptr_C;
    ElementC* ptr_D;

    int64_t* total_rows_before_expert;
    int64_t total_rows;
    int64_t gemm_n;
    int64_t gemm_k;

    WintQuantMethod quant_method;

    // Extra arguments for wint2.0
    uint8_t* local_scale;
    float* code_scale;
    float* code_zp;

    // Only used by device-level operator
    GemmCoord* host_problem_sizes;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments()
        : problem_count(0),
          threadblock_count(0),
          ptr_A(nullptr),
          ptr_B(nullptr),
          weight_scales(nullptr),
          ptr_C(nullptr),
          ptr_D(nullptr),
          total_rows_before_expert(nullptr),
          total_rows(-1),
          gemm_n(0),
          gemm_k(0),
          quant_method(WintQuantMethod::kNone),
          local_scale(nullptr),
          code_scale(nullptr),
          code_zp(nullptr),
          host_problem_sizes(nullptr) {}

    /// Ctor
    CUTLASS_HOST_DEVICE
    Arguments(int problem_count,
              int threadblock_count,
              typename EpilogueOutputOp::Params output_op,
              const ElementA* ptr_A,
              const ElementB* ptr_B,
              const ElementScale* weight_scales,
              const ElementC* ptr_C,
              ElementC* ptr_D,
              int64_t* total_rows_before_expert,
              int64_t total_rows,
              int64_t gemm_n,
              int64_t gemm_k,
              WintQuantMethod quant_method,
              const uint8_t* local_scale,
              const float* code_scale,
              const float* code_zp,
              GemmCoord* host_problem_sizes = nullptr)
        : problem_count(problem_count),
          threadblock_count(threadblock_count),
          output_op(output_op),
          ptr_A(const_cast<ElementA*>(ptr_A)),
          ptr_B(const_cast<ElementB*>(ptr_B)),
          weight_scales(const_cast<ElementScale*>(weight_scales)),
          ptr_C(const_cast<ElementC*>(ptr_C)),
          ptr_D(ptr_D),
          total_rows_before_expert(total_rows_before_expert),
          total_rows(total_rows),
          gemm_n(gemm_n),
          gemm_k(gemm_k),
          quant_method(quant_method),
          local_scale(const_cast<uint8_t*>(local_scale)),
          code_scale(const_cast<float*>(code_scale)),
          code_zp(const_cast<float*>(code_zp)),
          host_problem_sizes(nullptr) {
      if (quant_method != WintQuantMethod::kNone || platform::is_same<uint8_t, ElementB>::value ||
          platform::is_same<uint4b_t, ElementB>::value) {
        assert(weight_scales);
      }

      CUTLASS_TRACE_HOST("[Arguments] problem_count: " << problem_count << ", threadblock_count: " << threadblock_count << ", gemm_n: " << gemm_n << ", gemm_k: " << gemm_k);
      CUTLASS_TRACE_HOST("[Arguments] ptr_A: " << static_cast<void const*>(ptr_A));
      CUTLASS_TRACE_HOST("[Arguments] ptr_B: " << static_cast<void const*>(ptr_B));
      CUTLASS_TRACE_HOST("[Arguments] ptr_C: " << static_cast<void const*>(ptr_C));
      CUTLASS_TRACE_HOST("[Arguments] ptr_D: " << static_cast<void*>(ptr_D));
      CUTLASS_TRACE_HOST("[Arguments] weight_scales: " << static_cast<void const*>(weight_scales));
      CUTLASS_TRACE_HOST("[Arguments] total_rows_before_expert: " << static_cast<void*>(total_rows_before_expert));
      CUTLASS_TRACE_HOST("[Arguments] local_scale: " << static_cast<void const*>(local_scale));
      CUTLASS_TRACE_HOST("[Arguments] code_scale: " << static_cast<void const*>(code_scale));
      CUTLASS_TRACE_HOST("[Arguments] code_zp: " << static_cast<void const*>(code_zp));
      CUTLASS_TRACE_HOST("[Arguments] quant_method: " << static_cast<int>(quant_method));
      CUTLASS_TRACE_HOST("[Arguments] LayoutA: " << GetCutlassLayoutString<LayoutA>());
      CUTLASS_TRACE_HOST("[Arguments] LayoutB: " << GetCutlassLayoutString<LayoutB>());
      CUTLASS_TRACE_HOST("[Arguments] LayoutC: " << GetCutlassLayoutString<LayoutC>());
      CUTLASS_TRACE_HOST("[Arguments] Mma::IteratorA::AccessType::kElements:" << Mma::IteratorA::AccessType::kElements);
      CUTLASS_TRACE_HOST("[Arguments] Mma::IteratorB::AccessType::kElements:" << Mma::IteratorB::AccessType::kElements);
      CUTLASS_TRACE_HOST("[Arguments] SharedStorage Information:");
      CUTLASS_TRACE_HOST(" - ProblemVisitor::SharedStorage: " << sizeof(typename ProblemVisitor::SharedStorage) << " bytes");
      CUTLASS_TRACE_HOST(" - Mma::SharedStorage: " << sizeof(typename Mma::SharedStorage) << " bytes");
      CUTLASS_TRACE_HOST(" - Epilogue::SharedStorage: " << sizeof(typename Epilogue::SharedStorage) << " bytes");
    }
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {
    typename ProblemVisitor::Params problem_visitor;
    int threadblock_count;

    typename EpilogueOutputOp::Params output_op;

    ElementA* ptr_A;
    ElementB* ptr_B;
    ElementScale* weight_scales;
    ElementC* ptr_C;
    ElementC* ptr_D;

    WintQuantMethod quant_method;

    // Extra arguments for wint2.0
    uint8_t* local_scale;
    float* code_scale;
    float* code_zp;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params()
        : ptr_A(nullptr),
          ptr_B(nullptr),
          weight_scales(nullptr),
          ptr_C(nullptr),
          ptr_D(nullptr),
          quant_method(WintQuantMethod::kNone),
          local_scale(nullptr),
          code_scale(nullptr),
          code_zp(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(Arguments const& args,
           void* workspace = nullptr,
           int tile_count = 0)  // NOLINT
        : problem_visitor(args.total_rows_before_expert,
                          args.total_rows,
                          args.gemm_n,
                          args.gemm_k,
                          args.problem_count,
                          workspace,
                          tile_count),
          threadblock_count(args.threadblock_count),
          output_op(args.output_op),
          ptr_A(args.ptr_A),
          ptr_B(args.ptr_B),
          weight_scales(args.weight_scales),
          ptr_C(args.ptr_C),
          ptr_D(args.ptr_D),
          quant_method(args.quant_method),
          local_scale(args.local_scale),
          code_scale(args.code_scale),
          code_zp(args.code_zp) {}

    CUTLASS_HOST_DEVICE
    void update(Arguments const& args,
                void* workspace = nullptr,
                int tile_count = 0) {
      problem_visitor =
          typename ProblemVisitor::Params(args.total_rows_before_expert,
                                          args.total_rows,
                                          args.gemm_n,
                                          args.gemm_k,
                                          args.problem_count,
                                          workspace,
                                          tile_count);
      threadblock_count = args.threadblock_count;
      output_op = args.output_op;
      ptr_A = args.ptr_A;
      ptr_B = args.ptr_B;
      weight_scales = args.weight_scales;
      ptr_C = args.ptr_C;
      ptr_D = args.ptr_D;
      quant_method = args.quant_method;
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename ProblemVisitor::SharedStorage problem_visitor;
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

 public:
  //
  // Methods
  //

  CUTLASS_DEVICE
  MoeFCGemm() {}

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const& problem_size) {
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const& args) {
    if (args.quant_method != WintQuantMethod::kNone || platform::is_same<uint8_t, ElementB>::value ||
        platform::is_same<uint4b_t, ElementB>::value) {
      if (args.weight_scales == nullptr) {
        CUTLASS_TRACE_HOST(
            "MoeFCGemm::can_implement() - weight scales are required for "
            "uint8_t and uint4b_t");
        return Status::kInvalid;
      }
    } else if (args.weight_scales != nullptr) {
      CUTLASS_TRACE_HOST(
          "MoeFCGemm::can_implement() - weight scales are ignored for all "
          "types except uint8_t and uint4b_t");
      return Status::kInvalid;
    }
    return Status::kSuccess;
  }

  static size_t get_extra_workspace_size(
      Arguments const& args, cutlass::gemm::GemmCoord const& grid_tiled_shape) {
    return 0;
  }

  // The dummy template parameter is not used and exists so that we can compile
  // this code using a standard earlier than C++17. Prior to C++17, fully
  // specialized templates HAD to exists in a namespace
  template <WintQuantMethod QuantMethod, bool B, typename dummy = void>
  struct KernelRunner {
    CUTLASS_DEVICE
    static void run_kernel(Params const& params,
                           SharedStorage& shared_storage) {  // NOLINT
      CUTLASS_NOT_IMPLEMENTED();
    }
  };

  template <WintQuantMethod QuantMethod, typename dummy>
  struct KernelRunner<QuantMethod, true, dummy> {
    using WeightQuantTraits = WintQuantTraits<ElementA, QuantMethod>;
    using QuantArguments = typename WeightQuantTraits::Arguments;

    CUTLASS_DEVICE
    static QuantArguments get_quant_args(Params const& params, int32_t problem_idx, const int64_t gemm_k, const int64_t gemm_n) {
      QuantArguments quant_args;
      if constexpr (QuantMethod == WintQuantMethod::kWeightOnlyInt2) {
        quant_args.local_scale_ptr = params.local_scale + problem_idx * gemm_k * gemm_n / 128;
        quant_args.code_scale_ptr = params.code_scale + problem_idx * gemm_n;
        quant_args.code_zp_ptr = params.code_zp + problem_idx * gemm_n;
      }
      return quant_args;
    }

    CUTLASS_DEVICE
    static void run_kernel(Params const& params,
                           SharedStorage& shared_storage) {  // NOLINT
      //
      // These types shadow the type-level definitions and support the ability
      // to implement a 'transposed' GEMM that computes the transposed problems.
      //
      //CUTLASS_TRACE_DEVICE(" MoeFCGemm::KernelRunner::run_kernel()");

      using ElementA = typename Mma::IteratorA::Element;
      using LayoutA = typename Mma::IteratorA::Layout;
      using ElementB = typename Mma::IteratorB::Element;
      using LayoutB = typename Mma::IteratorB::Layout;
      using ElementC = typename Epilogue::OutputTileIterator::Element;
      using LayoutC = typename Epilogue::OutputTileIterator::Layout;
      using QuantElementB = typename WeightQuantTraits::WeightType;

      static constexpr int kInterleave =
          Mma::IteratorB::Shape::kRow / Mma::Shape::kK;
      static_assert(
          platform::is_same<LayoutB, layout::RowMajor>::value &&
                  kInterleave == 1 ||
              platform::is_same<LayoutB, layout::ColumnMajor>::value &&
                  kInterleave >= 1,
          "B must be row major/col major OR col major interleaved.");

      // LayoutB should be RowMajor
      using TileDequanterB = TileDequanter<ElementA, ElementScale, ThreadblockShape::kK, ThreadblockShape::kN, kThreadCount, QuantMethod>;
      __shared__ typename TileDequanterB::SharedStorage dequant_storage_B;

      //
      // Problem visitor.
      //
      ProblemVisitor problem_visitor(
          params.problem_visitor, shared_storage.problem_visitor, blockIdx.x);

      const int64_t gemm_k = params.problem_visitor.gemm_k;
      const int64_t gemm_n = params.problem_visitor.gemm_n;
      // wint2.5 and wint2.0 is quantized and packed along k dimension with group_size 64.
      const int64_t quant_gemm_k = WintQuantTraits<ElementA, QuantMethod>::CaclPackedDim(gemm_k);
      int64_t bytes_per_expert_matrix = (quant_gemm_k * gemm_n / 8) * cutlass::sizeof_bits<QuantElementB>::value;

      CUTLASS_TRACE_DEVICE(" gemm_k: %ld, quant_gemm_k: %ld, gemm_n: %ld, bytes_per_expert_matrix: %ld, kInterleave: %d", gemm_k, quant_gemm_k, gemm_n, bytes_per_expert_matrix, kInterleave);

      // Outer 'persistent' loop to iterate over tiles
      while (problem_visitor.next_tile()) {
        GemmCoord problem_size = problem_visitor.problem_size();
        int32_t problem_idx = problem_visitor.problem_index();
        int32_t cta_idx = int32_t(problem_visitor.threadblock_idx());
        //CUTLASS_TRACE_DEVICE(" problem_idx: %d, cta_idx: %d, problem_size: {%d, %d, %d}",
        //    problem_idx, cta_idx, static_cast<int>(problem_size.m()), static_cast<int>(problem_size.n()), static_cast<int>(problem_size.k()));

        //if (problem_idx > 5) {
        //  break;
        //}

        GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

        // threadblock_offset of C
        cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma::Shape::kM,  // NOLINT
            int(cta_idx % grid_shape.n()) * Mma::Shape::kN,  // NOLINT
            0);

        // begin address offset for weight_scale.
        ElementScale* weight_scale_ptr =
            params.weight_scales ? params.weight_scales + problem_idx * problem_size.n() : nullptr;
        // the begin threadblock_offset of scale, which holds the same column id with C, but with no row id
        cutlass::MatrixCoord tb_offset_scale{0, threadblock_offset.n()};

        // Load element pointers. Exchange pointers and strides if working on
        // the transpose
        int64_t rows_to_jump = 0;

        if (params.problem_visitor.total_rows < 0) {
          rows_to_jump = problem_idx == 0 ? 0 : params.problem_visitor.last_row_for_problem[problem_idx - 1];
        } else {
          rows_to_jump = problem_idx * (params.problem_visitor.total_rows / params.problem_visitor.problem_count);
        }

        // begin address offset for A for current tile
        ElementA* ptr_A =
            reinterpret_cast<ElementA*>(params.ptr_A) + rows_to_jump * gemm_k;
        typename LayoutA::LongIndex ldm_A = gemm_k;

        // Compute initial location in logical coordinates
        // the begin threadblock_offset of A, which holds the same row id with C
        cutlass::MatrixCoord tb_offset_A{
            threadblock_offset.m(),
            0,
        };

        // begin address offset for B for current problem_idx, totally num_experts problems
        char* byte_ptr_B = ((char*)params.ptr_B) +                 // NOLINT
                           problem_idx * bytes_per_expert_matrix;  // NOLINT
      
        typename LayoutB::LongIndex ldm_B =
            platform::is_same<layout::RowMajor, LayoutB>::value
                ? gemm_n
                : gemm_k * kInterleave;
        typename LayoutB::LongIndex ldm_B_shared = TileDequanterB::kColumns;

        // the begin threadblock_offset of B, which holds the same column id with C
        cutlass::MatrixCoord tb_offset_B{0,
                                         threadblock_offset.n() / kInterleave};

        cutlass::MatrixCoord extent_B{problem_size.k() * kInterleave, problem_size.n() / kInterleave};
        cutlass::MatrixCoord extent_B_shared{TileDequanterB::kRows, TileDequanterB::kColumns};

        QuantArguments quant_args = get_quant_args(params, problem_idx, gemm_k, gemm_n);
        TileDequanterB tile_dequanter_B(dequant_storage_B,
                                        byte_ptr_B,
                                        ldm_B,
                                        extent_B,
                                        tb_offset_B,
                                        weight_scale_ptr,
                                        tb_offset_scale,
                                        quant_args);
        ElementB* ptr_B = tile_dequanter_B.GetOutPtr();

        // Compute position within threadblock
        int thread_idx = threadIdx.x;

        // Construct iterators to A and B operands
        //CUTLASS_TRACE_DEVICE(" iterator_A, ptr_A=%p, ldm_A=%ld", ptr_A, ldm_A);
        typename Mma::IteratorA iterator_A(LayoutA(ldm_A),
                                           ptr_A,
                                           {problem_size.m(), problem_size.k()},
                                           thread_idx,
                                           tb_offset_A);

        //CUTLASS_TRACE_DEVICE(" iterator_B, ptr_B=%p, ldm_B=%ld", ptr_B, ldm_B);
        typename Mma::IteratorB iterator_B(
            LayoutB(TileDequanterB::kUseSharedMemory ? ldm_B_shared : ldm_B),
            ptr_B,
            TileDequanterB::kUseSharedMemory ? extent_B_shared : extent_B,
            thread_idx,
            TileDequanterB::kUseSharedMemory ? cutlass::make_Coord(0, 0) : tb_offset_B);

        typename Mma::FragmentC accumulators;

        accumulators.clear();

        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

        int lane_idx = threadIdx.x % 32;

#if 1
        //
        // Matrix multiply phase
        //

        // Construct thread-scoped matrix multiply
        auto CreateMMA = [&]() {
          if constexpr (use_dq_gemm<Mma>::value)
            return Mma(
                shared_storage.main_loop, -1, thread_idx, warp_idx, lane_idx);
          else
            return Mma(
                shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
        };
        Mma mma = CreateMMA();

        // Compute threadblock-scoped matrix multiply-add
        int gemm_k_iterations =
            (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
        //CUTLASS_TRACE_DEVICE(" gemm_k: %d, gemm_k_iterations: %d", problem_size.k(), gemm_k_iterations);

        // Wait for all threads to finish their epilogue phases from the
        // previous tile.
        __syncthreads();

        // Compute threadblock-scoped matrix multiply-add
        run_mma<Mma, TileDequanterB, QuantMethod>(mma,
                     gemm_k_iterations,
                     accumulators,
                     iterator_A,
                     iterator_B,
                     tile_dequanter_B,
                     accumulators,
                     weight_scale_ptr,
                     {1, problem_size.n()},
                     thread_idx,
                     tb_offset_scale);

        //
        // Epilogue
        //

        EpilogueOutputOp output_op(params.output_op);

        ElementC* ptr_C =
            params.ptr_C ? reinterpret_cast<ElementC*>(params.ptr_C) + problem_idx * gemm_n : nullptr;
        ElementC* ptr_D =
            reinterpret_cast<ElementC*>(params.ptr_D) + rows_to_jump * gemm_n;

        LayoutC layout_C(0);
        LayoutC layout_D(gemm_n);

        typename Epilogue::OutputTileIterator::Params params_C(layout_C);
        typename Epilogue::OutputTileIterator::Params params_D(layout_D);

        // Tile iterator loading from source tensor.
        //CUTLASS_TRACE_DEVICE(" iterator_C, ptr_C=%p", ptr_C);
        typename Epilogue::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn());

        // Tile iterator writing to destination tensor.
        //CUTLASS_TRACE_DEVICE(" iterator_D, ptr_D=%p", ptr_D);
        typename Epilogue::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn());

        Epilogue epilogue(
            shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

        // Execute the epilogue operator to update the destination tensor.
        epilogue(output_op, iterator_D, accumulators, iterator_C);
#endif

        // Next tile
        problem_visitor.advance(gridDim.x);
      }
    }
  };

  /*
    To improve compilation speed, we do not compile the device operator if the
    CUDA_ARCH does not correspond to the ArchTag of the cutlass kernel operator.
  */
  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const& params,
                  SharedStorage& shared_storage) {  // NOLINT
    static constexpr bool kCompileNeeded = NeedCompile<KernelArch>();
    if constexpr (std::is_same<ElementB, cutlass::bfloat16_t>::value || std::is_same<ElementB, cutlass::half_t>::value) {
      if (params.quant_method == WintQuantMethod::kWeightOnlyInt25) {
        KernelRunner<WintQuantMethod::kWeightOnlyInt25, kCompileNeeded>::run_kernel(params, shared_storage);
      } else if (params.quant_method == WintQuantMethod::kWeightOnlyInt2) {
        KernelRunner<WintQuantMethod::kWeightOnlyInt2, kCompileNeeded>::run_kernel(params, shared_storage);
      } else {
        KernelRunner<WintQuantMethod::kNone, kCompileNeeded>::run_kernel(params, shared_storage);
      }
    } else if constexpr (std::is_same<ElementB, uint8_t>::value) {
      KernelRunner<WintQuantMethod::kWeightOnlyInt8, kCompileNeeded>::run_kernel(params, shared_storage);
    } else if constexpr (std::is_same<ElementB, cutlass::uint4b_t>::value) {
      KernelRunner<WintQuantMethod::kWeightOnlyInt4, kCompileNeeded>::run_kernel(params, shared_storage);
    } else {
      CUTLASS_NOT_IMPLEMENTED();
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

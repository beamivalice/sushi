// Sushi Cooler NAX floating-point quantized kernels — instantiates MLX's
// MIT-licensed fp_quantized_nax.h (Copyright (c) 2025-2026 Apple Inc., see
// NOTICE) for mxfp4 and mxfp8 products in src/tensor/quantized.zig.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/fp_quantized_nax.h"

#define instantiate_fp_nax_aligned(mode, name, type, bm, bn, bk, wm, wn, aligned, group_size, bits) \
  instantiate_kernel(                                                                               \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_alN_" #aligned, \
      fp_##name, type, group_size, bits, aligned, bm, bk, bn, wm, wn)

#define instantiate_fp_nax_aligned_batched(mode, name, type, bm, bn, bk, wm, wn, aligned, batched, group_size, bits) \
  instantiate_kernel(                                                                                                \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_alN_" #aligned "_batch_" #batched, \
      fp_##name, type, group_size, bits, aligned, batched, bm, bk, bn, wm, wn)

#define instantiate_fp_nax_rhs_nt(mode, type, bm, bn, bk, wm, wn, group_size, bits)                 \
  instantiate_kernel(                                                                               \
      #mode "_gather_qmm_rhs_nax_nt_" #type "_gs_" #group_size "_b_" #bits "_bm_" #bm "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn, \
      fp_gather_qmm_rhs_nax, type, group_size, bits, bm, bn, bk, wm, wn, true)

#define instantiate_mx_nax_mode(type, mode, group_size, bits)                                                        \
  instantiate_fp_nax_aligned(mode, gather_qmm_t_nax, type, 64, 64, 64, 2, 2, true, group_size, bits)                 \
  instantiate_fp_nax_aligned(mode, gather_qmm_t_nax, type, 64, 64, 64, 2, 2, false, group_size, bits)                \
  instantiate_fp_nax_aligned_batched(mode, qmm_t_nax, type, 64, 64, 64, 2, 2, true, 1, group_size, bits)             \
  instantiate_fp_nax_aligned_batched(mode, qmm_t_nax, type, 64, 64, 64, 2, 2, true, 0, group_size, bits)             \
  instantiate_fp_nax_aligned_batched(mode, qmm_t_nax, type, 64, 64, 64, 2, 2, false, 1, group_size, bits)            \
  instantiate_fp_nax_aligned_batched(mode, qmm_t_nax, type, 64, 64, 64, 2, 2, false, 0, group_size, bits)            \
  instantiate_fp_nax_aligned_batched(mode, qmm_t_nax, type, 32, 64, 64, 2, 2, true, 1, group_size, bits)             \
  instantiate_fp_nax_aligned_batched(mode, qmm_t_nax, type, 32, 64, 64, 2, 2, true, 0, group_size, bits)             \
  instantiate_fp_nax_aligned_batched(mode, qmm_t_nax, type, 32, 64, 64, 2, 2, false, 1, group_size, bits)            \
  instantiate_fp_nax_aligned_batched(mode, qmm_t_nax, type, 32, 64, 64, 2, 2, false, 0, group_size, bits)            \
  instantiate_fp_nax_rhs_nt(mode, type, 64, 64, 64, 2, 2, group_size, bits)                                          \
  instantiate_fp_nax_rhs_nt(mode, type, 32, 64, 64, 2, 2, group_size, bits)

#define instantiate_fp_nax_types(type)            \
  instantiate_mx_nax_mode(type, mxfp8, 32, 8)     \
  instantiate_mx_nax_mode(type, mxfp4, 32, 4)

instantiate_fp_nax_types(float)
instantiate_fp_nax_types(bfloat16_t)
instantiate_fp_nax_types(float16_t)
// clang-format on

// Sushi Cooler SIMD floating-point quantized kernels — instantiates MLX's
// MIT-licensed fp_quantized.h (Copyright (c) 2025-2026 Apple Inc., see NOTICE)
// for mxfp4/mxfp8 products and mxfp4/mxfp8/nvfp4 dequantize in
// src/tensor/{stock,quantized}.zig.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/fp_quantized.h"

#define instantiate_fp_single(mode, name, type, group_size, bits) \
  instantiate_kernel(                                             \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits,   \
      fp_##name, type, group_size, bits)

#define instantiate_fp_batched(mode, name, type, batched, group_size, bits)         \
  instantiate_kernel(                                                               \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched,  \
      fp_##name, type, group_size, bits, batched)

#define instantiate_fp_aligned(mode, name, type, aligned, group_size, bits)         \
  instantiate_kernel(                                                               \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned,    \
      fp_##name, type, group_size, bits, aligned)

#define instantiate_fp_aligned_batched(mode, name, type, aligned, batched, group_size, bits)             \
  instantiate_kernel(                                                                                    \
      #mode "_" #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned "_batch_" #batched,      \
      fp_##name, type, group_size, bits, aligned, batched)

#define instantiate_fp_quad(mode, type, D, batched, group_size, bits)                                    \
  instantiate_kernel(                                                                                    \
      #mode "_qmv_quad_" #type "_gs_" #group_size "_b_" #bits "_d_" #D "_batch_" #batched,               \
      fp_qmv_quad, type, group_size, bits, D, batched)

#define instantiate_fp_wide(mode, type, nv, kl, group_size, bits, batched)                               \
  instantiate_kernel(                                                                                    \
      #mode "_qmv_wide_" #type "_gs_" #group_size "_b_" #bits "_nv_" #nv "_kl_" #kl "_batch_" #batched,  \
      fp_qmv_wide, type, group_size, bits, nv, kl, batched)

#define instantiate_fp_gather_rhs_nt(mode, type, group_size, bits)                                       \
  instantiate_kernel(                                                                                    \
      #mode "_gather_qmm_rhs_nt_" #type "_gs_" #group_size "_b_" #bits "_bm_16_bn_32_bk_32_wm_1_wn_2",   \
      fp_gather_qmm_rhs, type, group_size, bits, 16, 32, 32, 1, 2, true)

#define instantiate_fp_dequantize(type, mode, group_size, bits, hgs)                                     \
  instantiate_kernel(                                                                                    \
      #mode "_dequantize_" #type "_gs_" #group_size "_b_" #bits "_hgs_" #hgs,                            \
      fp_dequantize, type, group_size, bits, hgs)

#define instantiate_fp_wide_wrap(mode, type, nv, group_size, bits) \
  instantiate_fp_wide(mode, type, nv, 16, group_size, bits, 0)     \
  instantiate_fp_wide(mode, type, nv, 16, group_size, bits, 1)

#define instantiate_mx_mode(type, mode, group_size, bits)                             \
  instantiate_fp_dequantize(type, mode, group_size, bits, false)                      \
  instantiate_fp_single(mode, gather_qmv_fast, type, group_size, bits)                \
  instantiate_fp_single(mode, gather_qmv, type, group_size, bits)                     \
  instantiate_fp_batched(mode, qmv_fast, type, 1, group_size, bits)                   \
  instantiate_fp_batched(mode, qmv_fast, type, 0, group_size, bits)                   \
  instantiate_fp_batched(mode, qmv, type, 1, group_size, bits)                        \
  instantiate_fp_batched(mode, qmv, type, 0, group_size, bits)                        \
  instantiate_fp_quad(mode, type, 64, 1, group_size, bits)                            \
  instantiate_fp_quad(mode, type, 64, 0, group_size, bits)                            \
  instantiate_fp_quad(mode, type, 128, 1, group_size, bits)                           \
  instantiate_fp_quad(mode, type, 128, 0, group_size, bits)                           \
  instantiate_fp_wide_wrap(mode, type, 2, group_size, bits)                           \
  instantiate_fp_wide_wrap(mode, type, 3, group_size, bits)                           \
  instantiate_fp_wide_wrap(mode, type, 4, group_size, bits)                           \
  instantiate_fp_wide_wrap(mode, type, 5, group_size, bits)                           \
  instantiate_fp_aligned(mode, qmm_t_splitk, type, true, group_size, bits)            \
  instantiate_fp_aligned(mode, qmm_t_splitk, type, false, group_size, bits)           \
  instantiate_fp_aligned(mode, gather_qmm_t, type, true, group_size, bits)            \
  instantiate_fp_aligned(mode, gather_qmm_t, type, false, group_size, bits)           \
  instantiate_fp_aligned_batched(mode, qmm_t, type, true, 1, group_size, bits)        \
  instantiate_fp_aligned_batched(mode, qmm_t, type, true, 0, group_size, bits)        \
  instantiate_fp_aligned_batched(mode, qmm_t, type, false, 1, group_size, bits)       \
  instantiate_fp_aligned_batched(mode, qmm_t, type, false, 0, group_size, bits)       \
  instantiate_fp_gather_rhs_nt(mode, type, group_size, bits)

#define instantiate_fp_types(type)                          \
  instantiate_mx_mode(type, mxfp8, 32, 8)                   \
  instantiate_mx_mode(type, mxfp4, 32, 4)                   \
  instantiate_fp_dequantize(type, nvfp4, 16, 4, false)      \
  instantiate_fp_dequantize(type, nvfp4, 16, 4, true)

instantiate_fp_types(float)
instantiate_fp_types(bfloat16_t)
instantiate_fp_types(float16_t)
// clang-format on

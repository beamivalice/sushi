// Sushi Cooler SIMD affine quantized kernels — instantiates MLX's MIT-licensed
// quantized.h (Copyright (c) 2023-2024 Apple Inc., see NOTICE) for transposed
// affine products, quantize and dequantize in src/tensor/{stock,quantized}.zig.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#define instantiate_quantized(name, type, group_size, bits) \
  instantiate_kernel(                                       \
      #name "_" #type "_gs_" #group_size "_b_" #bits,       \
      name, type, group_size, bits)

#define instantiate_quantized_batched(name, type, group_size, bits, batched) \
  instantiate_kernel(                                                        \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_batch_" #batched,     \
      name, type, group_size, bits, batched)

#define instantiate_quantized_aligned(name, type, group_size, bits, aligned) \
  instantiate_kernel(                                                        \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned,       \
      name, type, group_size, bits, aligned)

#define instantiate_quantized_aligned_batched(name, type, group_size, bits, aligned, batched) \
  instantiate_kernel(                                                                         \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned "_batch_" #batched,     \
      name, type, group_size, bits, aligned, batched)

#define instantiate_quantized_quad(name, type, group_size, bits, D, batched)          \
  instantiate_kernel(                                                                 \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_d_" #D "_batch_" #batched,     \
      name, type, group_size, bits, D, batched)

#define instantiate_quantized_wide(name, type, group_size, bits, vecs_per_tg, k_lanes, batched)               \
  instantiate_kernel(                                                                                         \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_nv_" #vecs_per_tg "_kl_" #k_lanes "_batch_" #batched,  \
      name, type, group_size, bits, vecs_per_tg, k_lanes, batched)

#define instantiate_gather_qmm_rhs_nt(type, group_size, bits)                                         \
  instantiate_kernel(                                                                                 \
      "affine_gather_qmm_rhs_nt_" #type "_gs_" #group_size "_b_" #bits "_bm_16_bn_32_bk_32_wm_1_wn_2", \
      affine_gather_qmm_rhs, type, group_size, bits, 16, 32, 32, 1, 2, true)

#define instantiate_quantized_wide_wrap(type, group_size, bits, nv) \
  instantiate_quantized_wide(affine_qmv_wide, type, group_size, bits, nv, 8, 0) \
  instantiate_quantized_wide(affine_qmv_wide, type, group_size, bits, nv, 8, 1)

#define instantiate_quantized_common(type, group_size, bits)                      \
  instantiate_quantized(affine_quantize, type, group_size, bits)                  \
  instantiate_quantized(affine_dequantize, type, group_size, bits)                \
  instantiate_quantized(affine_gather_qmv_fast, type, group_size, bits)           \
  instantiate_quantized(affine_gather_qmv, type, group_size, bits)                \
  instantiate_quantized_batched(affine_qmv_fast, type, group_size, bits, 1)       \
  instantiate_quantized_batched(affine_qmv_fast, type, group_size, bits, 0)       \
  instantiate_quantized_batched(affine_qmv, type, group_size, bits, 1)            \
  instantiate_quantized_batched(affine_qmv, type, group_size, bits, 0)            \
  instantiate_quantized_aligned(affine_gather_qmm_t, type, group_size, bits, true)  \
  instantiate_quantized_aligned(affine_gather_qmm_t, type, group_size, bits, false) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, true, 1)  \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, true, 0)  \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, false, 1) \
  instantiate_quantized_aligned_batched(affine_qmm_t, type, group_size, bits, false, 0) \
  instantiate_quantized_wide_wrap(type, group_size, bits, 2)                      \
  instantiate_quantized_wide_wrap(type, group_size, bits, 3)                      \
  instantiate_quantized_wide_wrap(type, group_size, bits, 4)                      \
  instantiate_quantized_wide_wrap(type, group_size, bits, 5)                      \
  instantiate_quantized_aligned(affine_qmm_t_splitk, type, group_size, bits, true)  \
  instantiate_quantized_aligned(affine_qmm_t_splitk, type, group_size, bits, false) \
  instantiate_gather_qmm_rhs_nt(type, group_size, bits)

#define instantiate_quantized_quad_all(type, group_size, bits)                 \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 64, 1)   \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 64, 0)   \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 128, 1)  \
  instantiate_quantized_quad(affine_qmv_quad, type, group_size, bits, 128, 0)

#define instantiate_quantized_types(group_size, bits)          \
  instantiate_quantized_common(float, group_size, bits)        \
  instantiate_quantized_common(float16_t, group_size, bits)    \
  instantiate_quantized_common(bfloat16_t, group_size, bits)

#define instantiate_quantized_quad_types(group_size, bits)     \
  instantiate_quantized_quad_all(float, group_size, bits)      \
  instantiate_quantized_quad_all(float16_t, group_size, bits)  \
  instantiate_quantized_quad_all(bfloat16_t, group_size, bits)

#define instantiate_quantized_groups(bits)   \
  instantiate_quantized_types(128, bits)     \
  instantiate_quantized_types(64, bits)      \
  instantiate_quantized_types(32, bits)

#define instantiate_quantized_quad_groups(bits) \
  instantiate_quantized_quad_types(128, bits)   \
  instantiate_quantized_quad_types(64, bits)    \
  instantiate_quantized_quad_types(32, bits)

instantiate_quantized_groups(2)
instantiate_quantized_groups(3)
instantiate_quantized_groups(4)
instantiate_quantized_groups(5)
instantiate_quantized_groups(6)
instantiate_quantized_groups(8)

instantiate_quantized_quad_groups(2)
instantiate_quantized_quad_groups(4)
instantiate_quantized_quad_groups(8)
// clang-format on

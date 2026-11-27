// Sushi Cooler NAX affine quantized kernels — instantiates MLX's MIT-licensed
// quantized_nax.h (Copyright (c) 2023-2025 Apple Inc., see NOTICE) for
// transposed affine products in src/tensor/quantized.zig.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"

#define instantiate_quantized_aligned(name, type, group_size, bits, bm, bn, bk, wm, wn, aligned) \
  instantiate_kernel(                                                                            \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_alN_" #aligned, \
      name, type, group_size, bits, aligned, bm, bk, bn, wm, wn)

#define instantiate_quantized_aligned_batched(name, type, group_size, bits, bm, bn, bk, wm, wn, aligned, batched) \
  instantiate_kernel(                                                                                             \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn "_alN_" #aligned "_batch_" #batched, \
      name, type, group_size, bits, aligned, batched, bm, bk, bn, wm, wn)

#define instantiate_gather_qmm_rhs_nt(type, group_size, bits, bm, bn, bk, wm, wn)                                \
  instantiate_kernel(                                                                                            \
      "affine_gather_qmm_rhs_nax_nt_" #type "_gs_" #group_size "_b_" #bits "_bm_" #bm "_bn_" #bn "_bk_" #bk "_wm_" #wm "_wn_" #wn, \
      affine_gather_qmm_rhs_nax, type, group_size, bits, bm, bn, bk, wm, wn, true)

#define instantiate_quantized_funcs(type, group_size, bits)                                         \
  instantiate_quantized_aligned(affine_gather_qmm_t_nax, type, group_size, bits, 64, 64, 64, 2, 2, true)  \
  instantiate_quantized_aligned(affine_gather_qmm_t_nax, type, group_size, bits, 64, 64, 64, 2, 2, false) \
  instantiate_quantized_aligned_batched(affine_qmm_t_nax, type, group_size, bits, 64, 64, 64, 2, 2, true, 1)  \
  instantiate_quantized_aligned_batched(affine_qmm_t_nax, type, group_size, bits, 64, 64, 64, 2, 2, true, 0)  \
  instantiate_quantized_aligned_batched(affine_qmm_t_nax, type, group_size, bits, 64, 64, 64, 2, 2, false, 1) \
  instantiate_quantized_aligned_batched(affine_qmm_t_nax, type, group_size, bits, 64, 64, 64, 2, 2, false, 0) \
  instantiate_quantized_aligned_batched(affine_qmm_t_nax, type, group_size, bits, 32, 64, 64, 2, 2, true, 1)  \
  instantiate_quantized_aligned_batched(affine_qmm_t_nax, type, group_size, bits, 32, 64, 64, 2, 2, true, 0)  \
  instantiate_quantized_aligned_batched(affine_qmm_t_nax, type, group_size, bits, 32, 64, 64, 2, 2, false, 1) \
  instantiate_quantized_aligned_batched(affine_qmm_t_nax, type, group_size, bits, 32, 64, 64, 2, 2, false, 0) \
  instantiate_gather_qmm_rhs_nt(type, group_size, bits, 64, 64, 64, 2, 2)                           \
  instantiate_gather_qmm_rhs_nt(type, group_size, bits, 32, 64, 64, 2, 2)

#define instantiate_quantized_types(group_size, bits)         \
  instantiate_quantized_funcs(float, group_size, bits)        \
  instantiate_quantized_funcs(float16_t, group_size, bits)    \
  instantiate_quantized_funcs(bfloat16_t, group_size, bits)

#define instantiate_quantized_groups(bits)   \
  instantiate_quantized_types(128, bits)     \
  instantiate_quantized_types(64, bits)      \
  instantiate_quantized_types(32, bits)

instantiate_quantized_groups(2)
instantiate_quantized_groups(3)
instantiate_quantized_groups(4)
instantiate_quantized_groups(5)
instantiate_quantized_groups(6)
instantiate_quantized_groups(8)
// clang-format on

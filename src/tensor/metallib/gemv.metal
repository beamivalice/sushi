// Sushi Cooler GEMV and gathered GEMV kernels — instantiates MLX's MIT-licensed
// gemv.h (Copyright (c) 2023-2024 Apple Inc., see NOTICE) for src/tensor/matmul.zig
// and src/tensor/gather_mm.zig (axpby=0 only, no complex64).

#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/gemv.h"

using namespace metal;

// clang-format off
#define instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, nc) \
  instantiate_kernel(                                                    \
      "gemv_" #name "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn "_tm" #tm    \
      "_tn" #tn "_nc" #nc "_axpby0",                                     \
      gemv, itype, bm, bn, sm, sn, tm, tn, nc, 0)

#define instantiate_gemv(name, itype, bm, bn, sm, sn, tm, tn)      \
  instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, 0)  \
  instantiate_gemv_helper(name, itype, bm, bn, sm, sn, tm, tn, 1)

#define instantiate_gemv_blocks(name, itype)      \
  instantiate_gemv(name, itype, 1, 8, 1, 32, 4, 4) \
  instantiate_gemv(name, itype, 1, 8, 1, 32, 1, 4) \
  instantiate_gemv(name, itype, 1, 1, 8,  4, 4, 4) \
  instantiate_gemv(name, itype, 1, 1, 8,  4, 1, 4) \
  instantiate_gemv(name, itype, 4, 1, 1, 32, 1, 4) \
  instantiate_gemv(name, itype, 4, 1, 1, 32, 4, 4) \
  instantiate_gemv(name, itype, 8, 1, 1, 32, 4, 4)

instantiate_gemv_blocks(float32, float);
instantiate_gemv_blocks(float16, half);
instantiate_gemv_blocks(bfloat16, bfloat16_t);

#define instantiate_gemv_bs_helper(nm, itype, bm, bn, sm, sn, tm, tn) \
  instantiate_kernel(                                                 \
      "gemv_gather_" #nm "_bm" #bm "_bn" #bn "_sm" #sm                \
      "_sn" #sn "_tm" #tm "_tn" #tn,                                  \
      gemv_gather, itype, bm, bn, sm, sn, tm, tn)

#define instantiate_gemv_bs_blocks(name, itype)               \
  instantiate_gemv_bs_helper(name, itype, 4, 1, 1, 32, 1, 4)  \
  instantiate_gemv_bs_helper(name, itype, 4, 1, 1, 32, 4, 4)  \
  instantiate_gemv_bs_helper(name, itype, 8, 1, 1, 32, 4, 4)

instantiate_gemv_bs_blocks(float32, float);
instantiate_gemv_bs_blocks(float16, half);
instantiate_gemv_bs_blocks(bfloat16, bfloat16_t);

#define instantiate_gemv_t_helper(name, itype, bm, bn, sm, sn, tm, tn, nc) \
  instantiate_kernel(                                                      \
      "gemv_t_" #name "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn              \
      "_tm" #tm "_tn" #tn "_nc" #nc "_axpby0",                             \
      gemv_t, itype, bm, bn, sm, sn, tm, tn, nc, 0)

#define instantiate_gemv_t(name, itype, bm, bn, sm, sn, tm, tn)     \
  instantiate_gemv_t_helper(name, itype, bm, bn, sm, sn, tm, tn, 0) \
  instantiate_gemv_t_helper(name, itype, bm, bn, sm, sn, tm, tn, 1)

#define instantiate_gemv_t_blocks(name, itype)      \
  instantiate_gemv_t(name, itype, 1,  2, 8, 4, 4, 1) \
  instantiate_gemv_t(name, itype, 1,  2, 8, 4, 4, 4) \
  instantiate_gemv_t(name, itype, 1,  4, 8, 4, 4, 4) \
  instantiate_gemv_t(name, itype, 1, 16, 8, 4, 4, 4) \
  instantiate_gemv_t(name, itype, 1, 16, 4, 8, 4, 4)

instantiate_gemv_t_blocks(float32, float);
instantiate_gemv_t_blocks(float16, half);
instantiate_gemv_t_blocks(bfloat16, bfloat16_t);

#define instantiate_gemv_t_bs_helper(nm, itype, bm, bn, sm, sn, tm, tn) \
  instantiate_kernel(                                                   \
      "gemv_t_gather_" #nm "_bm" #bm "_bn" #bn "_sm" #sm                \
      "_sn" #sn "_tm" #tm "_tn" #tn,                                    \
      gemv_t_gather, itype, bm, bn, sm, sn, tm, tn)

#define instantiate_gemv_t_bs_blocks(name, itype)              \
  instantiate_gemv_t_bs_helper(name, itype, 1,  2, 8, 4, 4, 1) \
  instantiate_gemv_t_bs_helper(name, itype, 1,  2, 8, 4, 4, 4) \
  instantiate_gemv_t_bs_helper(name, itype, 1,  4, 8, 4, 4, 4) \
  instantiate_gemv_t_bs_helper(name, itype, 1, 16, 8, 4, 4, 4) \
  instantiate_gemv_t_bs_helper(name, itype, 1, 16, 4, 8, 4, 4)

instantiate_gemv_t_bs_blocks(float32, float);
instantiate_gemv_t_bs_blocks(float16, half);
instantiate_gemv_t_bs_blocks(bfloat16, bfloat16_t);

#define instantiate_gemv_wide_helper(name, itype, nv, kl) \
  instantiate_kernel(                                     \
      "gemv_wide_" #name "_nv" #nv "_kl" #kl,             \
      gemv_wide, itype, nv, kl)

#define instantiate_gemv_wide(name, itype, kl)    \
  instantiate_gemv_wide_helper(name, itype, 2, kl) \
  instantiate_gemv_wide_helper(name, itype, 3, kl) \
  instantiate_gemv_wide_helper(name, itype, 4, kl) \
  instantiate_gemv_wide_helper(name, itype, 5, kl)

instantiate_gemv_wide(float16, half, 16);
instantiate_gemv_wide(bfloat16, bfloat16_t, 16);
instantiate_gemv_wide(float16, half, 32);
instantiate_gemv_wide(bfloat16, bfloat16_t, 32);

#define instantiate_gemv_wide_gather_helper(name, itype, nv, kl) \
  instantiate_kernel(                                            \
      "gemv_wide_gather_" #name "_nv" #nv "_kl" #kl,             \
      gemv_wide_gather, itype, nv, kl)

#define instantiate_gemv_wide_gather(name, itype, kl)    \
  instantiate_gemv_wide_gather_helper(name, itype, 2, kl) \
  instantiate_gemv_wide_gather_helper(name, itype, 3, kl) \
  instantiate_gemv_wide_gather_helper(name, itype, 4, kl) \
  instantiate_gemv_wide_gather_helper(name, itype, 5, kl)

instantiate_gemv_wide_gather(float16, half, 16);
instantiate_gemv_wide_gather(bfloat16, bfloat16_t, 16);
instantiate_gemv_wide_gather(float16, half, 32);
instantiate_gemv_wide_gather(bfloat16, bfloat16_t, 32);
// clang-format on

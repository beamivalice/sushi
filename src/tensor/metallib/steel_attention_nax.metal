// Sushi Cooler NAX full-attention kernels — instantiates MLX's MIT-licensed
// steel_attention_nax.h (Copyright (c) 2024-2026 Apple Inc., see NOTICE) for
// the head-dim-64 and split-256 tiles src/tensor/attention.zig dispatches on NAX.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"

#define instantiate_sushi_attn_nax(tname, dtype)                                  \
  instantiate_kernel(                                                             \
      "steel_attention_" #tname "_bq64_bk32_bd64_bv64_wm4_wn1_mask" #tname,       \
      attention_nax, dtype, 64, 32, 64, 64, 4, 1, dtype, float)                   \
  instantiate_kernel(                                                             \
      "steel_attention_dsplit_" #tname "_bq64_bk32_bd256_bv256_wm4_wn2_mask" #tname, \
      attention_nax_dsplit, dtype, 64, 32, 256, 4, 2, dtype, float)

instantiate_sushi_attn_nax(float16, half);
instantiate_sushi_attn_nax(bfloat16, bfloat);
instantiate_sushi_attn_nax(float32, float);
// clang-format on

// Sushi Cooler SIMD full-attention kernels — instantiates MLX's MIT-licensed
// steel_attention.h (Copyright (c) 2024-2026 Apple Inc., see NOTICE) for the
// head-dim-64 tile src/tensor/attention.zig dispatches off NAX.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"

#define instantiate_sushi_attn(tname, dtype)                                      \
  instantiate_kernel(                                                             \
      "steel_attention_" #tname "_bq32_bk32_bd64_bv64_wm4_wn1_mask" #tname,       \
      attention, dtype, 32, 32, 64, 64, 4, 1, dtype, float)

instantiate_sushi_attn(float16, half);
instantiate_sushi_attn(bfloat16, bfloat16_t);
instantiate_sushi_attn(float32, float);
// clang-format on

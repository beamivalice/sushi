// Sushi Cooler dilated depthwise conv1d kernel — instantiates MLX's MIT-licensed
// steel_conv.h (Copyright (c) 2024 Apple Inc., see NOTICE) for src/tensor/stock.zig.

#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/mma.h"
#include "mlx/backend/metal/kernels/steel/conv/conv.h"
#include "mlx/backend/metal/kernels/steel/conv/params.h"
#include "mlx/backend/metal/kernels/steel/conv/kernels/steel_conv.h"

#define instantiate_sushi_conv_1ch(name, itype)                                            \
  template [[host_name("implicit_gemm_conv_2d_" #name                                      \
                       "_bm32_bn8_bk16_wm4_wn1_channel_1_filter_l")]] [[kernel]] void      \
  implicit_gemm_conv_2d<itype, 32, 8, 16, 4, 1, 1, false>(                                 \
      const device itype* A [[buffer(0)]],                                                 \
      const device itype* B [[buffer(1)]],                                                 \
      device itype* C [[buffer(2)]],                                                       \
      const constant MLXConvParams<2>* params [[buffer(3)]],                               \
      const constant ImplicitGemmConv2DParams* gemm_params [[buffer(4)]],                  \
      uint3 tid [[threadgroup_position_in_grid]],                                          \
      uint3 lid [[thread_position_in_threadgroup]],                                        \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                                    \
      uint simd_lid [[thread_index_in_simdgroup]]);

instantiate_sushi_conv_1ch(float32, float);
instantiate_sushi_conv_1ch(float16, half);
instantiate_sushi_conv_1ch(bfloat16, bfloat16_t);
// clang-format on

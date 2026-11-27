// Sushi Cooler binary stock kernels — instantiates MLX's MIT-licensed binary.h
// (Copyright (c) 2024 Apple Inc., see NOTICE) for the ops and dtypes
// src/tensor/stock.zig, ops.zig and route.zig dispatch.

#include <metal_integer>
#include <metal_math>

// clang-format off
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/binary_ops.h"
#include "mlx/backend/metal/kernels/binary.h"

#define instantiate_sushi_binary(op, tname, type)                    \
  instantiate_kernel("ss_" #op #tname, binary_ss, type, type, op)    \
  instantiate_kernel("sv_" #op #tname, binary_sv, type, type, op, 1) \
  instantiate_kernel("vs_" #op #tname, binary_vs, type, type, op, 1) \
  instantiate_kernel("vv_" #op #tname, binary_vv, type, type, op, 1) \
  instantiate_kernel("svn_" #op #tname, binary_sv, type, type, op)   \
  instantiate_kernel("vsn_" #op #tname, binary_vs, type, type, op)   \
  instantiate_kernel("vvn_" #op #tname, binary_vv, type, type, op)

#define instantiate_sushi_binary_float(op)          \
  instantiate_sushi_binary(op, float16, half)       \
  instantiate_sushi_binary(op, float32, float)      \
  instantiate_sushi_binary(op, bfloat16, bfloat16_t)

#define instantiate_sushi_binary_all(op)            \
  instantiate_sushi_binary(op, bool_, bool)         \
  instantiate_sushi_binary(op, uint8, uint8_t)      \
  instantiate_sushi_binary(op, uint16, uint16_t)    \
  instantiate_sushi_binary(op, uint32, uint32_t)    \
  instantiate_sushi_binary(op, int8, int8_t)        \
  instantiate_sushi_binary(op, int16, int16_t)      \
  instantiate_sushi_binary(op, int32, int32_t)      \
  instantiate_sushi_binary_float(op)

instantiate_sushi_binary_float(LogAddExp)
instantiate_sushi_binary_all(Power)
instantiate_sushi_binary_all(Remainder)
// clang-format on

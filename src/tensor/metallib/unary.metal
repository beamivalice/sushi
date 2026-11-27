// Sushi Cooler unary stock kernels — instantiates MLX's MIT-licensed unary.h
// (Copyright (c) 2024 Apple Inc., see NOTICE) for the ops and dtypes
// src/tensor/stock.zig and src/tensor/route.zig dispatch.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/unary_ops.h"
#include "mlx/backend/metal/kernels/unary.h"

#define instantiate_sushi_unary(op, tname, type)                             \
  instantiate_kernel("v_" #op #tname #tname, unary_v, type, type, op, 1)     \
  instantiate_kernel("vn_" #op #tname #tname, unary_v, type, type, op)

#define instantiate_sushi_unary_float(op)          \
  instantiate_sushi_unary(op, float16, half)       \
  instantiate_sushi_unary(op, float32, float)      \
  instantiate_sushi_unary(op, bfloat16, bfloat16_t)

#define instantiate_sushi_unary_all(op)            \
  instantiate_sushi_unary(op, bool_, bool)         \
  instantiate_sushi_unary(op, uint8, uint8_t)      \
  instantiate_sushi_unary(op, uint16, uint16_t)    \
  instantiate_sushi_unary(op, uint32, uint32_t)    \
  instantiate_sushi_unary(op, int8, int8_t)        \
  instantiate_sushi_unary(op, int16, int16_t)      \
  instantiate_sushi_unary(op, int32, int32_t)      \
  instantiate_sushi_unary_float(op)

instantiate_sushi_unary_float(Erf)
instantiate_sushi_unary_float(Tanh)
instantiate_sushi_unary_float(Cos)
instantiate_sushi_unary_float(Sin)
instantiate_sushi_unary_float(Log1p)
instantiate_sushi_unary_all(Sign)
instantiate_sushi_unary_all(Abs)
instantiate_sushi_unary_all(Floor)
// clang-format on

// Sushi Cooler float reduction kernels — instantiates MLX's MIT-licensed reduce.h
// (Copyright (c) 2024 Apple Inc., see NOTICE) for sum and max in src/tensor/reduce.zig.

#include <metal_atomic>
#include <metal_simdgroup>

// clang-format off
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/atomic.h"
#include "mlx/backend/metal/kernels/reduction/ops.h"
#include "mlx/backend/metal/kernels/reduce.h"

#define instantiate_sushi_reduce(name, type, op)                                  \
  instantiate_kernel("all_reduce_" #name, all_reduce, type, type, op)             \
  instantiate_kernel("row_reduce_simple_" #name, row_reduce_simple, type, type, op) \
  instantiate_kernel("row_reduce_small_1_reduce_" #name, row_reduce_small, type, type, op, int, 1) \
  instantiate_kernel("row_reduce_looped_1_reduce_" #name, row_reduce_looped, type, type, op, int, 1) \
  instantiate_kernel("col_reduce_small_1_reduce_" #name, col_reduce_small, type, type, op, int, 1) \
  instantiate_kernel("col_reduce_longcolumn_1_reduce_" #name, col_reduce_longcolumn, type, type, op, int, 1) \
  instantiate_kernel("col_reduce_2pass_1_32_32_reduce_" #name, col_reduce_2pass, type, type, op, int, 1, 32, 32) \
  instantiate_kernel("col_reduce_looped_1_32_32_reduce_" #name, col_reduce_looped, type, type, op, int, 1, 32, 32)

#define instantiate_sushi_reduce_float(name, op)                         \
  instantiate_sushi_reduce(name##float16, float16_t, op<float16_t>)      \
  instantiate_sushi_reduce(name##bfloat16, bfloat16_t, op<bfloat16_t>)   \
  instantiate_sushi_reduce(name##float32, float, op<float>)

instantiate_sushi_reduce_float(sum, Sum)
instantiate_sushi_reduce_float(max, Max)
// clang-format on

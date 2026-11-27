// Sushi Cooler cumulative-sum scan kernels — instantiates MLX's MIT-licensed scan.h
// (Copyright (c) 2023-2024 Apple Inc., see NOTICE) for src/tensor/scan.zig.

#include <metal_math>
#include <metal_simdgroup>

// clang-format off
using namespace metal;

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/scan.h"

#define instantiate_contiguous_scan(                                    \
    name, itype, otype, op, inclusive, reverse, nreads)                 \
  template [[host_name("contig_scan_" #name)]] [[kernel]] void          \
  contiguous_scan<itype, otype, op<otype>, nreads, inclusive, reverse>( \
      const device itype* in [[buffer(0)]],                             \
      device otype* out [[buffer(1)]],                                  \
      const constant size_t& axis_size [[buffer(2)]],                   \
      uint3 gid [[threadgroup_position_in_grid]],                       \
      uint3 gsize [[threadgroups_per_grid]],                            \
      uint3 lid [[thread_position_in_threadgroup]],                     \
      uint3 lsize [[threads_per_threadgroup]],                          \
      uint simd_lane_id [[thread_index_in_simdgroup]],                  \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_strided_scan(                                    \
    name, itype, otype, op, inclusive, reverse, nreads)              \
  template [[host_name("strided_scan_" #name)]] [[kernel]] void      \
  strided_scan<itype, otype, op<otype>, nreads, inclusive, reverse>( \
      const device itype* in [[buffer(0)]],                          \
      device otype* out [[buffer(1)]],                               \
      const constant size_t& axis_size [[buffer(2)]],                \
      const constant size_t& stride [[buffer(3)]],                   \
      const constant size_t& stride_blocks [[buffer(4)]],            \
      uint3 gid [[threadgroup_position_in_grid]],                    \
      uint3 gsize [[threadgroups_per_grid]],                         \
      uint3 lid [[thread_position_in_threadgroup]],                  \
      uint simd_lane_id [[thread_index_in_simdgroup]],               \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_scan_helper(name, itype, otype, op, nreads)                                \
  instantiate_contiguous_scan(inclusive_##name, itype, otype, op, true, false, nreads)         \
  instantiate_contiguous_scan(exclusive_##name, itype, otype, op, false, false, nreads)        \
  instantiate_contiguous_scan(reverse_inclusive_##name, itype, otype, op, true, true, nreads)  \
  instantiate_contiguous_scan(reverse_exclusive_##name, itype, otype, op, false, true, nreads) \
  instantiate_strided_scan(inclusive_##name, itype, otype, op, true, false, nreads)            \
  instantiate_strided_scan(exclusive_##name, itype, otype, op, false, false, nreads)           \
  instantiate_strided_scan(reverse_inclusive_##name, itype, otype, op, true, true, nreads)     \
  instantiate_strided_scan(reverse_exclusive_##name, itype, otype, op, false, true, nreads)

instantiate_scan_helper(sum_uint8_uint8,         uint8_t,     uint8_t,     CumSum, 4)
instantiate_scan_helper(sum_uint16_uint16,       uint16_t,    uint16_t,    CumSum, 4)
instantiate_scan_helper(sum_uint32_uint32,       uint32_t,    uint32_t,    CumSum, 4)
instantiate_scan_helper(sum_int8_int8,           int8_t,      int8_t,      CumSum, 4)
instantiate_scan_helper(sum_int16_int16,         int16_t,     int16_t,     CumSum, 4)
instantiate_scan_helper(sum_int32_int32,         int32_t,     int32_t,     CumSum, 4)
instantiate_scan_helper(sum_float16_float16,     half,        half,        CumSum, 4)
instantiate_scan_helper(sum_float32_float32,     float,       float,       CumSum, 4)
instantiate_scan_helper(sum_bfloat16_bfloat16,   bfloat16_t,  bfloat16_t,  CumSum, 4)
// clang-format on

// Sushi Cooler sort and searchsorted kernels — instantiates MLX's MIT-licensed
// sort.h and searchsorted.h (Copyright (c) 2023-2026 Apple Inc., see NOTICE)
// for the nine dtypes src/tensor/sort.zig dispatches.

#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sort.h"
#include "mlx/backend/metal/kernels/searchsorted.h"

#define instantiate_block_sort(                                            \
    name, itname, itype, otname, otype, arg_sort, bn, tn)                  \
  instantiate_kernel("c" #name "_" #itname "_" #otname "_bn" #bn "_tn" #tn, \
                     block_sort, itype, otype, arg_sort, bn, tn)           \
  instantiate_kernel("nc" #name "_" #itname "_" #otname "_bn" #bn "_tn" #tn, \
                     block_sort_nc, itype, otype, arg_sort, bn, tn)

#define instantiate_block_sort_tn(itname, itype, bn)                                        \
  instantiate_block_sort(_block_sort, itname, itype, itname, itype, false, bn, 4)           \
  instantiate_block_sort(arg_block_sort, itname, itype, uint32, uint32_t, true, bn, 4)

#define instantiate_multi_block_sort(vtname, vtype)                                         \
  instantiate_kernel("sort_mbsort_" #vtname "_uint32_bn512_tn4",                            \
                     mb_block_sort, vtype, uint32_t, true, 512, 4)                          \
  instantiate_kernel("partition_mbsort_" #vtname "_uint32_bn512_tn4",                       \
                     mb_block_partition, vtype, uint32_t, true, 512, 4)                     \
  instantiate_kernel("merge_mbsort_" #vtname "_uint32_bn512_tn4",                           \
                     mb_block_merge, vtype, uint32_t, true, 512, 4)

#define instantiate_sushi_sort(tname, type)                       \
  instantiate_block_sort_tn(tname, type, 32)                      \
  instantiate_block_sort_tn(tname, type, 64)                      \
  instantiate_block_sort_tn(tname, type, 128)                     \
  instantiate_block_sort_tn(tname, type, 256)                     \
  instantiate_block_sort_tn(tname, type, 512)                     \
  instantiate_multi_block_sort(tname, type)                       \
  instantiate_kernel("searchsorted_" #tname "_left", searchsorted, type, false) \
  instantiate_kernel("searchsorted_" #tname "_right", searchsorted, type, true)

instantiate_sushi_sort(uint8, uint8_t)
instantiate_sushi_sort(uint16, uint16_t)
instantiate_sushi_sort(uint32, uint32_t)
instantiate_sushi_sort(int8, int8_t)
instantiate_sushi_sort(int16, int16_t)
instantiate_sushi_sort(int32, int32_t)
instantiate_sushi_sort(float16, half)
instantiate_sushi_sort(float32, float)
instantiate_sushi_sort(bfloat16, bfloat16_t)
// clang-format on

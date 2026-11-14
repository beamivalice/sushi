#!/bin/bash
# sushi-test: tier=pack pack=default
# The DeepSeek engine tests as an M1 to M4 GPU runs them: MLX_METAL_GPU_ARCH names the GPU, so MLX, the device profile
# and the engine all dispatch the SIMD arms, and every stock tile is held to MLX's bytes for that GPU. Not a real
# chip: thread grants and clocks are the chip's own.
set -u
cd "$(dirname "$0")/../.."
ARCH=${1:-applegpu_g13s}
FIX=${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/test-fixtures
export DSV41_TINY=${DSV41_TINY:-$FIX/dsv41-dspark-tiny} DSV41_STQ_TINY=${DSV41_STQ_TINY:-$FIX/dsv41-stq-tiny} DSV41_EPI_TINY=${DSV41_EPI_TINY:-$FIX/dsv41-epi-tiny}
for d in "$DSV41_TINY" "$DSV41_STQ_TINY" "$DSV41_EPI_TINY"; do
    [ -d "$d" ] || { echo "SKIP: no fixture at $d"; exit 3; }
done
ZIG=${ZIG:-.zig-toolchain/zig}
"$ZIG" build test-build -Doptimize=ReleaseFast -Dmodel=dsv41 || { echo "FAIL: the test build"; exit 1; }
status=0
for bin in test-rest test-generate test-trunk; do
    log=$(mktemp)
    if MLX_METAL_GPU_ARCH=$ARCH "zig-out/tests/$bin" > "$log" 2>&1; then
        echo "PASS $bin as $ARCH: $(tail -1 "$log")"
    else
        echo "FAIL $bin as $ARCH:"; grep -a -B1 -E '\.\.\.FAIL|^FAIL' "$log" | cut -c1-240 | head -40
        status=1
    fi
    rm -f "$log"
done
exit $status

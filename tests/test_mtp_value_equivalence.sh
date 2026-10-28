#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 tests/test_mtp_equivalence_strict.py
.zig-toolchain/zig build test -Doptimize=ReleaseFast \
    -Dtest-filter=mtp_qmv \
    -Dtest-filter='GDN verify recurrence' \
    -Dtest-filter='affine8 projections are row-identical' \
    -Dtest-filter='dense projections are row-identical' \
    -Dtest-filter='causal verify attention rows are bit-identical' \
    -Dtest-filter='QSA verify'
if [ -n "${MTP_TEST_MODEL:-}" ]; then
    QWEN4_TEST_MODEL="$MTP_TEST_MODEL" .zig-toolchain/zig build test -Doptimize=ReleaseFast \
        -Dtest-filter='qwen4 checkpoint affine8 and dense projections' \
        -Dtest-filter='qwen4 verify history row and cache identity' \
        -Dtest-filter='qwen4 MTP verify token values' \
        -Dtest-filter='qwen4 deferred PLE: the captured hidden'
fi

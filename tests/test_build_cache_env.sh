#!/bin/bash
# sushi-test: tier=hermetic
# test_build_cache_env.sh — a test run is keyed by the fixture and engine env it reads: changing only such a variable
# runs the tests again instead of replaying a cached pass. A pack-tier test reads its fixture by getenv and skips without
# it, so a pass cached without the fixture would stand for a run with it.
#
# Usage: ./tests/test_build_cache_env.sh

set -u
cd "$(dirname "$0")/.." || exit 1
ZIG="${ZIG:-$PWD/.zig-toolchain/zig}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAIL=0
FILTER="format corpus"

run() { env "$@" "$ZIG" build test-host -Doptimize=ReleaseFast "-Dtest-filter=$FILTER" --summary all >"$TMP/out" 2>&1; }
ran() { grep -q "run test test-host [0-9]* pass" "$TMP/out"; }

for var in QWEN4_TEST_MODEL DSV41_STQ_TINY SUSHI_MODELS_DIR; do
  run "$var=$TMP/one" || { echo "  FAIL: test-host with $var set"; FAIL=$((FAIL + 1)); continue; }
  run "$var=$TMP/two" || { echo "  FAIL: test-host with $var changed"; FAIL=$((FAIL + 1)); continue; }
  ran || { echo "  FAIL: changing only $var replayed a cached run"; sed 's/^/    /' "$TMP/out" | tail -4; FAIL=$((FAIL + 1)); }
done

[ "$FAIL" -eq 0 ] || exit 1

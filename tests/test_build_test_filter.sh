#!/bin/bash
# sushi-test: tier=hermetic
# test_build_test_filter.sh — `zig build test -Dtest-filter=` configures for a filter that selects a test whose
# name holds an apostrophe (the response file a big test binary's args go through cannot carry one), and the pure
# filter helper in build.zig picks a quote-free filter that selects no other test. Configure only, except the filtered
# test-host runs, which compile and run the few host tests they select.
#
# Usage: ./tests/test_build_test_filter.sh

set -u
cd "$(dirname "$0")/.." || exit 1
ZIG="${ZIG:-$PWD/.zig-toolchain/zig}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAIL=0
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; sed 's/^/    /' "$TMP/out" | head -20; }

"$ZIG" test build.zig >"$TMP/out" 2>&1 || fail "build.zig helper tests"

for filter in "streamed request bill brackets" "Flash-Next's streamed request bill" "Flash-Next"; do
  "$ZIG" build -l -Doptimize=ReleaseFast "-Dtest-filter=$filter" >"$TMP/out" 2>&1 || fail "configure with -Dtest-filter=\"$filter\""
done

# A filtered test-host keeps to the host tier's own tests: a bare substring would also select the MLX-tier tests of the
# files the host files reach, which cannot link without MLX. "qsa" names such tests, and a filter that matches no host
# test must build no host binary instead of running them all.
for filter in qsa "no test is named like this"; do
  "$ZIG" build test-host "-Dtest-filter=$filter" >"$TMP/out" 2>&1 || fail "test-host links with -Dtest-filter=\"$filter\""
done

[ "$FAIL" -eq 0 ] || exit 1

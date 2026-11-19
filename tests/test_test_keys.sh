#!/bin/bash
# sushi-test: tier=hermetic
# test_test_keys.sh — tests/lib/test_keys.py's normalized binary hash, which tests/gate.sh --continue keys a run step
# by: two builds of the same code (another cache, an edit to a test the filter leaves out) hash alike although
# their bytes differ, and an edit to a test the binary holds changes the hash. Compile only, no GPU; it obeys the
# QUIET marker like any build.
#
# Usage: ./tests/test_test_keys.sh

set -u
cd "$(dirname "$0")/.." || exit 1
ZIG="${ZIG:-$PWD/.zig-toolchain/zig}"
QUIET=$HOME/.sushi/runs/QUIET
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAIL=0
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }

mkdir -p "$TMP/src"
# build <name> <kept test's body> <left-out test's body> <cache>: a ReleaseFast test binary of the kept test alone.
build() {
  printf '%s\n' 'const std = @import("std");' "test \"kept\" { $2 }" "test \"left out\" { $3 }" > "$TMP/src/t.zig"
  while [ -e "$QUIET" ]; do sleep 5; done
  (cd "$TMP/src" && taskpolicy -a "$ZIG" test -OReleaseFast t.zig --test-no-exec --test-filter kept \
    "-femit-bin=$TMP/$1" --cache-dir "$TMP/$4" --global-cache-dir "$TMP/g-$4" > "$TMP/out" 2>&1) \
    || { fail "the $1 binary builds"; sed 's/^/    /' "$TMP/out" | head -12; echo "FAIL"; exit 1; }
}
key() { python3 -I tests/lib/test_keys.py normalized "$TMP/$1"; }

PASSES='try std.testing.expect(true);'
SKIPS='return error.SkipZigTest;'
build base "$PASSES" "$PASSES" c1
build other_cache "$PASSES" "$PASSES" c2
build left_out_edit "$PASSES" "$SKIPS" c1
build kept_edit "$SKIPS" "$PASSES" c1

cmp -s "$TMP/base" "$TMP/other_cache" && fail "the two caches' builds should differ in bytes, or this test proves nothing"
[ "$(key base)" = "$(key other_cache)" ] || fail "a build of the same code in another cache should hash alike"
[ "$(key base)" = "$(key left_out_edit)" ] || fail "an edit to a test the binary leaves out should hash alike"
[ "$(key base)" != "$(key kept_edit)" ] || fail "an edit to a test the binary holds should change the hash"

[ "$FAIL" -eq 0 ] || { echo "FAIL"; exit 1; }
echo "PASS"

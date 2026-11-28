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

# metallib invalidation: only steps whose plan marks metallib=true (or omits it, fail-safe) hash zig-out/lib/sushi.metallib,
# and tracked .metal files do not spuriously bust pure-host / non-metallib steps.
REPO="$TMP/repo"
mkdir -p "$REPO/lib/mlx/lib" "$REPO/src/tensor/metallib" "$REPO/zig-out/tests" "$REPO/zig-out/lib"
git -C "$REPO" init -q
printf 'mlx' > "$REPO/lib/mlx/lib/libmlx.dylib"
printf 'kernel void k() {}\n' > "$REPO/src/tensor/metallib/k.metal"
git -C "$REPO" -c user.name=t -c user.email=t@t add -A
git -C "$REPO" -c user.name=t -c user.email=t@t commit -qm init
cp "$TMP/base" "$REPO/zig-out/tests/t"
printf 'mlib-v1' > "$REPO/zig-out/lib/sushi.metallib"
cat > "$REPO/zig-out/test-plan.json" << 'EOF'
[
  {"step":"step-no-mlib","binary":"tests/t","metallib":false,"env":[["SUSHI_TEST_STEP","step-no-mlib"]]},
  {"step":"step-uses-mlib","binary":"tests/t","metallib":true,"env":[["SUSHI_TEST_STEP","step-uses-mlib"]]},
  {"step":"step-default-mlib","binary":"tests/t","env":[["SUSHI_TEST_STEP","step-default-mlib"]]}
]
EOF
SUSHI_FIXTURE_ENVS="" python3 -I tests/lib/test_keys.py keys "$REPO/zig-out/test-plan.json" "$REPO" > "$TMP/k1.tsv"
printf 'mlib-v2' > "$REPO/zig-out/lib/sushi.metallib"
printf 'kernel void k2() {}\n' > "$REPO/src/tensor/metallib/k.metal"
git -C "$REPO" -c user.name=t -c user.email=t@t commit -qam "edit metal"
SUSHI_FIXTURE_ENVS="" python3 -I tests/lib/test_keys.py keys "$REPO/zig-out/test-plan.json" "$REPO" > "$TMP/k2.tsv"

kval() { awk -F'\t' -v s="$2" '$1 == s {print $2}' "$1"; }
[ "$(kval "$TMP/k1.tsv" step-no-mlib)" = "$(kval "$TMP/k2.tsv" step-no-mlib)" ] \
  || fail "a step with metallib=false should not be invalidated by sushi.metallib or .metal edits"
[ "$(kval "$TMP/k1.tsv" step-uses-mlib)" != "$(kval "$TMP/k2.tsv" step-uses-mlib)" ] \
  || fail "a step with metallib=true must be invalidated when sushi.metallib changes"
[ "$(kval "$TMP/k1.tsv" step-default-mlib)" != "$(kval "$TMP/k2.tsv" step-default-mlib)" ] \
  || fail "a step omitting metallib must default to hashing sushi.metallib"

[ "$FAIL" -eq 0 ] || { echo "FAIL"; exit 1; }
echo "PASS"

#!/bin/bash
# sushi-test: tier=hermetic needs=git
# test_arch_slice.sh — `zig build arch-slice -Dslice-arch=<name>` compiles src/<name>/ as a module of its own
# against the host stub: a folder that reaches the engine only through `arch_host` passes, and one that imports an
# engine file by path fails. The probe folders live in a temporary worktree of HEAD, so a kill leaves nothing in
# the tree and a concurrent build keeps its configure cache. Compile only, no GPU, no lock; it obeys the QUIET
# marker like any build.
#
# Usage: ./tests/test_arch_slice.sh

set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd -P)
ZIG="${ZIG:-$ROOT/.zig-toolchain/zig}"
QUIET=$HOME/.sushi/runs/QUIET
TMP="$(mktemp -d)"
cleanup() {
  git worktree remove --force "$TMP/wt" 2>/dev/null
  rm -rf "$TMP"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
FAIL=0
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; sed 's/^/    /' "$TMP/out" | head -12; }

git worktree add --quiet --detach "$TMP/wt" HEAD || exit 1
ln -s "$ROOT/.zig-toolchain" "$TMP/wt/.zig-toolchain"
ln -s "$ROOT/lib/mlx" "$TMP/wt/lib/mlx"
cd "$TMP/wt" || exit 1
mkdir src/zz_slice_ok src/zz_slice_bad
printf '%s\n' 'const host = @import("arch_host");' \
  'pub fn width() usize {' \
  '    return @sizeOf(host.mlx.mlx_array);' \
  '}' > src/zz_slice_ok/arch.zig
printf '%s\n' 'const log = @import("../log.zig");' \
  'pub fn say() void {' \
  '    log.info("x", .{});' \
  '}' > src/zz_slice_bad/arch.zig

while [ -e "$QUIET" ]; do sleep 5; done
slice() { taskpolicy -a "$ZIG" build arch-slice "$@" >"$TMP/out" 2>&1; }

slice -Dslice-arch=zz_slice_ok || fail "a folder that reaches the engine through arch_host alone should pass"
if slice -Dslice-arch=zz_slice_bad; then
  fail "a folder that imports an engine file by path should fail"
elif ! grep -q "module" "$TMP/out"; then
  fail "the failure should be the module boundary"
fi
if slice; then fail "no -Dslice-arch should fail"; elif ! grep -q -- "-Dslice-arch" "$TMP/out"; then fail "no -Dslice-arch should say so"; fi
if slice -Dslice-arch=zz_slice_missing; then fail "an unknown arch should fail"; elif ! grep -q "does not exist" "$TMP/out"; then fail "an unknown arch should say it is missing"; fi

[ "$FAIL" -eq 0 ] || { echo "FAIL"; exit 1; }
echo "PASS"

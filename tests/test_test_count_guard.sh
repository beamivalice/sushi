#!/bin/bash
# sushi-test: tier=hermetic
# test_test_count_guard.sh — scripts/test_count_guard.zig, which `zig build test` runs after each scanned test
# binary: a binary with no tests fails and names its bucket, one with a test passes silently, and the guard runs
# nothing of the binary's. Compile only, no GPU; it obeys the QUIET marker like any build.
#
# Usage: ./tests/test_test_count_guard.sh

set -u
cd "$(dirname "$0")/.." || exit 1
ZIG="${ZIG:-$PWD/.zig-toolchain/zig}"
QUIET=$HOME/.sushi/runs/QUIET
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAIL=0
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; sed 's/^/    /' "$TMP/out" | head -12; }

# The anonymous block of a root file survives every filter and must not count.
printf '%s\n' 'test {' '    _ = @import("std");' '}' > "$TMP/empty.zig"
printf '%s\n' 'const std = @import("std");' 'test "one" {' '    try std.testing.expect(true);' '}' \
  'test "two" {' '    std.process.exit(7);' '}' > "$TMP/some.zig"

# A real bucket lists thousands of names, far past any fixed read buffer.
for i in $(seq 1 400); do
  printf '%s\n' "test \"a test with a long enough name to fill the metadata message number $i\" {" '}' >> "$TMP/some.zig"
done

while [ -e "$QUIET" ]; do sleep 5; done
build() { taskpolicy -a "$ZIG" "$@" >"$TMP/out" 2>&1; }
build build-exe scripts/test_count_guard.zig "-femit-bin=$TMP/guard" || { fail "the guard builds"; echo "FAIL"; exit 1; }
build test "$TMP/empty.zig" --test-no-exec "-femit-bin=$TMP/empty" || { fail "the empty test binary builds"; echo "FAIL"; exit 1; }
build test "$TMP/some.zig" --test-no-exec "-femit-bin=$TMP/some" || { fail "the test binary builds"; echo "FAIL"; exit 1; }

# `two` would exit 7 if it ran: the guard only counts.
"$TMP/guard" some-bucket "$TMP/some" >"$TMP/out" 2>&1 || fail "a binary with tests should pass"
[ ! -s "$TMP/out" ] || fail "a passing guard prints nothing"
if "$TMP/guard" test-rest "$TMP/empty" >"$TMP/out" 2>&1; then
  fail "a binary with no tests should fail"
elif ! grep -q "test-rest" "$TMP/out"; then
  fail "the failure should name its bucket"
fi

[ "$FAIL" -eq 0 ] || { echo "FAIL"; exit 1; }
echo "PASS"

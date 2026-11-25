#!/bin/bash
# sushi-test: tier=hermetic
# test_build_test_step.sh — `-Dtest-step=<run step>` runs that test run step alone and compiles only its binary: `-Dmodel=<arch>` makes the folder's
# step and every shared one, and `zig build test` runs all it makes. A name that is no run step fails by name.
#
# Usage: ./tests/test_build_test_step.sh

set -u
cd "$(dirname "$0")/.." || exit 1
ZIG="${ZIG:-$PWD/.zig-toolchain/zig}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAIL=0
# One MLX-free test in each of two run steps, so the selection has something to leave out.
F=(-Dtest-filter="tool traffic" -Dtest-filter="the Load button tooltip keeps a long model name whole")

if "$ZIG" build test-plan -Doptimize=ReleaseFast -Dtest-step=no-such-step >"$TMP/out" 2>&1; then
  echo "  FAIL: -Dtest-step=no-such-step was accepted"; FAIL=$((FAIL + 1))
elif ! grep -q "Dtest-step=no-such-step names no test run step" "$TMP/out"; then
  echo "  FAIL: -Dtest-step=no-such-step failed without naming it"; tail -3 "$TMP/out" | sed 's/^/    /'; FAIL=$((FAIL + 1))
fi

"$ZIG" build test-plan -Doptimize=ReleaseFast "${F[@]}" >"$TMP/out" 2>&1 || { echo "  FAIL: test-plan"; tail -3 "$TMP/out"; exit 1; }
steps=$(python3 -I -c 'import json; print(" ".join(sorted(r["step"] for r in json.load(open("zig-out/test-plan.json")))))')
[ "$steps" = "test-host test-sushibar" ] || { echo "  FAIL: the filters plan '$steps', not test-host and test-sushibar"; exit 1; }

if ! "$ZIG" build test -Doptimize=ReleaseFast "${F[@]}" -Dtest-step=test-host -Dtest-log="$TMP/log" --summary all >"$TMP/out" 2>&1; then
  echo "  FAIL: zig build test -Dtest-step=test-host"; tail -5 "$TMP/out" | sed 's/^/    /'; FAIL=$((FAIL + 1))
else
  ran=$(cut -f2 "$TMP/log" | sort -u | tr '\n' ' ')
  [ "$ran" = "test-host " ] || { echo "  FAIL: -Dtest-step=test-host ran: $ran"; FAIL=$((FAIL + 1)); }
  # A binary none of whose steps runs is not compiled, not even for its count guard.
  if grep -q "compile test test-sushibar" "$TMP/out"; then
    echo "  FAIL: -Dtest-step=test-host compiled test-sushibar, whose step it leaves out"; FAIL=$((FAIL + 1))
  fi
fi

[ "$FAIL" -eq 0 ] && echo "PASS: test_build_test_step" || { echo "FAIL: test_build_test_step ($FAIL)"; exit 1; }

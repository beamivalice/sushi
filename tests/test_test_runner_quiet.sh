#!/bin/bash
# test_test_runner_quiet.sh — a passing test prints NOTHING, on either stream.
#
# CLAUDE.md already bans stdout from a test: under `zig build test` fd 1 is the
# build runner's protocol pipe and one stray byte hangs the runner forever.
# stderr is not free either. Zig 0.17 reports ANY stderr a test
# step produced through its failure renderer, so a suite that passed 69/69
# prints
#
#     test
#     +- run test w
#     <the test's own output>
#     failed command: .../test --cache-dir=./.zig-cache --seed=0x... --listen=-
#
# and no Build Summary, while exiting 0. That is indistinguishable from a real
# failure at a glance and has been read as one. So: benchmark and diagnostic
# lines in a test ride an env switch (SUSHI_EXL3_PARITY_TRACE for the EXL3
# ones); only a test that is about to FAIL may print.
#
# Usage: ./tests/test_test_runner_quiet.sh [test-filter]   (default: the whole suite)

set -u
cd "$(dirname "$0")/.." || exit 1

ZIG=${ZIG:-./.zig-toolchain/zig}
FILTER=${1:-}
STEP=${TEST_STEP:-test}

if [ ! -x "$ZIG" ]; then
  echo "SKIP: no $ZIG (run ./scripts/fetch-zig.sh)"
  exit 0
fi

# The run step is cached on success, so a second invocation would prove nothing.
# Drop the manifests that name a test binary; every other step stays cached.
# Zig 0.17 manifests are binary files with NUL-terminated paths and no extension.
LC_ALL=C grep -l -a -E 'o/[0-9a-f]+/(exl3-)?test' .zig-cache/h/* 2>/dev/null | xargs rm -f 2>/dev/null

if [ -n "$FILTER" ]; then
  WHAT="zig build $STEP -Dtest-filter=$FILTER"
  OUT=$("$ZIG" build "$STEP" -Doptimize=ReleaseFast -Dtest-filter="$FILTER" 2>&1)
else
  WHAT="zig build $STEP"
  OUT=$("$ZIG" build "$STEP" -Doptimize=ReleaseFast 2>&1)
fi
RC=$?

FAIL=0
if [ $RC -ne 0 ]; then
  echo "  FAIL: $WHAT exited $RC"
  FAIL=1
fi
if [ -n "$OUT" ]; then
  echo "  FAIL: the test step wrote $(printf '%s' "$OUT" | wc -l | tr -d ' ') lines; a passing suite is silent"
  printf '%s\n' "$OUT" | head -20 | sed 's/^/    | /'
  FAIL=1
fi

if [ $FAIL -eq 0 ]; then
  echo "  PASS: $WHAT is silent and green"
  echo "PASS"
  exit 0
fi
echo "FAIL"
exit 1

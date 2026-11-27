#!/bin/bash
# sushi-test: tier=hermetic
# test_build_cache_env.sh — a test run is keyed by the fixture and engine env it reads: changing only such a variable
# runs the tests again instead of replaying a cached pass. A pack-tier test reads its fixture by getenv and skips without
# it, so a pass cached without the fixture would stand for a run with it. And a run reads the value this invocation sets:
# the build's configure is cached, so a run step never carries the environment of the configure that made it (the runner
# gets only the step's own variables, `--env=`, and keeps the live environment); a variable the step owns is refused by name.
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
ran() { grep -q "run test-host [0-9]* pass" "$TMP/out"; }

for var in QWEN4_TEST_MODEL DSV41_STQ_TINY SUSHI_MODELS_DIR; do
  run "$var=$TMP/one" || { echo "  FAIL: test-host with $var set"; FAIL=$((FAIL + 1)); continue; }
  run "$var=$TMP/two" || { echo "  FAIL: test-host with $var changed"; FAIL=$((FAIL + 1)); continue; }
  ran || { echo "  FAIL: changing only $var replayed a cached run"; sed 's/^/    /' "$TMP/out" | tail -4; FAIL=$((FAIL + 1)); }
done

# What the test process saw: a host test writes its SUSHI_ variables to SUSHI_TEST_ENV_DUMP (same -D args every time).
PROBE="test env: SUSHI_TEST_ENV_DUMP"
probe() { env "$1=$2" SUSHI_TEST_ENV_DUMP="$TMP/dump" "$ZIG" build test-host -Doptimize=ReleaseFast "-Dtest-filter=$PROBE" >"$TMP/out" 2>&1; }
for var in SUSHI_CONTRACT_FIXTURE SUSHI_PREFIX_CACHE_DIR SUSHI_MODELS_DIR; do
  for value in one two; do
    rm -f "$TMP/dump"
    probe "$var" "$value" || { echo "  FAIL: probe with $var=$value"; sed 's/^/    /' "$TMP/out" | tail -6; FAIL=$((FAIL + 1)); continue; }
    grep -qx "$var=$value" "$TMP/dump" 2>/dev/null || { echo "  FAIL: $var=$value set at this invocation did not reach the test (it saw: $(grep "^$var=" "$TMP/dump" 2>/dev/null || echo nothing))"; FAIL=$((FAIL + 1)); }
  done
done

# A variable the run step owns is refused by name, never overridden in silence.
for var in SUSHI_TEST_STEP SUSHI_TEST_LOG; do
  rm -f "$TMP/dump"
  if env "$var=elsewhere" SUSHI_TEST_ENV_DUMP="$TMP/dump" "$ZIG" build test-host -Doptimize=ReleaseFast "-Dtest-filter=$PROBE" -Dtest-log="$TMP/log" >"$TMP/out" 2>&1; then
    echo "  FAIL: $var set at this invocation was accepted"; FAIL=$((FAIL + 1))
  elif ! grep -q "$var is set to 'elsewhere' at this invocation" "$TMP/out"; then
    echo "  FAIL: the refusal of $var does not name it"; sed 's/^/    /' "$TMP/out" | tail -6; FAIL=$((FAIL + 1))
  fi
done

[ "$FAIL" -eq 0 ] || exit 1

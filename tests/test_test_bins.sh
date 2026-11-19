#!/bin/bash
# sushi-test: tier=host
# test_test_bins.sh — the exl3 tests run in one binary per area, `test-exl3-experts`, `-trunk` and `-format`, each
# holding the tests of the files build.zig lists for it, and `test-encoder` builds only when the filter selects one
# of its tests. One CPU test per binary, read back from -Dtest-log.
#
# Usage: ./tests/test_test_bins.sh

set -u
cd "$(dirname "$0")/.." || exit 1
ZIG="${ZIG:-$PWD/.zig-toolchain/zig}"
QUIET=$HOME/.sushi/runs/QUIET
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FAIL=0
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; sed 's/^/    /' "$TMP/out" | head -20; }

EXPERTS="exl3 codebook names resolve and an unknown name is null"
TRUNK="trunk gemm: only a billed call takes the decoded path, and only from the minimum rows and within the width cap"
FORMAT="exl3 layout: tile-major inverse returns the pack order for odd and even grids"
ENCODER="convert cli: defaults and every flag parse"

# steps <log>: `<step> <status> <name>` of each test line, sorted, the anonymous root tests left out.
steps() { awk -F'\t' '$3 != "DONE" && $5 !~ /\.test_[0-9]+$/ {print $2 " " $3 " " $5}' "$1" | sort; }

while [ -e "$QUIET" ]; do sleep 5; done
taskpolicy -a "$ZIG" build test -Doptimize=ReleaseFast "-Dtest-filter=$EXPERTS" "-Dtest-filter=$TRUNK" \
  "-Dtest-filter=$FORMAT" "-Dtest-log=$TMP/exl3.tsv" >"$TMP/out" 2>&1 || fail "the three exl3 filters run green"
want="test-exl3-experts PASS expert_exl3.test.$EXPERTS
test-exl3-format PASS layout.test.$FORMAT
test-exl3-trunk PASS trunk_gemm.test.$TRUNK"
[ "$(steps "$TMP/exl3.tsv")" = "$want" ] || fail "each exl3 test runs in its area's binary; got: $(steps "$TMP/exl3.tsv")"
grep -q $'\ttest-encoder\t' "$TMP/exl3.tsv" && fail "a filter that selects no encoder test builds no test-encoder"

while [ -e "$QUIET" ]; do sleep 5; done
taskpolicy -a "$ZIG" build test -Doptimize=ReleaseFast "-Dtest-filter=$ENCODER" "-Dtest-log=$TMP/enc.tsv" \
  >"$TMP/out" 2>&1 || fail "the encoder filter runs green"
[ "$(steps "$TMP/enc.tsv")" = "test-encoder PASS cli.test.$ENCODER" ] \
  || fail "an encoder filter runs its test in test-encoder alone; got: $(steps "$TMP/enc.tsv")"

[ "$FAIL" -eq 0 ] || { echo "FAIL"; exit 1; }
echo "PASS"

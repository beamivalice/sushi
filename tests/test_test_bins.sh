#!/bin/bash
# sushi-test: tier=host
# test_test_bins.sh — the MLX tests run in one binary per domain of build.zig's `test_domains` (`test-cooler`,
# `test-qwen`, ...), and a test file no domain holds fails the configure; the exl3 tests run in one binary per area,
# `test-exl3-experts`, `-trunk` and `-format`, each holding the tests of the files build.zig lists for it, and
# `test-encoder` builds only when the filter selects one of its tests. One test per binary, read back from -Dtest-log.
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

# The MLX tests run in one binary per domain of `test_domains`, each test in the step its file's domain names.
COOLER="cooler contract runner: SUSHI_CONTRACT_FIXTURE names the fixtures a run takes, and without it a missing fixture skips the test before any load"
DEEPSEEK="dsv41 forward cap: on MLX, covers what every measured peak added to the state it started from, with no multiplier"
QWEN="qwen4 convolution transform rejects a non-unit channel axis"
GLM="GLM config maps compressed MLA and FP32 recurrent state without Qwen assumptions"
MIMO="a model stamped as mimo2 is dispatched to mimo2"
KERNELS="lastDim reads the innermost size and refuses a null handle"
LOAD="EXL3 shard stamp rejects invalid and nonfinite rates"
SERVER="routeRequestModel: a path names its own entry, an unknown name the default"
PREFIX="HotPrefixCache: shouldUse gates hybrid by enable_ssm_checkpoints"
while [ -e "$QUIET" ]; do sleep 5; done
taskpolicy -a "$ZIG" build test -Doptimize=ReleaseFast "-Dtest-filter=$COOLER" "-Dtest-filter=$DEEPSEEK" \
  "-Dtest-filter=$QWEN" "-Dtest-filter=$GLM" "-Dtest-filter=$MIMO" "-Dtest-filter=$KERNELS" "-Dtest-filter=$LOAD" \
  "-Dtest-filter=$SERVER" "-Dtest-filter=$PREFIX" "-Dtest-log=$TMP/domains.tsv" >"$TMP/out" 2>&1 \
  || fail "one test of each domain runs green"
want="cooler-load PASS cooler_contract_test.test.$COOLER
test-deepseek PASS dsv41.dsv41_config.test.$DEEPSEEK
test-glm PASS glm5.glm5_config.test.$GLM
test-kernels PASS mlx_util.test.$KERNELS
test-load PASS pack_index.test.$LOAD
test-mimo PASS mimo2.arch.test.$MIMO
test-prefix-cache PASS prefix_cache.test.$PREFIX
test-qwen PASS qwen4.qwen4_config.test.$QWEN
test-server PASS http_conn.test.$SERVER"
[ "$(steps "$TMP/domains.tsv")" = "$want" ] || fail "each test runs in its domain's step; got: $(steps "$TMP/domains.tsv")"

# A test of `alone_tests` runs in a step of its own, out of its file's step.
ALONE="engine kernels a request's lengths pick by size: an elementwise op's and a matrix-vector product's first use before the seal makes every size's, so a long prompt compiles nothing"
while [ -e "$QUIET" ]; do sleep 5; done
taskpolicy -a "$ZIG" build test -Doptimize=ReleaseFast "-Dtest-filter=$ALONE" "-Dtest-filter=$KERNELS" \
  "-Dtest-log=$TMP/alone.tsv" >"$TMP/out" 2>&1 || fail "an alone test and a test of its domain run green"
want="kernels-late-by-size PASS tensor.route.test.$ALONE
test-kernels PASS mlx_util.test.$KERNELS"
[ "$(steps "$TMP/alone.tsv")" = "$want" ] || fail "an alone test runs in its own step; got: $(steps "$TMP/alone.tsv")"

# A test file that no domain holds fails the configure by name: a copy of the tree whose src/ gains one.
mkdir -p "$TMP/tree/src"
for f in * .zig-toolchain; do [ "$f" = src ] || ln -s "$PWD/$f" "$TMP/tree/$f"; done
for f in src/*; do ln -s "$PWD/$f" "$TMP/tree/$f"; done
printf '%s\n' 'test "a file of no domain" {}' > "$TMP/tree/src/zz_unlisted.zig"
while [ -e "$QUIET" ]; do sleep 5; done
if (cd "$TMP/tree" && "$ZIG" build -h >"$TMP/out" 2>&1); then
  fail "a test file no domain holds should fail the configure"
elif ! grep -q "src/zz_unlisted.zig holds tests that no test_domains entry holds" "$TMP/out"; then
  fail "the configure failure should name the unlisted file"
fi

[ "$FAIL" -eq 0 ] || { echo "FAIL"; exit 1; }
echo "PASS"

#!/bin/bash
# sushi-test: tier=hermetic
# tests/gate.sh's logic, with no real build: a throwaway repo holds a copy of gate.sh, the real fixture env file and
# a stub hermetic_guards.sh; a stub zig (its .zig-toolchain), a stub taskpolicy and stub python3s stand in.
#   - green: every step runs, under taskpolicy -a, in a clean worktree of the resolved commit (an untracked file of the
#     checkout is not there), each zig build with --cache-dir <gate dir>/zig-cache, the test step with the fixture envs
#     and SUSHI_TEST_TIMES; the record is written for that sha with the toolchain, the summary counts, each step's
#     minutes and the per-test times; the worktree is gone
#   - a failing step stops the gate, names the step and writes no record
#   - a missing fixture dir and a python3 without PyYAML each refuse by name before any step runs
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PASS=0
FAIL=0
ok() { # name  actual  expected
  if [ "$2" = "$3" ]; then PASS=$((PASS + 1)); echo "  PASS $1"
  else FAIL=$((FAIL + 1)); echo "  FAIL $1 — expected [$3], got [$2]"; fi
}
has() { # name  haystack  needle
  if printf '%s' "$2" | grep -qF -- "$3"; then PASS=$((PASS + 1)); echo "  PASS $1"
  else FAIL=$((FAIL + 1)); echo "  FAIL $1 — [$3] missing from: $(printf '%s' "$2" | tail -5)"; fi
}
g() { git -c user.name=t -c user.email=t@t "$@"; }
field() { plutil -extract "$2" raw -o - "$1" 2> /dev/null; }

REAL_PY="$(command -v python3)"
CALLS="$TMP/calls.log"

# ---- stubs ----
mkdir -p "$TMP/bin" "$TMP/py-yes" "$TMP/py-no"
cat > "$TMP/bin/taskpolicy" << 'EOF'
#!/bin/bash
[ "$1" = -a ] || { echo "stub taskpolicy: expected -a, got $1" >&2; exit 2; }
shift
echo "taskpolicy $1" >> "$STUB_CALLS"
exec "$@"
EOF
for v in yes no; do
  rc=0; [ "$v" = no ] && rc=1
  cat > "$TMP/py-$v/python3" << EOF
#!/bin/bash
case "\$*" in *"import yaml"*) exit $rc ;; esac
exec "$REAL_PY" "\$@"
EOF
done
chmod +x "$TMP/bin/taskpolicy" "$TMP/py-yes/python3" "$TMP/py-no/python3"

# ---- the repo the gate runs from ----
REPO="$TMP/repo"
g init -q "$REPO"
mkdir -p "$REPO/tests/lib" "$REPO/scripts" "$REPO/.zig-toolchain" "$REPO/lib/mlx"
cp "$ROOT/tests/gate.sh" "$REPO/tests/" 2> /dev/null
cp "$ROOT/tests/lib/fixture_envs.sh" "$REPO/tests/lib/" 2> /dev/null
echo '// reads SUSHI_TEST_TIMES' > "$REPO/scripts/test_runner.zig"
echo 'tracked, as the real lib/ is' > "$REPO/lib/stb_image.h"
cat > "$REPO/tests/hermetic_guards.sh" << 'EOF'
#!/bin/bash
echo "guards $PWD python3=$(command -v python3) untracked=$([ -e tests/untracked_marker ] && echo yes || echo no) zcache=${ZIG_LOCAL_CACHE_DIR:-none}" >> "$STUB_CALLS"
[ "${STUB_FAIL:-}" != guards ]
EOF
printf '.zig-toolchain\nlib/mlx\n' > "$REPO/.gitignore"
cat > "$REPO/.zig-toolchain/zig" << 'EOF'
#!/bin/bash
if [ "$1" = version ]; then echo 0.17.1; exit 0; fi
step=$2
echo "zig $step $PWD QWEN4_STQ_TINY=${QWEN4_STQ_TINY:-unset} DSV41_EPI_TINY=${DSV41_EPI_TINY:-unset} args=$*" >> "$STUB_CALLS"
[ "${STUB_FAIL:-}" = "$step" ] && { echo "error: stub $step failed"; exit 1; }
if [ "$step" = test ]; then
  printf '12\tfirst: a test\n345\tsecond: "quoted" test\n' >> "$SUSHI_TEST_TIMES"
  echo "Build Summary: 22/22 steps succeeded; 939/1001 tests passed (62 skipped)"
fi
exit 0
EOF
chmod +x "$REPO/.zig-toolchain/zig"
g -C "$REPO" add -A
g -C "$REPO" commit -qm one
echo two > "$REPO/two"
g -C "$REPO" add two
g -C "$REPO" commit -qm two
touch "$REPO/tests/untracked_marker"
SHA1="$(git -C "$REPO" rev-parse HEAD~1)"
SHA2="$(git -C "$REPO" rev-parse HEAD)"

# ---- the fixtures the env file names ----
export SUSHI_MODELS_DIR="$TMP/models"
FIXTURE_ENVS="$(
  # shellcheck source=/dev/null
  source "$REPO/tests/lib/fixture_envs.sh" 2> /dev/null && for v in $SUSHI_FIXTURE_ENVS; do echo "$v=${!v}"; done
)"
for kv in $FIXTURE_ENVS; do mkdir -p "${kv#*=}"; done
QWEN_TINY="$(printf '%s\n' $FIXTURE_ENVS | sed -n 's/^QWEN4_STQ_TINY=//p')"

GATE_DIR="$TMP/gate"
gate() { # [VAR=value...] -- [gate args]: runs the copied gate with the stubs; output in $OUT, exit in $RC
  local vars=()
  while [ "$1" != -- ]; do vars+=("$1"); shift; done
  shift
  : > "$CALLS"
  OUT="$(env PATH="$TMP/bin:$PATH" STUB_CALLS="$CALLS" SUSHI_GATE_DIR="$GATE_DIR" \
    SUSHI_GATE_PYTHON="$TMP/py-yes/python3" ${vars[@]+"${vars[@]}"} bash "$REPO/tests/gate.sh" "$@" 2>&1)"
  RC=$?
}
worktrees() { git -C "$REPO" worktree list | wc -l | tr -d ' '; }

echo "── green ──"
gate SUSHI_TEST_TIMES= -- HEAD~1
ok "a green gate exits 0" "$RC" "0"
REC="$GATE_DIR/$SHA1.json"
ok "the record is named for the resolved sha" "$([ -f "$REC" ] && echo yes || echo no)" "yes"
ok "lib/mlx sits beside the gate's zig cache, where a cached test binary's rpath finds libmlxc" "$(readlink "$GATE_DIR/lib/mlx")" "$(cd "$REPO/lib/mlx" && pwd -P)"
ok "  and none for another commit" "$([ -e "$GATE_DIR/$SHA2.json" ] && echo yes || echo no)" "no"
ok "the record holds the sha" "$(field "$REC" sha)" "$SHA1"
ok "the record holds the toolchain" "$(field "$REC" zig)" "0.17.1"
ok "the record holds the tests passed" "$(field "$REC" build_summary.tests_passed)" "939"
ok "  the tests total" "$(field "$REC" build_summary.tests_total)" "1001"
ok "  the skips" "$(field "$REC" build_summary.skipped)" "62"
ok "  the steps succeeded" "$(field "$REC" build_summary.steps_succeeded)" "22"
for s in lint guards ubench test; do
  ok "the record holds $s's minutes" "$(field "$REC" minutes.$s > /dev/null && echo yes || echo no)" "yes"
done
ok "the record holds the slowest test first" "$(field "$REC" test_times.0.test)" 'second: "quoted" test'
ok "  with its milliseconds" "$(field "$REC" test_times.0.ms)" "345"
has "the record holds a summary line" "$(field "$REC" summary)" "939/1001"
ZIGS="$(grep '^zig ' "$CALLS" | cut -d' ' -f2 | tr '\n' ' ')"
ok "the steps run in order" "$(grep -E '^(zig|guards) ' "$CALLS" | cut -d' ' -f1-2 | sed 's#^guards .*#guards#' | tr '\n' ',')" "zig lint,guards,zig ubench,zig test,"
ok "every step runs under taskpolicy -a" "$(grep -c '^taskpolicy ' "$CALLS")" "4"
WT_DIR="$(sed -n 's/^zig lint \([^ ]*\) .*/\1/p' "$CALLS")"
ok "the steps run outside the checkout" "$([ -n "$WT_DIR" ] && [ "$WT_DIR" != "$REPO" ] && echo yes || echo no)" "yes"
has "the guards run in a tree without the checkout's untracked files" "$(grep '^guards ' "$CALLS")" "untracked=no"
has "the guards run with the PyYAML python3 first on PATH" "$(grep '^guards ' "$CALLS")" "python3=$TMP/py-yes/python3"
has "the guards' own zig build keeps its cache in the gate dir" "$(grep '^guards ' "$CALLS")" "zcache=$GATE_DIR/zig-cache"
has "the test step sees the fixture envs" "$(grep '^zig test ' "$CALLS")" "QWEN4_STQ_TINY=$QWEN_TINY"
has "  DSV41_EPI_TINY among them" "$(grep '^zig test ' "$CALLS")" "DSV41_EPI_TINY=$TMP/models/test-fixtures/dsv41-epi-tiny"
ok "the record names DSV41_EPI_TINY's fixture" "$(field "$REC" fixtures.DSV41_EPI_TINY)" "$TMP/models/test-fixtures/dsv41-epi-tiny"
for s in lint ubench test; do
  has "zig build $s keeps its cache in the gate dir" "$(grep "^zig $s " "$CALLS")" "--cache-dir $GATE_DIR/zig-cache"
done
ok "the worktree is removed" "$(worktrees)" "1"
ok "  and its directory" "$([ -n "$WT_DIR" ] && [ -e "$WT_DIR" ] && echo yes || echo no)" "no"

echo "── a failing step ──"
rm -rf "$GATE_DIR"
gate STUB_FAIL=ubench -- HEAD
ok "a failed step exits non-zero" "$([ "$RC" -ne 0 ] && echo yes || echo no)" "yes"
has "  naming the step" "$OUT" "step 'ubench' failed"
ok "  writing no record" "$(ls "$GATE_DIR"/*.json 2> /dev/null | wc -l | tr -d ' ')" "0"
ok "  and running no later step" "$(grep -c '^zig test ' "$CALLS")" "0"
ok "  and removing its worktree" "$(worktrees)" "1"

echo "── refusals ──"
gate SUSHI_GATE_PYTHON="$TMP/py-no/python3" -- HEAD
ok "a python3 without PyYAML refuses" "$([ "$RC" -ne 0 ] && echo yes || echo no)" "yes"
has "  naming PyYAML" "$OUT" "PyYAML"
ok "  before any step" "$(grep -c -E '^(zig|guards) ' "$CALLS")" "0"
ok "  writing no record" "$(ls "$GATE_DIR"/*.json 2> /dev/null | wc -l | tr -d ' ')" "0"

PM="$(printf '%s\n' $FIXTURE_ENVS | sed -n 's/^DSV41_STQ_TINY_PM=//p')"
rm -rf "$PM"
gate -- HEAD
ok "a missing fixture dir refuses" "$([ "$RC" -ne 0 ] && echo yes || echo no)" "yes"
has "  naming its env" "$OUT" "DSV41_STQ_TINY_PM"
has "  and its path" "$OUT" "$PM"
ok "  before any step" "$(grep -c -E '^(zig|guards) ' "$CALLS")" "0"
ok "  writing no record" "$(ls "$GATE_DIR"/*.json 2> /dev/null | wc -l | tr -d ' ')" "0"
ok "  and removing its worktree" "$(worktrees)" "1"
mkdir -p "$PM"

gate -- no-such-ref
ok "a ref that names no commit refuses" "$([ "$RC" -ne 0 ] && echo yes || echo no)" "yes"
has "  naming it" "$OUT" "no-such-ref"

echo
TOTAL=$((PASS + FAIL))
if [ "$FAIL" -eq 0 ]; then echo "PASS $TOTAL/$TOTAL"; exit 0
else echo "FAIL $FAIL/$TOTAL"; exit 1; fi

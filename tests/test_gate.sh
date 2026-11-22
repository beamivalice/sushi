#!/bin/bash
# sushi-test: tier=hermetic
# tests/gate.sh's logic, with no real build: a throwaway repo holds a copy of gate.sh, the real fixture env file and
# a stub hermetic_guards.sh; a stub zig (its .zig-toolchain), a stub taskpolicy and stub python3s stand in.
#   - green: every step runs, under taskpolicy -a, in a clean worktree of the resolved commit (an untracked file of the
#     checkout is not there), each zig build with --cache-dir <gate dir>/zig-cache, the guards beside the test step,
#     the test step with the fixture envs, -Dtest-log and --maxrss of the box's budget row; each test's log line shows
#     as it lands, then the per-step table and the slowest tests; the record is written for that sha with the
#     toolchain, the summary counts, each step's minutes and the per-test log; the worktree is gone
#   - a failing step stops the gate, names the step and writes no record; a failed test step repeats each failed test
#     last with its error block, and the guards still running beside it are stopped
#   - a held GPU lock, a box with no budget row, a missing fixture dir, a python3 without PyYAML and a commit whose
#     build.zig has no -Dtest-log each refuse by name before any step runs
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
cat > "$TMP/bin/sysctl" << 'EOF'
#!/bin/bash
[ "$*" = "-n hw.memsize" ] || exec /usr/sbin/sysctl "$@"
echo "${STUB_MEMSIZE:-137438953472}"
EOF
for v in yes no; do
  rc=0; [ "$v" = no ] && rc=1
  cat > "$TMP/py-$v/python3" << EOF
#!/bin/bash
case "\$*" in *"import yaml"*) exit $rc ;; esac
exec "$REAL_PY" "\$@"
EOF
done
chmod +x "$TMP/bin/taskpolicy" "$TMP/bin/sysctl" "$TMP/py-yes/python3" "$TMP/py-no/python3"

# ---- the repo the gate runs from ----
REPO="$TMP/repo"
g init -q "$REPO"
mkdir -p "$REPO/tests/lib" "$REPO/scripts" "$REPO/.zig-toolchain" "$REPO/lib/mlx"
cp "$ROOT/tests/gate.sh" "$REPO/tests/" 2> /dev/null
cp "$ROOT/tests/lib/fixture_envs.sh" "$ROOT/tests/lib/test_log.py" "$ROOT/tests/lib/test_keys.py" "$REPO/tests/lib/" 2> /dev/null
cp "$ROOT/scripts/gpu-lock.sh" "$REPO/scripts/" 2> /dev/null
printf '%s\n' 'const test_log = b.option([]const u8, "test-log", "");' 'const cut = b.option(bool, "fixture-cut", "");' \
  'const skip = b.option([]const []const u8, "test-skip-step", "");' > "$REPO/build.zig"
echo 'tracked, as the real lib/ is' > "$REPO/lib/stb_image.h"
# The guards wait for the test step to start: run before it instead of beside it, they find it never did.
cat > "$REPO/tests/hermetic_guards.sh" << 'EOF'
#!/bin/bash
echo $$ > "$STUB_CALLS.guards.pid"
for _ in $(seq 50); do [ -e "$STUB_CALLS.test-started" ] && break; sleep 0.1; done
beside=no; [ -e "$STUB_CALLS.test-started" ] && beside=yes
echo "guards $PWD python3=$(command -v python3) untracked=$([ -e tests/untracked_marker ] && echo yes || echo no) zcache=${ZIG_LOCAL_CACHE_DIR:-none} beside=$beside" >> "$STUB_CALLS"
sleep "${STUB_GUARDS_SLEEP:-0}"
[ "${STUB_FAIL:-}" != guards ]
EOF
printf '.zig-toolchain\nlib/mlx\n' > "$REPO/.gitignore"
cat > "$REPO/.zig-toolchain/zig" << 'EOF'
#!/bin/bash
if [ "$1" = version ]; then echo 0.17.1; exit 0; fi
step=$2
echo "zig $step $PWD QWEN4_STQ_TINY=${QWEN4_STQ_TINY:-unset} DSV41_EPI_TINY=${DSV41_EPI_TINY:-unset} args=$*" >> "$STUB_CALLS"
[ "${STUB_FAIL:-}" = "$step" ] && { echo "error: stub $step failed"; exit 1; }
# test-build + test-plan: the two binaries (their bytes from STUB_BIN_<name>) and the plan beside them.
if [ "$step" = test-build ]; then
  mkdir -p zig-out/tests
  printf '%s' "${STUB_BIN_HOST:-host v1}" > zig-out/tests/test-host
  printf '%s' "${STUB_BIN_REST:-rest v1}" > zig-out/tests/test-rest
  printf '%s' '[{"step":"test-host","binary":"tests/test-host","env":[["SUSHI_TEST_STEP","test-host"]]},' \
    '{"step":"test-rest","binary":"tests/test-rest","env":[["SUSHI_TEST_STEP","test-rest"]]}]' > zig-out/test-plan.json
fi
if [ "$step" = test ]; then
  touch "$STUB_CALLS.test-started"
  log=; skip=" "
  for a in "$@"; do case "$a" in -Dtest-log=*) log=${a#-Dtest-log=} ;; -Dtest-skip-step=*) skip+="${a#-Dtest-skip-step=} " ;; esac; done
  t=2026-11-19T07:13:02.000Z
  # Like zig, the summary counts only the run steps this invocation made, and has no tests part when it made none.
  p=0; sk=0
  if [ "$skip" = "${skip/ test-host /}" ]; then
    p=$((p + 1))
    printf '%s\ttest-host\tPASS\t12\tfirst: a test\n%s\ttest-host\tDONE\t12\t1 pass, 0 skip, 0 fail\n' $t $t >> "$log"
  fi
  if [ "$skip" = "${skip/ test-rest /}" ]; then
    p=$((p + 1)); sk=$((sk + 1))
    printf '%s\ttest-rest\tPASS\t345\tsecond: "quoted" test\n%s\ttest-rest\tSKIP\t0\tthird: a skip\n' $t $t >> "$log"
    if [ "${STUB_FAIL:-}" = a-test ]; then
      printf '%s\ttest-rest\tFAIL\t7\tfourth: a failure\n' $t >> "$log"
      printf "error: 'fourth: a failure' failed:\n       expected 1, found 2\nfailed command: test-rest --listen=-\n"
    fi
    printf '%s\ttest-rest\tDONE\t352\t1 pass, 1 skip, 0 fail\n' $t >> "$log"
  fi
  [ "${STUB_FAIL:-}" = a-test ] && exit 1
  if [ $((p + sk)) -eq 0 ]; then echo "Build Summary: 22/22 steps succeeded"
  else echo "Build Summary: 22/22 steps succeeded; $p/$((p + sk)) tests passed ($sk skipped)"; fi
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
  rm -f "$CALLS".*
  OUT="$(env PATH="$TMP/bin:$PATH" STUB_CALLS="$CALLS" SUSHI_GATE_DIR="$GATE_DIR" GPU_LOCK_DIR="$TMP/gpu.lock.d" \
    GPU_LOCK_QUIET="$TMP/QUIET" SUSHI_GATE_PYTHON="$TMP/py-yes/python3" ${vars[@]+"${vars[@]}"} \
    bash "$REPO/tests/gate.sh" "$@" 2>&1)"
  RC=$?
}
worktrees() { git -C "$REPO" worktree list | wc -l | tr -d ' '; }

echo "── green ──"
gate -- HEAD~1
ok "a green gate exits 0" "$RC" "0"
REC="$GATE_DIR/$SHA1.json"
ok "the record is named for the resolved sha" "$([ -f "$REC" ] && echo yes || echo no)" "yes"
ok "lib/mlx sits beside the gate's zig cache, where a cached test binary's rpath finds libmlxc" "$(readlink "$GATE_DIR/lib/mlx")" "$(cd "$REPO/lib/mlx" && pwd -P)"
ok "  and none for another commit" "$([ -e "$GATE_DIR/$SHA2.json" ] && echo yes || echo no)" "no"
ok "the record holds the sha" "$(field "$REC" sha)" "$SHA1"
ok "the record holds the toolchain" "$(field "$REC" zig)" "0.17.1"
ok "the record holds the tests passed" "$(field "$REC" build_summary.tests_passed)" "2"
ok "  the tests total" "$(field "$REC" build_summary.tests_total)" "3"
ok "  the skips" "$(field "$REC" build_summary.skipped)" "1"
ok "  the steps succeeded" "$(field "$REC" build_summary.steps_succeeded)" "22"
for s in lint guards ubench test; do
  ok "the record holds $s's minutes" "$(field "$REC" minutes.$s > /dev/null && echo yes || echo no)" "yes"
done
ok "the record holds the slowest test first" "$(field "$REC" test_times.0.test)" 'second: "quoted" test'
ok "  with its milliseconds" "$(field "$REC" test_times.0.ms)" "345"
ok "  its step" "$(field "$REC" test_times.0.step)" "test-rest"
ok "  and its status" "$(field "$REC" test_times.0.status)" "PASS"
ok "  and no DONE line among the tests" "$(field "$REC" test_times.3 > /dev/null && echo yes || echo no)" "no"
has "each test's line shows as it lands" "$OUT" 'test-rest                                    PASS     0.3 s  second: "quoted" test'
has "  and each step's DONE" "$OUT" "1 pass, 1 skip, 0 fail"
has "the end summary holds the per-step table" "$OUT" "  test-rest                                         2      1     1     0     0.4 s"
has "  and the slowest tests" "$OUT" '      0.3 s  test-rest                                    second: "quoted" test'
has "the test step logs to the gate's logs dir" "$(grep '^zig test ' "$CALLS")" "-Dtest-log=$GATE_DIR/$SHA1.logs/tests.tsv"
has "the test step runs under the M5 row's 72 GiB" "$(grep '^zig test ' "$CALLS")" "--maxrss 77309411328"
has "the gate names its budget" "$OUT" "test budget 72 GiB (hw.memsize 128 GiB)"
has "the test step declares the Qwen cut that the fixture envs name" "$(grep '^zig test ' "$CALLS")" "-Dfixture-cut=true"
ok "  and keeps zig's default job count on the M5 row" "$(grep '^zig test ' "$CALLS" | grep -c -- ' -j')" "0"
has "the record holds a summary line" "$(field "$REC" summary)" "2/3 tests passed"
ZIGS="$(grep '^zig ' "$CALLS" | cut -d' ' -f2 | tr '\n' ' ')"
ok "lint, then ubench, then the test binaries and plan, then the test step" "$(grep -E '^zig ' "$CALLS" | cut -d' ' -f1-2 | tr '\n' ',')" "zig lint,zig ubench,zig test-build,zig test,"
has "  the binaries and plan built with the test step's flags" "$(grep '^zig test-build ' "$CALLS")" "args=build test-build test-plan -Doptimize=ReleaseFast"
ok "the gate keeps each run step's key" "$(cut -f1 "$GATE_DIR/$SHA1.logs/step-keys.tsv" 2> /dev/null | tr '\n' ',')" "test-host,test-rest,"
ok "the steps run in the gate dir's one worktree path, so a rebuild of the same code is the same binary" "$(sed -n 's/^zig lint \([^ ]*\) .*/\1/p' "$CALLS")" "$GATE_DIR/worktree"
has "the guards run beside the test step" "$(grep '^guards ' "$CALLS")" "beside=yes"
ok "every step runs under taskpolicy -a" "$(grep -c '^taskpolicy ' "$CALLS")" "5"
WT_DIR="$(sed -n 's/^zig lint \([^ ]*\) .*/\1/p' "$CALLS")"
ok "the steps run outside the checkout" "$([ -n "$WT_DIR" ] && [ "$WT_DIR" != "$REPO" ] && echo yes || echo no)" "yes"
has "the guards run in a tree without the checkout's untracked files" "$(grep '^guards ' "$CALLS")" "untracked=no"
has "the guards run with the PyYAML python3 first on PATH" "$(grep '^guards ' "$CALLS")" "python3=$TMP/py-yes/python3"
has "the guards' own zig build keeps its cache in the gate dir" "$(grep '^guards ' "$CALLS")" "zcache=$GATE_DIR/zig-cache"
has "the test step sees the fixture envs" "$(grep '^zig test ' "$CALLS")" "QWEN4_STQ_TINY=$QWEN_TINY"
has "  DSV41_EPI_TINY among them" "$(grep '^zig test ' "$CALLS")" "DSV41_EPI_TINY=$TMP/models/tiny-models/dsv41-epi-tiny"
ok "the record names DSV41_EPI_TINY's fixture" "$(field "$REC" fixtures.DSV41_EPI_TINY)" "$TMP/models/tiny-models/dsv41-epi-tiny"
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

gate STUB_FAIL=a-test STUB_GUARDS_SLEEP=30 -- HEAD
ok "a failed test step exits non-zero" "$([ "$RC" -ne 0 ] && echo yes || echo no)" "yes"
has "  naming the step" "$OUT" "step 'test' failed"
has "  repeating the failed test last" "$OUT" "  FAIL [test-rest] fourth: a failure (0.0 s)"
has "  with its error block" "$OUT" "      expected 1, found 2"
ok "  writing no record" "$(ls "$GATE_DIR"/*.json 2> /dev/null | wc -l | tr -d ' ')" "0"
ok "  stopping the guards still running beside it" "$(kill -0 "$(cat "$CALLS.guards.pid")" 2> /dev/null && echo alive || echo gone)" "gone"
ok "  and removing its worktree" "$(worktrees)" "1"

gate STUB_FAIL=guards -- HEAD
ok "failed guards fail the gate" "$([ "$RC" -ne 0 ] && echo yes || echo no)" "yes"
has "  naming them" "$OUT" "step 'guards' failed"
ok "  writing no record" "$(ls "$GATE_DIR"/*.json 2> /dev/null | wc -l | tr -d ' ')" "0"

echo "── the test budget ──"
gate STUB_MEMSIZE=34359738368 -- HEAD
has "a 32 GiB box takes the M1 row's 20 GiB" "$(grep '^zig test ' "$CALLS")" "--maxrss 21474836480"
has "  and its 8 jobs" "$(grep '^zig test ' "$CALLS")" " -j8 "
gate STUB_MEMSIZE=137438953472 -- --test-budget-gib 30 HEAD
has "--test-budget-gib sets the budget" "$(grep '^zig test ' "$CALLS")" "--maxrss 32212254720"
gate STUB_MEMSIZE=17179869184 -- HEAD
ok "a box under every row refuses" "$([ "$RC" -ne 0 ] && echo yes || echo no)" "yes"
has "  naming --test-budget-gib" "$OUT" "--test-budget-gib"
ok "  before any step" "$(grep -c -E '^(zig|guards) ' "$CALLS")" "0"
gate -- --test-budget-gib abc HEAD
ok "a budget that is not a whole number of GiB refuses" "$([ "$RC" -ne 0 ] && echo yes || echo no)" "yes"
has "  naming it" "$OUT" "--test-budget-gib abc"
gate -- --no-such-flag HEAD
ok "an unknown flag refuses" "$([ "$RC" -ne 0 ] && echo yes || echo no)" "yes"
has "  naming it" "$OUT" "--no-such-flag"

echo "── a stopped gate's worktree ──"
git -C "$REPO" worktree add --quiet --detach "$GATE_DIR/worktree" HEAD
sleep 0 & DEAD=$!
wait "$DEAD"
echo "$DEAD" > "$GATE_DIR/worktree.pid"
gate -- HEAD
ok "a worktree whose gate is gone is removed and the gate runs" "$RC" "0"
has "  saying so" "$OUT" "gate: removed the worktree of a stopped gate (pid $DEAD)"
ok "  and leaves none behind" "$(worktrees)" "1"
git -C "$REPO" worktree add --quiet --detach "$GATE_DIR/worktree" HEAD
rm -f "$GATE_DIR/worktree.pid"
gate -- HEAD
ok "a worktree with no gate pid is a stopped gate's too" "$RC" "0"
git -C "$REPO" worktree add --quiet --detach "$GATE_DIR/worktree" HEAD
printf '#!/bin/bash\nsleep 30\n' > "$TMP/live-gate.sh"
bash "$TMP/live-gate.sh" & LIVE=$!
echo "$LIVE" > "$GATE_DIR/worktree.pid"
gate -- HEAD
ok "a worktree a running gate holds refuses" "$([ "$RC" -ne 0 ] && echo yes || echo no)" "yes"
has "  naming its pid" "$OUT" "gate pid $LIVE"
ok "  before any step" "$(grep -c -E '^(zig|guards) ' "$CALLS")" "0"
ok "  and leaves its worktree alone" "$([ -d "$GATE_DIR/worktree" ] && echo yes || echo no)" "yes"
kill "$LIVE" 2> /dev/null
wait "$LIVE" 2> /dev/null
git -C "$REPO" worktree remove --force "$GATE_DIR/worktree"
rm -f "$GATE_DIR/worktree.pid"

echo "── continue ──"
rm -rf "$GATE_DIR"
gate -- HEAD
PRIOR="$SHA2"
mkdir -p "$REPO/src"
echo 'pub const a = 1;' > "$REPO/src/a.zig"
g -C "$REPO" add src/a.zig
g -C "$REPO" commit -qm "a zig-only change"
SHA3="$(git -C "$REPO" rev-parse HEAD)"
skips() { grep '^zig test ' "$CALLS" | tr ' ' '\n' | sed -n 's/^-Dtest-skip-step=//p' | tr '\n' ','; }
gate -- --continue "$PRIOR" HEAD
ok "a continued gate passes" "$RC" "0"
ok "  leaving out every step whose binary and inputs are the prior gate's" "$(skips)" "test-host,test-rest,"
ok "  with their prior lines in its log" "$(grep -c $'\ttest-rest\t' "$GATE_DIR/$SHA3.logs/tests.tsv")" "3"
ok "  the record names the gate it reused" "$(field "$GATE_DIR/$SHA3.json" reused.from)" "$PRIOR"
ok "  and the steps" "$(field "$GATE_DIR/$SHA3.json" reused.steps.1)" "test-rest"
has "  and says so" "$OUT" "gate: reused from $PRIOR: test-host test-rest"
CREC="$GATE_DIR/$SHA3.json"
ok "  the record counts the reused tests, zig having run none" "$(field "$CREC" summary | grep -c '2/3 tests passed (1 skipped)')" "1"
ok "  the passes" "$(field "$CREC" build_summary.tests_passed)" "2"
ok "  the total" "$(field "$CREC" build_summary.tests_total)" "3"
ok "  the skips" "$(field "$CREC" build_summary.skipped)" "1"
ok "  lint and the guards still run" "$(grep -c -E '^(zig lint|guards) ' "$CALLS")" "2"
gate STUB_BIN_REST="rest v2" -- --continue "$PRIOR" HEAD
ok "a step whose binary changed runs again" "$(skips)" "test-host,"
ok "  and the record lists only the reused one" "$(field "$GATE_DIR/$SHA3.json" reused.steps.1 > /dev/null && echo two || echo one)" "one"
ok "  and counts the rerun step's tests with the reused one's" "$(field "$GATE_DIR/$SHA3.json" summary | grep -c '2/3 tests passed (1 skipped)')" "1"
gate STUB_FAIL=a-test -- HEAD
echo 'pub const a = 2;' > "$REPO/src/a.zig"
g -C "$REPO" commit -qam "another zig-only change"
gate -- --continue "$SHA3" HEAD
ok "a step that failed in the prior gate runs again" "$(skips)" "test-host,"
SHA4="$(git -C "$REPO" rev-parse HEAD)"
gate -- --continue "$SHA4" HEAD
ok "a continue onto the same commit reuses its own last run" "$(skips)" "test-host,test-rest,"
echo changed >> "$QWEN_TINY/weights"
gate -- --continue "$SHA4" HEAD
ok "a fixture whose files changed reruns every step" "$(skips)" ""
echo data > "$REPO/data.txt"
g -C "$REPO" add data.txt
g -C "$REPO" commit -qm "a tracked data file"
gate -- --continue "$SHA4" HEAD
ok "a tracked file that is not Zig or docs reruns every step" "$(skips)" ""
gate -- --continue no-such-ref HEAD
ok "a --continue ref that names no commit refuses" "$([ "$RC" -ne 0 ] && echo yes || echo no)" "yes"
has "  naming it" "$OUT" "no-such-ref"
gate -- --continue "$SHA1" HEAD
ok "a prior gate with no step keys refuses" "$([ "$RC" -ne 0 ] && echo yes || echo no)" "yes"
has "  naming the sha" "$OUT" "$SHA1"
ok "  before any step" "$(grep -c -E '^(zig|guards) ' "$CALLS")" "0"
gate -- --continue
ok "--continue with no ref refuses" "$([ "$RC" -ne 0 ] && echo yes || echo no)" "yes"

echo "── refusals ──"
rm -rf "$GATE_DIR"
mkdir -p "$TMP/gpu.lock.d"
echo "bench-agent" > "$TMP/gpu.lock.d/owner"
gate -- HEAD
ok "a held GPU lock refuses" "$([ "$RC" -ne 0 ] && echo yes || echo no)" "yes"
has "  naming its holder" "$OUT" "bench-agent"
ok "  before any step" "$(grep -c -E '^(zig|guards) ' "$CALLS")" "0"
ok "  leaving no worktree" "$(worktrees)" "1"
rm -rf "$TMP/gpu.lock.d"

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

mv "$TMP/models/tiny-models" "$TMP/models/test-fixtures"
gate -- HEAD
ok "tiny models still in the old test-fixtures folder refuse" "$([ "$RC" -ne 0 ] && echo yes || echo no)" "yes"
has "  naming the mv" "$OUT" "mv \"$TMP/models/test-fixtures\" \"$TMP/models/tiny-models\""
ok "  before any step" "$(grep -c -E '^(zig|guards) ' "$CALLS")" "0"
mv "$TMP/models/test-fixtures" "$TMP/models/tiny-models"

gate -- no-such-ref
ok "a ref that names no commit refuses" "$([ "$RC" -ne 0 ] && echo yes || echo no)" "yes"
has "  naming it" "$OUT" "no-such-ref"

echo 'const test_log = b.option([]const u8, "test-log", "");' > "$REPO/build.zig"
g -C "$REPO" commit -qam "three"
gate -- HEAD
ok "a commit whose build.zig has no -Dfixture-cut refuses" "$([ "$RC" -ne 0 ] && echo yes || echo no)" "yes"
has "  naming it" "$OUT" "-Dfixture-cut"
ok "  before any step" "$(grep -c -E '^(zig|guards) ' "$CALLS")" "0"

echo 'no test log here' > "$REPO/build.zig"
g -C "$REPO" commit -qam "four"
gate -- HEAD
ok "a commit whose build.zig has no -Dtest-log refuses" "$([ "$RC" -ne 0 ] && echo yes || echo no)" "yes"
has "  naming it" "$OUT" "-Dtest-log"
ok "  before any step" "$(grep -c -E '^(zig|guards) ' "$CALLS")" "0"
ok "  and removing its worktree" "$(worktrees)" "1"

echo
TOTAL=$((PASS + FAIL))
if [ "$FAIL" -eq 0 ]; then echo "PASS $TOTAL/$TOTAL"; exit 0
else echo "FAIL $FAIL/$TOTAL"; exit 1; fi

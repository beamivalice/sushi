#!/bin/bash
# sushi-test: tier=tool
# The local release gate: every check a local run can prove, run once on one commit in a clean worktree of it.
#   tests/gate.sh [--test-budget-gib N] [<ref>]
#                             the ref as this repo has it (default origin/main: fetch first); N is the test step's
#                             memory budget (zig build --maxrss), by default the box's row: 72 GiB on 96 GiB or more
#                             (the M5 Max), 20 GiB and -j8 on 32 GiB or more (the M1 Max); a smaller box names N itself
# It refuses while the GPU lock is held: a full model load owns the box, and the suite's fixture loads would join it.
# Steps, each under `taskpolicy -a`, stopping at the first failure: zig build lint, zig build ubench, then
# zig build test -Doptimize=ReleaseFast --summary all with the tiny-fixture envs of tests/lib/fixture_envs.sh,
# -Dtest-log and, when they name the Qwen cut, -Dfixture-cut=true, with tests/hermetic_guards.sh running beside it. Each test's log line shows as it lands; at the end the
# per-step table, the 10 slowest tests and, last, every failed test with its error block (tests/lib/test_log.py).
# On success it writes ${SUSHI_GATE_DIR:-$HOME/.sushi/runs/gate}/<sha>.json, the record ./release.sh requires; each
# step's log and the per-test log (tests.tsv) stay in <sha>.logs/ beside it. The worktree is removed either way.
# Every zig build keeps its local cache in <gate dir>/zig-cache, so a fresh worktree is not a cold build; after a
# toolchain or SDK change remove it (configure-time output is cached).
# The worktree borrows this checkout's .zig-toolchain and lib/mlx (symlinks). SUSHI_GATE_PYTHON names the python3
# the hermetic guards run with (default: python3 on PATH); it must have PyYAML, since the guards fail on its skip.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE_DIR="${SUSHI_GATE_DIR:-$HOME/.sushi/runs/gate}"

die() { echo "gate.sh: $*" >&2; exit 1; }

REF=""
BUDGET_GIB=""
while [ $# -gt 0 ]; do
  case "$1" in
    --test-budget-gib)
      [[ "${2:-}" =~ ^[1-9][0-9]*$ ]] || die "refused: --test-budget-gib ${2:-} is not a whole number of GiB"
      BUDGET_GIB=$2
      shift 2
      ;;
    -*) die "refused: unknown flag $1" ;;
    *)
      [ -z "$REF" ] || die "refused: a second ref '$1' (one ref is gated at a time)"
      REF=$1
      shift
      ;;
  esac
done
REF="${REF:-origin/main}"

SHA="$(git -C "$ROOT" rev-parse --verify --quiet "$REF^{commit}")" || die "refused: '$REF' names no commit in $ROOT"
for d in .zig-toolchain lib/mlx; do
  [ -e "$ROOT/$d" ] || die "refused: $ROOT/$d is missing; the gate's worktree borrows it"
done
PY="$(command -v "${SUSHI_GATE_PYTHON:-python3}" || true)"
[ -n "$PY" ] || die "refused: no python3 '${SUSHI_GATE_PYTHON:-python3}' to run the hermetic guards with"
"$PY" -c 'import yaml' 2> /dev/null \
  || die "refused: $PY has no PyYAML, and the hermetic guards fail on its skip; set SUSHI_GATE_PYTHON to a python3 that has it"
LOCK="$("$ROOT/scripts/gpu-lock.sh" status 2> /dev/null | sed -n 1p)" || LOCK=""
[ "$LOCK" = free ] || die "refused: the GPU lock is held (${LOCK:-status unreadable}): a full model load is running"

MEM_GIB=$(($(sysctl -n hw.memsize) >> 30))
BUDGET_FROM="hw.memsize $MEM_GIB GiB"
JOBS=()
if [ -n "$BUDGET_GIB" ]; then
  BUDGET_FROM="--test-budget-gib; $BUDGET_FROM"
elif [ "$MEM_GIB" -ge 96 ]; then
  BUDGET_GIB=72
elif [ "$MEM_GIB" -ge 32 ]; then
  BUDGET_GIB=20
  JOBS=(-j8)
else
  die "refused: no test budget row for a $MEM_GIB GiB box; name one with --test-budget-gib N"
fi
echo "gate: test budget $BUDGET_GIB GiB ($BUDGET_FROM)${JOBS[*]+, ${JOBS[*]}}"

WT="$(mktemp -d "${TMPDIR:-/tmp}/sushi-gate.XXXXXX")"
GUARDS_PID=""
FOLLOW_PID=""
cleanup() {
  [ -z "$FOLLOW_PID" ] || kill "$FOLLOW_PID" 2> /dev/null || true
  if [ -n "$GUARDS_PID" ]; then
    kill -TERM -- "-$GUARDS_PID" 2> /dev/null || true
    wait "$GUARDS_PID" 2> /dev/null || true
  fi
  git -C "$ROOT" worktree remove --force "$WT" 2> /dev/null || rm -rf "$WT"
}
trap cleanup EXIT
git -C "$ROOT" worktree add --quiet --detach "$WT" "$SHA"
ln -s "$(cd "$ROOT/.zig-toolchain" && pwd -P)" "$WT/.zig-toolchain"
ln -s "$(cd "$ROOT/lib/mlx" && pwd -P)" "$WT/lib/mlx"
cd "$WT"

[ -f tests/lib/fixture_envs.sh ] || die "refused: $SHA has no tests/lib/fixture_envs.sh"
grep -q '"test-log"' build.zig || die "refused: $SHA's build.zig has no -Dtest-log, which the gate's per-test log needs"
grep -q '"fixture-cut"' build.zig || die "refused: $SHA's build.zig has no -Dfixture-cut, which declares the Qwen cut's memory"
# shellcheck source=lib/fixture_envs.sh
source tests/lib/fixture_envs.sh || die "refused: tests/lib/fixture_envs.sh failed (above)"
for v in $SUSHI_FIXTURE_ENVS; do
  [ -d "${!v}" ] || die "refused: fixture $v=${!v} is not a folder"
done
CUT=()
[ -z "${QWEN4_TEST_MODEL:-}" ] || CUT=(-Dfixture-cut=true)

ZIG_CACHE="$GATE_DIR/zig-cache"
# A test binary in <cache>/o/<hash>/ finds libmlxc through its rpath @loader_path/../../../lib/mlx/lib.
mkdir -p "$GATE_DIR/lib"
ln -sfn "$(cd "$ROOT/lib/mlx" && pwd -P)" "$GATE_DIR/lib/mlx"
LOGS="$GATE_DIR/$SHA.logs"
rm -rf "$LOGS"
mkdir -p "$LOGS"
export PATH="$WT/.zig-toolchain:$PATH"
STEP_SECONDS=""

# run_step <name> <cmd...>: under taskpolicy -a, output in $LOGS/<name>.log; a failure ends the gate naming the step.
run_step() {
  local name=$1 t0 rc=0
  shift
  echo "gate: $name"
  t0=$(date +%s)
  taskpolicy -a "$@" > "$LOGS/$name.log" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    tail -n 40 "$LOGS/$name.log" >&2
    die "step '$name' failed (exit $rc) on $SHA; log: $LOGS/$name.log; no record written"
  fi
  STEP_SECONDS="$STEP_SECONDS $name=$(($(date +%s) - t0))"
}

# The guards run beside the test step, in a process group of their own (set -m), which a failure stops whole.
start_guards() {
  echo "gate: guards (beside the test step)"
  set -m
  (
    rc=0
    t0=$(date +%s)
    taskpolicy -a env PATH="$(dirname "$PY"):$PATH" ZIG_LOCAL_CACHE_DIR="$ZIG_CACHE" bash tests/hermetic_guards.sh \
      > "$LOGS/guards.log" 2>&1 || rc=$?
    echo "$rc $(($(date +%s) - t0))" > "$LOGS/guards.rc"
  ) &
  GUARDS_PID=$!
  set +m
}

join_guards() {
  local rc secs
  wait "$GUARDS_PID" || true
  GUARDS_PID=""
  read -r rc secs < "$LOGS/guards.rc"
  if [ "$rc" -ne 0 ]; then
    tail -n 40 "$LOGS/guards.log" >&2
    die "step 'guards' failed (exit $rc) on $SHA; log: $LOGS/guards.log; no record written"
  fi
  STEP_SECONDS="$STEP_SECONDS guards=$secs"
}

# The test step, its log lines shown as they land, then the summary; a failure ends the gate naming the step.
TEST_LOG="$LOGS/tests.tsv"
run_tests() {
  local t0 rc=0
  echo "gate: test"
  "$PY" -I "$ROOT/tests/lib/test_log.py" follow "$TEST_LOG" "$LOGS/tests.done" &
  FOLLOW_PID=$!
  t0=$(date +%s)
  taskpolicy -a zig build test -Doptimize=ReleaseFast --summary all --cache-dir "$ZIG_CACHE" \
    --maxrss $((BUDGET_GIB << 30)) ${JOBS[@]+"${JOBS[@]}"} ${CUT[@]+"${CUT[@]}"} "-Dtest-log=$TEST_LOG" \
    > "$LOGS/test.log" 2>&1 || rc=$?
  touch "$LOGS/tests.done"
  wait "$FOLLOW_PID" || true
  FOLLOW_PID=""
  local failed=()
  [ "$rc" -eq 0 ] || failed=(--step-failed)
  "$PY" -I "$ROOT/tests/lib/test_log.py" summary "$TEST_LOG" "$LOGS/test.log" ${failed[@]+"${failed[@]}"} \
    | tee "$LOGS/summary.txt"
  [ "$rc" -eq 0 ] || die "step 'test' failed (exit $rc) on $SHA; log: $LOGS/test.log; no record written"
  STEP_SECONDS="$STEP_SECONDS test=$(($(date +%s) - t0))"
}

run_step lint zig build lint --cache-dir "$ZIG_CACHE"
run_step ubench zig build ubench --cache-dir "$ZIG_CACHE"
start_guards
run_tests
join_guards

SUMMARY_LINE="$(grep '^Build Summary:' "$LOGS/test.log" | tail -n 1)" || die "the test step printed no Build Summary: $LOGS/test.log"
mkdir -p "$GATE_DIR"
GATE_SHA="$SHA" GATE_REF="$REF" GATE_ZIG="$(zig version)" GATE_SUMMARY="$SUMMARY_LINE" GATE_STEPS="$STEP_SECONDS" \
  GATE_TIMES="$TEST_LOG" GATE_FIXTURES="$SUSHI_FIXTURE_ENVS" GATE_OUT="$GATE_DIR/$SHA.json" "$PY" -I - << 'EOF'
import datetime, json, os, re

env = os.environ
line = env["GATE_SUMMARY"]
steps_part, _, tests_part = line.partition(";")
build = {"line": line}
m = re.search(r"(\d+)/(\d+) steps succeeded", steps_part)
if m:
    build["steps_succeeded"], build["steps_total"] = int(m[1]), int(m[2])
m = re.search(r"(\d+)/(\d+) tests passed", tests_part)
if m:
    build["tests_passed"], build["tests_total"] = int(m[1]), int(m[2])
for n, kind in re.findall(r"(\d+) (\w+)", tests_part.split("passed", 1)[-1]):
    build[kind] = int(n)

minutes = {}
for kv in env["GATE_STEPS"].split():
    name, secs = kv.split("=")
    minutes[name] = round(int(secs) / 60, 1)

times = []
with open(env["GATE_TIMES"], encoding="utf-8") as f:
    for row in f:
        _, step, status, ms, test = row.rstrip("\n").split("\t", 4)
        if status != "DONE":
            times.append({"ms": int(ms), "test": test, "step": step, "status": status})
times.sort(key=lambda t: -t["ms"])

finished = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
counts = f"{build.get('tests_passed', '?')}/{build.get('tests_total', '?')} tests passed"
extra = ", ".join(f"{build[k]} {k}" for k in build if k not in ("line", "steps_succeeded", "steps_total", "tests_passed", "tests_total"))
record = {
    "sha": env["GATE_SHA"],
    "ref": env["GATE_REF"],
    "finished": finished,
    "zig": env["GATE_ZIG"],
    "fixtures": {name: env[name] for name in env["GATE_FIXTURES"].split()},
    "build_summary": build,
    "minutes": minutes,
    "test_times": times,
    "summary": f"{env['GATE_SHA'][:12]} green at {finished}, zig {env['GATE_ZIG']}: {counts}"
    + (f" ({extra})" if extra else "") + "; " + ", ".join(f"{k} {v} min" for k, v in minutes.items()),
}
tmp = env["GATE_OUT"] + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(record, f, indent=1)
    f.write("\n")
os.replace(tmp, env["GATE_OUT"])
print(record["summary"])
EOF
echo "gate: record $GATE_DIR/$SHA.json"

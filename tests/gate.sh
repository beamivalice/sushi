#!/bin/bash
# sushi-test: tier=tool
# The local release gate: every check a local run can prove, run once on one commit in a clean worktree of it.
#   tests/gate.sh [<ref>]     the ref as this repo has it (default origin/main: fetch first)
# Steps, each under `taskpolicy -a`, stopping at the first failure: zig build lint, tests/hermetic_guards.sh,
# zig build ubench, then zig build test -Doptimize=ReleaseFast --summary all with the tiny-fixture envs of
# tests/lib/fixture_envs.sh. On success it writes ${SUSHI_GATE_DIR:-$HOME/.sushi/runs/gate}/<sha>.json, the record
# ./release.sh requires; each step's log stays in <sha>.logs/ beside it. The worktree is removed either way.
# Every zig build keeps its local cache in <gate dir>/zig-cache, so a fresh worktree is not a cold build; after a
# toolchain or SDK change remove it (configure-time output is cached).
# The worktree borrows this checkout's .zig-toolchain and lib/mlx (symlinks). SUSHI_GATE_PYTHON names the python3
# the hermetic guards run with (default: python3 on PATH); it must have PyYAML, since the guards fail on its skip.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE_DIR="${SUSHI_GATE_DIR:-$HOME/.sushi/runs/gate}"
REF="${1:-origin/main}"

die() { echo "gate.sh: $*" >&2; exit 1; }

SHA="$(git -C "$ROOT" rev-parse --verify --quiet "$REF^{commit}")" || die "refused: '$REF' names no commit in $ROOT"
for d in .zig-toolchain lib/mlx; do
  [ -e "$ROOT/$d" ] || die "refused: $ROOT/$d is missing; the gate's worktree borrows it"
done
PY="$(command -v "${SUSHI_GATE_PYTHON:-python3}" || true)"
[ -n "$PY" ] || die "refused: no python3 '${SUSHI_GATE_PYTHON:-python3}' to run the hermetic guards with"
"$PY" -c 'import yaml' 2> /dev/null \
  || die "refused: $PY has no PyYAML, and the hermetic guards fail on its skip; set SUSHI_GATE_PYTHON to a python3 that has it"

WT="$(mktemp -d "${TMPDIR:-/tmp}/sushi-gate.XXXXXX")"
cleanup() { git -C "$ROOT" worktree remove --force "$WT" 2> /dev/null || rm -rf "$WT"; }
trap cleanup EXIT
git -C "$ROOT" worktree add --quiet --detach "$WT" "$SHA"
ln -s "$(cd "$ROOT/.zig-toolchain" && pwd -P)" "$WT/.zig-toolchain"
ln -s "$(cd "$ROOT/lib/mlx" && pwd -P)" "$WT/lib/mlx"
cd "$WT"

[ -f tests/lib/fixture_envs.sh ] || die "refused: $SHA has no tests/lib/fixture_envs.sh"
# shellcheck source=lib/fixture_envs.sh
source tests/lib/fixture_envs.sh
for v in $SUSHI_FIXTURE_ENVS; do
  [ -d "${!v}" ] || die "refused: fixture $v=${!v} is not a folder"
done

ZIG_CACHE="$GATE_DIR/zig-cache"
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

TIMES=""
grep -q SUSHI_TEST_TIMES scripts/test_runner.zig 2> /dev/null && TIMES="$LOGS/test-times.tsv"
TEST_ENV=()
[ -z "$TIMES" ] || TEST_ENV=("SUSHI_TEST_TIMES=$TIMES")

run_step lint zig build lint --cache-dir "$ZIG_CACHE"
run_step guards env PATH="$(dirname "$PY"):$PATH" ZIG_LOCAL_CACHE_DIR="$ZIG_CACHE" bash tests/hermetic_guards.sh
run_step ubench zig build ubench --cache-dir "$ZIG_CACHE"
run_step test env ${TEST_ENV[@]+"${TEST_ENV[@]}"} zig build test -Doptimize=ReleaseFast --summary all --cache-dir "$ZIG_CACHE"

SUMMARY_LINE="$(grep '^Build Summary:' "$LOGS/test.log" | tail -n 1)" || die "the test step printed no Build Summary: $LOGS/test.log"
mkdir -p "$GATE_DIR"
GATE_SHA="$SHA" GATE_REF="$REF" GATE_ZIG="$(zig version)" GATE_SUMMARY="$SUMMARY_LINE" GATE_STEPS="$STEP_SECONDS" \
  GATE_TIMES="$TIMES" GATE_FIXTURES="$SUSHI_FIXTURE_ENVS" GATE_OUT="$GATE_DIR/$SHA.json" "$PY" -I - << 'EOF'
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

times = None
if env["GATE_TIMES"]:
    times = []
    with open(env["GATE_TIMES"], encoding="utf-8") as f:
        for row in f:
            ms, _, test = row.rstrip("\n").partition("\t")
            times.append({"ms": int(ms), "test": test})
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

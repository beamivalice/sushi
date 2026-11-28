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

if "$ZIG" build test-plan -Doptimize=ReleaseFast -Dtest-step=test-host -Dtest-skip-step=test-host >"$TMP/out" 2>&1; then
  echo "  FAIL: -Dtest-step=test-host -Dtest-skip-step=test-host was accepted"; FAIL=$((FAIL + 1))
elif ! grep -q "Dtest-step=test-host is also named by -Dtest-skip-step" "$TMP/out"; then
  echo "  FAIL: a step named by both -Dtest-step and -Dtest-skip-step failed without naming it"; tail -3 "$TMP/out" | sed 's/^/    /'; FAIL=$((FAIL + 1))
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

# Split domain and fault-sweep group aliases still resolve cleanly via -Dtest-step=<group>, and test-plan.json records
# the split steps plus exact metallib dependency flags.
"$ZIG" build test-plan -Doptimize=ReleaseFast -Dfixture-cut=true >"$TMP/out" 2>&1 || { echo "  FAIL: full test-plan"; tail -3 "$TMP/out"; exit 1; }
python3 -I - <<'PY' || FAIL=$((FAIL + 1))
import json, sys
plan = {r["step"]: r for r in json.load(open("zig-out/test-plan.json"))}
need = [
    "test-deepseek-stream", "test-deepseek-generate", "test-deepseek-core",
    "test-hy4-plan", "test-hy4-request", "test-hy4-stq", "test-hy4-core",
    "cooler-requests-fault-spec", "cooler-requests-fault-rows",
    "cooler-load-fault-mlx", "cooler-load-fault-pool",
]
missing = [s for s in need if s not in plan]
if missing:
    print(f"  FAIL: test-plan missing split steps: {missing}")
    sys.exit(1)
no_mlib = ["test-host", "test-sushibar", "test-encoder", "test-glm", "test-mimo", "test-exl3-format"]
bad_no = [s for s in no_mlib if plan[s].get("metallib") is not False]
if bad_no:
    print(f"  FAIL: expected metallib=false for {bad_no}")
    sys.exit(1)
yes_mlib = ["test-server", "cooler-load", "test-deepseek-core", "test-hy4-plan", "test-exl3-experts", "test-exl3-trunk"]
bad_yes = [s for s in yes_mlib if plan[s].get("metallib") is not True]
if bad_yes:
    print(f"  FAIL: expected metallib=true for {bad_yes}")
    sys.exit(1)
no_cut = ["cooler-load", "cooler-manifest", "cooler-requests", "cooler-load-fault-mlx", "cooler-load-fault-pool",
          "cooler-requests-fault-spec", "cooler-requests-fault-rows", "cooler-restore",
          "qwen-late-gdn-captured-verify", "qwen-late-gdn-final-state", "qwen-late-vision-tower",
          "qwen-late-qsa-split", "qwen-late-qsa-row-class"]
bad_cut = [s for s in no_cut if any(k == "SUSHI_TEST_FIXTURE_CUT" for k, _ in plan[s]["env"])]
if bad_cut:
    print(f"  FAIL: non-cut steps still carry SUSHI_TEST_FIXTURE_CUT: {bad_cut}")
    sys.exit(1)
PY

for args in \
  "-Dmodel=dsv41 -Dtest-step=test-deepseek" \
  "-Dmodel=hy4flash -Dtest-step=test-hy4" \
  "-Dtest-step=cooler-requests-fault" \
  "-Dtest-step=cooler-load-fault"; do
  # shellcheck disable=SC2086
  if ! "$ZIG" build test-plan -Doptimize=ReleaseFast $args >"$TMP/out" 2>&1; then
    echo "  FAIL: group alias failed to configure: $args"; tail -5 "$TMP/out" | sed 's/^/    /'; FAIL=$((FAIL + 1))
  fi
done

[ "$FAIL" -eq 0 ] && echo "PASS: test_build_test_step" || { echo "FAIL: test_build_test_step ($FAIL)"; exit 1; }

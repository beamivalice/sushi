#!/bin/bash
# sushi-test: tier=hermetic needs=git
# test_arch_drop.sh — an architecture is droppable: delete src/<arch>/ and its registry line, and the rest still
# builds and passes the host tier. No GPU, no model, no lock; it obeys the QUIET marker like any build.
#
# For each arch in DROPPABLE, in a temporary git worktree of HEAD:
#   1. delete src/<arch>/ and the registry lines marked `// arch:<arch>`
#   2. zig build -Darchs=<the rest>               (Debug, compile only)
#   3. zig build test-build -Darchs=<the rest>    (every test binary compiles, none runs)
#   4. zig build test-host -Darchs=<the rest>
#   5. tests/test_doc_links.sh
# A stray reference outside the folder fails step 2. `-Darchs` alone cannot catch one: the folder still exists.
#
# An arch joins DROPPABLE in its S4 stage. DROP_TRY=<arch> runs the same check for an arch not in the list (it is
# expected to fail while shared code still names it); DROP_KEEP=1 keeps the worktree for a look. DROP_ONLY=<arch>
# drops only that arch, for a local run; the registry check still covers every arch.
#
# Usage: ./tests/test_arch_drop.sh

set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd -P)

DROPPABLE=(mimo2 dsv41 glm5 qwen4)

REGISTRY=src/arch/registry.zig
ZIG=${ZIG:-$ROOT/.zig-toolchain/zig}
QUIET=$HOME/.sushi/runs/QUIET

registered=$(grep -o '// arch:[a-z0-9_][a-z0-9_]*' "$REGISTRY" | sed 's#// arch:##')
FAIL=0

# Each registered arch is one marked line, or a later drop cannot delete it.
for arch in $registered; do
  n=$(grep -c "// arch:$arch\$" "$REGISTRY")
  if [ "$n" -ne 1 ]; then
    echo "  FAIL: $REGISTRY has $n lines marked '// arch:$arch', want 1"
    FAIL=1
  fi
  if [ ! -d "src/$arch" ]; then
    echo "  FAIL: $REGISTRY registers $arch but src/$arch/ is missing"
    FAIL=1
  fi
  if ! grep -q "\"$arch\"" <<< "$(grep "// arch:$arch\$" "$REGISTRY")"; then
    echo "  FAIL: the line marked '// arch:$arch' does not register \"$arch\""
    FAIL=1
  fi
done
for arch in "${DROPPABLE[@]+"${DROPPABLE[@]}"}" ${DROP_TRY:-} ${DROP_ONLY:-}; do
  if ! grep -qx "$arch" <<< "$registered"; then
    echo "  FAIL: $arch is not a registered architecture"
    FAIL=1
  fi
done

WT_DIR=
cleanup() {
  [ -n "$WT_DIR" ] && [ -z "${DROP_KEEP:-}" ] || return 0
  git worktree remove --force "$WT_DIR/wt" 2>/dev/null
  rm -rf "$WT_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

drop_one() {
  local arch=$1 wt rest start rc=0
  if [ ! -x "$ZIG" ]; then
    echo "  SKIP: no $ZIG (run ./scripts/fetch-zig.sh)"
    return 0
  fi
  rest=$(grep -vx "$arch" <<< "$registered" | paste -sd, -)
  wt=$(mktemp -d "${TMPDIR:-/tmp}/arch-drop-$arch.XXXXXX")
  WT_DIR=$wt
  git worktree add --quiet --detach "$wt/wt" HEAD || return 1
  ln -s "$ROOT/.zig-toolchain" "$wt/wt/.zig-toolchain"
  ln -s "$ROOT/lib/mlx" "$wt/wt/lib/mlx"
  rm -rf "$wt/wt/src/$arch"
  sed -i.bak "/\/\/ arch:$arch\$/d" "$wt/wt/$REGISTRY" && rm -f "$wt/wt/$REGISTRY.bak"

  while [ -e "$QUIET" ]; do sleep 5; done
  start=$(date +%s)
  (
    cd "$wt/wt" || exit 1
    taskpolicy -a "$ZIG" build -Darchs="$rest" 2>&1 || exit 1
    taskpolicy -a "$ZIG" build test-build -Darchs="$rest" 2>&1 || exit 1
    taskpolicy -a "$ZIG" build test-host -Darchs="$rest" 2>&1 || exit 1
    ./tests/test_doc_links.sh 2>&1 || exit 1
  ) > "$wt/log" 2>&1 || rc=$?
  if [ $rc -ne 0 ]; then
    echo "  FAIL: dropping $arch ($(( $(date +%s) - start ))s); last lines of the log:"
    tail -15 "$wt/log" | sed 's/^/    | /'
  else
    echo "  PASS: dropping $arch builds and passes the host tier ($(( $(date +%s) - start ))s)"
  fi
  if [ -n "${DROP_KEEP:-}" ]; then echo "  kept: $wt/wt"; else cleanup; fi
  WT_DIR=
  return $rc
}

for arch in "${DROPPABLE[@]+"${DROPPABLE[@]}"}" ${DROP_TRY:-}; do
  [ -z "${DROP_ONLY:-}" ] || [ "$arch" = "$DROP_ONLY" ] || continue
  drop_one "$arch" || FAIL=1
done

if [ $FAIL -eq 0 ]; then
  echo "  PASS: ${#DROPPABLE[@]} droppable architectures of $(wc -w <<< "$registered" | tr -d " ") registered"
  echo "PASS"
  exit 0
fi
echo "FAIL"
exit 1

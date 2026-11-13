#!/bin/bash
# sushi-test: tier=hermetic needs=git
# test_arch_drop.sh — an architecture is droppable: delete src/<arch>/ and its registry line, and the rest still
# builds and passes the host tier. No GPU, no model, no lock; it obeys the QUIET marker like any build.
#
# For each arch it takes, in a temporary git worktree of HEAD:
#   1. delete src/<arch>/ and the registry lines marked `// arch:<arch>`
#   2. zig build -Darchs=<the rest>               (Debug, compile only)
#   3. zig build test-build -Darchs=<the rest>    (every test binary compiles, none runs)
#   4. zig build test-host -Darchs=<the rest>
#   5. tests/test_doc_links.sh
# A stray reference outside the folder fails step 2. `-Darchs` alone cannot catch one: the folder still exists.
#
# And for each arch it takes, `zig build -Doptimize=ReleaseFast -Darchs=<it>` in the checkout, into a directory of
# its own and without the host tier: a build with only one arch compiles every comptime branch an empty union or list
# takes, which the drop (Debug, the other archs still present) does not.
#
# The drops and the single-arch builds are independent, so they all run at once: the wall time is the slowest one's.
#
# Which archs: every one in DROPPABLE (and every registered one alone) by default. `--since <ref>` takes only the
# archs whose folder src/<arch>/ changed after that commit, the run a series' end and a version bump owe; a series
# that touched no arch folder drops nothing. `--plan` prints the choice and builds nothing.
#
# An arch joins DROPPABLE in its S4 stage. DROP_TRY=<arch> runs the same check for an arch not in the list (it is
# expected to fail while shared code still names it); DROP_KEEP=1 keeps the worktrees for a look. DROP_ONLY=<arch>
# drops and builds only that arch, for a local run; the registry check still covers every arch.
#
# Usage: ./tests/test_arch_drop.sh [--since <ref>] [--plan]

set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd -P)

DROPPABLE=(mimo2 dsv41 glm5 qwen4)

REGISTRY=src/arch/registry.zig
ZIG=${ZIG:-$ROOT/.zig-toolchain/zig}
QUIET=$HOME/.sushi/runs/QUIET

SINCE=
PLAN=
while [ $# -gt 0 ]; do
  case "$1" in
    --since) [ $# -ge 2 ] || { echo "--since takes a commit"; exit 2; }; SINCE=$2; shift 2 ;;
    --plan) PLAN=1; shift ;;
    *) echo "unknown argument: $1 (usage: $0 [--since <ref>] [--plan])"; exit 2 ;;
  esac
done
if [ -n "$SINCE" ] && ! git rev-parse --verify --quiet "$SINCE^{commit}" > /dev/null; then
  echo "--since $SINCE: not a commit"
  exit 2
fi

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

# The archs this run takes: one named, those a series touched, or all.
if [ -n "${DROP_ONLY:-}" ]; then
  chosen=$DROP_ONLY
elif [ -n "$SINCE" ]; then
  changed=$(git diff --name-only "$SINCE" HEAD -- src | sed -n 's#^src/\([^/]*\)/.*#\1#p' | sort -u)
  chosen=$(for arch in $registered; do grep -qx "$arch" <<< "$changed" && echo "$arch"; done)
else
  chosen=$registered
fi
drops=
for arch in "${DROPPABLE[@]+"${DROPPABLE[@]}"}"; do
  grep -qx "$arch" <<< "$chosen" && drops="$drops $arch"
done
[ -z "${DROP_TRY:-}" ] || drops="$drops $DROP_TRY"
alone=
for arch in $registered; do
  grep -qx "$arch" <<< "$chosen" && alone="$alone $arch"
done
drops=${drops# }
alone=${alone# }
echo "  plan: drop ${drops:-(none)}; alone ${alone:-(none)}"
if [ -n "$PLAN" ]; then
  [ $FAIL -eq 0 ] && exit 0
  exit 1
fi
if [ -z "$drops$alone" ]; then
  if [ $FAIL -eq 0 ]; then
    echo "  PASS: no architecture's folder changed since $SINCE: nothing to drop"
    echo "PASS"
    exit 0
  fi
  echo "FAIL"
  exit 1
fi
if [ ! -x "$ZIG" ]; then
  echo "  SKIP: no $ZIG (run ./scripts/fetch-zig.sh)"
  [ $FAIL -eq 0 ] && exit 0
  exit 1
fi

RUN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/arch-drop.XXXXXX")
cleanup() {
  [ -z "${DROP_KEEP:-}" ] || return 0
  local wt
  for wt in "$RUN_DIR"/drop-*/wt; do
    [ -d "$wt" ] && git worktree remove --force "$wt" 2>/dev/null
  done
  rm -rf "$RUN_DIR"
}
trap cleanup EXIT
# A stop reaches every job, and each job the step it is running.
trap 'kill $(jobs -p) 2>/dev/null; wait; exit 130' INT TERM

# One step of a job, stoppable while it runs.
step() {
  "$@" &
  wait $!
}

# The worktree of HEAD without `arch`. Serial: `git worktree add` takes the repository's lock.
prepare_drop() {
  local arch=$1 dir=$RUN_DIR/drop-$1
  mkdir -p "$dir"
  git worktree add --quiet --detach "$dir/wt" HEAD || return 1
  ln -s "$ROOT/.zig-toolchain" "$dir/wt/.zig-toolchain"
  ln -s "$ROOT/lib/mlx" "$dir/wt/lib/mlx"
  rm -rf "$dir/wt/src/$arch"
  sed -i.bak "/\/\/ arch:$arch\$/d" "$dir/wt/$REGISTRY" && rm -f "$dir/wt/$REGISTRY.bak"
  # lint fails an allowlist entry whose file is gone; arch-tests.allow names shared tests, which a drop leaves in place.
  sed -i.bak "/^[a-z-]* src\/$arch\//d" "$dir/wt/tests/guards/lint.allow" && rm -f "$dir/wt/tests/guards/lint.allow.bak"
}

# Writes "<exit status> <seconds>" into the job's `result`. Runs as a background job: its steps are its own children.
build_drop() {
  local arch=$1 dir=$RUN_DIR/drop-$1 rest start rc=0
  trap 'kill $(jobs -p) 2>/dev/null; exit 143' TERM
  rest=$(grep -vx "$arch" <<< "$registered" | paste -sd, -)
  start=$(date +%s)
  cd "$dir/wt" || exit 1
  {
    step taskpolicy -a "$ZIG" build -Darchs="$rest" &&
      step taskpolicy -a "$ZIG" build test-build -Darchs="$rest" &&
      step taskpolicy -a "$ZIG" build test-host -Darchs="$rest" &&
      step ./tests/test_doc_links.sh
  } > "$dir/log" 2>&1 || rc=$?
  echo "$rc $(( $(date +%s) - start ))" > "$dir/result"
}

build_only() {
  local arch=$1 dir=$RUN_DIR/only-$1 start rc=0
  trap 'kill $(jobs -p) 2>/dev/null; exit 143' TERM
  mkdir -p "$dir"
  start=$(date +%s)
  step taskpolicy -a "$ZIG" build -Doptimize=ReleaseFast -Darchs="$arch" -p "$dir/out" > "$dir/log" 2>&1 || rc=$?
  echo "$rc $(( $(date +%s) - start ))" > "$dir/result"
}

# report <job dir> <what passed> <what failed>
report() {
  local rc secs
  if ! read -r rc secs < "$1/result" 2>/dev/null; then
    echo "  FAIL: $3 (no result: the job was stopped)"
    return 1
  fi
  if [ "$rc" -ne 0 ]; then
    echo "  FAIL: $3 (${secs}s); last lines of the log:"
    tail -15 "$1/log" | sed 's/^/    | /'
    return 1
  fi
  echo "  PASS: $2 (${secs}s)"
}

while [ -e "$QUIET" ]; do sleep 5; done
began=$(date +%s)
ready=
for arch in $drops; do
  if prepare_drop "$arch"; then
    ready="$ready $arch"
  else
    echo "  FAIL: no worktree to drop $arch from"
    FAIL=1
  fi
done
for arch in $ready; do build_drop "$arch" & done
for arch in $alone; do build_only "$arch" & done
wait
for arch in $ready; do
  report "$RUN_DIR/drop-$arch" "dropping $arch builds and passes the host tier" "dropping $arch" || FAIL=1
done
for arch in $alone; do
  report "$RUN_DIR/only-$arch" "building only $arch (ReleaseFast)" "building only $arch" || FAIL=1
done
[ -z "${DROP_KEEP:-}" ] || echo "  kept: $RUN_DIR"

if [ $FAIL -eq 0 ]; then
  echo "  PASS: $(wc -w <<< "$ready" | tr -d " ") dropped and $(wc -w <<< "$alone" | tr -d " ") built alone, of $(wc -w <<< "$registered" | tr -d " ") registered, at once ($(( $(date +%s) - began ))s)"
  echo "PASS"
  exit 0
fi
echo "FAIL"
exit 1

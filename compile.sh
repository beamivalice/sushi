#!/usr/bin/env bash
# Compile the local checkout (including uncommitted changes) to zig-out/bin/sushi and zig-out/bin/sushibar;
# zig build compiles the two in parallel.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [ ! -x .zig-toolchain/zig ]; then
  echo "compile.sh: missing Zig toolchain; run ./scripts/fetch-zig.sh first" >&2
  exit 1
fi

.zig-toolchain/zig build -Doptimize=ReleaseFast --prefix "$REPO_ROOT/zig-out" "$@"
echo "Built $REPO_ROOT/zig-out/bin/sushi and $REPO_ROOT/zig-out/bin/sushibar"

#!/bin/bash
# sushi-test: tier=hermetic
# Static guard: the Brewfile lists every Homebrew formula the build needs.
# CI installs only what the Brewfile names, and GitHub's runner image carries extra tools, so a missing
# formula stays green there while a clean Mac fails. Silent when clean; HERMETIC.
set -euo pipefail
cd "$(dirname "$0")/.."

BREWFILE=${BREWFILE:-Brewfile}
status=0

listed() { grep -qE "^[[:space:]]*brew[[:space:]]+\"$1\"" "$BREWFILE"; }

# Tools macOS does not ship, named as the formula that provides them.
for tool in cmake ninja pkg-config jq wget autoconf automake; do
    if grep -v -E '^[[:space:]]*#' scripts/*.sh build.zig compile.sh \
        | grep -qE "(^|[^A-Za-z0-9_./-])$tool([[:space:]]|\$)" && ! listed "$tool"; then
        echo "FAIL: scripts/ or build.zig invoke '$tool' but $BREWFILE has no brew \"$tool\"" >&2
        status=1
    fi
done

# Every dependency build.zig verifies at build time.
for dep in $(sed -n '/required_brew_deps/,/^};/p' build.zig | sed -n 's/.*\.name = "\([^"]*\)".*/\1/p'); do
    if ! listed "$dep"; then
        echo "FAIL: build.zig requires brew '$dep' but $BREWFILE has no brew \"$dep\"" >&2
        status=1
    fi
done

exit "$status"

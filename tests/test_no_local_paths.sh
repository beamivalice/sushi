#!/bin/bash
# sushi-test: tier=hermetic
# Repo-hygiene gate: no tracked file names a development box's own directories.
# Tests and scripts find packs under ${SUSHI_MODELS_DIR:-$HOME/.sushi/models}; raw measurement files stay private.
# Prints the offending file:line and fails; prints nothing when clean.
set -euo pipefail
cd "$(dirname "$0")/.."

pattern='/Users/(beam|david)|~/llm/|\$HOME/llm/|claude-tmp|/tmp/claude-|scratchpad/|session scratchpad'
status=0
hits=$(git grep -n -I -E "$pattern" -- . ':!tests/test_no_local_paths.sh') || status=$?
if [ "$status" -gt 1 ]; then
    echo "FAIL: git grep exited $status" >&2
    exit 1
fi
if [ -n "$hits" ]; then
    printf '%s\n' "$hits" | cut -c1-200 >&2
    echo "FAIL: tracked files name local directories (above)" >&2
    exit 1
fi

#!/bin/bash
# sushi-test: tier=hermetic
# Repo-hygiene gate: every tests/<file> path that docs/, CLAUDE.md, README.md, CONTRIBUTING.md, src/, .github/ or
# tests/CLAUDE.md names exists. A directory mention (tests/lib/) has no extension and is not a link.
# Prints file:line and the path; silent when clean.
set -euo pipefail
cd "$(dirname "$0")/.."

hits=$(git grep -n -o -I -E 'tests/[A-Za-z0-9_./-]+\.(sh|py|cjs|mjs|json|md|jinja|txt|csv)' -- \
    docs CLAUDE.md README.md CONTRIBUTING.md src .github tests/CLAUDE.md || true)

status=0
while IFS=: read -r file line path; do
    [ -n "$path" ] || continue
    [ -e "$path" ] && continue
    echo "$file:$line: $path does not exist"
    status=1
done <<< "$hits"
exit "$status"

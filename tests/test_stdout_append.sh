#!/bin/bash
# sushi-test: tier=host
# `sushi --version >> f` and `sushi --help >> f` append: stdout is written in streaming mode, never at file offset 0.
# Usage: ./tests/test_stdout_append.sh
set -u
. "$(dirname "$0")/private_cache.sh"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${SUSHI_BIN:-$ROOT/zig-out/bin/sushi}"
PASS=0
FAIL=0
check() {
    if [ "$2" = "1" ]; then PASS=$((PASS + 1)); echo "  PASS $1"; else FAIL=$((FAIL + 1)); echo "  FAIL $1"; fi
}
yes_if() { if eval "$1"; then echo 1; else echo 0; fi; }

[ -x "$BIN" ] || { echo "FAIL: $BIN missing (zig build -Doptimize=ReleaseFast)"; exit 1; }
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export HOME="$WORK/home"
mkdir -p "$HOME"

for flag in --version --help; do
    f="$WORK/out.txt"
    printf 'line-one-xxxx\nline-two\n' > "$f"
    "$BIN" "$flag" >> "$f" 2>/dev/null
    check "sushi $flag >> f keeps the first line" "$(yes_if '[ "$(sed -n 1p "$f")" = "line-one-xxxx" ]')"
    check "sushi $flag >> f keeps the second line" "$(yes_if '[ "$(sed -n 2p "$f")" = "line-two" ]')"
    check "sushi $flag >> f appends its text after them" "$(yes_if '[ "$(wc -l < "$f")" -gt 2 ]')"
done

echo "== $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]

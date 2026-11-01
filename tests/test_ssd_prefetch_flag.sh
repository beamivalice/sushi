#!/usr/bin/env bash
# sushi-test: tier=host
. "$(dirname "$0")/private_cache.sh"
# `--ssd-prefetch` takes an integer 0..100: anything else exits non-zero with the flag's own message before any model
# loads, a good value passes the parser, and `--help` lists the flag. Hermetic: the model path never exists.
set -uo pipefail

BIN=${SUSHI_BIN:-zig-out/bin/sushi}
[ -x "$BIN" ] || { echo "FAIL: missing binary $BIN" >&2; exit 1; }
MISSING=$(mktemp -d)/no-such-pack
FAILED=""

for bad in 101 -1 abc "" 6.5 +5 1000; do
    out=$("$BIN" --serve --model "$MISSING" --ssd-prefetch "$bad" 2>&1); rc=$?
    if [ "$rc" -eq 0 ] || ! grep -q -- "--ssd-prefetch: expected an integer in 0..100" <<<"$out"; then
        echo "FAIL: --ssd-prefetch '$bad' (exit $rc) was not refused by its own message" >&2; FAILED=1
    fi
done

for good in 0 65 100; do
    out=$("$BIN" --serve --model "$MISSING" --ssd-prefetch "$good" 2>&1) || true
    if grep -q -- "--ssd-prefetch:" <<<"$out"; then
        echo "FAIL: --ssd-prefetch $good was refused" >&2; FAILED=1
    fi
done

"$BIN" --help 2>&1 | grep -q -- "--ssd-prefetch <n>" || { echo "FAIL: --help does not list --ssd-prefetch" >&2; FAILED=1; }

[ -z "$FAILED" ] && echo "PASS: --ssd-prefetch parse"
exit ${FAILED:+1}

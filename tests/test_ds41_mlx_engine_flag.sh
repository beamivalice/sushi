#!/usr/bin/env bash
# sushi-test: tier=host
# `--ds41-mlx-engine` takes no value: it passes the parser (a stray value after it is the next argument, not its
# own), and `--help` lists it with Sushi Cooler's name and the report link. Hermetic: the model path never exists.
set -uo pipefail
. "$(dirname "$0")/private_cache.sh"

BIN=${SUSHI_BIN:-zig-out/bin/sushi}
[ -x "$BIN" ] || { echo "FAIL: missing binary $BIN" >&2; exit 1; }
MISSING=$(mktemp -d)/no-such-pack
FAILED=""

out=$("$BIN" --serve --model "$MISSING" --port 18990 --log-file off --ds41-mlx-engine 2>&1) || true
if grep -qi "unrecognized\|unknown argument" <<<"$out"; then
    echo "FAIL: --ds41-mlx-engine was not recognized: $out" >&2; FAILED=1
fi

help=$("$BIN" --help 2>&1)
grep -q -- "--ds41-mlx-engine" <<<"$help" || { echo "FAIL: --help does not list --ds41-mlx-engine" >&2; FAILED=1; }
grep -q "Sushi Cooler" <<<"$help" || { echo "FAIL: --help does not name Sushi Cooler" >&2; FAILED=1; }
grep -q "github.com/beamivalice/sushi/issues/new" <<<"$help" || { echo "FAIL: --help does not give the report link" >&2; FAILED=1; }

[ -z "$FAILED" ] && echo "PASS: --ds41-mlx-engine parse and help"
exit ${FAILED:+1}

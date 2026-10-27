#!/usr/bin/env bash
. "$(dirname "$0")/private_cache.sh"
# Boot one model, fire the decode forward micro-bench at load, print the lines,
# shut down. Diagnostic helper for decode-perf work — NOT a test.
#
#   tests/fwd_ubench.sh <model-dir> [iters] [extra sushi flags...]
#
# Env passthrough: any SUSHI_* already exported reaches the server.
set -uo pipefail

MODEL="${1:?usage: fwd_ubench.sh <model-dir> [iters] [flags...]}"
ITERS="${2:-20}"
shift 2 2>/dev/null || shift 1
PORT="${PORT:-8099}"
BIN="${BIN:-./zig-out/bin/sushi}"
LOG="${LOG:-/tmp/fwd-ubench-$PORT.log}"

# Only the server this script starts is ever killed, by its PID.
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | grep -q LISTEN; then
  echo "port $PORT is already in use; stop that server or set PORT" >&2
  exit 1
fi

SUSHI_DECODE_FWD_UBENCH="$ITERS" "$BIN" --model "$MODEL" --serve --port "$PORT" \
  --log-level info "$@" >"$LOG" 2>&1 &
SRV=$!

for _ in $(seq 1 600); do
  grep -qE "\[fwd-ubench\] done" "$LOG" && break
  kill -0 "$SRV" 2>/dev/null || break
  sleep 1
done
sleep 2

grep -E "\[fwd-ubench\]|\[moe\]|\[dtype\]|engaged|declined" "$LOG"

kill "$SRV" 2>/dev/null
wait "$SRV" 2>/dev/null

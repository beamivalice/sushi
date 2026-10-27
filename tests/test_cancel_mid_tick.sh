#!/bin/bash
. "$(dirname "$0")/private_cache.sh"
# Stopping one stream while another MTP stream decodes must not crash the server:
# the handler frees its sampling state (`think_bound`) once `complete` returns,
# and a tick that already holds the slot still reads it (`Slot.in_pass`).
#
# Usage: CANCEL_TEST_MODEL=<MTP + thinking pack> ./tests/test_cancel_mid_tick.sh [port]
set -u
MODEL="${CANCEL_TEST_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw}"
PORT="${1:-11282}"
BIN="${SUSHI_BINARY:-./zig-out/bin/sushi}"
N="${CANCELS:-12}"
if [ ! -d "$MODEL" ]; then echo "skip: model not found ($MODEL)"; exit 0; fi

LOG=$(mktemp)
KEEP=
"$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --mtp --temp 0.7 --log-level info >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV $KEEP 2>/dev/null; rm -f "$LOG"' EXIT
for _ in $(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "FAIL: server never became healthy" >&2
    tail -20 "$LOG" >&2
    exit 1
fi

# reasoning_effort arms the decode-time think bound.
body() { printf '{"model":"x","stream":true,"enable_mtp":true,"reasoning_effort":"low","max_tokens":1500,"messages":[{"role":"user","content":"%s"}]}' "$1"; }
curl -sN -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(body 'Write a very long detailed essay about the history of bridges.')" >/dev/null &
KEEP=$!
for i in $(seq 1 "$N"); do
    curl -sN -m 60 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "$(body "Explain in detail how a compiler works, part $i.")" >/dev/null &
    V=$!
    sleep 3
    kill $V 2>/dev/null
    wait $V 2>/dev/null
    sleep 0.3
    if ! kill -0 $SRV 2>/dev/null; then
        echo "FAIL: server died at cancel $i" >&2
        tail -5 "$LOG" >&2
        exit 1
    fi
done
# Each request logs its dispatch at admission; `[spec-stats]` comes only at a request's end,
# which a cancelled or still-running stream never reaches.
if [ "$(grep -c "mtp=enabled" "$LOG")" -lt 2 ]; then
    echo "FAIL: fewer than two MTP streams in the log; the cancels never raced an MTP tick" >&2
    grep -E "mtp|spec" "$LOG" | tail -20 >&2
    exit 1
fi
echo "PASS: survived $N cancels beside a decoding MTP stream"

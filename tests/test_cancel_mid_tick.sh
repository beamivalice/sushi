#!/bin/bash
# sushi-test: tier=live boot=api
# Stopping one stream while another MTP stream decodes must not crash the server:
# the handler frees its sampling state (`think_bound`) once `complete` returns,
# and a tick that already holds the slot still reads it (`Slot.in_pass`).
# SKIPs where the pass has no MTP head or cannot batch (the cancelled stream must race a decoding one).
. "$(dirname "$0")/lib/boot.sh"; sushi_attach
sushi_require mtp
sushi_require batching
set -u
BASE=$SUSHI_URL
N="${CANCELS:-12}"
KEEP=
trap 'kill $KEEP 2>/dev/null' EXIT

# reasoning_effort arms the decode-time think bound.
body() { printf '{"model":"x","stream":true,"enable_mtp":true,"temperature":0.7,"reasoning_effort":"low","max_tokens":1500,"messages":[{"role":"user","content":"%s %s"}]}' "$SUSHI_NONCE" "$1"; }
curl -sN -m 900 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(body 'Write a very long detailed essay about the history of bridges.')" >/dev/null &
KEEP=$!
for i in $(seq 1 "$N"); do
    curl -sN -m 60 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "$(body "Explain in detail how a compiler works, part $i.")" >/dev/null &
    V=$!
    sleep 3
    kill $V 2>/dev/null
    wait $V 2>/dev/null
    sleep 0.3
    if ! curl -sf -m 10 "$BASE/health" >/dev/null 2>&1; then
        echo "FAIL: server died at cancel $i" >&2
        sushi_log_since | tail -5 >&2
        exit 1
    fi
done
# Each request logs its dispatch at admission; `[spec-stats]` comes only at a request's end,
# which a cancelled or still-running stream never reaches.
if [ "$(sushi_log_since | grep -c "mtp=enabled")" -lt 2 ]; then
    echo "FAIL: fewer than two MTP streams in the log; the cancels never raced an MTP tick" >&2
    sushi_log_since | grep -E "mtp|spec" | tail -20 >&2
    exit 1
fi
echo "PASS: survived $N cancels beside a decoding MTP stream"

#!/bin/bash
# sushi-test: tier=live boot=api
# Regression tests for client-disconnect handling during long prefills.
#
# Pins the 2026-06-10 live failure: Claude Code with a huge MCP toolset sent a
# ~40K-token prompt to gemma-4-12b, timed out waiting on the multi-minute cold
# prefill, disconnected, and RETRIED — but the server only checks the peer
# socket after a token arrives, so the handler sat blocked in waitNext for the
# whole prefill. Abandoned prefills piled up serially; `lsof` showed ZERO
# clients connected while the GPU ground away for minutes. The server looked
# dead ("not returning anything") and memory climbed with each ghost KV.
#
# Two contracts:
#   1. Keepalive: during a long prefill, streaming responses emit periodic
#      SSE pings (Anthropic `event: ping` / OpenAI `: keepalive` comments) so
#      clients don't hit their idle timeout in the first place.
#   2. Disconnect-cancel: when the client vanishes mid-prefill, the server
#      notices within one keepalive interval, cancels the slot (aborting the
#      prefill at the next chunk boundary), and the next request isn't stuck
#      behind a ghost.
#
# Cancellation takes effect at a chunk boundary and the boot's chunk is auto-sized, so no check holds a
# wall-clock bound: the cancel is proven by its log line and by the session list going idle.
. "$(dirname "$0")/lib/boot.sh"; sushi_attach
set -u

BASE="$SUSHI_URL"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
PASS=0
FAIL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

check() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then
        PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $desc"
    else
        FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $desc"
    fi
}

# requests the server still prefills or decodes
live_sessions() {
    curl -s -m 10 "$BASE/metrics.json" | python3 -c 'import sys,json; print(sum(1 for x in json.load(sys.stdin).get("sessions",[]) if x.get("phase") in ("prefill","decode")))'
}
# waits (at most 180 s) for the session list to go idle; the last count stays in $LIVE
wait_idle() {
    for _ in $(seq 1 180); do
        LIVE=$(live_sessions)
        [ "$LIVE" = 0 ] && return 0
        sleep 1
    done
    return 1
}
count_log() { sushi_log_since | grep -c "$1"; }

# ~36K-token unique-ish prompts, long enough on every arch that keepalives and the
# disconnect window are clearly observable. Distinct seeds per check: a shared prompt would let
# check 2's ghost ride check 1's prefix-cache entry and skip the cold prefill entirely.
big_body() { # $1 = seed -> JSON on stdout
    python3 - "$1" <<'EOF'
import json, os, random, sys
random.seed(int(sys.argv[1]))
words = ["alpha","bridge","cobalt","delta","ember","fjord","glacier","harbor",
         "isotope","jasper","kelvin","lumen","meridian","nectar","onyx","prism"]
text = " ".join(random.choice(words) + str(i % 97) for i in range(12000))
print(json.dumps({
    "model": "m", "max_tokens": 40, "stream": True,
    "messages": [{"role": "user", "content": os.environ["SUSHI_NONCE"] + " Summarize this in one word: " + text}],
}))
EOF
}
big_body 7 > "$WORK/big.json"
big_body 1234 > "$WORK/big2.json"
OK_BODY='{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"'"$SUSHI_NONCE"' Say OK."}]}'

echo "1. keepalive pings flow during a long prefill (/v1/messages stream)"
curl -sN -m 600 "$BASE/v1/messages" -H 'Content-Type: application/json' \
    -d @"$WORK/big.json" > "$WORK/keepalive.sse"
PINGS=$(grep -c '"type":"ping"' "$WORK/keepalive.sse")
echo "    -> $PINGS ping event(s)"
check "≥2 ping events (initial + ≥1 during prefill)" "$([ "$PINGS" -ge 2 ] && echo 1 || echo 0)"
grep -q '"type":"message_stop"' "$WORK/keepalive.sse"
check "request still completes normally" "$([ $? -eq 0 ] && echo 1 || echo 0)"

echo "2. disconnect mid-prefill cancels the ghost; next request not blocked"
CANCELS_BEFORE=$(count_log "client disconnected")
# Client gives up after 3s — mirrors Claude Code's timeout+retry behavior.
curl -sN -m 3 "$BASE/v1/messages" -H 'Content-Type: application/json' \
    -d @"$WORK/big2.json" > /dev/null 2>&1
SMALL=$(curl -s -m 600 "$BASE/v1/messages" -H 'Content-Type: application/json' -d "$OK_BODY")
echo "$SMALL" | grep -q '"type":"message"'
check "follow-up request succeeded" "$([ $? -eq 0 ] && echo 1 || echo 0)"
wait_idle
echo "    -> requests still generating after the follow-up: $LIVE"
check "no request still prefilling for the departed client" "$([ "$LIVE" = 0 ] && echo 1 || echo 0)"
# Pre-fix, the ghost ground through its whole prefill and never logged a cancel.
check "server logged the disconnect-cancel" "$([ "$(count_log "client disconnected")" -gt "$CANCELS_BEFORE" ] && echo 1 || echo 0)"

echo "3. disconnect mid-decode cancels a non-stream ghost (/v1/messages stream=false)"
# A non-stream request never idles once tokens flow, so an idle-only peer probe misses a client that left mid-decode.
curl -s -m 5 "$BASE/v1/messages" -H 'Content-Type: application/json' \
    -d '{"model":"m","max_tokens":6000,"stream":false,"messages":[{"role":"user","content":"'"$SUSHI_NONCE"' Count from 1 to 3000, one number per line."}]}' > /dev/null 2>&1
# A batching server serves a follow-up beside a ghost, so timing cannot tell; the live session list can.
sleep 8
LIVE=$(live_sessions)
echo "    -> requests still generating 8s after the client left: $LIVE"
check "no request still decoding for the departed client" "$([ "$LIVE" = 0 ] && echo 1 || echo 0)"
SMALL=$(curl -s -m 600 "$BASE/v1/messages" -H 'Content-Type: application/json' -d "$OK_BODY")
echo "$SMALL" | grep -q '"type":"message"'
check "follow-up request succeeded" "$([ $? -eq 0 ] && echo 1 || echo 0)"
sushi_log_since | grep -q "client disconnected while decoding (non-stream)"
check "server logged the non-stream decode cancel" "$([ $? -eq 0 ] && echo 1 || echo 0)"

echo ""
echo "===== $PASS passed, $FAIL failed ====="
[ "$FAIL" -eq 0 ] || exit 1

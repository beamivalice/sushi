#!/bin/bash
# sushi-test: tier=live boot=api
# Streaming-silence regression test (class guard).
#
# CLASS: a streaming surface that BUFFERS tokens must never let the socket go
# silent for longer than the keepalive interval.
#
# The server buffers generated tokens whenever it might be looking at a tool
# call (`chat.streamShouldBufferForTools`) or an unclosed thinking block
# (`chat.streamThinkGate` -> .hold_thinking). For a large tool call — an agent
# one-shotting a whole file into `write_file` — that buffer holds for the
# ENTIRE generation, and pre-fix the server wrote zero bytes for its whole
# duration. Clients with an idle-body timeout tear such a connection down
# mid-generation: Node's `fetch` (undici) defaults to `bodyTimeout: 300_000`,
# so any tool call that takes >5 min to generate dies as `TypeError: terminated`.
# Live failure 2026-07-08: a pi agent session building a JS game lost two
# ~5-minute `write` calls exactly this way (~10 min of GPU discarded).
#
# The keepalive used to fire only on `.idle` — i.e. only while WAITING for the
# first token (long prefill). Once tokens flowed into a buffer, it never fired.
#
# This test asserts, for EVERY streaming surface (chat / messages / responses):
#   1. the max inter-chunk gap stays under MAX_GAP_S even though the model
#      spends far longer than that inside one buffered tool call
#   2. at least one keepalive (`: keepalive` comment / `event: ping`) arrives
#   3. the tool call still parses — name intact, arguments are valid JSON
#      (the injected keepalive bytes must not corrupt the SSE stream)
#
# A surface whose generation finishes faster than MIN_BUFFER_S never exercised
# the buffer, so it SKIPs rather than passing vacuously; so does one where the
# checkpoint chose not to call the tool. All three skipping makes the script a SKIP.
#
# Attaches to the shared api boot: tests/run.sh --only test_stream_keepalive.sh --model <m>.

set -e

. "$(dirname "$0")/lib/boot.sh"
sushi_attach
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'
PORT=${SUSHI_URL##*:}

# Keepalive interval is 5s (Conn.STREAM_KEEPALIVE_MS). Allow generous slack for
# a busy GPU: what we are proving is "seconds, not minutes".
MAX_GAP_S=${MAX_GAP_S:-15}
# A generation shorter than this never filled the buffer long enough to prove
# anything, so the surface SKIPs instead of passing vacuously.
MIN_BUFFER_S=${MIN_BUFFER_S:-8}

FAILURES=0
SKIPS=0
for SURFACE in chat messages responses; do
    echo "── surface: $SURFACE ──"
    set +e
    OUT=$(MAX_GAP_S="$MAX_GAP_S" MIN_BUFFER_S="$MIN_BUFFER_S" \
        python3 "$(dirname "$0")/lib/stream_keepalive_probe.py" "$PORT" "$SURFACE" sushi "$SUSHI_NONCE")
    RC=$?
    set -e
    echo "$OUT" | sed 's/^/  /'
    case $RC in
        0) echo -e "  ${GREEN}PASS${NC} $SURFACE" ;;
        2) echo -e "  ${YELLOW}SKIP${NC} $SURFACE (no buffered tool call to time)"; SKIPS=$((SKIPS + 1)) ;;
        *) echo -e "  ${RED}FAIL${NC} $SURFACE"; FAILURES=$((FAILURES + 1)) ;;
    esac
done

echo
if [ $FAILURES -eq 0 ]; then
    if [ $SKIPS -eq 3 ]; then
        echo "SKIP $(basename "$0"): no surface produced a buffered tool call long enough to time"
        exit 3
    fi
    echo -e "${GREEN}ALL PASS${NC} — no streaming surface goes silent during a buffered tool call"
    exit 0
fi
echo -e "${RED}$FAILURES surface(s) FAILED${NC}"
exit 1

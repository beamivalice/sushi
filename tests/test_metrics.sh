#!/bin/bash
. "$(dirname "$0")/private_cache.sh"
# Integration tests for the opt-in observability layer (--metrics):
#   * GET /metrics       — Prometheus text exposition (headless scraping)
#   * GET /metrics.json  — open JSON feed of the same counters
#
# There is NO admin dashboard and NO admin mutations — the feeds are read-only.
#
# Tests:
#  1. Without --metrics: /metrics + /metrics.json → 503.
#  2. With    --metrics: /metrics → 200 Prometheus text; /metrics.json → 200 JSON.
#  3. After one chat request: counters/histograms increment; live-gauge holds
#                             (live > 0 after a request, live == total at rest).
#
# Usage: ./tests/test_metrics.sh [model_dir] [port]
#   Starts its own servers. Default model: the Flash-Next EXL3 pack.

set -u

MODEL="${1:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw}"
PORT="${2:-11291}"
BASE="http://127.0.0.1:$PORT"
BINARY="${BINARY:-./zig-out/bin/sushi}"
LOG=/tmp/test_metrics.log
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

if [ ! -d "$MODEL" ]; then
    echo "SKIP: model dir not found: $MODEL (pass as first arg)"
    exit 0
fi

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | grep -q LISTEN; then
    echo "port $PORT is already in use; stop that server or pass another port" >&2
    exit 1
fi

wait_health() {
    for _ in $(seq 1 90); do
        curl -sf "$BASE/health" >/dev/null 2>&1 && return 0
        sleep 1
    done
    echo "FAIL: server never became healthy on port $PORT"
    return 1
}

# ════════════════════════════════════════════════════════════════════════════
# Phase 1: Without --metrics, /metrics* return 503
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 1: without --metrics ──"

"$BINARY" --model "$MODEL" --serve --port "$PORT" --log-level warn > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true' EXIT
wait_health

STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/metrics")
check "GET /metrics without --metrics → 503" "$([ "$STATUS" = "503" ] && echo 1 || echo 0)"

BODY=$(curl -s "$BASE/metrics")
check "503 body mentions 'not enabled'" "$(echo "$BODY" | grep -q "not enabled" && echo 1 || echo 0)"

STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/metrics.json")
check "GET /metrics.json without --metrics → 503" "$([ "$STATUS" = "503" ] && echo 1 || echo 0)"

kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true
sleep 1

# ════════════════════════════════════════════════════════════════════════════
# Phase 2: With --metrics, both endpoints answer
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 2: with --metrics (idle — no requests yet) ──"

# --log-level info so the "Prometheus metrics: ENABLED" startup line is visible.
"$BINARY" --model "$MODEL" --serve --port "$PORT" --metrics --log-level info > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true' EXIT
wait_health

check "startup log: 'Prometheus metrics: ENABLED'" \
    "$(grep -q "Prometheus metrics: ENABLED" "$LOG" && echo 1 || echo 0)"

STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/metrics")
check "GET /metrics with --metrics → 200" "$([ "$STATUS" = "200" ] && echo 1 || echo 0)"

CT=$(curl -s -D - -o /dev/null "$BASE/metrics" | grep -i "^content-type:" | tr -d '\r')
check "Content-Type is Prometheus text MIME" \
    "$(echo "$CT" | grep -q "text/plain" && echo "$CT" | grep -q "version=0.0.4" && echo 1 || echo 0)"

BODY=$(curl -s "$BASE/metrics")
check "# HELP vllm:prompt_tokens_total present" \
    "$(echo "$BODY" | grep -q "# HELP vllm:prompt_tokens_total" && echo 1 || echo 0)"
check "# TYPE vllm:prompt_tokens_total counter" \
    "$(echo "$BODY" | grep -q "# TYPE vllm:prompt_tokens_total counter" && echo 1 || echo 0)"
check "# TYPE vllm:time_to_first_token_seconds histogram" \
    "$(echo "$BODY" | grep -q "# TYPE vllm:time_to_first_token_seconds histogram" && echo 1 || echo 0)"
check "TTFT +Inf bucket present" \
    "$(echo "$BODY" | grep -q 'vllm:time_to_first_token_seconds_bucket{le="+Inf"}' && echo 1 || echo 0)"
check "vllm:num_requests_running gauge present" \
    "$(echo "$BODY" | grep -q "# TYPE vllm:num_requests_running gauge" && echo 1 || echo 0)"
check "sushi:gpu_utilization_pct gauge present" \
    "$(echo "$BODY" | grep -q "sushi:gpu_utilization_pct" && echo 1 || echo 0)"
check "sushi:memory_mb gauge present (TYPE line)" \
    "$(echo "$BODY" | grep -q "# TYPE sushi:memory_mb gauge" && echo 1 || echo 0)"
check "sushi:generation_tokens_live gauge present (TYPE line)" \
    "$(echo "$BODY" | grep -q "# TYPE sushi:generation_tokens_live gauge" && echo 1 || echo 0)"

check "request_success_total is 0 before any requests" \
    "$(echo "$BODY" | grep "^vllm:request_success_total " | grep -q " 0$" && echo 1 || echo 0)"
check "prompt_tokens_total is 0 before any requests" \
    "$(echo "$BODY" | grep "^vllm:prompt_tokens_total " | grep -q " 0$" && echo 1 || echo 0)"

# JSON feed shape
JSTATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/metrics.json")
check "GET /metrics.json with --metrics → 200" "$([ "$JSTATUS" = "200" ] && echo 1 || echo 0)"
JCT=$(curl -s -D - -o /dev/null "$BASE/metrics.json" | grep -i "^content-type:" | tr -d '\r')
check "/metrics.json Content-Type is application/json" \
    "$(echo "$JCT" | grep -q "application/json" && echo 1 || echo 0)"
JBODY=$(curl -s "$BASE/metrics.json")
check "/metrics.json has 'counters' key"   "$(echo "$JBODY" | grep -q '"counters"' && echo 1 || echo 0)"
check "/metrics.json has 'gauges' key"     "$(echo "$JBODY" | grep -q '"gauges"' && echo 1 || echo 0)"
check "/metrics.json has 'histograms' key" "$(echo "$JBODY" | grep -q '"histograms"' && echo 1 || echo 0)"
check "/metrics.json has 'generation_tokens_live'" \
    "$(echo "$JBODY" | grep -q '"generation_tokens_live"' && echo 1 || echo 0)"
check "/metrics.json has 'bucket_counts'"  "$(echo "$JBODY" | grep -q '"bucket_counts"' && echo 1 || echo 0)"
check "/metrics.json 'sessions' is an empty array when idle" \
    "$(echo "$JBODY" | python3 -c "
import json,sys
s = json.load(sys.stdin).get('sessions')
print(1 if isinstance(s, list) and len(s) == 0 else 0)" 2>/dev/null)"


# ════════════════════════════════════════════════════════════════════════════
# Phase 3: After one chat request, counters are non-zero
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 3: after one chat completion ──"

CHAT=$(curl -s -X POST "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"sushi","messages":[{"role":"user","content":"Reply with one word: OK"}],"max_tokens":5,"temperature":0}')

check "chat completion returned a response" \
    "$(echo "$CHAT" | grep -q '"choices"' && echo 1 || echo 0)"

BODY2=$(curl -s "$BASE/metrics")

check "request_success_total == 1 after one request" \
    "$(echo "$BODY2" | grep "^vllm:request_success_total " | grep -q " 1$" && echo 1 || echo 0)"

PT=$(echo "$BODY2" | grep "^vllm:prompt_tokens_total " | awk '{print $2}')
check "prompt_tokens_total > 0 after one request" \
    "$([ -n "$PT" ] && [ "$PT" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"

check "vllm:time_to_first_token_seconds_count == 1" \
    "$(echo "$BODY2" | grep "^vllm:time_to_first_token_seconds_count " | grep -q " 1$" && echo 1 || echo 0)"

check "vllm:e2e_request_latency_seconds_count == 1" \
    "$(echo "$BODY2" | grep "^vllm:e2e_request_latency_seconds_count " | grep -q " 1$" && echo 1 || echo 0)"

TTFT_INF=$(echo "$BODY2" | grep 'vllm:time_to_first_token_seconds_bucket{le="+Inf"}' | awk '{print $2}')
check "TTFT +Inf bucket == 1" \
    "$([ "$TTFT_INF" = "1" ] && echo 1 || echo 0)"

check "request_cancelled_total == 0" \
    "$(echo "$BODY2" | grep "^vllm:request_cancelled_total " | grep -q " 0$" && echo 1 || echo 0)"

# memory_mb must reflect the loaded model footprint (phys_footprint, not
# resident_size). Any loaded model footprints >500 MB.
MEM=$(echo "$BODY2" | grep "^sushi:memory_mb " | awk '{print $2}')
check "sushi:memory_mb > 500 (phys_footprint, not resident_size)" \
    "$([ -n "$MEM" ] && [ "$MEM" -gt 500 ] 2>/dev/null && echo 1 || echo 0)"

# generation_tokens_live (live tok/s source) = completed + in-flight. The gauge
# sampler ticks every 2s, so wait one cadence. With nothing decoding at scrape
# time it must equal generation_tokens_total AND be > 0.
sleep 3
BODY3=$(curl -s "$BASE/metrics")
GEN=$(echo "$BODY3" | grep "^vllm:generation_tokens_total " | awk '{print $2}')
LIVE=$(echo "$BODY3" | grep "^sushi:generation_tokens_live " | awk '{print $2}')
check "generation_tokens_live > 0 after one request (sampler ticked)" \
    "$([ -n "$LIVE" ] && [ "$LIVE" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"
check "generation_tokens_live == generation_tokens_total at rest (no slots decoding)" \
    "$([ -n "$LIVE" ] && [ -n "$GEN" ] && [ "$LIVE" = "$GEN" ] && echo 1 || echo 0)"

# ── Phase 4: prefill is visible WHILE it runs, not only when the request ends ──
#
# Regression: `prompt_tokens_total` and the prefill_time histogram only advance
# at request completion, and generated tokens only accrue during decode. So a
# multi-minute prefill pinned the GPU while the panel showed 0 tok/s decode and
# "—" prefill — the user could not tell a long prefill from a hung server.
# `sushi:prefill_tokens_live` is the missing signal.
echo ""
echo "── Phase 4: live prefill gauge ──"

# At rest, no prefill is in flight.
sleep 3
IDLE_PRE=$(curl -s "$BASE/metrics" | grep "^sushi:prefill_tokens_live " | awk '{print $2}')
check "prefill_tokens_live == 0 at rest" \
    "$([ "$IDLE_PRE" = "0" ] && echo 1 || echo 0)"

# Build a prompt big enough that prefill spans several chunks (chunk = 8192
# tokens), then poll the gauge WHILE the request is still in flight.
BIG=$(python3 -c "print(('The quick brown fox jumps over the lazy dog. ' * 2600).strip())")
REQ=$(python3 -c "
import json,sys
print(json.dumps({'model':'sushi','stream':False,'max_tokens':1,'temperature':0,
                  'messages':[{'role':'user','content':sys.stdin.read()}]}))" <<< "$BIG")

curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ" >/dev/null &
CURL_PID=$!

SAW_LIVE=0
SAW_PHASE=0
MAX_SEEN=0
for _ in $(seq 1 200); do
    kill -0 $CURL_PID 2>/dev/null || break     # request finished
    read -r V P <<< "$(curl -s -m 2 "$BASE/metrics.json" | python3 -c "
import json,sys
try:
    g = json.load(sys.stdin)['gauges']
    print(g.get('prefill_tokens_live', 0), g.get('requests_prefilling', 0))
except Exception: print(0, 0)" 2>/dev/null)"
    [ -n "$P" ] && [ "$P" -gt 0 ] 2>/dev/null && SAW_PHASE=1
    [ -n "$V" ] && [ "$V" -gt "$MAX_SEEN" ] 2>/dev/null && MAX_SEEN=$V
    [ -n "$V" ] && [ "$V" -gt 0 ] 2>/dev/null && SAW_LIVE=1 && break
    sleep 0.5
done
wait $CURL_PID 2>/dev/null

check "prefill_tokens_live > 0 DURING a long prefill (saw $MAX_SEEN tokens in flight)" "$SAW_LIVE"
check "requests_prefilling was 1 during the prefill (phase visible before the first chunk)" "$SAW_PHASE"

# ...and it returns to 0 once the prefill is done.
sleep 3
DONE_PRE=$(curl -s "$BASE/metrics" | grep "^sushi:prefill_tokens_live " | awk '{print $2}')
DONE_PHASE=$(curl -s "$BASE/metrics" | grep "^sushi:requests_prefilling " | awk '{print $2}')
check "prefill_tokens_live back to 0 after the request completes" \
    "$([ "$DONE_PRE" = "0" ] && echo 1 || echo 0)"
check "requests_prefilling back to 0 after the request completes" \
    "$([ "$DONE_PHASE" = "0" ] && echo 1 || echo 0)"

# ── Phase 5: prefill throughput must exclude prefix-cache restores ──
#
# `prompt_tokens_total` bills every prompt token; `prefill_time_seconds` only
# ticks for tokens actually forwarded. Dividing the first by the second inflated
# the panel's prefill tok/s by prompt/(prompt-cached) — 10.6x on a warm
# multi-turn 35B MoE session (9.8K tok/s reported, ~220 real).
echo ""
echo "── Phase 5: prefill tok/s excludes cached tokens ──"

read_counter() { curl -s "$BASE/metrics" | grep "^$1 " | awk '{print $2}'; }

# Same prompt twice: the second request must hit the hot prefix cache.
WARM='{"model":"sushi","max_tokens":1,"temperature":0,"messages":[{"role":"user","content":"Count slowly and describe each number in one clause: one two three four five six seven eight nine ten eleven twelve."}]}'
curl -s -m 120 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$WARM" >/dev/null
P1=$(read_counter "vllm:prompt_tokens_total")
F1=$(read_counter "sushi:prefill_tokens_total")

curl -s -m 120 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$WARM" >/dev/null
P2=$(read_counter "vllm:prompt_tokens_total")
F2=$(read_counter "sushi:prefill_tokens_total")
C2=$(read_counter "sushi:prefix_cache_tokens_total")

DP=$((P2 - P1))   # billed prompt tokens of the warm request
DF=$((F2 - F1))   # tokens it actually forwarded

check "warm request still bills its full prompt (prompt_tokens_total +$DP)" \
    "$([ "$DP" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"
check "warm request forwards FEWER tokens than it bills ($DF < $DP)" \
    "$([ "$DF" -lt "$DP" ] 2>/dev/null && echo 1 || echo 0)"
check "prefix_cache_tokens_total > 0 after a cache hit" \
    "$([ -n "$C2" ] && [ "$C2" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"
# The invariant that makes prefill tok/s trustworthy.
check "forwarded + restored == billed ($F2 + $C2 == $P2)" \
    "$([ $((F2 + C2)) -eq "$P2" ] 2>/dev/null && echo 1 || echo 0)"

# The panel divides by this counter; it must never exceed the billed total.
check "prefill_tokens_total <= prompt_tokens_total" \
    "$([ "$F2" -le "$P2" ] 2>/dev/null && echo 1 || echo 0)"

# ── Phase 6: per-request live sessions ──
#
# `/metrics.json` ends with one row per live request (mlx-serve port): phase,
# context occupancy against the model's limit, and a submit-stable request_id;
# then one `cached` row per hot-cache entry no live row restored from.
echo ""
echo "── Phase 6: live per-request sessions ──"

REQ6=$(python3 -c "
import json,sys
print(json.dumps({'model':'sushi','stream':False,'max_tokens':64,'temperature':0,
                  'messages':[{'role':'user','content':sys.stdin.read()}]}))" <<< "$BIG")
curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ6" >/dev/null &
CURL_PID=$!

SAW_ROW=0
STABLE_ID=0
GOOD_ROW=0
PREV_ID=""
for _ in $(seq 1 200); do
    kill -0 $CURL_PID 2>/dev/null || break     # request finished
    read -r R N G <<< "$(curl -s -m 2 "$BASE/metrics.json" | python3 -c "
import json,sys
try:
    ss = [r for r in json.load(sys.stdin)['sessions'] if r['phase'] != 'cached']
    if ss:
        r = ss[0]
        good = (r['context_length'] > 0 and r['max_tokens'] == 64 and
                r['elapsed_seconds'] > 0 and r['phase'] in ('prefill', 'decode'))
        print(r['request_id'], len(ss), 1 if good else 0)
    else: print(0, 0, 0)
except Exception: print(0, 0, 0)" 2>/dev/null)"
    if [ -n "$R" ] && [ "$R" != "0" ]; then
        SAW_ROW=1
        [ "$G" = "1" ] && GOOD_ROW=1
        if [ -z "$PREV_ID" ]; then PREV_ID=$R
        elif [ "$PREV_ID" = "$R" ]; then STABLE_ID=1; fi
        [ "$STABLE_ID" = "1" ] && break
    fi
    sleep 0.2
done
wait $CURL_PID 2>/dev/null

check "sessions row appears while a request is live" "$SAW_ROW"
check "row carries context_length, requested max_tokens, age, prefill|decode phase" "$GOOD_ROW"
check "request_id is stable across two polls of one request" "$STABLE_ID"

sleep 2
REST6=$(curl -s "$BASE/metrics.json")
check "no live rows again at rest (no resurrected rows)" \
    "$(echo "$REST6" | python3 -c "
import json,sys
print(1 if all(r['phase'] == 'cached' for r in json.load(sys.stdin)['sessions']) else 0)" 2>/dev/null)"
check "the finished request left cached rows with zero request fields and an entry's bytes" \
    "$(echo "$REST6" | python3 -c "
import json,sys
ss = json.load(sys.stdin)['sessions']
ok = any(r['state_bytes'] > 0 for r in ss) and all(
    r['request_id'] == 0 and r['max_tokens'] == 0 and r['elapsed_seconds'] == 0 and
    r['generated_tokens'] == 0 and r['context_tokens'] == r['cached_tokens'] > 0 and
    r['context_length'] > 0 for r in ss)
print(1 if ok else 0)" 2>/dev/null)"

# A repeated prompt restores from the entry its first run committed: while it decodes, that
# entry is listed once, as the live row, never also as a cached row.
cached_lens() { curl -s "$BASE/metrics.json" | python3 -c "
import json,sys
print(' '.join(str(r['context_tokens']) for r in json.load(sys.stdin)['sessions'] if r['phase'] == 'cached'))"; }
REQ7='{"model":"sushi","stream":false,"max_tokens":512,"temperature":0,"messages":[{"role":"user","content":"Count from one to three hundred in words, separated by commas."}]}'
BEFORE7=$(cached_lens)
curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ7" >/dev/null
sleep 1
SEED_LEN=$(python3 -c "
import sys
before, after = sys.argv[1].split(), sys.argv[2].split()
for x in before:
    if x in after: after.remove(x)
print(max(map(int, after)) if after else 0)" "$BEFORE7" "$(cached_lens)")
check "the first run committed a cached entry ($SEED_LEN tokens)" \
    "$([ "$SEED_LEN" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"

curl -s -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ7" >/dev/null &
CURL_PID=$!
RESTORED_POLLS=0
DUP_POLLS=0
for _ in $(seq 1 300); do
    kill -0 $CURL_PID 2>/dev/null || break
    read -r LIVE DUP <<< "$(curl -s -m 2 "$BASE/metrics.json" | python3 -c "
import json,sys
try:
    ss = json.load(sys.stdin)['sessions']
    # Early in the decode only: a poll between the final commit and the cull may see both.
    live = [r for r in ss if r['phase'] == 'decode' and r['cached_tokens'] > 0 and r['generated_tokens'] <= 256]
    dup = any(r['phase'] == 'cached' and r['context_tokens'] == $SEED_LEN for r in ss)
    print(1 if live else 0, 1 if live and dup else 0)
except Exception: print(0, 0)" 2>/dev/null)"
    [ "$LIVE" = "1" ] && RESTORED_POLLS=$((RESTORED_POLLS + 1))
    [ "$DUP" = "1" ] && DUP_POLLS=$((DUP_POLLS + 1))
    sleep 0.2
done
wait $CURL_PID 2>/dev/null
check "the repeated prompt decoded from a restore ($RESTORED_POLLS polls)" \
    "$([ "$RESTORED_POLLS" -gt 0 ] && echo 1 || echo 0)"
check "its restored entry was never listed beside it ($DUP_POLLS duplicate polls)" \
    "$([ "$RESTORED_POLLS" -gt 0 ] && [ "$DUP_POLLS" -eq 0 ] && echo 1 || echo 0)"

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
TOTAL=$((PASS + FAIL))
if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}PASS${NC} $TOTAL/$TOTAL tests passed"
    exit 0
else
    echo -e "${RED}FAIL${NC} $FAIL/$TOTAL tests failed"
    echo ""
    echo "--- Server log (last 20 lines) ---"
    tail -20 "$LOG" 2>/dev/null || true
    exit 1
fi

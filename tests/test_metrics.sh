#!/bin/bash
# sushi-test: tier=live boot=api
# Integration tests for the opt-in observability layer (--metrics):
#   * GET /metrics       — Prometheus text exposition (headless scraping)
#   * GET /metrics.json  — open JSON feed of the same counters
#
# There is NO admin dashboard and NO admin mutations — the feeds are read-only.
#
# The api boot serves with --metrics (the 503-when-off arm is test_metrics_enable.sh [1]). The server
# is shared with other scripts, so counters are asserted as deltas around this script's own requests.
#
# Tests:
#  2. With    --metrics: /metrics → 200 Prometheus text; /metrics.json → 200 JSON.
#  3. After one chat request: counters/histograms increment; live-gauge holds
#                             (live > 0 after a request, live == total at rest).

set -u

. "$(dirname "$0")/lib/boot.sh"; sushi_attach
BASE=$SUSHI_URL
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

# Live rows of /metrics.json: requests another script left running must finish before an at-rest check.
live_rows() {
    curl -s -m 2 "$BASE/metrics.json" | python3 -c "
import json,sys
try: print(sum(1 for r in json.load(sys.stdin)['sessions'] if r['phase'] != 'cached'))
except Exception: print(-1)" 2>/dev/null
}
wait_rest() {
    for _ in $(seq 1 120); do
        [ "$(live_rows)" = "0" ] && return 0
        sleep 0.5
    done
    echo "  NOTE  /metrics.json still lists live sessions after 60s"
}

# A counter's value as an integer; empty when the series is missing.
counter() { curl -s "$BASE/metrics" | grep "^$1 " | awk '{print $2}' | cut -d. -f1; }
is_int() { case "$1" in '' | *[!0-9]*) return 1 ;; *) return 0 ;; esac; }

# ════════════════════════════════════════════════════════════════════════════
# Phase 2: With --metrics, both endpoints answer
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 2: with --metrics (idle — no requests yet) ──"

# The boot predates the attach, so the startup line is read from the whole log.
check "startup log: 'Prometheus metrics: ENABLED'" \
    "$(grep -q "Prometheus metrics: ENABLED" "$SUSHI_LOG" && echo 1 || echo 0)"

wait_rest
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

check "request_success_total is a non-negative integer" \
    "$(is_int "$(counter vllm:request_success_total)" && echo 1 || echo 0)"
check "prompt_tokens_total is a non-negative integer" \
    "$(is_int "$(counter vllm:prompt_tokens_total)" && echo 1 || echo 0)"

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
check "/metrics.json 'sessions' is an array with no live row when idle" \
    "$(echo "$JBODY" | python3 -c "
import json,sys
s = json.load(sys.stdin).get('sessions')
print(1 if isinstance(s, list) and all(r['phase'] == 'cached' for r in s) else 0)" 2>/dev/null)"


# ════════════════════════════════════════════════════════════════════════════
# Phase 3: After one chat request, counters are non-zero
# ════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 3: after one chat completion ──"

SUCCESS0=$(counter vllm:request_success_total)
PT0=$(counter vllm:prompt_tokens_total)
TTFT0=$(counter vllm:time_to_first_token_seconds_count)
E2E0=$(counter vllm:e2e_request_latency_seconds_count)
INF0=$(counter 'vllm:time_to_first_token_seconds_bucket{le="+Inf"}')
CANCEL0=$(counter vllm:request_cancelled_total)

CHAT=$(curl -s -X POST "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"sushi\",\"messages\":[{\"role\":\"user\",\"content\":\"$SUSHI_NONCE Reply with one word: OK\"}],\"max_tokens\":5,\"temperature\":0}")

check "chat completion returned a response" \
    "$(echo "$CHAT" | grep -q '"choices"' && echo 1 || echo 0)"

BODY2=$(curl -s "$BASE/metrics")

delta() { echo $(($(counter "$1") - $2)); }

check "request_success_total grew by 1 after one request" \
    "$([ "$(delta vllm:request_success_total "$SUCCESS0")" = "1" ] && echo 1 || echo 0)"

check "prompt_tokens_total grew after one request" \
    "$([ "$(delta vllm:prompt_tokens_total "$PT0")" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"

check "vllm:time_to_first_token_seconds_count grew by 1" \
    "$([ "$(delta vllm:time_to_first_token_seconds_count "$TTFT0")" = "1" ] && echo 1 || echo 0)"

check "vllm:e2e_request_latency_seconds_count grew by 1" \
    "$([ "$(delta vllm:e2e_request_latency_seconds_count "$E2E0")" = "1" ] && echo 1 || echo 0)"

check "TTFT +Inf bucket grew by 1" \
    "$([ "$(delta 'vllm:time_to_first_token_seconds_bucket{le="+Inf"}' "$INF0")" = "1" ] && echo 1 || echo 0)"

check "request_cancelled_total did not move" \
    "$([ "$(delta vllm:request_cancelled_total "$CANCEL0")" = "0" ] && echo 1 || echo 0)"

# memory_mb must reflect the loaded model footprint (phys_footprint, not
# resident_size). Any loaded model footprints >500 MB.
MEM=$(echo "$BODY2" | grep "^sushi:memory_mb " | awk '{print $2}')
check "sushi:memory_mb > 500 (phys_footprint, not resident_size)" \
    "$([ -n "$MEM" ] && [ "$MEM" -gt 500 ] 2>/dev/null && echo 1 || echo 0)"

# generation_tokens_live (live tok/s source) = completed + in-flight. The gauge
# sampler ticks every 2s, so wait one cadence. With nothing decoding at scrape
# time it must equal generation_tokens_total AND be > 0.
wait_rest
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
wait_rest
sleep 3
IDLE_PRE=$(curl -s "$BASE/metrics" | grep "^sushi:prefill_tokens_live " | awk '{print $2}')
check "prefill_tokens_live == 0 at rest" \
    "$([ "$IDLE_PRE" = "0" ] && echo 1 || echo 0)"

# Build a prompt big enough that prefill spans several chunks (chunk = 8192
# tokens), then poll the gauge WHILE the request is still in flight.
BIG=$(python3 -c "import sys; print(sys.argv[1] + ' ' + ('The quick brown fox jumps over the lazy dog. ' * 2600).strip())" "$SUSHI_NONCE")
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

# Restores below 256 tokens never come from the SSD tier (the api boot has no RAM tier), so the prompt is
# longer than that. The first run's SSD write is asynchronous: repeat until a request reports a restore.
WARM=$(python3 -c "
import json,sys
print(json.dumps({'model':'sushi','max_tokens':1,'temperature':0,
                  'messages':[{'role':'user','content':sys.argv[1] + ' Count slowly: ' + ' '.join('n%d' % i for i in range(700))}]}))" "$SUSHI_NONCE")
CACHED=0
for _ in $(seq 1 8); do
    P1=$(counter vllm:prompt_tokens_total)
    F1=$(counter sushi:prefill_tokens_total)
    C1=$(counter sushi:prefix_cache_tokens_total)
    CACHED=$(curl -s -m 120 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$WARM" \
        | python3 -c "
import json,sys
try: print(json.load(sys.stdin)['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0))
except Exception: print(0)")
    P2=$(counter vllm:prompt_tokens_total)
    F2=$(counter sushi:prefill_tokens_total)
    C2=$(counter sushi:prefix_cache_tokens_total)
    [ "$CACHED" -gt 0 ] 2>/dev/null && break
    sleep 1
done

DP=$((P2 - P1))   # billed prompt tokens of the last request
DF=$((F2 - F1))   # tokens it actually forwarded
DC=$((C2 - C1))   # tokens it restored

check "warm request still bills its full prompt (prompt_tokens_total +$DP)" \
    "$([ "$DP" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"
check "warm request forwards FEWER tokens than it bills ($DF < $DP)" \
    "$([ "$DF" -lt "$DP" ] 2>/dev/null && echo 1 || echo 0)"
check "prefix_cache_tokens_total grew after a cache hit (+$DC)" \
    "$([ "$DC" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"
# The invariant that makes prefill tok/s trustworthy.
check "forwarded + restored == billed ($DF + $DC == $DP)" \
    "$([ $((DF + DC)) -eq "$DP" ] 2>/dev/null && echo 1 || echo 0)"

# The panel divides by this counter; it must never exceed the billed total.
check "prefill_tokens_total grew by no more than prompt_tokens_total" \
    "$([ "$DF" -le "$DP" ] 2>/dev/null && echo 1 || echo 0)"

# ── Phase 6: per-request live sessions ──
#
# `/metrics.json` ends with one row per live request (mlx-serve port): phase,
# context occupancy against the model's limit, and a submit-stable request_id;
# then one `cached` row per hot-cache entry no live row restored from.
echo ""
echo "── Phase 6: live per-request sessions ──"

BIG6=$(python3 -c "import sys; print(sys.argv[1] + ' six ' + ('The quick brown fox jumps over the lazy dog. ' * 1300).strip())" "$SUSHI_NONCE")
REQ6=$(python3 -c "
import json,sys
print(json.dumps({'model':'sushi','stream':False,'max_tokens':64,'temperature':0,
                  'messages':[{'role':'user','content':sys.stdin.read()}]}))" <<< "$BIG6")
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

wait_rest
sleep 2
REST6=$(curl -s "$BASE/metrics.json")
check "no live rows again at rest (no resurrected rows)" \
    "$(echo "$REST6" | python3 -c "
import json,sys
print(1 if all(r['phase'] == 'cached' for r in json.load(sys.stdin)['sessions']) else 0)" 2>/dev/null)"

# Cached rows list the RAM hot cache, which the SSD-only api boot does not keep.
RAM_ON=$(curl -s "$BASE/props" | python3 -c "
import json,sys
print(1 if json.load(sys.stdin)['settings']['prefix_cache']['ram_enabled'] else 0)" 2>/dev/null)
if [ "$RAM_ON" != "1" ]; then
    echo "  NOTE  RAM prefix tier off: the cached-row and restored-entry checks are skipped"
else
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
REQ7="{\"model\":\"sushi\",\"stream\":false,\"max_tokens\":512,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"$SUSHI_NONCE Count from one to three hundred in words, separated by commas.\"}]}"
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
fi

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
    sushi_log_since | tail -20
    exit 1
fi

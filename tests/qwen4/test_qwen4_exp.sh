#!/usr/bin/env bash
# sushi-test: tier=live boot=exact
# Qwen3.8-Flash-Next (qwen4_exp) live end-to-end on the exact boot: architecture advertised, a short greedy
# answer, a tool call + round-trip, streaming delta cleanliness, a prompt past the QSA budget (2048 tokens)
# asserted through the server's own `[qsa] sparse attention engaged` line since a dense fallback answers
# plausibly too, the MTP head, and batched decode (plain slots, long slots, an MTP slot beside a plain one).
set -u
. "$(dirname "$0")/../lib/boot.sh"
sushi_attach
[ -f "$SUSHI_PACK/ngram_table.bin" ] || { echo "SKIP $(basename "$0"): pack has no ngram_table.bin"; exit 3; }
U="$SUSHI_URL"
LOG="$SUSHI_LOG"
N="$SUSHI_NONCE"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
# Engagement lines are logged once per process, so a count over $LOG sees an earlier script's line;
# per-request lines (spec-stats, QsaHistoryGap) are counted over this script's own slice.
slice_count() { sushi_log_since | grep -c "$1"; }
pass=0; fail=0
# Tie-aware equivalence (test_mtp_equivalence.sh bar): prints 1 when `other`
# equals the serial greedy answer for body `$1`, or first diverges at a token
# whose serial top-2 gap is <= 0.15 nats.
same_or_tie() {
  local body="$1" other="$2"
  local lp; lp=$(echo "$body" | python3 -c "import sys,json; d=json.load(sys.stdin); d['logprobs']=True; d['top_logprobs']=2; d['enable_mtp']=False; print(json.dumps(d))")
  curl -s -m 1200 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$lp" | python3 -c "
import sys,json; d=json.load(sys.stdin); other=sys.argv[1]; acc=''; ok=1
for n,e in enumerate(d['choices'][0]['logprobs']['content'][:30]):
    if not other.startswith((acc+e['token']).lstrip()):  # content is lead-trimmed, tokens are not
        t=e['top_logprobs']; gap=(t[0]['logprob']-t[1]['logprob']) if len(t)>1 else 99
        print('  diverged at token %d, serial top-2 gap %.3f nats' % (n, gap), file=sys.stderr)
        ok=1 if gap <= 0.15 else 0; break
    acc+=e['token']
print(ok)" "$other"
}
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail+1)); fi; }
# The exact boot pins depth 3 (every MTP round verifies 4 rows, so [5b] exercises the array-mask row split:
# S >= 3 at gqa 12 is MLX's unfused fallback), batches four slots for [8]-[10] and keeps the prefix cache off,
# so the serial reruns those arms compare against never restore (hybrid restore class 0.14-0.30 nats > the near-tie bar).
echo "[1] architecture + n-gram table"
arch=$(curl -s "$U/v1/models" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['meta'].get('architecture',''))")
check "architecture" "$arch" "qwen4_exp"
check "ngram table log" "$(grep -c '\[qwen4\] n-gram table' "$LOG" | sed 's/^[1-9][0-9]*$/1/')" "1"
echo "[2] greedy short answer"
ans=$(curl -s -m 300 "$U/v1/chat/completions" -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"'"$N"' What is the capital of France? Answer with one word."}],"max_tokens":64,"temperature":0,"enable_thinking":false}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])")
check "mentions Paris" "$(echo "$ans" | grep -ci paris | sed 's/^0$/0/;s/^[1-9][0-9]*$/1/')" "1"
echo "[3] tool call + round trip"
body='{"messages":[{"role":"user","content":"'"$N"' What is the weather in Berlin right now? Use the tool."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":512,"temperature":0,"enable_thinking":false}'
tc=$(curl -s -m 600 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$body")
name=$(echo "$tc" | python3 -c "import sys,json; d=json.load(sys.stdin); c=d['choices'][0]['message'].get('tool_calls') or []; print(c[0]['function']['name'] if c else '')")
args_ok=$(echo "$tc" | python3 -c "import sys,json; d=json.load(sys.stdin); c=d['choices'][0]['message'].get('tool_calls') or []; a=json.loads(c[0]['function']['arguments']) if c else {}; print('1' if 'berlin' in json.dumps(a).lower() else '0')")
check "tool name" "$name" "get_weather"
check "args carry the city" "$args_ok" "1"
echo "[4] streaming deltas are clean"
st=$(curl -s -m 300 -N "$U/v1/chat/completions" -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"'"$N"' Count from 1 to 5."}],"max_tokens":48,"temperature":0,"stream":true,"enable_thinking":false}')
check "stream terminates" "$(echo "$st" | grep -c '^data: \[DONE\]')" "1"
check "no markup leak" "$(echo "$st" | grep -c '<tool_call>\|<think>\|<|im_')" "0"
echo "[5] past the QSA budget"
long=$(python3 -c "
import json,sys
filler=' '.join(f'Fact {i}: the sky over city number {i} is blue.' for i in range(700))
print(json.dumps({'messages':[{'role':'user','content':sys.argv[1]+' '+filler+' The secret code is PELICAN-42. '+filler+' What is the secret code? Answer with the code only.'}],'max_tokens':32,'temperature':0,'enable_thinking':False}))" "$N")
la=$(curl -s -m 1200 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$long" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['prompt_tokens'], '|', d['choices'][0]['message']['content'])")
echo "  prompt_tokens|answer: $la"
check "qsa engaged line" "$(grep -c '\[qsa\] sparse attention engaged' "$LOG" | sed 's/^[1-9][0-9]*$/1/')" "1"
check "needle recovered" "$(echo "$la" | grep -c 'PELICAN-42')" "1"
# Assert the invariant (some QSA attention arm engaged), never one kernel's name.
check "qsa prefill attention arm engaged (gather kernel or sushi_attn_p256 mask arm)" "$(grep -cE '\[qsa-gather\] engaged|\[qsa-fused\] engaged' "$LOG" | sed 's/^[1-9][0-9]*$/1/')" "1"
# Decode (S=1) is served by the fused split-K kernel or the decode gather.
check "qsa decode attention arm engaged (S=1 subset rows)" "$(grep -cE '\[qsa-decode-gather\] engaged|\[qsa-attn\] engaged' "$LOG" | sed 's/^[1-9][0-9]*$/1/')" "1"
echo "[5b] MTP past the QSA budget (verify rows under the QSA mask)"
longm=$(echo "$long" | python3 -c "import sys,json; d=json.load(sys.stdin); d['enable_mtp']=True; print(json.dumps(d))")
lm=$(curl -s -m 1200 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$longm" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])")
check "needle recovered under MTP" "$(echo "$lm" | grep -c 'PELICAN-42')" "1"
# Verify widths: the fused kernel serves a quantized cache (this boot's kv8 default); a dense
# cache takes the union gather or the masked split arm. Any of the three satisfies the invariant.
check "verify-width masked attention arm engaged" "$(grep -cE '\[qsa-attn\] engaged|\[qsa-verify-gather\] engaged|\[sdpa-split\] masked arm engaged' "$LOG" | sed 's/^[1-9][0-9]*$/1/')" "1"
echo "[6] MTP head: engagement + greedy equivalence"
base=$(curl -s -m 600 "$U/v1/chat/completions" -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"'"$N"' Write a limerick about a cat."}],"max_tokens":80,"temperature":0,"enable_thinking":false,"enable_mtp":false}' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
mtp=$(curl -s -m 600 "$U/v1/chat/completions" -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"'"$N"' Write a limerick about a cat."}],"max_tokens":80,"temperature":0,"enable_thinking":false,"enable_mtp":true}' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])")
check "mtp engaged" "$(slice_count 'spec-stats\] mode=mtp' | sed 's/^[1-9][0-9]*$/1/')" "1"
if [ "${mtp:0:60}" = "${base:0:60}" ]; then check "mtp == serial (first 60 chars)" 1 1; else
  # Tie-aware acquittal (test_mtp_equivalence.sh): a verify width moves ties, a
  # plumbing bug diverges at a confident position. Serial top-2 gap at the first
  # divergent token must be <= 0.15 nats.
  gap=$(curl -s -m 600 "$U/v1/chat/completions" -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"'"$N"' Write a limerick about a cat."}],"max_tokens":80,"temperature":0,"enable_thinking":false,"enable_mtp":false,"logprobs":true,"top_logprobs":2}' | python3 -c "
import sys,json; d=json.load(sys.stdin); mtp=sys.argv[1]; acc=''
for e in d['choices'][0]['logprobs']['content']:
    if not mtp.startswith(acc+e['token']):
        t=e['top_logprobs']; print(round(t[0]['logprob']-t[1]['logprob'],3) if len(t)>1 else 99); break
    acc+=e['token']
else: print(0)" "$mtp")
  echo "  mtp diverged; serial top-2 gap at the first divergent token: $gap nats"
  check "mtp == serial or near-tie divergence (gap <= 0.15)" "$(python3 -c "print(1 if float('${gap:-99}')<=0.15 else 0)")" "1"
fi
apr=$(sushi_log_since | grep 'spec-stats\] mode=mtp' | tail -1 | sed -E 's/.*avg_per_round=([0-9.]+).*/\1/')
echo "  avg accepted per round: $apr"
check "acceptance floor 0.5/round" "$(python3 -c "print(1 if float('${apr:-0}')>=0.5 else 0)")" "1"
# Widest batched decode group the scheduler reports while $c1 and $c2 run.
peak_group() {
  local m=0 g
  while kill -0 $c1 2>/dev/null || kill -0 $c2 2>/dev/null; do
    g=$(curl -s -m 2 "$U/metrics.json" | grep -o '"batched_group_size":[0-9]*' | cut -d: -f2)
    [ -n "$g" ] && [ "$g" -gt "$m" ] && m=$g
    sleep 0.05
  done
  echo "$m"
}
echo "[8] two concurrent plain requests batch-decode (one forward for both slots)"
pa='{"messages":[{"role":"user","content":"'"$N"' Write a short story about a lighthouse keeper who finds a message in a bottle."}],"max_tokens":400,"temperature":0,"enable_thinking":false,"enable_mtp":false}'
pb='{"messages":[{"role":"user","content":"'"$N"' Explain how a bicycle stays upright while moving, step by step."}],"max_tokens":400,"temperature":0,"enable_thinking":false,"enable_mtp":false}'
curl -s -m 1200 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$pa" > "$T/8a" &
c1=$!
curl -s -m 1200 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$pb" > "$T/8b" &
c2=$!
g8=$(peak_group)
wait $c1 $c2
ta=$(python3 -c "import sys,json; print(json.load(open(sys.argv[1]))['choices'][0]['message']['content'])" "$T/8a")
tb=$(python3 -c "import sys,json; print(json.load(open(sys.argv[1]))['choices'][0]['message']['content'])" "$T/8b")
echo "  A: $(echo "$ta" | tr '\n' ' ' | cut -c1-70)"
echo "  B: $(echo "$tb" | tr '\n' ' ' | cut -c1-70)"
check "batched decode engaged (slots=2)" "$g8" "2"
check "A == serial (30 tokens, tie-aware)" "$(same_or_tie "$pa" "$ta")" "1"
check "B == serial (30 tokens, tie-aware)" "$(same_or_tie "$pb" "$tb")" "1"
echo "[9] two concurrent long prompts: batched decode under the QSA mask"
mk_long() { python3 -c "
import json,sys
code=sys.argv[1]; nonce=sys.argv[2]
filler=' '.join(f'Note {i}: the {nonce} river near town number {i} runs east.' for i in range(700))
print(json.dumps({'messages':[{'role':'user','content':sys.argv[3]+' '+filler+' The secret code is '+code+'. '+filler+' What is the secret code? Answer with the code only.'}],'max_tokens':24,'temperature':0,'enable_thinking':False,'enable_mtp':False}))" "$1" "$2" "$N"; }
la9=$(mk_long HERON-77 amber); lb9=$(mk_long OTTER-31 cobalt)
curl -s -m 1200 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$la9" > "$T/9a" &
c1=$!
curl -s -m 1200 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$lb9" > "$T/9b" &
c2=$!
g9=$(peak_group)
wait $c1 $c2
ta=$(python3 -c "import sys,json; print(json.load(open(sys.argv[1]))['choices'][0]['message']['content'])" "$T/9a")
tb=$(python3 -c "import sys,json; print(json.load(open(sys.argv[1]))['choices'][0]['message']['content'])" "$T/9b")
echo "  A: $(echo "$ta" | tr '\n' ' ' | cut -c1-60)  B: $(echo "$tb" | tr '\n' ' ' | cut -c1-60)"
check "needle A" "$(echo "$ta" | grep -c 'HERON-77')" "1"
check "needle B" "$(echo "$tb" | grep -c 'OTTER-31')" "1"
check "batched decode engaged again past the budget" "$g9" "2"
check "A == serial (tie-aware)" "$(same_or_tie "$la9" "$ta")" "1"
check "B == serial (tie-aware)" "$(same_or_tie "$lb9" "$tb")" "1"
echo "[10] an MTP slot stays exclusive while a plain slot decodes beside it"
nm0=$(slice_count 'spec-stats\] mode=mtp')
pm='{"messages":[{"role":"user","content":"'"$N"' List ten European capitals with one fact each."}],"max_tokens":200,"temperature":0,"enable_thinking":false,"enable_mtp":true}'
pp='{"messages":[{"role":"user","content":"'"$N"' Describe the water cycle in five sentences."}],"max_tokens":200,"temperature":0,"enable_thinking":false,"enable_mtp":false}'
curl -s -m 1200 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$pm" > "$T/10m" &
c1=$!
sleep 0.5
curl -s -m 1200 "$U/v1/chat/completions" -H 'content-type: application/json' -d "$pp" > "$T/10p" &
c2=$!
wait $c1 $c2
tm=$(python3 -c "import sys,json; print(json.load(open(sys.argv[1]))['choices'][0]['message']['content'])" "$T/10m")
tp=$(python3 -c "import sys,json; print(json.load(open(sys.argv[1]))['choices'][0]['message']['content'])" "$T/10p")
echo "  mtp: $(echo "$tm" | tr '\n' ' ' | cut -c1-60)  plain: $(echo "$tp" | tr '\n' ' ' | cut -c1-60)"
check "exactly one more mtp engagement" "$(python3 -c "print($(slice_count 'spec-stats\] mode=mtp') - $nm0)")" "1"
check "mtp answer == serial (tie-aware)" "$(same_or_tie "$pm" "$tm")" "1"
check "plain answer == serial (tie-aware)" "$(same_or_tie "$pp" "$tp")" "1"
# The head is module-owned; anything that hands it between slots risks `QsaHistoryGap`.
check "no QSA history gap anywhere in the run" "$(slice_count 'QsaHistoryGap')" "0"
# If the adaptive switch fired, the release is a one-shot per request.
n_rel=$(slice_count 'sticky-serial: module head released')
if [ "$n_rel" -gt 0 ]; then
  echo "  sticky-serial releases this run: $n_rel"
  check "release never repeats within a request" "$(python3 -c "print(1 if $n_rel <= $(slice_count 'adaptive: .* -> serial (from the next round)') else 0)")" "1"
else
  echo "  (adaptive serial switch did not fire this run — release path not exercised)"
fi
echo "passed $pass failed $fail"
[ "$fail" = 0 ]

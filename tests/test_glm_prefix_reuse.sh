#!/bin/bash
. "$(dirname "$0")/private_cache.sh"
# test_glm_prefix_reuse.sh — GLM-5.3-Flash prefix reuse: KDA checkpoints on the prefill chunk grid
# and at the prompt end, MLA rows below them (docs/glm5-prefix-cache.md#glm). Pins, greedy, one
# hot entry:
#
#  1. An identical re-issue restores at the prompt-end checkpoint and answers byte for byte as its
#     cold run (DFlash2 on).
#  2. A turn that diverges far inside the prompt restores on the chunk grid and its logprobs equal
#     the same prompt prefilled cold, bit for bit.
#  3. A turn that appends to the conversation restores at the prompt-end checkpoint, faster than
#     cold, and its logprobs stay within the measured bound of the cold run up to any flip.
#  4. A BF16-latent request never restores a kv8 entry.
#  5. A stream cancelled mid-decode still commits its prompt for the next request.
#
# Every cold arm starts from a cache holding nothing it shares: an unrelated raw completion
# evicts the one entry first.
#
# Env: SUSHI_MODELS_DIR (default $HOME/.sushi/models), GLM_MODEL, PORT (default 19091), BINARY.

set -uo pipefail

MODEL="${GLM_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/GLM-5.3-Flash-Sushi-2.3bpw}"
PORT="${PORT:-19091}"
BIN="${BINARY:-./zig-out/bin/sushi}"
BASE="http://127.0.0.1:$PORT"
BOUND="${GLM_RESTORE_BOUND_NATS:-0.5}"

[ -d "$MODEL" ] || { echo "SKIP: model dir not found: $MODEL"; exit 0; }
[ -x "$BIN" ]   || { echo "fail: build sushi first"; exit 1; }
command -v jq >/dev/null || { echo "needs jq"; exit 1; }
curl -sf --max-time 2 "$BASE/health" >/dev/null 2>&1 && { echo "fail: port $PORT is busy"; exit 1; }

LOG="${GLM_REUSE_LOG:-$(mktemp)}"
SERVER_PID=""
trap '[ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; [ -z "${GLM_REUSE_LOG:-}" ] && rm -f "$LOG"; true' EXIT

"$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
    --prefix-cache-entries 1 --prefix-cache-mem 1GB --prefix-cache-disk off --log-level info > "$LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 900); do
    curl -sf --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"' && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "fail: server died:"; tail -20 "$LOG"; exit 1; }
    sleep 1
done
grep -q '\[glm\] prefix cache' "$LOG" || { echo "fail: the GLM prefix cache did not arm"; grep -E '\[glm\]|prefix' "$LOG" | head; exit 1; }

words() { python3 -c "import random,sys; r=random.Random($1); w='agent tool file cache ring window layer prompt reply token session restore commit budget chunk kernel latent pool state'.split(); print(' '.join(r.choice(w) for _ in range($2)))"; }
SYSTEM="You are a careful engineer. Notes: $(words 1 2200)"
Q1="Task: $(words 2 400). Summarise the notes in two sentences."
Q1_EDIT="Task: $(words 3 400). Name three words the notes repeat most."
REPLY="The notes repeat a handful of systems words in random order."
Q2="Now list them as a numbered list."
Q3="Task: $(words 4 400). Write a long essay about these notes."

ask() {
    curl -sf --max-time 1800 -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$1"
}
# kv: per-request kv_quant (empty = the server default); lp: logprobs (serial decode when on).
chat() {
    jq -nc --arg s "$SYSTEM" --arg q "$1" --arg r "$2" --arg q2 "$3" --argjson lp "$4" --arg kv "${5:-}" '
        {messages:([{role:"system",content:$s},{role:"user",content:$q}]
            + (if $r == "" then [] else [{role:"assistant",content:$r},{role:"user",content:$q2}] end)),
         max_tokens:(if $lp then 400 else 96 end),temperature:0,stream:false,reasoning_effort:"low"}
        + (if $lp then {logprobs:true,top_logprobs:1} else {} end)
        + (if $kv == "" then {} else {kv_quant:($kv|tonumber)} end)'
}
unrelated() { curl -sf --max-time 600 -X POST "$BASE/v1/completions" -H 'Content-Type: application/json' \
    -d '{"prompt":"Lorem ipsum dolor sit amet, consectetur adipiscing elit.","max_tokens":4,"temperature":0}' >/dev/null; }
text() { echo "$1" | jq -r '(.choices[0].message.reasoning_content // "") + "\u0001" + (.choices[0].message.content // "")'; }
field() { echo "$1" | jq -r "$2"; }
drift() {
    python3 - "$1" "$2" <<'PY'
import json, sys
a, b = (json.loads(x)["choices"][0]["logprobs"]["content"] for x in sys.argv[1:3])
n, worst = 0, 0.0
while n < min(len(a), len(b)) and a[n]["token"] == b[n]["token"]:
    worst = max(worst, abs(a[n]["logprob"] - b[n]["logprob"]))
    n += 1
print(f"{n} {len(a)} {worst:.6f}")
PY
}

EC=0
fail() { echo "FAIL: $*"; EC=1; }
need() { [ -n "$1" ] || { echo "fail: $2"; tail -30 "$LOG"; exit 1; }; }

# Cold references first, each behind an eviction.
T2_COLD=$(ask "$(chat "$Q1" "$REPLY" "$Q2" true)"); need "$T2_COLD" "cold turn 2"
unrelated || { echo "fail: unrelated completion"; exit 1; }
E_COLD=$(ask "$(chat "$Q1_EDIT" "" "" true)"); need "$E_COLD" "cold edited turn"
unrelated || { echo "fail: unrelated completion"; exit 1; }
T1_COLD=$(ask "$(chat "$Q1" "" "" false)"); need "$T1_COLD" "cold turn 1"

# 1. Identical re-issue.
T1_WARM=$(ask "$(chat "$Q1" "" "" false)"); need "$T1_WARM" "warm turn 1"
# 2. The edited task diverges early in the user message: the restore lands on the chunk grid.
E_WARM=$(ask "$(chat "$Q1_EDIT" "" "" true)"); need "$E_WARM" "warm edited turn"
# 3. The appended turn restores off turn 1's entry, re-seated after an eviction.
unrelated || { echo "fail: unrelated completion"; exit 1; }
T1_AGAIN=$(ask "$(chat "$Q1" "" "" false)"); need "$T1_AGAIN" "turn 1 again"
T2_WARM=$(ask "$(chat "$Q1" "$REPLY" "$Q2" true)"); need "$T2_WARM" "warm turn 2"
# 4. Another latent width.
BF16=$(ask "$(chat "$Q1" "$REPLY" "$Q2" false 16)"); need "$BF16" "BF16 turn 2"

# 5. Cancel a stream once it decodes, then ask the same again.
unrelated || { echo "fail: unrelated completion"; exit 1; }
python3 - "$BASE" "$(jq -nc --arg s "$SYSTEM" --arg q "$Q3" '{messages:[{role:"system",content:$s},{role:"user",content:$q}],max_tokens:2000,temperature:0,stream:true,reasoning_effort:"low"}')" <<'PY'
import sys, urllib.request
req = urllib.request.Request(sys.argv[1] + "/v1/chat/completions", data=sys.argv[2].encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=1800) as r:
    deltas = 0
    for line in r:
        if line.startswith(b"data: ") and b'"delta"' in line:
            deltas += 1
            if deltas >= 24:
                break
PY
sleep 3
CANCELLED_AGAIN=$(ask "$(jq -nc --arg s "$SYSTEM" --arg q "$Q3" '{messages:[{role:"system",content:$s},{role:"user",content:$q}],max_tokens:8,temperature:0,stream:false,reasoning_effort:"low"}')"); need "$CANCELLED_AGAIN" "re-issue after a cancel"

for n in T1_COLD T1_WARM E_COLD E_WARM T2_COLD T2_WARM BF16 CANCELLED_AGAIN; do
    echo "$n: prompt_n=$(field "${!n}" .timings.prompt_n) cached_n=$(field "${!n}" .timings.cached_n) prompt_ms=$(field "${!n}" .timings.prompt_ms) predicted_per_second=$(field "${!n}" .timings.predicted_per_second)"
done

for n in T1_COLD E_COLD T2_COLD; do [ "$(field "${!n}" .timings.cached_n)" = 0 ] || fail "$n restored a prefix"; done
[ "$(field "$T1_WARM" .timings.cached_n)" -gt 0 ] || fail "identical re-issue cold-prefilled"
[ "$(text "$T1_COLD")" = "$(text "$T1_WARM")" ] || fail "identical re-issue diverged from its cold run"

[ "$(field "$E_WARM" .timings.cached_n)" -gt 2047 ] || fail "edited turn did not restore the system prompt"
[ $(( $(field "$E_WARM" .timings.cached_n) % 2048 )) = 0 ] || fail "edited turn restored off the chunk grid"
read -r AGREE TOTAL WORST <<< "$(drift "$E_COLD" "$E_WARM")"
echo "edited turn cold vs warm: tokens agree for $AGREE of $TOTAL; max |dlogprob| $WORST nats"
# Logprobs describe the answer only; the reasoning before it is compared as text.
[ "$TOTAL" -gt 0 ] || fail "the edited turn returned no answer tokens to compare"
[ "$(text "$E_COLD")" = "$(text "$E_WARM")" ] || fail "the edited turn's reasoning or answer differs from cold"
[ "$AGREE" = "$TOTAL" ] && [ "$WORST" = "0.000000" ] || fail "a restore on the chunk grid is not bit-identical to cold"

[ "$(field "$T2_WARM" .timings.cached_n)" -gt 0 ] || fail "appended turn cold-prefilled"
read -r AGREE TOTAL WORST <<< "$(drift "$T2_COLD" "$T2_WARM")"
echo "appended turn cold vs warm: tokens agree for $AGREE of $TOTAL; max |dlogprob| $WORST nats"
[ "$TOTAL" -gt 0 ] || fail "the appended turn returned no answer tokens to compare"
python3 -c "import sys; sys.exit(0 if float('$WORST') <= $BOUND else 1)" || fail "restored turn 2 drifts past $BOUND nats"
python3 -c "import sys; sys.exit(0 if $(field "$T2_WARM" .timings.prompt_ms) < 0.5 * $(field "$T2_COLD" .timings.prompt_ms) else 1)" \
    || fail "warm turn 2 prefill not under 0.5x cold"
[ "$(field "$BF16" .timings.cached_n)" = 0 ] || fail "a BF16 request restored a kv8 entry"
[ "$(field "$CANCELLED_AGAIN" .timings.cached_n)" -gt 2047 ] || fail "a stream cancelled mid-decode did not commit its prompt"

grep -E '\[hot-cache\]|\[spec-stats\] (accept|glm)' "$LOG" | head -40
[ $EC = 0 ] && echo "PASS"
exit $EC

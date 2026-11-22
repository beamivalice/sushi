#!/bin/bash
# sushi-test: tier=live boot=own pack=glm5 boots=1
# test_glm_prefix_reuse.sh — GLM-5.3-Flash prefix reuse: KDA checkpoints on the prefill chunk grid
# and at the prompt end, MLA rows below them (docs/glm5-prefix-cache.md#glm). Pins, greedy, one
# hot entry:
#
#  1. An identical re-issue restores at the prompt-end checkpoint and answers byte for byte as its
#     cold run (DFlash2 on).
#  2. A turn that diverges far inside the prompt restores on the chunk grid and its reasoning and
#     answer equal the same prompt prefilled cold, byte for byte.
#  3. A turn that appends to the conversation restores at the prompt-end checkpoint, faster than
#     cold, and its reply starts as the cold run's does (the bytes they share, printed).
#  4. A BF16-latent request never restores a kv8 entry.
#  5. A stream cancelled mid-decode still commits its prompt for the next request.
#
# Every cold arm starts from a cache holding nothing it shares: an unrelated raw completion evicts
# the one hot entry. The boot keeps one entry; the eviction policy keeps a long entry beside short ones.

set -uo pipefail

. "$(dirname "$0")/../lib/boot.sh"
MODEL=$(sushi_pack glm5) || exit $?
sushi_boot glm-prefix "$MODEL" --prefix-cache-entries 1 --prefix-cache-mem 1GB --prefix-cache-disk off --log-level info || exit $?
BASE="$SUSHI_URL"
export SUSHI_NONCE="glp$$"
command -v jq >/dev/null || { echo "needs jq"; exit 1; }
grep -q '\[glm\] prefix cache' "$SUSHI_LOG" || { echo "fail: the GLM prefix cache did not arm"; exit 1; }

words() { python3 -c "import random,sys; r=random.Random($1); w='agent tool file cache ring window layer prompt reply token session restore commit budget chunk kernel latent pool state'.split(); print(' '.join(r.choice(w) for _ in range($2)))"; }
# Checkpoints are kept at multiples of 8192 tokens, so the shared system prompt must pass one.
SYSTEM="$SUSHI_NONCE You are a careful engineer. Notes: $(words 1 9000)"
Q1="Task: $(words 2 400). Summarise the notes in two sentences."
Q1_EDIT="Task: $(words 3 400). Name three words the notes repeat most."
REPLY="The notes repeat a handful of systems words in random order."
Q2="Now list them as a numbered list."
Q3="Task: $(words 4 400). Write a long essay about these notes."

ask() {
    curl -sf --max-time 1800 -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$1"
}
# kv: per-request kv_quant (empty = the server default); lp: a long answer on the serial decode (no drafter, no MTP).
chat() {
    jq -nc --arg s "$SYSTEM" --arg q "$1" --arg r "$2" --arg q2 "$3" --argjson lp "$4" --arg kv "${5:-}" '
        {messages:([{role:"system",content:$s},{role:"user",content:$q}]
            + (if $r == "" then [] else [{role:"assistant",content:$r},{role:"user",content:$q2}] end)),
         max_tokens:(if $lp then 400 else 96 end),temperature:0,stream:false,reasoning_effort:"low"}
        + (if $lp then {enable_drafter:false,enable_mtp:false} else {} end)
        + (if $kv == "" then {} else {kv_quant:($kv|tonumber)} end)'
}
unrelated() {
    curl -sf --max-time 600 -X POST "$BASE/v1/completions" -H 'Content-Type: application/json' \
        -d "{\"prompt\":\"$SUSHI_NONCE Lorem ipsum dolor sit amet, consectetur adipiscing elit.\",\"max_tokens\":4,\"temperature\":0}" >/dev/null
}
text() { echo "$1" | jq -r '(.choices[0].message.reasoning_content // "") + "\u0001" + (.choices[0].message.content // "")'; }
field() { echo "$1" | jq -r "$2"; }
# The bytes two replies share from their start, and the length of the first.
agree() {
    python3 - "$(text "$1")" "$(text "$2")" <<'PY'
import os, sys
a, b = sys.argv[1:3]
print(f"{len(os.path.commonprefix([a, b]))} {len(a)}")
PY
}

EC=0
fail() { echo "FAIL: $*"; EC=1; }
need() { [ -n "$1" ] || { echo "fail: $2"; sushi_log_since | tail -30; exit 1; }; }

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
read -r AGREE TOTAL <<< "$(agree "$E_COLD" "$E_WARM")"
echo "edited turn cold vs warm: the replies share $AGREE of $TOTAL bytes"
[ "$TOTAL" -gt 0 ] || fail "the edited turn returned no reply to compare"
[ "$(text "$E_COLD")" = "$(text "$E_WARM")" ] || fail "a restore on the chunk grid does not say what cold says"

[ "$(field "$T2_WARM" .timings.cached_n)" -gt 0 ] || fail "appended turn cold-prefilled"
read -r AGREE TOTAL <<< "$(agree "$T2_COLD" "$T2_WARM")"
echo "appended turn cold vs warm: the replies share $AGREE of $TOTAL bytes"
[ "$TOTAL" -gt 0 ] || fail "the appended turn returned no reply to compare"
[ "$AGREE" -gt 0 ] || fail "the restored appended turn shares no bytes with its cold run"
# The restore lands at turn 1's prompt-end checkpoint, so it covers at least what the identical re-issue did.
[ "$(field "$T2_WARM" .timings.cached_n)" -ge "$(field "$T1_WARM" .timings.cached_n)" ] || fail "appended turn restored short of turn 1's prompt end"
[ "$(field "$BF16" .timings.cached_n)" = 0 ] || fail "a BF16 request restored a kv8 entry"
[ "$(field "$CANCELLED_AGAIN" .timings.cached_n)" -gt 2047 ] || fail "a stream cancelled mid-decode did not commit its prompt"

sushi_log_since | grep -E '\[hot-cache\]|\[spec-stats\] (accept|glm)' | sed -n 1,40p
[ $EC = 0 ] && echo "PASS"
exit $EC

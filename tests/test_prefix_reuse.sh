#!/bin/bash
# sushi-test: tier=live boot=api
# Prefix reuse through the api boot's SSD tier (no RAM tier), on every arch:
#
#  A. A repeat of a prompt answers byte-for-byte what its first request did (greedy), and its `cached_n` covers the
#     prompt: the first request of a fresh nonce text is cold, its repeat restores from the SSD entry.
#  B. The next turn of a conversation (turn 1 + its reply + a follow-up) restores turn 1's prompt.
#  C. DeepSeek only: its encoder writes the results of parallel tool calls in the order of the calls. A conversation
#     whose results arrive in another order is the same prompt, so it reuses the prompt of the one sent in call order,
#     on all three APIs. Other models render the client's order: the section is skipped there.
#
# Restore points sit at each prompt's end less a 30-token backoff, so turn 2 is held to turn 1's prompt less a slack,
# not to the reply's tokens: those re-render, and only a tool loop's thought point covers them.
#
# Attaches to the shared api boot: tests/run.sh --tier live --model <m> --only test_prefix_reuse.sh

. "$(dirname "$0")/lib/boot.sh"
sushi_attach
BASE=$SUSHI_URL
FAIL=0
SLACK=64

command -v jq >/dev/null || { echo "SKIP test_prefix_reuse.sh: jq is required"; exit 3; }
if [ "$(curl -sf "$BASE/props" | jq -r '(.settings.prefix_cache.disk_bytes > 0) and (.settings.prefix_cache.ram_enabled | not)')" != true ]; then
    echo "SKIP test_prefix_reuse.sh: the boot has no SSD-only prefix tier ($SUSHI_ARCH)"
    exit 3
fi

# A model that cannot switch thinking off (GLM) answers at its lowest effort instead.
if curl -sf "$BASE/v1/models" | jq -e '.data[0].reasoning_efforts | index("off")' > /dev/null; then
    THINK_OBJ='{"enable_thinking":false}'
else
    THINK_OBJ='{"reasoning_effort":"low"}'
fi

check() { # <description> <condition exit status> [detail]
    if [ "$2" -eq 0 ]; then
        echo "  PASS: $1"
    else
        echo "  FAIL: $1${3:+ ($3)}"
        FAIL=1
    fi
}

# Long enough to pass the SSD tier's 512-token floor (kv_disk_cache.MIN_PERSIST_TOKENS).
background() { # <repeats>
    python3 -I -c 'import sys
para = ("The kingdom of Avalon was beset by trials. Each season brought new "
        "challenges to its people, but the king remained steadfast. ")
print("Background:\n" + para * int(sys.argv[1]))' "$1"
}

chat_body() { # <messages json> <max_tokens>
    jq -nc --argjson msgs "$1" --argjson cap "$2" --argjson think "$THINK_OBJ" \
        '{model:"sushi", messages:$msgs, max_tokens:$cap, temperature:0, stream:false} + $think'
}

# ask <messages json> <max_tokens>: sets R (body), TEXT (reasoning + content), PN, CN, CACHED, REQ_LOG (log slice
# of this request).
ask() {
    local off
    off=$(wc -c < "$SUSHI_LOG" | tr -d ' ')
    R=$(curl -sf -m 900 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$(chat_body "$1" "$2")")
    REQ_LOG=$(tail -c +$((off + 1)) "$SUSHI_LOG")
    TEXT=$(printf '%s' "$R" | jq -r '[.choices[0].message.reasoning_content // "", .choices[0].message.content // ""] | join("\u0001")')
    PN=$(printf '%s' "$R" | jq -r '.usage.prompt_tokens // 0')
    CN=$(printf '%s' "$R" | jq -r '.usage.completion_tokens // 0')
    CACHED=$(printf '%s' "$R" | jq -r '.usage.prompt_tokens_details.cached_tokens // 0')
}

# The SSD entry lands in the background after the response; a repeat before it lands would be cold.
persisted_count() { sushi_log_since | grep -c 'complete on disk'; }
wait_persisted() { # <count before the request>
    local i
    for i in $(seq 1 40); do
        [ "$(persisted_count)" -gt "$1" ] && return 0
        sleep 0.5
    done
    echo "  note: no 'complete on disk' line within 20 s; the restore below decides"
}

# The one restore line every request logs ([dsv41-prefix] on DeepSeek), with a non-empty restore from SSD.
restored_from_ssd() { printf '%s\n' "$REQ_LOG" | grep -Eq '\[(dsv41-)?prefix\] reused [1-9][0-9]* of [0-9]+ prompt tokens from (SSD|the live state)'; }

echo "=== Prefix reuse ($SUSHI_ARCH) ==="

# ── A. Warm equals cold ──
echo
echo "--- A: a repeat answers the first request's bytes and restores from the SSD entry ---"
MSG_A=$(jq -nc --arg p "${SUSHI_NONCE}a Continue the story in exactly 4 sentences.

$(background 60)" '[{role:"user",content:$p}]')

BEFORE=$(persisted_count)
ask "$MSG_A" 40
[ -n "$R" ] || { echo "  FAIL: cold request returned nothing"; exit 1; }
COLD_TEXT=$TEXT; COLD_PN=$PN; COLD_CN=$CN; COLD_CACHED=$CACHED
echo "  cold: prompt_tokens=$COLD_PN completion_tokens=$COLD_CN cached_n=$COLD_CACHED"
check "the first request of a fresh nonce text is cold (cached_n under half the prompt)" "$((COLD_CACHED * 2 < COLD_PN ? 0 : 1))" "cached_n=$COLD_CACHED of $COLD_PN"
check "the cold request generated tokens" "$((COLD_CN > 0 ? 0 : 1))"

wait_persisted "$BEFORE"
ask "$MSG_A" 40
[ -n "$R" ] || { echo "  FAIL: warm request returned nothing"; exit 1; }
echo "  warm: prompt_tokens=$PN completion_tokens=$CN cached_n=$CACHED"
check "warm cached_n > 0" "$((CACHED > 0 ? 0 : 1))" "cached_n=$CACHED"
check "warm cached_n covers at least half the prompt" "$((CACHED * 2 >= PN ? 0 : 1))" "cached_n=$CACHED of $PN"
check "the warm request logs a restore from SSD" "$(restored_from_ssd; echo $?)" "no '[prefix] reused N of M prompt tokens from SSD' line"
if [ "$TEXT" = "$COLD_TEXT" ] && [ "$CN" = "$COLD_CN" ]; then
    check "warm output is byte-identical to cold" 0
else
    check "warm output is byte-identical to cold" 1 "cold: $(printf '%s' "$COLD_TEXT" | head -c 120) | warm: $(printf '%s' "$TEXT" | head -c 120)"
fi

# ── B. The next turn reuses the previous turn ──
echo
echo "--- B: turn 2 (turn 1 + its reply + a follow-up) restores turn 1's prompt ---"
SYS="${SUSHI_NONCE}b You are a helpful assistant. When asked to count, count out loud from 1 to 50, separated by commas, on a single line.

$(background 40)"
M1=$(jq -nc --arg s "$SYS" --arg u "${SUSHI_NONCE}b Count from 1 to 50 as instructed." \
    '[{role:"system",content:$s},{role:"user",content:$u}]')

BEFORE=$(persisted_count)
ask "$M1" 120
[ -n "$R" ] || { echo "  FAIL: turn 1 returned nothing"; exit 1; }
T1_PN=$PN; T1_CN=$CN
echo "  turn 1: prompt_tokens=$T1_PN completion_tokens=$T1_CN cached_n=$CACHED"
check "turn 1 generated a substantial reply (>= 30 tokens)" "$((T1_CN >= 30 ? 0 : 1))" "completion_tokens=$T1_CN"
M2=$(jq -nc --argjson m1 "$M1" --argjson r "$R" '$m1 + [
        ({role:"assistant", content:($r.choices[0].message.content // "")}
         + (if ($r.choices[0].message.reasoning_content // "") != "" then {reasoning_content:$r.choices[0].message.reasoning_content} else {} end)),
        {role:"user", content:"Now do the same but with squares of those numbers."}]')

wait_persisted "$BEFORE"
ask "$M2" 120
[ -n "$R" ] || { echo "  FAIL: turn 2 returned nothing"; exit 1; }
FLOOR=$((T1_PN - SLACK))
echo "  turn 2: prompt_tokens=$PN completion_tokens=$CN cached_n=$CACHED (floor $FLOOR)"
check "turn 2 reuses turn 1's prompt (cached_n >= prompt_tokens - $SLACK)" "$((CACHED >= FLOOR ? 0 : 1))" "cached_n=$CACHED, turn-1 prompt $T1_PN"
check "turn 2 logs a restore from SSD" "$(restored_from_ssd; echo $?)" "no '[prefix] reused N of M prompt tokens from SSD' line"
if [ "$CACHED" -ge $((T1_PN + T1_CN - 8)) ]; then
    echo "  note: turn 2 also reused turn 1's generated tokens (cached_n $CACHED >= $((T1_PN + T1_CN - 8)))"
fi

# ── C. Tool results in another order than the calls ──
# post <path> <body>: sets R and REQ_LOG.
post() {
    local off
    off=$(wc -c < "$SUSHI_LOG" | tr -d ' ')
    R=$(curl -sf -m 900 "$BASE$1" -H 'Content-Type: application/json' -d "$2")
    REQ_LOG=$(tail -c +$((off + 1)) "$SUSHI_LOG")
}
# "<reused> <prompt>" of the request's restore line, "0 0" without one.
reused() { printf '%s\n' "$REQ_LOG" | sed -nE 's/.*prefix\] reused ([0-9]+) of ([0-9]+) prompt tokens.*/\1 \2/p' | tail -1 | grep . || echo "0 0"; }

# order_pair <api name> <path> <body in call order> <body with the two results swapped>
order_pair() {
    local before n m
    before=$(persisted_count)
    post "$2" "$3"
    [ -n "$R" ] || { check "$1: the conversation in call order answers" 1; return; }
    wait_persisted "$before"
    post "$2" "$4"
    [ -n "$R" ] || { check "$1: the conversation with its results swapped answers" 1; return; }
    read -r n m <<< "$(reused)"
    echo "  $1: results swapped, reused $n of $m prompt tokens"
    check "$1: results sent out of call order reuse the in-order prompt (reused >= prompt - $SLACK)" "$((m > 0 && n >= m - SLACK ? 0 : 1))" "reused $n of $m"
}

echo
if [ "$SUSHI_ARCH" != deepseek_v41 ]; then
    echo "--- C: skipped ($SUSHI_ARCH renders tool results in the client's order) ---"
else
    echo "--- C: tool results sent out of call order are the in-order prompt ---"
    # Each result is far longer than the slack, so a prompt that kept the client's order shares neither.
    RES_A="File a.txt: $(background 12)"
    RES_B="File b.txt: $(background 12)"
    ARGS_A='{"path":"a.txt"}'
    ARGS_B='{"path":"b.txt"}'
    PARAMS='{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}'

    oai() { # <first result id> <second result id>
        jq -nc --arg sys "${SUSHI_NONCE}c1 You read files when asked." --arg a "$RES_A" --arg b "$RES_B" --arg aa "$ARGS_A" --arg ab "$ARGS_B" \
            --argjson params "$PARAMS" --arg first "$1" '
            {call_a: {role:"tool", tool_call_id:"call_a", content:$a}, call_b: {role:"tool", tool_call_id:"call_b", content:$b}} as $res
            | {model:"sushi", max_tokens:8, temperature:0, stream:false,
               tools:[{type:"function", function:{name:"read_file", description:"Read a file.", parameters:$params}}],
               messages:([{role:"system", content:$sys}, {role:"user", content:"Read both files."},
                          {role:"assistant", content:"", tool_calls:[
                              {id:"call_a", type:"function", function:{name:"read_file", arguments:$aa}},
                              {id:"call_b", type:"function", function:{name:"read_file", arguments:$ab}}]}]
                         + (if $first == "call_a" then [$res.call_a, $res.call_b] else [$res.call_b, $res.call_a] end))}'
    }
    order_pair "chat completions" /v1/chat/completions "$(oai call_a)" "$(oai call_b)"

    ant() {
        jq -nc --arg sys "${SUSHI_NONCE}c2 You read files when asked." --arg a "$RES_A" --arg b "$RES_B" --argjson aa "$ARGS_A" --argjson ab "$ARGS_B" \
            --argjson params "$PARAMS" --arg first "$1" '
            {call_a: {type:"tool_result", tool_use_id:"call_a", content:$a}, call_b: {type:"tool_result", tool_use_id:"call_b", content:$b}} as $res
            | {model:"sushi", max_tokens:8, temperature:0, system:$sys,
               tools:[{name:"read_file", description:"Read a file.", input_schema:$params}],
               messages:[{role:"user", content:"Read both files."},
                         {role:"assistant", content:[{type:"tool_use", id:"call_a", name:"read_file", input:$aa},
                                                     {type:"tool_use", id:"call_b", name:"read_file", input:$ab}]},
                         {role:"user", content:(if $first == "call_a" then [$res.call_a, $res.call_b] else [$res.call_b, $res.call_a] end)}]}'
    }
    order_pair "messages" /v1/messages "$(ant call_a)" "$(ant call_b)"

    rsp() {
        jq -nc --arg sys "${SUSHI_NONCE}c3 You read files when asked." --arg a "$RES_A" --arg b "$RES_B" --arg aa "$ARGS_A" --arg ab "$ARGS_B" \
            --argjson params "$PARAMS" --arg first "$1" '
            {call_a: {type:"function_call_output", call_id:"call_a", output:$a}, call_b: {type:"function_call_output", call_id:"call_b", output:$b}} as $res
            | {model:"sushi", max_output_tokens:16, temperature:0, store:false, instructions:$sys,
               tools:[{type:"function", name:"read_file", description:"Read a file.", parameters:$params}],
               input:([{role:"user", content:"Read both files."},
                       {type:"function_call", call_id:"call_a", name:"read_file", arguments:$aa},
                       {type:"function_call", call_id:"call_b", name:"read_file", arguments:$ab}]
                      + (if $first == "call_a" then [$res.call_a, $res.call_b] else [$res.call_b, $res.call_a] end))}'
    }
    order_pair "responses" /v1/responses "$(rsp call_a)" "$(rsp call_b)"
fi

echo
if [ "$FAIL" -ne 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL TESTS PASSED"

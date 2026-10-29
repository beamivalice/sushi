#!/bin/bash
# sushi-test: tier=live boot=api
# Guard: continuing a partial assistant reply extends it — it does not restart.
#
# A conversation ending in an assistant message used to render that message as
# HISTORY and then let add_generation_prompt open a SECOND assistant turn, so
# the model answered the doubled turn by starting over. Continuation ends the
# prompt mid-turn instead, on the partial text.
#
# The observable contract, which is what this pins:
#   - the reply CONTINUES rather than repeating the prefix back
#   - the two surfaces ask for it their own documented way: an explicit
#     `continue_final_message` on /v1/chat/completions, an implicit trailing
#     assistant message on /v1/messages (Anthropic's own behaviour)
#   - without the flag, /v1/chat/completions is unchanged — a trailing
#     assistant message is history and a fresh turn is answered
#
# Greedy (temperature 0) throughout: the assertions are about WHERE the prompt
# ends, and a sampled reply makes a "did it restart?" check a coin flip.
#
# Thinking is turned off (GLM refuses that: a 16-token budget), so the continuation is visible text.
. "$(dirname "$0")/lib/boot.sh"; sushi_attach
set -u

BASE="$SUSHI_URL"
PASS=0
FAIL=0
TOTAL=0
command -v jq >/dev/null 2>&1 || { echo "FAIL: jq is required"; exit 1; }
if [ "${SUSHI_ARCH:-}" = glm5_next ]; then
    NT_CHAT='"reasoning_effort": "low", "reasoning_budget_tokens": 16,'
    NT_MSG='"thinking": {"type": "enabled", "budget_tokens": 16},'
else
    NT_CHAT='"enable_thinking": false,'
    NT_MSG='"thinking": {"type": "disabled"},'
fi
USER_MSG="$SUSHI_NONCE Name the three primary colors in one sentence."

run_test() {
    TOTAL=$((TOTAL + 1))
    if [ "$2" = PASS ]; then PASS=$((PASS + 1)); echo "  PASS: $1"
    else FAIL=$((FAIL + 1)); echo "  FAIL: $1 — $3"; fi
}

PARTIAL="The three primary colors are red, blue, and"

echo "=== /v1/chat/completions: continue_final_message ==="
BODY=$(curl -s -m 300 "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"sushi\",
    \"messages\": [
        {\"role\": \"user\", \"content\": \"$USER_MSG\"},
        {\"role\": \"assistant\", \"content\": \"$PARTIAL\"}
    ],
    \"continue_final_message\": true, $NT_CHAT
    \"temperature\": 0, \"max_tokens\": 64, \"stream\": false
}")
CONT=$(echo "$BODY" | jq -r '.choices[0].message.content // ""')

if [ -n "$CONT" ]; then
    run_test "continuation returns content" PASS ""
else
    run_test "continuation returns content" FAIL "empty: $(echo "$BODY" | head -c 200)"
fi

# The whole point: the model is handed its own unfinished sentence, so what
# comes back is the REST of it — never the prefix again. A restart is the
# doubled-turn bug, and it looks exactly like the prefix being repeated.
if echo "$CONT" | grep -qi "three primary colors are"; then
    run_test "continuation does not restate the prefix" FAIL "restarted: $(echo "$CONT" | head -c 120)"
else
    run_test "continuation does not restate the prefix" PASS ""
fi

echo "=== /v1/chat/completions: WITHOUT the flag, the prompt is a full turn ==="
# Measured on the PROMPT, not on the reply. The first cut of this compared the
# two replies and demanded they differ — but "yellow." is the natural next word
# whether the model is finishing the sentence or answering the question afresh,
# so a working feature failed the test. What the flag changes is where the
# prompt ENDS: a continuation stops mid-turn, an ordinary turn closes the
# assistant message and opens a new one, which costs the close tag plus a turn
# header. Structural, and true of every checkpoint.
TOKENS_CONT=$(echo "$BODY" | jq -r '.usage.prompt_tokens // 0')
TOKENS_PLAIN=$(curl -s -m 300 "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"sushi\",
    \"messages\": [
        {\"role\": \"user\", \"content\": \"$USER_MSG\"},
        {\"role\": \"assistant\", \"content\": \"$PARTIAL\"}
    ], $NT_CHAT
    \"temperature\": 0, \"max_tokens\": 64, \"stream\": false
}" | jq -r '.usage.prompt_tokens // 0')

if [ "$TOKENS_CONT" -gt 0 ] && [ "$TOKENS_PLAIN" -gt "$TOKENS_CONT" ]; then
    run_test "the flag shortens the prompt (no close tag, no second turn header)" PASS \
        "" # measured 27 vs 32 on llama-3.2
else
    run_test "the flag shortens the prompt (no close tag, no second turn header)" FAIL \
        "continuation=$TOKENS_CONT plain=$TOKENS_PLAIN — the flag is not reaching the renderer"
fi

echo "=== /v1/messages: a trailing assistant message IS the request ==="
ABODY=$(curl -s -m 300 "$BASE/v1/messages" -H "Content-Type: application/json" \
    -H "anthropic-version: 2023-06-01" -d "{
    \"model\": \"sushi\",
    \"messages\": [
        {\"role\": \"user\", \"content\": \"$USER_MSG\"},
        {\"role\": \"assistant\", \"content\": \"$PARTIAL\"}
    ], $NT_MSG
    \"temperature\": 0, \"max_tokens\": 64, \"stream\": false
}")
ACONT=$(echo "$ABODY" | jq -r '[.content[]? | select(.type=="text") | .text] | join("")')

if [ -n "$ACONT" ]; then
    run_test "/v1/messages continues with no flag" PASS ""
else
    run_test "/v1/messages continues with no flag" FAIL "empty: $(echo "$ABODY" | head -c 200)"
fi
if echo "$ACONT" | grep -qi "three primary colors are"; then
    run_test "/v1/messages does not restate the prefix" FAIL "restarted: $(echo "$ACONT" | head -c 120)"
else
    run_test "/v1/messages does not restate the prefix" PASS ""
fi

echo
echo "=== $PASS/$TOTAL passed ==="
[ "$FAIL" -eq 0 ] || exit 1

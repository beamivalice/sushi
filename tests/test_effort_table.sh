#!/bin/bash
# sushi-test: tier=live boot=own pack=each boots=1
# The per-family thinking-effort table, live on the pass model (Qwen, DeepSeek, GLM; MiMo has no table and skips). One
# boot at --log-level debug with a private HOME; every change goes through POST /v1/settings and reaches the next
# request with no reload:
#   [1] a word remapped (Qwen high -> medium, DeepSeek medium -> 40, GLM low -> max) with cap 64: /v1/models lists its value
#       and cap, the next request renders the new value (`[chat] template kwargs`) and arms the cap of 64, with no
#       effort notice;
#   [2] the request's own `reasoning_budget_tokens` outranks the table's cap.
#   A cap is checked as an invariant, never as how long the model must think: a thought that reaches the cap is closed
#   by `[think-bound]` within the forced close; a model that stops first passes, with a note.
#   [3] `offered: false` refuses the word with a 400 and /v1/models drops it; a built-in fold (Qwen max, GLM xhigh)
#       still folds with its notice;
#   [4] a reset brings the built-in table back: /v1/models equals the boot's rows, the word renders its built-in value
#       and the cap is gone; the file loses its section; the model never reloaded.
#   [5] a numeric effort (`reasoning_effort`, Responses `reasoning.effort`, Anthropic `output_config.effort`) is a 400
#       naming the accepted words, as an unknown word is.
set -uo pipefail
. "$(dirname "$0")/lib/boot.sh"

command -v jq > /dev/null || { echo "needs jq"; exit 1; }
KEY=$(sushi_model_key "${SUSHI_PASS_MODEL:-qwen4}") || exit 1
MODEL=$(sushi_pack "$KEY") || exit $?
ARCH=$(sushi_arch "$MODEL")
case "$ARCH" in
    qwen4_exp) WORD=high ROW='{"offered":true,"sends":"medium","cap":64}' NEW='"reasoning_effort":"medium"' OLD='"reasoning_effort":"xhigh"' VALUE='"medium"' FOLD=max ;;
    deepseek_v41) WORD=medium ROW='{"sends":40,"cap":64}' NEW='"reasoning_effort":40' OLD='"reasoning_effort":62' VALUE=40 FOLD= ;;
    glm5_next) WORD=low ROW='{"sends":"max","cap":64}' NEW='"reasoning_effort":"max"' OLD='"reasoning_effort":"low"' VALUE='"max"' FOLD=xhigh ;;
    *) echo "SKIP $(basename "$0"): $ARCH has no effort table"; exit 3 ;;
esac
REFUSE=low
PROMPT='How many prime numbers are there below 200? Work it out carefully before answering.'
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
pass=0; fail=0
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass + 1)); else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail + 1)); fi; }
has() { if grep -qF -- "$3" <<< "$2"; then echo "  ok   $1"; pass=$((pass + 1)); else echo "  FAIL $1: no '$3'"; fail=$((fail + 1)); fi; }
# capped <cap> <label>: the cap was armed, and either the thought reached it and `[think-bound]` closed it (its
# reasoning tokens within the cap plus the forced close), or the model stopped thinking below it.
CLOSE_SLACK=48
capped() {
    local cap="$1" label="$2" used
    used=$(jq -r '.usage.completion_tokens_details.reasoning_tokens // -1' "$WORK/reply.json")
    has "$label: the cap is armed" "$SEG" "reasoning budget $cap: enforced in-stream"
    if grep -qF "[think-bound] reasoning budget $cap reached" <<< "$SEG"; then
        check "$label: the thought closed at the cap ($used reasoning tokens)" "$((used >= 0 && used <= cap + CLOSE_SLACK))" 1
    else
        echo "  note $label: the model stopped thinking at $used tokens, before the cap of $cap"
        check "$label: the thought ended below the cap" "$((used >= 0 && used <= cap))" 1
    fi
}
lacks() { if grep -qF -- "$3" <<< "$2"; then echo "  FAIL $1: found '$3'"; fail=$((fail + 1)); else echo "  ok   $1"; pass=$((pass + 1)); fi; }

sushi_boot effort "$MODEL" --log-level debug || exit $?
HOME_FILE="$SUSHI_RUN_DIR/effort.home/.sushi/model-settings.json"
ID=$(curl -s "$SUSHI_URL/v1/models" | jq -r '.data[0].id')
row() { curl -s "$SUSHI_URL/v1/models" | jq -c --arg id "$ID" '.data[] | select(.id == $id) | {reasoning_efforts, reasoning_effort_budgets, reasoning_effort_values, default_reasoning_effort}'; }
settings() { curl -s -o "$WORK/post.json" -w '%{http_code}' -H 'Content-Type: application/json' -d "$1" "$SUSHI_URL/v1/settings"; }
# chat <word> [extra fields] [max_tokens]: sets CODE (the HTTP status) and SEG (the request's log); the body lands in $WORK/reply.json.
# Called directly, never inside $(...): the boot state and these results live in this shell.
# The counter keeps two requests in one second apart: an identical render skips the `template kwargs` log line.
SEQ=0
chat() {
    local off
    SEQ=$((SEQ + 1))
    off=$(_sushi_size "$SUSHI_LOG")
    CODE=$(curl -s -o "$WORK/reply.json" -w '%{http_code}' -H 'Content-Type: application/json' -d "{\"model\":\"$ID\",\"messages\":[{\"role\":\"user\",\"content\":\"$(date +%s).$SEQ $PROMPT\"}],\"reasoning_effort\":\"$1\",\"max_tokens\":${3:-700},\"temperature\":0${2:+,$2}}" "$SUSHI_URL/v1/chat/completions")
    SEG=$(_sushi_tail_from "$SUSHI_LOG" "$off")
}
BUILTIN_ROW=$(row)
echo "[$ARCH] built-in row: $BUILTIN_ROW"
check "GET /v1/settings carries the $ARCH table" "$(curl -s "$SUSHI_URL/v1/settings" | jq -r --arg f "$ARCH" '[.effort.families[] | select(.family == $f)] | length')" 1

echo "[1] $WORD remapped with cap 64"
check "the save is accepted" "$(settings "{\"effort\":{\"$ARCH\":{\"$WORD\":$ROW}}}")" 200
check "it applies at once" "$(jq -r '.applied.effort' "$WORK/post.json")" "applies at once, from the next request"
check "/v1/models lists its value" "$(row | jq -c --arg w "$WORD" '.reasoning_effort_values[$w]')" "$VALUE"
check "/v1/models lists its cap" "$(row | jq -r --arg w "$WORD" '.reasoning_effort_budgets[$w]')" 64
chat "$WORD"
check "the request is served" "$CODE" 200
echo "$SEG" | grep -E '\[chat\] template kwargs|\[think-bound\]|\[effort\]' | sed 's/^/      log: /'
has "the request renders the new value" "$SEG" "$NEW"
capped 64 "the table's cap"
check "a mapped word carries no notice" "$(jq -r '.effort_notice // "none"' "$WORK/reply.json")" none

echo "[2] the request's own budget outranks the cap"
chat "$WORD" '"reasoning_budget_tokens":128'
check "the request is served" "$CODE" 200
echo "$SEG" | grep -E '\[think-bound\]' | sed 's/^/      log: /'
capped 128 "the request's budget"

echo "[3] $REFUSE turned off"
check "the save is accepted" "$(settings "{\"effort\":{\"$ARCH\":{\"$REFUSE\":{\"offered\":false}}}}")" 200
chat "$REFUSE"
check "a request for $REFUSE is refused" "$CODE" 400
has "with the usual message" "$(cat "$WORK/reply.json")" "reasoning effort '$REFUSE' is not supported by"
check "/v1/models drops it" "$(row | jq -r --arg w "$REFUSE" '.reasoning_efforts | index($w) // "gone"')" gone
if [ -n "$FOLD" ]; then
    chat "$FOLD" '"reasoning_budget_tokens":32'
    check "$FOLD, which this family has no level for, is still served" "$CODE" 200
    has "with its effort notice" "$(jq -r '.effort_notice // ""' "$WORK/reply.json")" "has no \`$FOLD\` effort"
fi

echo "[4] reset to built-in"
check "the reset is accepted" "$(settings "{\"effort\":{\"$ARCH\":null}}")" 200
check "/v1/models is the boot's row again" "$(row)" "$BUILTIN_ROW"
chat "$WORD" "" 200
check "the request is served" "$CODE" 200
echo "$SEG" | grep -E '\[chat\] template kwargs|\[think-bound\]|\[effort\]' | sed 's/^/      log: /'
has "it renders the built-in value" "$SEG" "$OLD"
lacks "the table's cap is gone" "$SEG" "[think-bound] reasoning budget 64 reached"
check "the file holds no $ARCH table" "$(jq -r --arg s "effort:$ARCH" 'has($s)' "$HOME_FILE" 2> /dev/null || echo false)" false
check "the model loaded once, never reloaded" "$(grep -c 'Model ready' "$SUSHI_LOG")" 1

echo "[5] a numeric effort is refused on every surface, as an unknown word is"
for surface in chat responses messages; do
    case "$surface" in
        chat) path=/v1/chat/completions body="{\"model\":\"$ID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"reasoning_effort\":40,\"max_tokens\":8}" ;;
        responses) path=/v1/responses body="{\"model\":\"$ID\",\"input\":\"hi\",\"reasoning\":{\"effort\":40},\"max_output_tokens\":8}" ;;
        messages) path=/v1/messages body="{\"model\":\"$ID\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"output_config\":{\"effort\":40}}" ;;
    esac
    code=$(curl -s -o "$WORK/refused.json" -w '%{http_code}' -H 'Content-Type: application/json' -d "$body" "$SUSHI_URL$path")
    check "$surface: a numeric effort is a 400" "$code" 400
    has "$surface: the refusal names the accepted words" "$(cat "$WORK/refused.json")" "use one of:"
done

echo "effort table ($ARCH): $pass passed, $fail failed"
[ "$fail" -eq 0 ]

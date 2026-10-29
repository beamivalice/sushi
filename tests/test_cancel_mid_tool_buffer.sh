#!/bin/bash
# sushi-test: tier=live boot=api
# Guard: disconnecting WHILE the stream is buffering for tool detection leaves
# no leak and no wedge.
#
# With tools present the server emits nothing until the pattern resolves — it
# holds tokens, sometimes for a long one-shot write_file. That buffering window
# is the one place a client can vanish while the server owns an unflushed
# buffer AND a live slot, and it is not covered by test_disconnect_cancel.sh
# (which cancels a plain prose stream, where every token has already gone out).
#
# The test drops the connection mid-buffer, several times, then asserts:
#   • the server cancels the slot rather than generating to max_tokens
#   • memory does not climb across the cycles (the buffer is freed)
#   • the server still answers, with tools and without, afterwards
#
# Requests turn thinking off so the 3 s window lands in the buffered tool call; GLM refuses that
# (400), so it runs under a 16-token thought. Whether the model calls the tool is its own choice.
#   CANCEL_ROUNDS=5
. "$(dirname "$0")/lib/boot.sh"; sushi_attach
set -u

ROUNDS="${CANCEL_ROUNDS:-5}"
BASE="$SUSHI_URL"
RED='\033[0;31m'; GREEN='\033[0;32m'; DIM='\033[2m'; NC='\033[0m'
PASS=0; FAIL=0
ok()  { echo -e "  ${GREEN}PASS${NC} $1"; PASS=$((PASS+1)); }
bad() { echo -e "  ${RED}FAIL${NC} $1"; shift; for l in "$@"; do echo "        $l"; done; FAIL=$((FAIL+1)); }

if [ "${SUSHI_ARCH:-}" = glm5_next ]; then
    NT='{"reasoning_effort": "low", "reasoning_budget_tokens": 16}'
else
    NT='{"enable_thinking": false}'
fi

mem_bytes() { curl -sf -m 10 "$BASE/props" 2>/dev/null \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print((d.get('memory') or {}).get('active_bytes') or 0)" 2>/dev/null || echo 0; }

# A request whose answer is ONE big tool call: the server buffers the whole
# thing for detection, so a disconnect lands mid-buffer with high probability.
REQ=$(NT="$NT" python3 <<'PY'
import json, os
body = {"model": "sushi",
 "messages": [{"role": "user", "content": os.environ["SUSHI_NONCE"] + " " +
   "Use write_file RIGHT NOW to create a complete standalone HTML page saved as "
   "mars.html about Mars: full <!DOCTYPE html>, a <head> with a <title>, an "
   "embedded <style> block, an <h1>, and at least eight <p> paragraphs. Call the "
   "tool now; do not ask questions."}],
 "tools": [{"type": "function", "function": {
   "name": "write_file", "description": "Write a file",
   "parameters": {"type": "object", "properties": {
     "path": {"type": "string"}, "content": {"type": "string"}},
     "required": ["path", "content"]}}}],
 "max_tokens": 3000, "temperature": 0.7, "stream": True}
body.update(json.loads(os.environ["NT"]))
print(json.dumps(body))
PY
)

MEM_BEFORE=$(mem_bytes)

for r in $(seq 1 "$ROUNDS"); do
    # --max-time drops the socket mid-generation; the buffer is still unflushed
    # because nothing is emitted until the tool pattern resolves.
    curl -sN -m 3 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
        --data-binary "$REQ" >/dev/null 2>&1
    echo -e "  ${DIM}round $r: disconnected after 3s${NC}"
    sleep 2
done

# Give the server a moment to notice the last drop.
sleep 5

NEW_CANCELS=$(sushi_log_since | grep -c "\[client_disconnect\]")
if [ "$NEW_CANCELS" -ge "$ROUNDS" ]; then
    ok "every disconnect ended its request as [client_disconnect] ($NEW_CANCELS for $ROUNDS rounds)"
else
    bad "disconnects cancel the slot" "$NEW_CANCELS [client_disconnect] endings for $ROUNDS rounds — a slot generated on after the client left"
fi

# Stronger than the log line: the slot must have STOPPED, not run to max_tokens.
# 3 s of generation is a few hundred tokens; max_tokens is 3000.
OVERRUN=$(sushi_log_since | grep -aoE "<- [0-9]+\+([0-9]+) tokens streamed" 2>/dev/null \
    | sed -E 's/.*\+([0-9]+) tokens.*/\1/' | awk '$1 > 2000' | wc -l | tr -d ' ')
if [ "${OVERRUN:-0}" -eq 0 ]; then
    ok "no cancelled request generated on toward max_tokens"
    sushi_log_since | grep -aoE "<- [0-9]+\+[0-9]+ tokens streamed.*\[client_disconnect\]" 2>/dev/null | tail -2 | sed 's/^/        /'
else
    bad "a cancelled request kept generating" "$OVERRUN request(s) produced >2000 of 3000 tokens after the client left"
fi

# A cancelled buffer must be freed, not retained per round.
sleep 3
MEM_AFTER=$(mem_bytes)
if [ "${MEM_BEFORE:-0}" -gt 0 ] && [ "${MEM_AFTER:-0}" -gt 0 ]; then
    GROWTH=$(( (MEM_AFTER - MEM_BEFORE) * 100 / MEM_BEFORE ))
    if [ "$GROWTH" -lt 25 ]; then
        ok "no runaway retention across $ROUNDS cancelled buffers (active ${GROWTH}%)"
    else
        bad "cancelled tool buffers are retained" "active grew ${GROWTH}% over $ROUNDS rounds ($MEM_BEFORE -> $MEM_AFTER)"
    fi
else
    echo -e "  ${DIM}note: /props memory unavailable — retention not measured${NC}"
fi

# Not wedged: the next request must work, with tools and without.
PLAIN=$(NT="$NT" python3 -c "
import json, os
b = {'model': 'sushi', 'messages': [{'role': 'user', 'content': os.environ['SUSHI_NONCE'] + ' Say OK.'}], 'max_tokens': 8, 'temperature': 0}
b.update(json.loads(os.environ['NT'])); print(json.dumps(b))")
R=$(curl -sf -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$PLAIN" 2>/dev/null \
    | python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; print(((m.get('content') or '') + (m.get('reasoning_content') or '')).strip())" 2>/dev/null)
[ -n "$R" ] && ok "plain request works after the cancels (${R:0:30})" || bad "server wedged: plain request returned nothing"

R2=$(echo "$REQ" | python3 -c "
import json,sys
b=json.load(sys.stdin); b['stream']=False; b['max_tokens']=400; print(json.dumps(b))" \
    | curl -sf -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' --data-binary @- 2>/dev/null \
    | python3 -c "
import json,sys
m=json.load(sys.stdin)['choices'][0]['message']
tcs=m.get('tool_calls') or []
if not tcs: print('nocall'); raise SystemExit
try: json.loads(tcs[0]['function']['arguments']); print('ok')
except Exception: print('badjson')" 2>/dev/null)
case "$R2" in
    ok)      ok "tool-calling still works after the cancels" ;;
    nocall)  echo -e "  ${DIM}note: the model answered in text instead of calling the tool (its own choice)${NC}" ;;
    badjson) bad "tool arguments invalid after the cancels" ;;
    *)       bad "tools request failed after the cancels" "$R2" ;;
esac

sushi_log_since | grep -qiE "panic|segmentation fault|SIGBUS|unreachable" \
    && bad "crash signature in the server log" "$(sushi_log_since | grep -iE 'panic|segmentation|SIGBUS|unreachable' | head -2)" \
    || ok "no crash signature in the server log"

echo
echo "cancel mid-tool-buffer: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]

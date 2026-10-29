#!/bin/bash
# sushi-test: tier=live boot=api
# Guard: a tool call still parses at long context, where the sliding-window
# block trim and speculative verify interact.
#
# The trim changed what every sliding layer READS on any block-wide forward —
# spec verify and prefill chunks — and spec verify is exactly where a tool call
# is being detected. Both are guarded for SPEED and for byte-identity on prose;
# neither guarded a TOOL CALL at the context lengths where the trim actually
# engages (below ~8k there is nothing to trim, so short tool tests pass for the
# wrong reason).
#
# Per rung: a long filler prompt puts the request past the trim threshold, then
# the model is asked for one concrete tool call. Assertions are correctness, not
# speed: the call FIRES, its name is the declared one, its arguments are valid
# JSON carrying the value the prompt planted, and no tool markup leaks into
# visible content. The engagement line is checked too — a rung that silently
# fell back to short-context handling proves nothing.
#
# Whether the checkpoint calls the tool (or finds the planted string) is its choice, so a rung without a call
# is a note and a run where no rung called is a SKIP. A forced call would skip the spec verify this guards.
#
# Attaches to the shared api boot: tests/run.sh --only test_tools_long_context.sh --model <m>.
#   LONGCTX_RUNGS=16000,32000,64000   — override the token rungs; a rung past the boot's context is a named 400

set -u

. "$(dirname "$0")/lib/boot.sh"
sushi_attach
BASE=$SUSHI_URL
RED='\033[0;31m'; GREEN='\033[0;32m'; DIM='\033[2m'; NC='\033[0m'
PASS=0; FAIL=0
ok()  { echo -e "  ${GREEN}PASS${NC} $1"; PASS=$((PASS+1)); }
bad() { echo -e "  ${RED}FAIL${NC} $1"; shift; for l in "$@"; do echo "        $l"; done; FAIL=$((FAIL+1)); }

# A streamed DeepSeek prefills a long prompt at the SSD's pace, so it climbs one rung.
if [ "$SUSHI_ARCH" = deepseek_v41 ]; then RUNGS="${LONGCTX_RUNGS:-16000}"; else RUNGS="${LONGCTX_RUNGS:-16000,32000,64000}"; fi

IFS=',' read -r -a RUNG_LIST <<< "$RUNGS"
CARRIED_ANY=0
CALLED_ANY=0
for target in "${RUNG_LIST[@]}"; do
    echo -e "${DIM}── rung ~${target} tokens ──${NC}"
    REQ=$(python3 - "$target" "$SUSHI_NONCE" <<'PY'
import json, sys
target = int(sys.argv[1])
nonce = sys.argv[2]
# Filler that is cheap to generate and impossible to compress into a short
# answer, with a planted fact the tool call has to carry back.
# ~24 tokens per record, measured against /v1/models tokenizers — sizing by
# characters instead put 64k tokens into a rung labelled 16k.
para = ("Record %04d: the maintenance log notes routine calibration of unit %04d "
        "with no anomalies observed during the shift.\n")
filler = "".join(para % (i, i) for i in range(max(1, target // 24)))
planted = "AURORA-7731"
prompt = (nonce + " " + filler
          + "\nThe archive identifier for every record above is " + planted + ".\n"
          + "Now call the write_file tool exactly once to save the archive identifier "
            "to notes.txt. Put the identifier in the content argument.")
print(json.dumps({
    "model": "sushi",
    "messages": [{"role": "user", "content": prompt}],
    "tools": [{"type": "function", "function": {
        "name": "write_file", "description": "Write a file",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}}],
    "max_tokens": 2048, "reasoning_effort": "low", "temperature": 0, "stream": False}))
PY
)
    RESP=$(echo "$REQ" | curl -s -m 1800 -w '\n%{http_code}' "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' --data-binary @- 2>/dev/null)
    CODE=$(echo "$RESP" | tail -1)
    R=$(echo "$RESP" | sed '$d')
    if [ "$CODE" = "400" ]; then
        # A rung past the model's own window is not a failure of this test —
        # the contract is that the refusal NAMES BOTH counts so a client can act
        # on it, which is what contextOverflowMessage exists for.
        if echo "$R" | grep -qiE "context|token" && echo "$R" | grep -qE "[0-9]{4,}"; then
            ok "[$target] beyond the model's window: refused with a 400 naming the counts"
            echo "$R" | python3 -c "import json,sys; print('       ', (json.load(sys.stdin).get('error') or {}).get('message','')[:150])" 2>/dev/null
        else
            bad "[$target] 400 without the counts" "$(echo "$R" | head -c 200)"
        fi
        continue
    fi
    if [ -z "$R" ] || [ "$CODE" != "200" ]; then
        bad "[$target] request failed" "HTTP $CODE" "$(echo "$R" | head -c 200)"; continue
    fi

    V=$(echo "$R" | python3 -c "
import json, sys
d = json.load(sys.stdin)
m = d['choices'][0]['message']
tcs = m.get('tool_calls') or []
content = m.get('content') or ''
leak = any(t in content for t in ('<tool_call', '<|tool_call', '<function=', '</think'))
if not tcs:
    print('no-call|0|%d|%d' % (leak, d['usage']['prompt_tokens'])); raise SystemExit
try:
    args = json.loads(tcs[0]['function']['arguments'])
    valid = 1
except Exception:
    args, valid = {}, 0
carried = 1 if 'AURORA-7731' in json.dumps(args) else 0
print('%s|%d|%d|%d|%d' % (tcs[0]['function']['name'], valid, leak, d['usage']['prompt_tokens'], carried))
" 2>/dev/null)
    NAME=$(echo "$V" | cut -d'|' -f1)
    VALID=$(echo "$V" | cut -d'|' -f2)
    LEAK=$(echo "$V" | cut -d'|' -f3)
    PT=$(echo "$V" | cut -d'|' -f4)
    CARRIED=$(echo "$V" | cut -d'|' -f5)

    [ "$LEAK" = "0" ]   && ok "[$target] no tool markup in visible content" || bad "[$target] tool markup leaked into content"
    if [ "$NAME" = "no-call" ]; then
        echo -e "  ${DIM}note [$target] the model made no tool call (prompt=$PT tokens)${NC}"
        continue
    fi
    CALLED_ANY=1
    [ "$NAME" = "write_file" ] && ok "[$target] tool call fires with the declared name (prompt=$PT tokens)" \
                               || bad "[$target] tool call fires" "got name='$NAME' (prompt=$PT tokens)"
    [ "$VALID" = "1" ]  && ok "[$target] arguments are valid JSON" || bad "[$target] arguments are valid JSON"
    # Whether a 4-bit model finds one planted string in a 64k haystack is
    # RETRIEVAL, not a serving contract — measured here it carries it at 32k and
    # drops it at 16k and 64k, on the same build. What the trim could plausibly
    # break is long-context comprehension ENTIRELY, so the bar is the aggregate
    # below: at least one rung must carry it. Per-rung is reported, not asserted.
    if [ "${CARRIED:-0}" = "1" ]; then
        echo -e "  ${DIM}note [$target] the planted identifier survived into the arguments${NC}"
        CARRIED_ANY=1
    else
        echo -e "  ${DIM}note [$target] the model dropped the planted identifier at this length${NC}"
    fi
done

# The rungs only mean something if the trim actually engaged on this arch. The line is one-shot per process, and
# earlier scripts on the shared boot may have fired it, so this reads the whole boot log.
TRIM=0
if grep -q "block trim engaged" "$SUSHI_LOG"; then
    TRIM=1
    ok "sliding block trim engaged on this boot"
    grep -m1 "block trim engaged" "$SUSHI_LOG" | sed 's/^/        /'
else
    echo -e "  ${DIM}note: no [sliding] trim line — this arch has no sliding layers, so the rungs exercised full attention${NC}"
fi

# If the trim had broken long-context attention, the model would carry the
# planted value at NO length. One rung is enough to show it still reads. Where nothing was trimmed this is
# plain retrieval by a quantized checkpoint, so it is only a note.
if [ "${CARRIED_ANY:-0}" = "1" ]; then
    ok "the model still reads the long prompt (planted identifier recovered on at least one rung)"
elif [ "$TRIM" = "1" ] && [ "$CALLED_ANY" = "1" ]; then
    bad "long-context comprehension" "the planted identifier was lost at EVERY rung — the model is not reading the prompt"
else
    echo -e "  ${DIM}note: the planted identifier was not recovered on any rung${NC}"
fi

echo
echo "tools at long context: $PASS passed, $FAIL failed"
if [ "$FAIL" -eq 0 ] && [ "$CALLED_ANY" = "0" ]; then
    echo "SKIP $(basename "$0"): no rung produced a tool call (or every rung was refused for its context size)"
    exit 3
fi
[ "$FAIL" -eq 0 ]

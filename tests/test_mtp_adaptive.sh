#!/bin/bash
# sushi-test: tier=live boot=api
# Adaptive serial switch: past some context a speculative round costs more per token than the
# plain step it replaces, so the server measures a plain token per KV bucket and stops
# speculating when the planned round loses. Pins the INVARIANT, never the model's choice:
#   [1] with the switch on, the long request's log slice carries `[mtp] adaptive:` or a
#       measured `serial_cell=` > 0;
#   [2] speculation still ENGAGES there (`mode=mtp` with attempts > 0);
#   [3] a burst of short requests never emits `-> serial` (the MIN_KV floor).
# The ~40k-token prompt is generated in-script from a fixed seed. Round-cost persistence is opt-in, so
# the shared boot keeps no table. Needs an MTP head (SKIPs otherwise).

set -u
. "$(dirname "$0")/lib/boot.sh"; sushi_attach
sushi_require mtp
# DeepSeek reports its DSpark head as an MTP head in /props but never plans MTP rounds.
if [ "$SUSHI_ARCH" = deepseek_v41 ]; then
    echo "SKIP $(basename "$0"): DeepSeek speculates with DSpark, not MTP rounds"
    exit 3
fi
BASE=$SUSHI_URL
WORK=$(mktemp -d /tmp/mtp_adaptive.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

# The vote starts at 32k of KV, and at 64k on MiMo (MTP_ADAPTIVE_MIN_KV_MIMO): the prompt must clear it.
case "$SUSHI_ARCH" in
    mimo_v2) LONG_CHARS="${LONG_CHARS:-330000}"; PROMPT_TOKENS_MIN="${PROMPT_TOKENS_MIN:-66000}" ;;
    *)       LONG_CHARS="${LONG_CHARS:-165000}"; PROMPT_TOKENS_MIN="${PROMPT_TOKENS_MIN:-30000}" ;;
esac
# Long enough that the long cells run tens of speculative rounds: the price
# window is 16 rounds and the serial cell needs MIN_SAMPLES ticks.
MAX_TOKENS=200
# Below this the long request cannot have exercised the mechanism, whatever the
# mechanism did — a fixture failure, reported as one.
MIN_LONG_COMPLETION="${MIN_LONG_COMPLETION:-60}"

CTX=$(curl -s "$BASE/v1/models" | python3 -c '
import json, sys
print((json.load(sys.stdin)["data"][0].get("meta") or {}).get("context_length") or 0)')
if [ "$CTX" -gt 0 ] && [ "$CTX" -lt $((LONG_CHARS / 3 + MAX_TOKENS)) ]; then
    echo "SKIP $(basename "$0"): the boot serves $CTX tokens of context, the long prompt needs more"
    exit 3
fi

PASS=0
FAIL=0
ok()   { echo "PASS [$1]"; PASS=$((PASS+1)); }
bad()  { echo "FAIL [$1]: $2"; FAIL=$((FAIL+1)); }

# Deterministic prompt bodies: an LCG walk over a fixed word list.
python3 - "$WORK" "$LONG_CHARS" "$MAX_TOKENS" "$SUSHI_NONCE" <<'PY'
import json, sys, pathlib
work, chars, max_tokens, nonce = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]

WORDS = ("harbour lantern quiet ledger orchard sandstone drifting compass meadow "
         "tinker rope kettle furrow beacon willow granite shallow ember thistle "
         "marsh cobble weather almanac hollow river bramble shutter pasture flint "
         "lichen tallow bridge cartwright saddle furnace ripple heather quarry "
         "spindle brine oaken hedgerow smithy trellis warren coppice fallow gale").split()

state = 20260904          # fixed seed: the prompt must be a constant
def nxt(n):
    global state
    state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (state >> 33) % n

out, para = [], 0
while sum(len(p) for p in out) < chars:
    para += 1
    sentences = []
    for _ in range(4 + nxt(4)):
        w = [WORDS[nxt(len(WORDS))] for _ in range(8 + nxt(11))]
        w[0] = w[0].capitalize()
        sentences.append(" ".join(w) + ".")
    out.append(f"Section {para}. " + " ".join(sentences) + "\n\n")
long_text = "".join(out)[:chars]

def body(content, mtp):
    return {"model": "default", "stream": False, "temperature": 0, "top_p": 1.0,
            "max_tokens": max_tokens, "enable_thinking": False,
            "enable_pld": False, "enable_drafter": False,
            "enable_mtp": mtp,
            "messages": [{"role": "user", "content": content}]}

# The answer length is part of the fixture: a two-token reply measures nothing.
long_q = (nonce + " " + long_text + "\n\nDescribe the material above in detail: its overall "
          "structure, how the sections are numbered, the kind of vocabulary it "
          "uses, and how one section differs from another. Keep writing until "
          "you have covered all four points thoroughly.")
short_q = (nonce + " Write a Python function that reverses a linked list in place. "
           "Return only the code.")

(work / "long_serial.json").write_text(json.dumps(body(long_q, False)))
(work / "long_mtp.json").write_text(json.dumps(body(long_q, True)))
(work / "short_mtp.json").write_text(json.dumps(body(short_q, True)))
print(f"generated long prompt: {len(long_q)} chars", file=sys.stderr)
PY
[ -f "$WORK/long_mtp.json" ] || { echo "FAIL: could not generate prompts"; exit 1; }

# req <body-file> <slice-out> -> echoes "<prompt_tokens> <completion_tokens> <content chars>" and
# leaves this request's own slice of the server log in <slice-out>.
req() {
    local bodyf="$1" slicef="$2"
    local start end counts
    start=$(wc -c < "$SUSHI_LOG" | tr -d ' ')
    counts=$(curl -s -m 3600 "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' --data-binary "@$bodyf" |
        python3 -c "
import json, sys
d = json.load(sys.stdin)
if d.get('error'):
    print('ERR', d['error'], file=sys.stderr); raise SystemExit(2)
u = d.get('usage') or {}
c = (d.get('choices') or [{}])[0].get('message', {}).get('content') or ''
print(u.get('prompt_tokens', 0), u.get('completion_tokens', 0), len(c))
") || return 1
    end=$(wc -c < "$SUSHI_LOG" | tr -d ' ')
    tail -c "+$((start + 1))" "$SUSHI_LOG" | head -c "$((end - start))" > "$slicef"
    echo "$counts"
}

# The largest measured serial cell reported in a log slice (0 if none).
max_serial_cell() { # $1 = slice
    grep -o "serial_cell=[0-9.]*" "$1" | cut -d= -f2 |
        sort -g | tail -1 | sed 's/^$/0/'
}
gt_zero() { python3 -c "import sys; sys.exit(0 if float(sys.argv[1] or 0) > 0 else 1)" "$1"; }

echo "== adaptive serial ON (the default) =="

# The serial request first: a bucket with no measured plain token cannot be decided.
A_SERIAL=$(req "$WORK/long_serial.json" "$WORK/a_serial.slice") ||
    { echo "FAIL: long serial request failed"; exit 1; }
A_PROMPT_N=$(echo "$A_SERIAL" | awk '{print $1}')
echo "  long serial: prompt_tokens=$A_PROMPT_N completion=$(echo "$A_SERIAL" | awk '{print $2}')"

if [ "${A_PROMPT_N:-0}" -ge "$PROMPT_TOKENS_MIN" ]; then
    ok "prompt reaches the long-context regime ($A_PROMPT_N >= $PROMPT_TOKENS_MIN tokens)"
else
    bad "prompt length" "only $A_PROMPT_N prompt tokens (< $PROMPT_TOKENS_MIN); raise LONG_CHARS"
fi

A_LONG=$(req "$WORK/long_mtp.json" "$WORK/a_long.slice") ||
    { echo "FAIL: long MTP request failed"; exit 1; }
A_LONG_N=$(echo "$A_LONG" | awk '{print $2}')
echo "  long mtp:    prompt_tokens=$(echo "$A_LONG" | awk '{print $1}') completion=$A_LONG_N"
[ "$(echo "$A_LONG" | awk '{print $3}')" -gt 0 ] ||
    bad "content" "the long MTP request returned empty content"
# A short reply cannot exercise a 16-round window or fill a serial cell, so
# without this the next assertion blames the mechanism for the prompt.
if [ "${A_LONG_N:-0}" -lt "$MIN_LONG_COMPLETION" ]; then
    bad "fixture" "long reply was only $A_LONG_N tokens (< $MIN_LONG_COMPLETION): too few rounds to measure anything — fix the PROMPT, not the controller"
else
    ok "long reply is long enough to price ($A_LONG_N tokens)"
fi

A_SHORT=$(req "$WORK/short_mtp.json" "$WORK/a_short.slice") ||
    { echo "FAIL: short request failed"; exit 1; }
echo "  short mtp:   prompt_tokens=$(echo "$A_SHORT" | awk '{print $1}') completion=$(echo "$A_SHORT" | awk '{print $2}')"

# llmprobe-shaped burst: the `<2k` bucket only matures over many short requests.
SHORT_BURST="${SHORT_BURST:-12}"
echo "  short burst: $SHORT_BURST requests (llmprobe-shaped)"
burst_start=$(wc -c < "$SUSHI_LOG" | tr -d ' ')
bi=1
while [ "$bi" -le "$SHORT_BURST" ]; do
    req "$WORK/short_mtp.json" "$WORK/a_burst_$bi.slice" >/dev/null ||
        { echo "FAIL: short burst request $bi failed"; exit 1; }
    bi=$((bi + 1))
done
burst_end=$(wc -c < "$SUSHI_LOG" | tr -d ' ')
tail -c "+$((burst_start + 1))" "$SUSHI_LOG" | head -c "$((burst_end - burst_start))" > "$WORK/a_burst.slice"
sushi_log_since > "$WORK/a_all.slice"

# [2] Engagement: the long request ran speculative rounds. Without this, [1]
#     could be satisfied by a server that never speculated at all.
if grep -q "\[spec-stats\] mode=mtp" "$WORK/a_long.slice"; then
    ATTEMPTS=$(grep -o "mode=mtp attempts=[0-9]*" "$WORK/a_long.slice" | tail -1 | grep -o "[0-9]*$")
    if [ "${ATTEMPTS:-0}" -gt 0 ]; then
        ok "speculation engaged on the long request (attempts=$ATTEMPTS)"
    else
        bad "engagement" "mode=mtp logged with attempts=0 on the long request"
    fi
else
    bad "engagement" "no '[spec-stats] mode=mtp' for the long request (dispatch hole)"
fi

# [1] The mechanism ran: a decision line, or a measured serial cell.
A_CELL=$(max_serial_cell "$WORK/a_all.slice")
if grep -q "\[mtp\] adaptive:" "$WORK/a_long.slice"; then
    ok "adaptive controller reported a decision on the long request"
    grep "\[mtp\] adaptive:" "$WORK/a_long.slice" | sed 's/^/      /'
elif gt_zero "$A_CELL"; then
    ok "serial cell measured with the switch on (serial_cell=$A_CELL ms/tok)"
else
    bad "mechanism" "no '[mtp] adaptive:' line and serial_cell=$A_CELL — nothing was measured or decided"
fi

# [3] Short context never switches — the single fixture AND the burst.
if grep -q -- "-> serial" "$WORK/a_short.slice"; then
    bad "short context" "the short code request left speculation (false positive)"
    grep -- "\[mtp\] adaptive:" "$WORK/a_short.slice" | sed 's/^/      /'
else
    ok "short code request stayed on speculation"
fi
BURST_SWITCHES=$(grep -c -- "-> serial" "$WORK/a_burst.slice" 2>/dev/null || true)
if [ "${BURST_SWITCHES:-0}" -gt 0 ]; then
    bad "short burst" "$BURST_SWITCHES of $SHORT_BURST short requests left speculation (the <2k regression)"
    grep -- "\[mtp\] adaptive:" "$WORK/a_burst.slice" | head -5 | sed 's/^/      /'
else
    ok "$SHORT_BURST-request short burst: zero switches (kv floor holds)"
fi

echo
echo "── $PASS passed, $FAIL failed ──"
[ "$FAIL" -eq 0 ] || exit 1

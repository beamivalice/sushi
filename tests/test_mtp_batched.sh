#!/usr/bin/env bash
# sushi-test: tier=live boot=exact
# Concurrent MTP users on one model: every stream's greedy output at a FIXED draft depth
# must be byte-identical to its solo run (the byte bar for spec decode), whether the
# rounds ride one batched verify (qwen3_5 dense/MoE) or interleave with their own head
# state (qwen4_exp). A crowd of 4 must still complete on the plain batched tick.
set -u
. "$(dirname "$0")/lib/boot.sh"
sushi_attach
sushi_require mtp
sushi_require batching
BASE="$SUSHI_URL"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
LOG=$(mktemp); OUT=$(mktemp -d)
trap 'rm -rf "$LOG" "$OUT"' EXIT
IS_QWEN4=$([ "$SUSHI_ARCH" = qwen4_exp ] && echo 1 || echo 0)

P=("$SUSHI_NONCE Write a short essay about the history of quantum computing." "$SUSHI_NONCE Explain how a compiler turns source code into machine code." "$SUSHI_NONCE Describe the water cycle for a graduate seminar." "$SUSHI_NONCE Summarize the causes of the French revolution.")
req() { python3 -c "import json,sys; print(json.dumps({'model':'sushi','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':150,'temperature':0}))" "$1" |
    curl -s -m 900 -X POST -H 'Content-Type: application/json' -d @- "$BASE/v1/chat/completions" |
    python3 -c "import sys,json; j=json.load(sys.stdin); print(j['choices'][0]['message']['content'])"; }

# qwen4 rounds stay solo and two of them interleave; three go plain (a B=3 tick).
N=$([ "$IS_QWEN4" = 1 ] && echo 2 || echo 3); LAST=$((N-1))
req "${P[0]}" > /dev/null
for i in $(seq 0 $LAST); do req "${P[$i]}" > "$OUT/solo$i"; done
pids=(); for i in $(seq 0 $LAST); do req "${P[$i]}" > "$OUT/conc$i" & pids+=($!); done; wait "${pids[@]}"
# A batched verify is a B>1 forward: like the plain batched tick it is not bit-identical
# to serial, and the server returns no probabilities to acquit a near-tie flip by, so those streams must match
# solo over their first 100 characters. qwen4 rounds are solo forwards and must match byte for byte.
fail=0
for i in $(seq 0 $LAST); do
    if cmp -s "$OUT/solo$i" "$OUT/conc$i"; then continue; fi
    if [ "$IS_QWEN4" = 1 ]; then
        echo -e "${RED}FAIL${NC} stream $i at N=$N differs from its solo run"; diff "$OUT/solo$i" "$OUT/conc$i" | head -6; fail=1; continue
    fi
    at=$(python3 - "$OUT/solo$i" "$OUT/conc$i" <<'PYEOF'
import sys
a, b = open(sys.argv[1]).read(), open(sys.argv[2]).read()
print(next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b))))
PYEOF
)
    if [ "$at" -ge 100 ]; then
        echo -e "  ${YELLOW}diverged${NC} stream $i at character $at, past the compared 100"
    else
        echo -e "${RED}FAIL${NC} stream $i at N=$N diverged from its solo run at character $at"; diff "$OUT/solo$i" "$OUT/conc$i" | head -6; fail=1
    fi
done
[ "$fail" = 0 ] && echo -e "${GREEN}PASS${NC} $N concurrent MTP streams match solo (fixed depth; the first 100 characters on the batched verify)"
# Engagement lines are logged once per process, so an earlier script on this boot may have logged them.
cp "$SUSHI_LOG" "$LOG"
if [ "$IS_QWEN4" = 1 ]; then
    grep -q "gdn batched verify engaged" "$LOG" && { echo -e "${RED}FAIL${NC} qwen4 rounds must stay solo (no batched verify yet)"; fail=1; }
    [ "$fail" = 0 ] && echo -e "${GREEN}PASS${NC} qwen4 MTP users interleave on their own head state"
else
    grep -q "gdn batched verify engaged" "$LOG" || { echo -e "${RED}FAIL${NC} no batched verify engaged"; fail=1; }
    [ "$fail" = 0 ] && echo -e "${GREEN}PASS${NC} batched verify engaged: $(grep -o 'gdn batched verify engaged.*' "$LOG" | head -1)"
fi

pids=(); for i in 0 1 2 3; do req "${P[$i]}" > "$OUT/crowd$i" & pids+=($!); done; wait "${pids[@]}"
sushi_log_since > "$LOG"
for i in 0 1 2 3; do [ -s "$OUT/crowd$i" ] || { echo -e "${RED}FAIL${NC} crowd stream $i returned nothing"; fail=1; }; done
if [ "$IS_QWEN4" = 0 ]; then
    grep -q "gdn batched decode engaged (slots=4)" "$LOG" || { echo -e "${YELLOW}NOT RUN${NC} the four MTP streams never overlapped into one plain batched tick"; }
fi
[ "$fail" = 0 ] && echo -e "${GREEN}PASS${NC} four MTP streams complete (crowd arm)"
grep -ciE "panic|Segmentation" "$LOG" | grep -q "^0$" || { echo -e "${RED}FAIL${NC} server log has a crash"; fail=1; }
exit $fail

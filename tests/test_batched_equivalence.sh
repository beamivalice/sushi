#!/bin/bash
# sushi-test: tier=live boot=own pack=default boots=2
# Batched-kernel byte-equivalence. The default scheduler routes a single-slot tick through `runSingleDecodeTick`;
# `SUSHI_FORCE_BATCHED=1` sends even N=1 through `forwardBatchedDecode`, a kernel that lays tensors out differently
# ([N,1,d] against [1,1,d]), so a silent shape/cache/RoPE bug that shows only there flips token ids.
#
# Boot 1 (default path) records the references; boot 2 (`SUSHI_FORCE_BATCHED=1`, kv8 default) then asserts:
#   [1] the short prompt is byte-identical to the default path;
#   [2] the first FIRST_N_TOKENS tokens of a long greedy recital match (past them an INT4 near-tie may flip, see
#       docs/engine-kv-cache.md "Byte stability");
#   [3] two genuinely concurrent streams (batch != 1, where the fused QK-norm+RoPE gates differ) each match the
#       serial answer, byte for byte on a per-row trunk, for FIRST_N_TOKENS with near-ties acquitted otherwise;
#   [4] /props and /v1/models report batching, and a logprobs slot beside a batched one names its serial reason;
#   [5] the batched kernel under kv-quant (the default kv8) leaves the server alive with no MLX error: it once read
#       the packed quantized words raw and killed the server on the first concurrent decode.
# --no-mtp: an MTP slot runs speculative rounds and never reaches the plain batched tick under test.
set -u
. "$(dirname "$0")/lib/boot.sh"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
FIRST_N_TOKENS=30
MODEL=$(sushi_pack qwen4) || exit $?
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

SHORT_PAYLOAD='{"model":"sushi","messages":[{"role":"user","content":"What is 2+2? Answer with just the number, no explanation."}],"max_tokens":32,"temperature":0.0,"stream":false}'
LONG_PAYLOAD='{"model":"sushi","messages":[{"role":"user","content":"Recite the first paragraph of \"A Tale of Two Cities\" by Charles Dickens."}],"max_tokens":200,"temperature":0.0,"stream":false}'

post() { curl -s -m 180 -X POST -H "Content-Type: application/json" -d @- "$SUSHI_URL/v1/chat/completions"; }
text_of() { python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"; }
tokens_of() { # completion text on stdin -> comma-separated token ids
    python3 -c "import json,sys; print(json.dumps({'content': sys.stdin.read().rstrip('\n')}))" |
        curl -s -X POST -H "Content-Type: application/json" -d @- "$SUSHI_URL/tokenize" |
        python3 -c "import sys,json; print(','.join(str(t) for t in json.load(sys.stdin)['tokens']))"
}
fail() { echo -e "${RED}FAIL${NC} $*"; exit 1; }

# A GatedDeltaNet trunk routes the batched tick through a DIFFERENT kernel; without this probe the test passes
# vacuously if the batching gate ever stops admitting it, since both sides become the same serial path.
IS_GDN=$(python3 - "$MODEL" <<'PYEOF'
import json, sys, pathlib
try:
    c = json.loads((pathlib.Path(sys.argv[1]) / "config.json").read_text())
except Exception:
    print("0"); raise SystemExit
t = c.get("text_config") or c
print("1" if t.get("full_attention_interval", 0) else "0")
PYEOF
)

echo "== boot 1: the default single-slot path (references) =="
sushi_boot batched-ref "$MODEL" --no-mtp --max-concurrent 4 || exit $?
echo "$SHORT_PAYLOAD" | post | text_of > "$WORK/short.single"
echo "$LONG_PAYLOAD" | post | text_of > "$WORK/long.single"
tokens_of < "$WORK/long.single" > "$WORK/long.single.toks"
sushi_stop 0
echo "  single-slot: short $(wc -c < "$WORK/short.single" | tr -d ' ') bytes, long $(tr ',' '\n' < "$WORK/long.single.toks" | wc -l | tr -d ' ') tokens"

echo "== boot 2: SUSHI_FORCE_BATCHED=1 =="
SUSHI_FORCE_BATCHED=1 sushi_boot batched-forced "$MODEL" --no-mtp --max-concurrent 4 || exit $?
grep -q "force_batched=on" "$SUSHI_LOG" || fail "expected force_batched=on log line not found"

echo "$SHORT_PAYLOAD" | post | text_of > "$WORK/short.batched"
if [ "$IS_GDN" = 1 ] && ! grep -qE "gdn batched decode engaged|qwen4 per-row decode engaged" "$SUSHI_LOG"; then
    fail "GatedDeltaNet trunk never entered the batched kernel: this comparison would be serial-vs-serial and pass for free"
fi
if cmp -s "$WORK/short.single" "$WORK/short.batched"; then
    echo -e "${GREEN}PASS${NC} [1] short prompt byte-identical (single vs batched)"
else
    echo -e "${RED}FAIL${NC} [1] outputs differ:"; diff "$WORK/short.single" "$WORK/short.batched" | sed 's/^/    /'; exit 1
fi

echo "$LONG_PAYLOAD" | post | text_of > "$WORK/long.batched"
tokens_of < "$WORK/long.batched" > "$WORK/long.batched.toks"
DIVERGENCE=$(python3 - "$WORK/long.single.toks" "$WORK/long.batched.toks" "$FIRST_N_TOKENS" <<'PY'
import sys
a, b = (open(p).read().strip().split(",") for p in sys.argv[1:3])
n = int(sys.argv[3])
a, b = a[:n], b[:n]
if len(a) < n or len(b) < n:
    print(f"SHORT len(single)={len(a)} len(batched)={len(b)} need>={n}")
else:
    d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), -1)
    print("OK" if d < 0 else f"DIFF at index {d}: single={a[d]} batched={b[d]}")
PY
)
[ "$DIVERGENCE" = "OK" ] || fail "[2] first-${FIRST_N_TOKENS}-tokens divergence: $DIVERGENCE"
echo -e "${GREEN}PASS${NC} [2] first ${FIRST_N_TOKENS} tokens byte-identical (single vs batched)"

echo "== real N=2 concurrency (batch != 1) =="
OFF=$(wc -c < "$SUSHI_LOG" | tr -d ' ')
echo "$LONG_PAYLOAD" | post > "$WORK/conc.a" &
CA=$!
echo "$LONG_PAYLOAD" | post > "$WORK/conc.b" &
CB=$!
wait $CA; wait $CB
SLICE=$(tail -c +$((OFF + 1)) "$SUSHI_LOG")
# Output equality alone cannot tell a batched run from two serial ones, and two concurrent requests are not
# guaranteed to overlap: the log line is the only proof the batched path ran at N>1.
if ! echo "$SLICE" | grep -qE "\[batched\] (gdn batched decode|batched decode|qwen4 per-row decode) engaged \(slots=[2-9]"; then
    echo -e "  ${YELLOW}NOT RUN${NC} the two requests never overlapped into a batch of >= 2"
    CONC_SKIPPED=1
fi
ROW_EXACT=0
echo "$SLICE" | grep -q "\[batched\] qwen4 per-row decode engaged" && ROW_EXACT=1
if [ -z "${CONC_SKIPPED:-}" ]; then
    for f in "$WORK/conc.a" "$WORK/conc.b"; do
        txt=$(python3 -c "import sys,json; print(json.load(open(sys.argv[1]))['choices'][0]['message']['content'])" "$f" 2> /dev/null || true)
        [ -n "$txt" ] || { cat "$f"; fail "concurrent request returned no completion"; }
        toks=$(printf '%s' "$txt" | tokens_of)
        CMP_N=$FIRST_N_TOKENS
        [ "$ROW_EXACT" = 1 ] && CMP_N=1000000
        verdict=$(python3 - "$(cat "$WORK/long.single.toks")" "$toks" "$CMP_N" <<'PYEOF'
import sys
a = sys.argv[1].split(",") if sys.argv[1] else []
b = sys.argv[2].split(",") if sys.argv[2] else []
if int(sys.argv[3]) > 1000 and len(a) != len(b):
    print(f"DIFF at index {min(len(a), len(b))}: length single={len(a)} concurrent={len(b)}")
    sys.exit(0)
n = min(int(sys.argv[3]), len(a), len(b))
for i in range(n):
    if a[i] != b[i]:
        print(f"DIFF at index {i}: single={a[i]} concurrent={b[i]}")
        break
else:
    print("OK" if n > 0 else "EMPTY")
PYEOF
)
        if [ "$verdict" != "OK" ] && [ "$ROW_EXACT" = 1 ]; then
            fail "per-row concurrent stream is not the serial answer: $verdict"
        fi
        if [ "$verdict" != "OK" ]; then
            # A batch of >= 2 is not bit-identical to serial (B=2 tiles accumulate in a different order). Acquit a
            # divergence whose serial top-2 gap is a near-tie (<= 0.15 nats, the MTP equivalence bar).
            idx=$(echo "$verdict" | sed -n 's/DIFF at index \([0-9]*\):.*/\1/p')
            gap=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); d['logprobs']=True; d['top_logprobs']=2; print(json.dumps(d))" "$LONG_PAYLOAD" | post |
                python3 -c "
import sys,json
c=json.load(sys.stdin)['choices'][0]['logprobs']['content']; i=int(sys.argv[1])
t=c[i]['top_logprobs']; print(round(t[0]['logprob']-t[1]['logprob'],4))" "$idx" 2> /dev/null || echo 99)
            if python3 -c "import sys; sys.exit(0 if float('$gap') <= 0.15 else 1)"; then
                echo -e "  ${YELLOW}near-tie${NC} $verdict acquitted: serial top-2 gap at index $idx = $gap nats"
            else
                fail "[3] N=2 concurrent stream diverged from serial: $verdict (serial top-2 gap $gap nats)"
            fi
        fi
    done
    if [ "$ROW_EXACT" = 1 ]; then
        echo -e "${GREEN}PASS${NC} [3] both concurrent streams are the serial answer byte for byte (per-row batch >= 2)"
    else
        echo -e "${GREEN}PASS${NC} [3] both concurrent streams match serial for ${FIRST_N_TOKENS} tokens (batch >= 2; near-ties acquitted)"
    fi
fi

# What a user reads without the log: /props and /v1/models say the loaded model batches, and a slot that cannot
# (logprobs) names its reason once beside live company.
curl -s "$SUSHI_URL/props" | grep -q '"batching":{"supported":true' || fail "[4] /props does not report batching.supported=true"
curl -s "$SUSHI_URL/v1/models" | grep -q '"batched_decode":true' || fail "[4] /v1/models row lacks batched_decode:true"
OFF=$(wc -c < "$SUSHI_LOG" | tr -d ' ')
python3 -c "import json,sys; d=json.loads(sys.argv[1]); d['logprobs']=True; d['top_logprobs']=1; print(json.dumps(d))" "$LONG_PAYLOAD" | post > /dev/null &
LA=$!
echo "$LONG_PAYLOAD" | post > /dev/null &
LB=$!
wait $LA; wait $LB
if tail -c +$((OFF + 1)) "$SUSHI_LOG" | grep -q "\[batched\] slot serial: logprobs"; then
    echo -e "${GREEN}PASS${NC} [4] a logprobs slot beside a batched one names its serial reason"
else
    echo -e "  ${YELLOW}NOT RUN${NC} the logprobs request never overlapped a live slot (no serial-reason line)"
fi

curl -sf "$SUSHI_URL/health" > /dev/null || fail "[5] server died under the batched kernel at kv8"
grep -q "MLX error" "$SUSHI_LOG" && fail "[5] MLX error in the server log during batched decode at kv8"
echo -e "${GREEN}PASS${NC} [5] batched decode survives the default kv8 (server alive, no MLX error)"
exit 0

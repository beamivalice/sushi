#!/bin/bash
# sushi-test: tier=live boot=exact
# Per-request `kv_quant` body field, on one boot whose process default is the shipping kv8.
#   1. plumbing: no field, 4, "off", 8 and 6 on /v1/chat/completions each complete with content, and the log
#      names every override the scheduler applied; /v1/completions honours the field too.
#   2. equivalence: the same greedy request under kv_quant 4, 6 and 8 matches kv_quant "off" for its first N
#      tokens (30; kv4 is the lossiest and takes 20). KV quantization is lossy, so the bar catches kernel bugs (wrong dequantize, off-by-one slice
#      update, NaN), not the float-reduction tail (docs/engine-kv-cache.md, "Byte stability").
# A scheme the arch refuses (kv4 on GLM) is left out; DeepSeek has no KV cache to quantize.
set -u
. "$(dirname "$0")/lib/boot.sh"
sushi_attach
case "$SUSHI_ARCH" in
    deepseek_v41) echo "SKIP $(basename "$0"): deepseek_v41 has no KV cache to quantize"; exit 3 ;;
    glm5_next) BITS="6 8" ;;
    *) BITS="4 6 8" ;;
esac

BASE="$SUSHI_URL"
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
FAIL=0

fire() {
    # $1 = label, $2 = optional kv_quant field value (empty = no field)
    local label="$1" extra="" body content
    [ -z "$2" ] || extra=", \"kv_quant\": $2"
    body=$(curl -s -m 600 -X POST -H "Content-Type: application/json" \
        -d "{\"model\":\"sushi\",\"messages\":[{\"role\":\"user\",\"content\":\"$SUSHI_NONCE Say hello in one short sentence.\"}],\"max_tokens\":32,\"temperature\":0.0,\"stream\":false${extra}}" \
        "$BASE/v1/chat/completions")
    content=$(echo "$body" | python3 -c "import sys,json; m=json.load(sys.stdin)['choices'][0]['message']; print((m.get('reasoning_content') or '') + (m.get('content') or ''))" 2>/dev/null || echo "")
    if [ -z "$content" ]; then
        echo -e "${RED}FAIL${NC} $label: empty/null content"
        echo "  body: $(echo "$body" | head -c 400)"
        FAIL=1
        return
    fi
    echo -e "${GREEN}PASS${NC} $label: $(echo "$content" | head -c 80)..."
}

echo "== per-request kv_quant override =="
fire "request without kv_quant (inherits the process default)" ""
fire "kv_quant \"off\" (explicit dense)" '"off"'
for b in $BITS; do
    fire "kv_quant $b" "$b"
done

echo
echo "== server log: kv-quant override lines =="
LOGS=$(sushi_log_since)
for b in $BITS; do
    if grep -q "kv-quant override: affine $b-bit" <<< "$LOGS"; then
        echo -e "${GREEN}PASS${NC} per-request $b-bit override observed in server log"
    else
        echo -e "${RED}FAIL${NC} expected 'kv-quant override: affine $b-bit' line in server log"
        FAIL=1
    fi
done
if grep -q "kv-quant override: off" <<< "$LOGS"; then
    echo -e "${GREEN}PASS${NC} per-request 'off' override observed in server log"
else
    echo -e "${RED}FAIL${NC} expected 'kv-quant override: off' line in server log"
    FAIL=1
fi

# /v1/completions once ignored the field: the override line must appear for it too.
echo
echo "== /v1/completions per-request kv_quant =="
before=$(sushi_log_since | grep -c "kv-quant override: affine 8-bit")
curl -s -m 300 -X POST -H "Content-Type: application/json" \
    -d "{\"model\":\"sushi\",\"prompt\":\"$SUSHI_NONCE Say hello.\",\"max_tokens\":8,\"temperature\":0.0,\"kv_quant\":8}" \
    "$BASE/v1/completions" > /dev/null
sleep 1
after=$(sushi_log_since | grep -c "kv-quant override: affine 8-bit")
if [ "$after" -gt "$before" ]; then
    echo -e "${GREEN}PASS${NC} /v1/completions honoured the per-request 8-bit override"
else
    echo -e "${RED}FAIL${NC} /v1/completions ignored kv_quant: 8"
    FAIL=1
fi

echo
echo "== kv_quant equivalence against off =="
python3 - "$BASE" "$SUSHI_NONCE" "$BITS" <<'PY' || FAIL=1
import json, os, sys, urllib.request

base, nonce, bits = sys.argv[1], sys.argv[2], sys.argv[3].split()
PROMPT = nonce + ' Recite the first paragraph of "A Tale of Two Cities" by Charles Dickens.'


def post(path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def tokens(kv):
    body = {"model": "sushi", "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 200,
            "temperature": 0.0, "stream": False, "enable_mtp": False, "kv_quant": kv}
    m = post("/v1/chat/completions", body)["choices"][0]["message"]
    text = (m.get("reasoning_content") or "") + (m.get("content") or "")
    return text, post("/tokenize", {"content": text})["tokens"]


ref_text, ref = tokens("off")
fail = False
if not ref:
    print("FAIL: kv_quant off produced no tokens")
    sys.exit(1)
for b in bits:
    text, toks = tokens(int(b))
    n = int(os.environ.get(f"KV_QUANT_FIRST_N_{b}BIT", "20" if b == "4" else "30"))
    if not text:
        print(f"FAIL: kv{b}: empty completion, likely NaN or kernel error")
        fail = True
        continue
    if len(ref) < n or len(toks) < n:
        print(f"FAIL: kv{b} vs off: SHORT len(ref)={len(ref)} len(quant)={len(toks)} need>={n}")
        fail = True
        continue
    diff = next((i for i in range(n) if ref[i] != toks[i]), -1)
    if diff < 0:
        print(f"PASS kv{b} vs off: first {n} tokens byte-identical")
    else:
        print(f"FAIL: kv{b} vs off: DIFF at index {diff}: ref={ref[diff]} quant={toks[diff]}")
        fail = True
sys.exit(1 if fail else 0)
PY

exit $FAIL

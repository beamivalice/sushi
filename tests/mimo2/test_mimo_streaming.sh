#!/bin/bash
# sushi-test: tier=live boot=own pack=mimo2-src boots=2
# Live MiMo MXFP4 streaming on the native source checkpoint (or any converted MiMo pack set by
# SUSHI_TEST_PACK_MIMO2_SRC): boot 1 pins the sigmoid lossy pick (`--expert-pick-tolerance 0.2` swaps ids on a decode);
# boot 2 pins the budget and cache engagement, the 80% per-layer cache preload before readiness, the text-only model
# row, deterministic greedy repeats without prefix caching, a prefill that crosses the sliding window and the named
# MTP refusal. MIMO_SSD_BUDGET_GB selects total residency (default 60 GiB).
set -euo pipefail
. "$(dirname "$0")/../lib/boot.sh"

MODEL=$(sushi_pack mimo2-src) || exit $?
if [[ ! -f "$MODEL/model.safetensors.index.json" ]]; then
    echo "SKIP $(basename "$0"): $MODEL is not a MiMo MXFP4 checkpoint (no model.safetensors.index.json)"
    exit 3
fi
BUDGET=${MIMO_SSD_BUDGET_GB:-60}
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

# A sigmoid router takes the lossy pick on its sigmoid probabilities: a decode engages the swap.
sushi_boot mimo-pick "$MODEL" --ssd-budget-gb "$BUDGET" --no-mtp --no-vision --expert-pick-tolerance 0.2 || exit $?
curl --max-time 600 -fsS "$SUSHI_URL/v1/chat/completions" -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"Count from one to twenty in words."}],"max_tokens":32,"temperature":0}' > /dev/null
grep -q 'expert-swap\] engaged: tolerance=0.20' "$SUSHI_LOG"
grep -Eq 'expert-swap\] swapped [1-9][0-9]* of [0-9]+ routed ids' "$SUSHI_LOG"
sushi_stop 0

sushi_boot mimo-stream "$MODEL" --ssd-budget-gb "$BUDGET" --no-mtp --no-vision --kv-quant off \
    --ctx-size 4096 --prefill-chunk 512 --prefix-cache-entries 0 --metrics --log-level debug || exit $?
BASE="$SUSHI_URL"
curl --max-time 10 -fsS "$BASE/v1/models" > "$OUT/models.json"
ID=$(jq -er '.data[] | select(.loaded == true and .streaming == true and .input_modalities == ["text"]) | .id' "$OUT/models.json")
if jq -e '.quantization_config.store_dtype == "mxfp4"' "$MODEL/config.json" > /dev/null; then
    jq -e --arg id "$ID" '.data[] | select(.id == $id) | .streaming_required == true' "$OUT/models.json" > /dev/null
    grep -q '\[mimo-source\] loading original shards' "$SUSHI_LOG"
fi
python3 - "$SUSHI_LOG" "$MODEL/config.json" <<'PY'
import json
import re
import sys

log = open(sys.argv[1]).read()
config = json.load(open(sys.argv[2]))
warm = re.search(
    r"\[expert-stream\] cache warm complete: slots=(\d+)/(\d+) per layer, layers=(\d+), bytes=(\d+)",
    log,
)
assert warm, "model became ready without preloading its expert cache"
slots, capacity, layers, size = map(int, warm.groups())
assert slots == capacity * 4 // 5, (slots, capacity)
pattern = config["moe_layer_freq"]
assert layers == sum(bool(x) for x in pattern), (layers, pattern)
assert size > 0, "cache preload read no expert bytes"
PY
jq -nc --arg model "$ID" '{model:$model,messages:[{role:"user",content:"Write one short sentence about rain."}],temperature:0,seed:1234,max_tokens:16,enable_thinking:false,stream:false}' > "$OUT/request.json"
for n in 1 2; do
    curl --connect-timeout 5 --max-time 1800 -fsS -H 'Content-Type: application/json' \
        -d @"$OUT/request.json" "$BASE/v1/chat/completions" > "$OUT/reply$n.json"
    jq -e '.choices[0].message.content | type == "string"' "$OUT/reply$n.json" > /dev/null
    jq -S '.choices[0].message | {content,reasoning_content}' "$OUT/reply$n.json" > "$OUT/text$n.json"
done
cmp "$OUT/text1.json" "$OUT/text2.json"

jq -nc --arg model "$ID" '{model:$model,messages:[{role:"user",content:([range(64)|"word\(.): rain falls."]|join(" "))}],temperature:0,max_tokens:4,enable_thinking:false,stream:false}' |
    curl --connect-timeout 5 --max-time 1800 -fsS -H 'Content-Type: application/json' \
        -d @- "$BASE/v1/chat/completions" > "$OUT/long.json"
jq -e '.usage.prompt_tokens > 128 and (.choices[0].message.content | type == "string")' "$OUT/long.json" > /dev/null

jq '.enable_mtp=true' "$OUT/request.json" > "$OUT/mtp-request.json"
STATUS=$(curl --max-time 30 -sS -o "$OUT/mtp.json" -w '%{http_code}' \
    -H 'Content-Type: application/json' -d @"$OUT/mtp-request.json" "$BASE/v1/chat/completions")
[[ "$STATUS" == 400 ]]
jq -e '.error.type == "invalid_request_error" and (.error.message | contains("MTP speculative decode is not supported"))' "$OUT/mtp.json" > /dev/null
grep -q '\[expert-stream\] ssd budget' "$SUSHI_LOG"
grep -q '\[expert-stream\] cache' "$SUSHI_LOG"
echo "PASS: MiMo streaming discovery, sigmoid lossy pick, greedy determinism, window-crossing prefill, and MTP refusal"

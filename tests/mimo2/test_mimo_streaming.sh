#!/usr/bin/env bash
. "$(dirname "$0")/../../private_cache.sh"
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MODEL=${MIMO_STREAM_MODEL:-}
PORT=${1:-}
BUDGET=${MIMO_SSD_BUDGET_GB:-60}
if [[ -z "$MODEL" || ! -f "$MODEL/config.json" || ! -f "$MODEL/model.safetensors.index.json" ]]; then
    echo "SKIP: set MIMO_STREAM_MODEL to an original or converted MiMo MXFP4 checkpoint"
    exit 0
fi
if [[ ! "$PORT" =~ ^[0-9]+$ || "$PORT" -lt 1 || "$PORT" -gt 65535 ]]; then
    echo "usage: MIMO_STREAM_MODEL=<pack> $0 PORT" >&2
    exit 2
fi
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | grep -q LISTEN; then
    echo "port $PORT is already in use" >&2
    exit 1
fi
BIN="$ROOT/zig-out/bin/sushi"
[[ -x "$BIN" ]] || { echo "Build ReleaseFast first" >&2; exit 1; }
mkdir -p "$ROOT/.zig-cache"
OUT=$(mktemp -d "$ROOT/.zig-cache/mimo-http.XXXXXX")
PID=
cleanup() {
    if [[ -n "$PID" ]]; then
        kill "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
    fi
    rm -rf "$OUT"
}
trap cleanup EXIT
trap 'cat "$OUT/pick.log" "$OUT/server.log" >&2 2>/dev/null' ERR
mkdir "$OUT/home"
# A sigmoid router takes the lossy pick on its sigmoid probabilities: a decode engages the swap.
HOME="$OUT/home" "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
    --ssd-budget-gb "$BUDGET" --no-mtp --no-vision --expert-pick-tolerance 0.2 \
    --log-file "$OUT/pick.log" >"$OUT/pick-console.log" 2>&1 &
PID=$!
for ((i=0; i<1200; i++)); do
    if curl --connect-timeout 1 --max-time 2 -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
    kill -0 "$PID"
    sleep 1
done
curl --max-time 600 -fsS "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"Count from one to twenty in words."}],"max_tokens":32,"temperature":0}' >/dev/null
grep -q 'expert-swap\] engaged: tolerance=0.20' "$OUT/pick.log"
grep -Eq 'expert-swap\] swapped [1-9][0-9]* of [0-9]+ routed ids' "$OUT/pick.log"
kill "$PID"
wait "$PID" 2>/dev/null || true
PID=
HOME="$OUT/home" "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
    --ssd-budget-gb "$BUDGET" --no-mtp --no-vision --kv-quant off \
    --ctx-size 4096 --prefill-chunk 512 --prefix-cache-entries 0 --metrics \
    --log-file "$OUT/server.log" >"$OUT/console.log" 2>&1 &
PID=$!
BASE="http://127.0.0.1:$PORT"
for ((i=0; i<1200; i++)); do
    if curl --connect-timeout 1 --max-time 2 -fsS "$BASE/health" >/dev/null 2>&1; then break; fi
    kill -0 "$PID"
    sleep 1
done
curl --connect-timeout 2 --max-time 5 -fsS "$BASE/health" >/dev/null
curl --max-time 10 -fsS "$BASE/v1/models" >"$OUT/models.json"
ID=$(jq -er '.data[] | select(.loaded == true and .streaming == true and .input_modalities == ["text"]) | .id' "$OUT/models.json")
if jq -e '.quantization_config.store_dtype == "mxfp4"' "$MODEL/config.json" >/dev/null; then
    jq -e --arg id "$ID" '.data[] | select(.id == $id) | .streaming_required == true' "$OUT/models.json" >/dev/null
    grep -q '\[mimo-source\] loading original shards' "$OUT/server.log"
fi
python3 - "$OUT/server.log" "$MODEL/config.json" <<'PY'
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
jq -nc --arg model "$ID" '{model:$model,messages:[{role:"user",content:"Write one short sentence about rain."}],temperature:0,seed:1234,max_tokens:16,enable_thinking:false,stream:false}' >"$OUT/request.json"
for n in 1 2; do
    curl --connect-timeout 5 --max-time 1800 -fsS -H 'Content-Type: application/json' \
        -d @"$OUT/request.json" "$BASE/v1/chat/completions" >"$OUT/reply$n.json"
    jq -e '.choices[0].message.content | type == "string"' "$OUT/reply$n.json" >/dev/null
    jq -S '.choices[0].message | {content,reasoning_content}' "$OUT/reply$n.json" >"$OUT/text$n.json"
done
cmp "$OUT/text1.json" "$OUT/text2.json"

jq -nc --arg model "$ID" '{model:$model,messages:[{role:"user",content:([range(64)|"word\(.): rain falls."]|join(" "))}],temperature:0,max_tokens:4,enable_thinking:false,stream:false}' |
    curl --connect-timeout 5 --max-time 1800 -fsS -H 'Content-Type: application/json' \
        -d @- "$BASE/v1/chat/completions" >"$OUT/long.json"
jq -e '.usage.prompt_tokens > 128 and (.choices[0].message.content | type == "string")' "$OUT/long.json" >/dev/null

jq '.enable_mtp=true' "$OUT/request.json" >"$OUT/mtp-request.json"
STATUS=$(curl --max-time 30 -sS -o "$OUT/mtp.json" -w '%{http_code}' \
    -H 'Content-Type: application/json' -d @"$OUT/mtp-request.json" "$BASE/v1/chat/completions")
[[ "$STATUS" == 400 ]]
jq -e '.error.type == "invalid_request_error" and (.error.message | contains("MTP speculative decode is not supported"))' "$OUT/mtp.json" >/dev/null
grep -q '\[expert-stream\] ssd budget' "$OUT/server.log"
grep -q '\[expert-stream\] cache' "$OUT/server.log"
echo "PASS: MiMo streaming discovery, sigmoid lossy pick, greedy determinism, window-crossing prefill, and MTP refusal"

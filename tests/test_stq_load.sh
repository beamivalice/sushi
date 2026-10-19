#!/usr/bin/env bash
# Boots one STQ pack and checks the engine engaged its codebook and answers deterministically.
. "$(dirname "$0")/private_cache.sh"
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PACK=${1:?usage: test_stq_load.sh PACK PORT}
PORT=${2:?missing port}
[[ "$PORT" =~ ^[0-9]+$ && "$PORT" -ge 1 && "$PORT" -le 65535 ]] || exit 2
CODEBOOK=$(jq -er 'select(.expert_quant.format == "stq") | .expert_quant.codebook' "$PACK/config.json")
RUN=$(mktemp -d /tmp/sushi-stq-load.XXXXXX)
PID=
LOCKED=0
OWNER="stq-load-$$"
cleanup() {
    local status=$?
    [[ -z "$PID" ]] || { kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; }
    [[ "$LOCKED" != 1 ]] || "$ROOT/scripts/gpu-lock.sh" release "$OWNER" >>"$RUN/lock.log" 2>&1
    if [[ "$status" != 0 ]]; then
        cat "$RUN"/*.log >&2 2>/dev/null || true
        printf 'STQ load test failed; artifacts: %s\n' "$RUN" >&2
    else
        rm -rf "$RUN"
    fi
}
trap cleanup EXIT
cd "$ROOT"
.zig-toolchain/zig build -Doptimize=ReleaseFast >"$RUN/build.log" 2>&1
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >"$RUN/port.log" 2>&1; then
    printf 'port %s is already in use\n' "$PORT" >&2
    exit 1
fi
"$ROOT/scripts/gpu-lock.sh" acquire "$OWNER" >>"$RUN/lock.log" 2>&1
LOCKED=1
./zig-out/bin/sushi --serve --model "$PACK" --host 127.0.0.1 --port "$PORT" --no-mtp --no-vision \
    --ctx-size 2048 --prefill-chunk 512 --max-concurrent 1 --prefix-cache-entries 0 --prefix-cache-mem 0 \
    --prefix-cache-disk 0 >"$RUN/serve.log" 2>&1 &
PID=$!
ready=0
for ((attempt=0; attempt<1200; attempt++)); do
    if curl --connect-timeout 1 --max-time 2 -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    kill -0 "$PID" 2>/dev/null || exit 1
    sleep 1
done
[[ "$ready" == 1 ]]
grep -q "\[expert-exl3\] engaged .*codebook=$CODEBOOK " "$RUN/serve.log"
MODEL=$(curl --connect-timeout 2 --max-time 10 -fsS "http://127.0.0.1:$PORT/v1/models" | jq -er '[.data[] | select(.loaded == true) | .id][0]')
jq -nc --arg model "$MODEL" '{model:$model,messages:[{role:"user",content:"Write one short sentence about a library."}],temperature:0,seed:1234,enable_thinking:false,max_tokens:24,stream:false}' >"$RUN/request.json"
for pass in 1 2; do
    curl --connect-timeout 5 --max-time 600 -fsS -H 'Content-Type: application/json' \
        -d @"$RUN/request.json" "http://127.0.0.1:$PORT/v1/chat/completions" >"$RUN/reply-$pass.json"
    jq -e '.choices[0].message.content | type == "string" and length > 0' "$RUN/reply-$pass.json" >/dev/null
done
cmp -s <(jq -S -c '.choices[0].message' "$RUN/reply-1.json") <(jq -S -c '.choices[0].message' "$RUN/reply-2.json")

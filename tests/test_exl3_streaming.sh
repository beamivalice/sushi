#!/usr/bin/env bash
. "$(dirname "$0")/private_cache.sh"
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PACK=${1:?usage: test_exl3_streaming.sh PACK BUDGET_GIB PORT}
BUDGET=${2:?missing SSD budget in GiB}
PORT=${3:?missing port}
[[ "$BUDGET" =~ ^[1-9][0-9]*$ && "$PORT" =~ ^[0-9]+$ && "$PORT" -ge 1 && "$PORT" -le 65535 ]] || exit 2
RUN=$(mktemp -d /tmp/sushi-exl3-stream.XXXXXX)
PID=
LOCKED=0
OWNER="exl3-stream-$$"
stop() {
    if [[ -n "$PID" ]]; then
        kill "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
        PID=
    fi
    if [[ "$LOCKED" == 1 ]]; then
        "$ROOT/scripts/gpu-lock.sh" release "$OWNER" >>"$RUN/lock.log" 2>&1
        LOCKED=0
    fi
}
cleanup() {
    local status=$?
    stop
    if [[ "$status" != 0 ]]; then
        for log in "$RUN"/*.log; do [[ ! -f "$log" ]] || cat "$log" >&2; done
        printf 'EXL3 streaming test failed; artifacts: %s\n' "$RUN" >&2
    else
        rm -rf "$RUN"
    fi
}
trap cleanup EXIT
python3 - "$PACK" "$RUN/pack" <<'PY'
import json, pathlib, sys
source = pathlib.Path(sys.argv[1]).resolve(strict=True)
config = json.loads((source / 'config.json').read_text())
assert config.get('expert_quant', {}).get('format') in ('exl3', 'stq'), 'expected an EXL3 or STQ pack'
dest = pathlib.Path(sys.argv[2])
dest.mkdir()
for child in source.iterdir():
    if child.name != 'model-settings.json':
        (dest / child.name).symlink_to(child)
PY
cd "$ROOT"
.zig-toolchain/zig build -Doptimize=ReleaseFast >"$RUN/build.log" 2>&1
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >"$RUN/port.log" 2>&1; then
    printf 'port %s is already in use\n' "$PORT" >&2
    exit 1
fi
for arm in resident streamed; do
    "$ROOT/scripts/gpu-lock.sh" acquire "$OWNER" >>"$RUN/lock.log" 2>&1
    LOCKED=1
    FLAGS=()
    if [[ "$arm" == streamed ]]; then FLAGS=(--ssd-budget-gb "$BUDGET"); fi
    ./zig-out/bin/sushi --serve --model "$RUN/pack" --host 127.0.0.1 --port "$PORT" \
        --no-mtp --no-vision --ctx-size 2048 --prefill-chunk 512 --max-concurrent 1 \
        --prefix-cache-entries 0 --prefix-cache-mem 0 --prefix-cache-disk 0 \
        --kv-quant 8 --metrics ${FLAGS[@]+"${FLAGS[@]}"} >"$RUN/$arm.log" 2>&1 &
    PID=$!
    ready=0
    for ((attempt=0; attempt<1200; attempt++)); do
        if curl --connect-timeout 1 --max-time 2 -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
        kill -0 "$PID" 2>/dev/null || exit 1
        sleep 1
    done
    [[ "$ready" == 1 ]]
    if [[ "$arm" == streamed ]]; then
        grep -q '\[expert-stream\] ssd budget' "$RUN/$arm.log"
        grep -q '\[expert-stream\] cache .*fallback_imports=0' "$RUN/$arm.log"
        grep -q '\[expert-stream\].*warm' "$RUN/$arm.log"
    else
        if grep -q '\[expert-stream\] cache ' "$RUN/$arm.log"; then exit 1; fi
    fi
    MODEL=$(curl --connect-timeout 2 --max-time 10 -fsS "http://127.0.0.1:$PORT/v1/models" | jq -er '[.data[] | select(.loaded == true) | .id][0]')
    jq -nc --arg model "$MODEL" '{model:$model,messages:[{role:"user",content:(("The library keeps books about rivers, forests, mountains, and cities. " * 12) + "Write one short sentence about a library.")}],temperature:0,seed:1234,enable_thinking:false,max_tokens:32,stream:false,logprobs:true,top_logprobs:20}' >"$RUN/request.json"
    for pass in 1 2; do
        curl --connect-timeout 5 --max-time 1800 -fsS -H 'Content-Type: application/json' \
            -d @"$RUN/request.json" "http://127.0.0.1:$PORT/v1/chat/completions" >"$RUN/$arm-$pass.json"
        jq -e '(.choices | length == 1) and (.choices[0].logprobs.content | type == "array" and length > 0 and all(.[]; (.top_logprobs | length) == 20))' "$RUN/$arm-$pass.json" >/dev/null
        jq -S -c '{message:.choices[0].message,logprobs:.choices[0].logprobs,finish_reason:.choices[0].finish_reason,completion_tokens:.usage.completion_tokens}' \
            "$RUN/$arm-$pass.json" >"$RUN/$arm-$pass.reply"
    done
    cmp -s "$RUN/$arm-1.reply" "$RUN/$arm-2.reply"
    stop
done
cmp -s "$RUN/resident-1.reply" "$RUN/streamed-1.reply"

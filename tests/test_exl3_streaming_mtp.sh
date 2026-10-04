#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PACK=${1:?usage: test_exl3_streaming_mtp.sh PACK BUDGET_GIB PORT}
BUDGET=${2:?missing SSD budget in GiB}
PORT=${3:?missing port}
[[ "$BUDGET" =~ ^[1-9][0-9]*$ && "$PORT" =~ ^[0-9]+$ && "$PORT" -ge 1 && "$PORT" -le 65535 ]] || exit 2
RUN=$(mktemp -d /tmp/sushi-exl3-stream-mtp.XXXXXX)
PID=
LOCKED=0
OWNER="exl3-stream-mtp-$$"
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
        for log in "$RUN"/*.log; do [[ ! -f "$log" ]] || tail -40 "$log" >&2; done
        printf 'EXL3 streaming MTP test failed; artifacts: %s\n' "$RUN" >&2
    else
        rm -rf "$RUN"
    fi
}
trap cleanup EXIT
python3 - "$PACK" "$RUN/pack" <<'PY'
import json, pathlib, sys
source = pathlib.Path(sys.argv[1]).resolve(strict=True)
config = json.loads((source / 'config.json').read_text())
assert config.get('expert_quant', {}).get('format') == 'exl3', 'expected an EXL3 pack'
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
ask() { # arm, enable_mtp
    jq -nc --arg model "$MODEL" --argjson mtp "$2" '{model:$model,messages:[{role:"user",content:"Write a Python class LRUCache(capacity) with get(key) (-1 when missing) and put(key, value), both O(1). Only the code."}],temperature:0,seed:1234,enable_thinking:false,enable_mtp:$mtp,max_tokens:400,stream:false}' >"$RUN/request.json"
    curl --connect-timeout 5 --max-time 1800 -fsS -H 'Content-Type: application/json' \
        -d @"$RUN/request.json" "http://127.0.0.1:$PORT/v1/chat/completions" | jq -er '.choices[0].message.content' >"$RUN/$1.reply"
}
for arm in resident streamed; do
    "$ROOT/scripts/gpu-lock.sh" acquire "$OWNER" >>"$RUN/lock.log" 2>&1
    LOCKED=1
    FLAGS=()
    if [[ "$arm" == streamed ]]; then FLAGS=(--ssd-budget-gb "$BUDGET"); fi
    ./zig-out/bin/sushi --serve --model "$RUN/pack" --host 127.0.0.1 --port "$PORT" \
        --mtp --no-vision --ctx-size 4096 --max-concurrent 1 \
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
    MODEL=$(curl --connect-timeout 2 --max-time 10 -fsS "http://127.0.0.1:$PORT/v1/models" | jq -er '[.data[] | select(.loaded == true) | .id][0]')
    ask "$arm" true
    if [[ "$arm" == streamed ]]; then
        grep -q '\[expert-stream\] MTP head resident' "$RUN/$arm.log"
        grep -q '\[qwen4\] MTP head loaded' "$RUN/$arm.log"
        grep -q 'spec-stats\] mode=mtp' "$RUN/$arm.log"
        ask "$arm-serial" false
    fi
    stop
done
cmp -s "$RUN/resident.reply" "$RUN/streamed.reply"
cmp -s "$RUN/streamed.reply" "$RUN/streamed-serial.reply"
printf 'streamed MTP reply equals resident MTP and streamed serial (%s bytes)\n' "$(wc -c <"$RUN/streamed.reply" | tr -d ' ')"

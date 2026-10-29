#!/usr/bin/env bash
# sushi-test: tier=live boot=own pack=default boots=2
# One EXL3 or STQ pack (rate-grouped Flash-Next layers included) booted resident, then streamed under
# `--ssd-budget-gb` (SUSHI_TEST_STREAM_BUDGET_GB, default 60). The same requests run on both boots:
#   * serial (`enable_mtp:false`) greedy replies plus top-20 logprobs: identical across two passes and identical
#     between the resident and the streamed boot;
#   * streaming engagement lines (`[expert-stream] ssd budget`, a warm cache with fallback_imports=0), and none on the
#     resident boot;
#   * with an MTP head on the pack: `[expert-stream] MTP head resident`, `[qwen4] MTP head loaded` and
#     `[spec-stats] mode=mtp` on the streamed boot, and the streamed MTP reply equals resident MTP and streamed serial.
set -euo pipefail
. "$(dirname "$0")/lib/boot.sh"

PACK=$(sushi_pack qwen4) || exit $?
BUDGET=${SUSHI_TEST_STREAM_BUDGET_GB:-60}
RUN=$(mktemp -d)
trap 'rm -rf "$RUN"' EXIT

# A symlink clone without the pack's own model-settings.json, so the launch flags below are the only settings.
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

post() { curl --connect-timeout 5 --max-time 1800 -fsS -H 'Content-Type: application/json' -d @- "$SUSHI_URL/v1/chat/completions"; }
LOGPROBS_REQ() {
    jq -nc --arg model "$MODEL" '{model:$model,messages:[{role:"user",content:(("The library keeps books about rivers, forests, mountains, and cities. " * 12) + "Write one short sentence about a library.")}],temperature:0,seed:1234,enable_thinking:false,enable_mtp:false,max_tokens:32,stream:false,logprobs:true,top_logprobs:20}'
}
CODE_REQ() { # enable_mtp
    jq -nc --arg model "$MODEL" --argjson mtp "$1" '{model:$model,messages:[{role:"user",content:"Write a Python class LRUCache(capacity) with get(key) (-1 when missing) and put(key, value), both O(1). Only the code."}],temperature:0,seed:1234,enable_thinking:false,enable_mtp:$mtp,max_tokens:400,stream:false}'
}

for arm in resident streamed; do
    FLAGS=()
    [[ "$arm" == streamed ]] && FLAGS=(--ssd-budget-gb "$BUDGET")
    sushi_boot "exl3-$arm" "$RUN/pack" --no-vision --ctx-size 4096 --prefill-chunk 512 --max-concurrent 1 \
        --prefix-cache-entries 0 --prefix-cache-mem 0 --prefix-cache-disk 0 --kv-quant 8 --metrics ${FLAGS[@]+"${FLAGS[@]}"} || exit $?
    if [[ "$arm" == streamed ]]; then
        grep -q '\[expert-stream\] ssd budget' "$SUSHI_LOG"
        grep -q '\[expert-stream\] cache .*fallback_imports=0' "$SUSHI_LOG"
        grep -q '\[expert-stream\].*warm' "$SUSHI_LOG"
    else
        if grep -q '\[expert-stream\] cache ' "$SUSHI_LOG"; then echo "the resident boot streamed" >&2; exit 1; fi
    fi
    MODEL=$(curl --connect-timeout 2 --max-time 10 -fsS "$SUSHI_URL/v1/models" | jq -er '[.data[] | select(.loaded == true) | .id][0]')

    for pass in 1 2; do
        LOGPROBS_REQ | post > "$RUN/$arm-$pass.json"
        jq -e '(.choices | length == 1) and (.choices[0].logprobs.content | type == "array" and length > 0 and all(.[]; (.top_logprobs | length) == 20))' "$RUN/$arm-$pass.json" > /dev/null
        jq -S -c '{message:.choices[0].message,logprobs:.choices[0].logprobs,finish_reason:.choices[0].finish_reason,completion_tokens:.usage.completion_tokens}' \
            "$RUN/$arm-$pass.json" > "$RUN/$arm-$pass.reply"
    done
    cmp -s "$RUN/$arm-1.reply" "$RUN/$arm-2.reply" || { echo "$arm: two greedy passes differ" >&2; exit 1; }

    MTP=$(curl -fsS "$SUSHI_URL/props" | jq -r '.settings.mtp.loaded')
    if [[ "$MTP" == true ]]; then
        CODE_REQ true | post | jq -er '.choices[0].message.content' > "$RUN/$arm.mtp.reply"
        if [[ "$arm" == streamed ]]; then
            grep -q '\[expert-stream\] MTP head resident' "$SUSHI_LOG"
            grep -q 'spec-stats\] mode=mtp' "$SUSHI_LOG"
            CODE_REQ false | post | jq -er '.choices[0].message.content' > "$RUN/streamed.serial.reply"
        fi
    else
        echo "note: the pack has no MTP head loaded; the MTP rows are skipped" >&2
    fi
    sushi_stop 0
done
cmp -s "$RUN/resident-1.reply" "$RUN/streamed-1.reply" || { echo "streamed greedy reply or top-20 logprobs differ from resident" >&2; exit 1; }
if [[ -f "$RUN/resident.mtp.reply" ]]; then
    cmp -s "$RUN/resident.mtp.reply" "$RUN/streamed.mtp.reply" || { echo "streamed MTP reply differs from resident MTP" >&2; exit 1; }
    cmp -s "$RUN/streamed.mtp.reply" "$RUN/streamed.serial.reply" || { echo "streamed MTP reply differs from streamed serial" >&2; exit 1; }
    printf 'streamed MTP reply equals resident MTP and streamed serial (%s bytes)\n' "$(wc -c < "$RUN/streamed.mtp.reply" | tr -d ' ')"
fi
echo "PASS exl3 streaming: streamed greedy replies and top-20 logprobs equal resident"

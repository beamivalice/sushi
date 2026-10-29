#!/usr/bin/env bash
# sushi-test: tier=live boot=api
# Checks the booted STQ pack engaged its codebook and answers deterministically.
set -euo pipefail
. "$(dirname "$0")/lib/boot.sh"; sushi_attach
sushi_require stq
BASE=$SUSHI_URL
RUN=$(mktemp -d /tmp/sushi-stq-load.XXXXXX)
trap 'rm -rf "$RUN"' EXIT
CODEBOOK=$(jq -er '.expert_quant.codebook' "$SUSHI_PACK/config.json")
# The engaged line is printed at load, before the attach point: read the whole boot log, not the slice.
grep -q "\[expert-exl3\] engaged .*codebook=$CODEBOOK " "$SUSHI_LOG" || {
    echo "FAIL: no '[expert-exl3] engaged ... codebook=$CODEBOOK' line in the boot log" >&2
    exit 1
}
MODEL=$(curl --connect-timeout 2 --max-time 10 -fsS "$BASE/v1/models" | jq -er '[.data[] | select(.loaded == true) | .id][0]')
jq -nc --arg model "$MODEL" --arg nonce "$SUSHI_NONCE" '{model:$model,messages:[{role:"user",content:($nonce + " Write one short sentence about a library.")}],temperature:0,seed:1234,enable_thinking:false,max_tokens:24,stream:false}' >"$RUN/request.json"
for pass in 1 2; do
    curl --connect-timeout 5 --max-time 600 -fsS -H 'Content-Type: application/json' \
        -d @"$RUN/request.json" "$BASE/v1/chat/completions" >"$RUN/reply-$pass.json"
    jq -e '.choices[0].message.content | type == "string" and length > 0' "$RUN/reply-$pass.json" >/dev/null
done
cmp -s <(jq -S -c '.choices[0].message' "$RUN/reply-1.json") <(jq -S -c '.choices[0].message' "$RUN/reply-2.json") || {
    echo "FAIL: two greedy completions of one request differ" >&2
    exit 1
}

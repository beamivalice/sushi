#!/usr/bin/env bash
# sushi-test: tier=live boot=own pack=default boots=1
# The first chat after a load that streams a Flash-Next pack's experts (the 32 GB Macs' case) is a 200, and the load
# raises no MLX error: an error the load latched failed the first request as "generation failed" until a retry.
#   * no `[mlx]` line in the log before the model is ready, and no `[load] MLX raised` line;
#   * the FIRST request answers 200 with text, with no `prefill failed` or `[slot]` failure line;
#   * a second identical greedy request answers the same text.
# SUSHI_TEST_STREAM_BUDGET_GB (default 12) must sit below the pack's routed experts, or nothing streams.
set -euo pipefail
. "$(dirname "$0")/lib/boot.sh"

PACK=$(sushi_pack "${SUSHI_TEST_FIRST_PACK:-Qwen3.8-Flash-Next-Sushi-4bpw}") || exit $?
BUDGET=${SUSHI_TEST_STREAM_BUDGET_GB:-12}

sushi_boot first-streamed "$PACK" --ssd-budget-gb "$BUDGET" || exit $?
MODEL=$(curl -fsS "$SUSHI_URL/v1/models" | jq -r '.data[0].id')

grep -q '\[expert-stream\] ssd budget' "$SUSHI_LOG" || { echo "the boot did not stream its experts" >&2; exit 1; }
if grep -q '^\[mlx\]' "$SUSHI_LOG"; then
    echo "the load raised an MLX error:" >&2
    grep '^\[mlx\]' "$SUSHI_LOG" | sort | uniq -c >&2
    exit 1
fi

REQ=$(jq -nc --arg model "$MODEL" '{model:$model,messages:[{role:"user",content:"hello"}],max_tokens:32,temperature:0,stream:false}')
first=$(curl --max-time 600 -sS -w '\n%{http_code}' -H 'Content-Type: application/json' -d "$REQ" "$SUSHI_URL/v1/chat/completions")
code=${first##*$'\n'}
body=${first%$'\n'*}
if [[ "$code" != 200 ]]; then
    echo "the first request after the load answered $code: $body" >&2
    exit 1
fi
text=$(jq -r '.choices[0].message | (.content // "") + (.reasoning_content // "")' <<<"$body")
[[ -n "$text" ]] || { echo "the first reply is empty" >&2; exit 1; }
if grep -q 'prefill failed\|\[slot\] request' "$SUSHI_LOG"; then
    echo "the first request failed a slot:" >&2
    grep 'prefill failed\|\[slot\] request' "$SUSHI_LOG" >&2
    exit 1
fi

second=$(curl --max-time 600 -fsS -H 'Content-Type: application/json' -d "$REQ" "$SUSHI_URL/v1/chat/completions" | jq -r '.choices[0].message | (.content // "") + (.reasoning_content // "")')
[[ "$second" == "$text" ]] || { echo "a retry answered different text" >&2; exit 1; }
echo "first request after a streamed load: 200, no MLX error in the load"

#!/bin/bash
# sushi-test: tier=live boot=own pack=qwen4-bf16 boots=1
# The BF16 source checkpoint streamed from SSD: a 3-prompt KLD against the teacher fixture, then one boot that
# answers a short prompt twice with the same greedy bytes and takes a 4k-token prompt.
set -euo pipefail
. "$(dirname "$0")/../lib/boot.sh"

CHECKPOINT=$(sushi_pack qwen4-bf16) || exit $?
FIXTURE=${QWEN4_TEACHER_FIXTURE:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/kld-teacher/mlx-serve-bf16-60x64}
if [[ ! -f "$FIXTURE/baseline.json" ]]; then
    echo "SKIP $(basename "$0"): teacher fixture missing at $FIXTURE"
    exit 3
fi
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# The KLD run loads the checkpoint itself, so it holds the lock on its own and lets go before the server boots.
"$SUSHI_ROOT/scripts/gpu-lock.sh" acquire tests-bf16-kld
taskpolicy -a "$SUSHI_BIN" kld compare --model "$CHECKPOINT" --fixture "$FIXTURE" --limit 3 --ssd-budget-gb 60 --no-mtp --kv-quant off --json "$WORK/kld.json" \
    || { "$SUSHI_ROOT/scripts/gpu-lock.sh" release tests-bf16-kld; exit 1; }
"$SUSHI_ROOT/scripts/gpu-lock.sh" release tests-bf16-kld
jq -e '.mean_kld_to_eos < 0.02 and .mean_top1_to_eos > 0.95' "$WORK/kld.json" > /dev/null

sushi_boot bf16 "$CHECKPOINT" --ssd-budget-gb 60 --no-mtp --kv-quant 8 --ctx-size 65536 --metrics --no-vision || exit $?
MODEL_ID=$(curl --connect-timeout 2 --max-time 10 -fsS "$SUSHI_URL/v1/models" | jq -er '[.data[] | select(.loaded == true) | .id][0]')

post() { curl --connect-timeout 5 --max-time 1800 -fsS -H 'Content-Type: application/json' -d @- "$SUSHI_URL/v1/chat/completions"; }
REQ=$(jq -nc --arg model "$MODEL_ID" '{model:$model,messages:[{role:"user",content:"Write one short sentence about Bangkok."}],temperature:0,max_tokens:16,seed:1234,stream:false}')
echo "$REQ" | post > "$WORK/1.json"
echo "$REQ" | post > "$WORK/2.json"
jq -e '.choices[0].message.content | type == "string"' "$WORK/1.json" > /dev/null
cmp <(jq -r '.choices[0].message.content' "$WORK/1.json") <(jq -r '.choices[0].message.content' "$WORK/2.json")
LONG_PROMPT=$(for _ in $(seq 1 12); do cat "$FIXTURE/prompts/00_wikitext2-test-00-robert-boulter/prompt.txt"; done)
jq -n --arg model "$MODEL_ID" --arg content "$LONG_PROMPT" '{model:$model,messages:[{role:"user",content:$content}],temperature:0,max_tokens:4,stream:false}' | post > "$WORK/3.json"
jq -e '.choices[0].message.content | type == "string"' "$WORK/3.json" > /dev/null
grep -q '\[expert-stream\]' "$SUSHI_LOG"
echo "PASS bf16 streaming: kld compare, boot, short prompt, 4k prompt and greedy determinism"

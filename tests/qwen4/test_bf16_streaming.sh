#!/usr/bin/env bash
# sushi-test: tier=live boot=own pack=qwen4-bf16 boots=1
# The BF16 Flash-Next source streamed from SSD: `sushi kld compare` against the teacher fixture (3 prompts), then a
# served boot with a short prompt, a 4k prompt and greedy determinism.
set -euo pipefail
. "$(dirname "$0")/../lib/boot.sh"

FIXTURE=${QWEN4_TEACHER_FIXTURE:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/kld-teacher/mlx-serve-bf16-60x64}
CHECKPOINT=$(sushi_pack qwen4-bf16) || exit $?
[[ -f "$FIXTURE/baseline.json" ]] || { echo "SKIP $(basename "$0"): teacher fixture $FIXTURE absent"; exit 3; }
BODY1=$(mktemp)
BODY2=$(mktemp)
BODY3=$(mktemp)
KLD_JSON=$(mktemp)
trap 'rm -f "$BODY1" "$BODY2" "$BODY3" "$KLD_JSON"' EXIT

# The compare loads the checkpoint itself, so it holds the lock for its own run.
"$SUSHI_ROOT/scripts/gpu-lock.sh" acquire "${GPU_LOCK_OWNER:-tests}-bf16-kld"
rc=0
"$SUSHI_BIN" kld compare --model "$CHECKPOINT" --fixture "$FIXTURE" --limit 3 --ssd-budget-gb 60 --no-mtp --kv-quant off --json "$KLD_JSON" || rc=$?
"$SUSHI_ROOT/scripts/gpu-lock.sh" release "${GPU_LOCK_OWNER:-tests}-bf16-kld"
[[ "$rc" = 0 ]] || exit "$rc"
jq -e '.mean_kld_to_eos < 0.02 and .mean_top1_to_eos > 0.95' "$KLD_JSON" >/dev/null

sushi_boot bf16 "$CHECKPOINT" --ssd-budget-gb 60 --no-mtp --kv-quant 8 --ctx-size 65536 --metrics --no-vision
PORT=$SUSHI_PORT
MODEL_ID=$(curl --connect-timeout 2 --max-time 10 -fsS "http://127.0.0.1:$PORT/v1/models" | jq -er '[.data[] | select(.loaded == true) | .id][0]')

REQ=$(jq -nc --arg model "$MODEL_ID" '{model:$model,messages:[{role:"user",content:"Write one short sentence about Bangkok."}],temperature:0,max_tokens:16,seed:1234,stream:false}')
curl --connect-timeout 5 --max-time 1800 -fsS -H 'Content-Type: application/json' -d "$REQ" "http://127.0.0.1:$PORT/v1/chat/completions" >"$BODY1"
curl --connect-timeout 5 --max-time 1800 -fsS -H 'Content-Type: application/json' -d "$REQ" "http://127.0.0.1:$PORT/v1/chat/completions" >"$BODY2"
jq -e '.choices[0].message.content | type == "string"' "$BODY1" >/dev/null
cmp <(jq -r '.choices[0].message.content' "$BODY1") <(jq -r '.choices[0].message.content' "$BODY2")
LONG_PROMPT=$(for _ in $(seq 1 12); do sed -n '1,$p' "$FIXTURE/prompts/00_wikitext2-test-00-robert-boulter/prompt.txt"; done)
jq -n --arg model "$MODEL_ID" --arg content "$LONG_PROMPT" '{model:$model,messages:[{role:"user",content:$content}],temperature:0,max_tokens:4,stream:false}' | curl --connect-timeout 5 --max-time 1800 -fsS -H 'Content-Type: application/json' -d @- "http://127.0.0.1:$PORT/v1/chat/completions" >"$BODY3"
jq -e '.choices[0].message.content | type == "string"' "$BODY3" >/dev/null
grep -E '\[expert-stream\]|Prompt processing|tokens/sec' "$SUSHI_LOG" | tail -80
echo "PASS bf16 streaming kld compare, boot, short prompt, 4k prompt, and greedy determinism"

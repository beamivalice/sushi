#!/usr/bin/env bash
# sushi-test: tier=pack pack=default
# Hy4-Flash's calibration contract end to end on the tiny STQ source (HY4_BF16_TINY_STQ): `sushi kld capture
# --layer-major` writes calib/ over 64 windows, and `sushi convert exl3 --capture` reads it whole. Every bank whose input
# the capture taps rounds on its own sum (none under 64 rows); only the embedding and the DFly2 drafter, which have no
# tap, round against the identity. SUSHI_CAPTURE_BIN runs the capture with another build (default SUSHI_BIN).
set -euo pipefail
cd "$(dirname "$0")/.."
. tests/private_cache.sh
SRC=${HY4_BF16_TINY_STQ:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/tiny-models/hy4-bf16-tiny-stq}
[ -d "$SRC" ] || { echo "SKIP: no fixture at $SRC"; exit 3; }
BIN=${SUSHI_BIN:-zig-out/bin/sushi}
CAPTURE_BIN=${SUSHI_CAPTURE_BIN:-$BIN}
RUN=$(mktemp -d)
trap 'rm -rf "$RUN"' EXIT

for w in $(seq 0 63); do
    ids=$(seq 0 47 | awk -v w="$w" '{ printf "%s%d", (NR > 1 ? "," : ""), 1000 + (w * 37 + $1 * 101) % 100000 }')
    printf '{"id":"w%02d","prompt_ids":[%s]}\n' "$w" "$ids"
done > "$RUN/prompts.jsonl"

# Run as a user does: MLX_ENABLE_TF32 unset (the capture turns TF32 off itself).
env -u MLX_ENABLE_TF32 "$CAPTURE_BIN" kld capture --model "$SRC" --prompts "$RUN/prompts.jsonl" --out "$RUN/cap" --tokens 1 \
    --no-template --layer-major > "$RUN/capture.log" 2>&1 || { echo "FAIL: capture"; tail -5 "$RUN/capture.log"; exit 1; }
"$BIN" convert exl3 --model "$SRC" --capture "$RUN/cap/calib" --out "$RUN/pack" --expert-codebook stq-pair1 --experts-k 4 \
    > "$RUN/convert.log" 2>&1 || { echo "FAIL: convert"; tail -5 "$RUN/convert.log"; exit 1; }

untapped=$(python3 - "$SRC" <<'PY'
import json, pathlib, struct, sys
n = 0
for f in pathlib.Path(sys.argv[1]).glob('*.safetensors'):
    raw = f.read_bytes()
    head = json.loads(raw[8:8 + struct.unpack('<Q', raw[:8])[0]])
    for k, v in head.items():
        if k == '__metadata__' or not k.endswith('.weight') or v['dtype'] != 'BF16' or len(v['shape']) != 2:
            continue
        if (k == 'model.embed_tokens.weight' or k.startswith('model.dfly_layers.')) and all(d % 128 == 0 for d in v['shape']):
            n += 1
print(n)
PY
)
want="against the identity: $untapped banks the capture taps no input of (the embedding, the DFly2 drafter), 0 banks under 64 rows"
grep -qF "$want" "$RUN/convert.log" || { echo "FAIL: expected '$want'"; tail -3 "$RUN/convert.log"; exit 1; }
grep -Eq '^wrote pack capture-[0-9a-f]{12} ' "$RUN/convert.log" || { echo "FAIL: no capture pack line"; exit 1; }
echo "PASS: $(grep -c . "$RUN/prompts.jsonl") windows captured and converted, $untapped untapped banks"

#!/usr/bin/env bash
# sushi-test: tier=pack pack=default
# The Hy4-Flash STQ pack recipe end to end on the tiny BF16 source (HY4_BF16_TINY_STQ), every command as the release
# run takes it: the BF16 teacher (`kld capture --kv-quant off`), the layer-major calibration capture, the encode
# (PAIR1 K4 experts, K6 trunk, `--drafter-k 4`), a stop and `--resume` back to the uninterrupted pack's bytes (the
# encoder writes the pack order), `sushi repack` into the pair-major pack Hy4-Flash serves, and that pack loaded and
# scored against the teacher (`kld compare`). The tiny's weights are random, so no KLD is asserted.
set -euo pipefail
cd "$(dirname "$0")/.."
. tests/private_cache.sh
SRC=${HY4_BF16_TINY_STQ:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/tiny-models/hy4-bf16-tiny-stq}
[ -d "$SRC" ] || { echo "SKIP: no fixture at $SRC"; exit 3; }
BIN=${SUSHI_BIN:-zig-out/bin/sushi}
RUN=$(mktemp -d)
trap 'rm -rf "$RUN"' EXIT
ids() { seq 0 $(($2 - 1)) | awk -v w="$1" '{ printf "%s%d", (NR > 1 ? "," : ""), 1000 + (w * 37 + $1 * 101) % 100000 }'; }
for w in $(seq 0 3); do printf '{"id":"p%d","prompt_ids":[%s]}\n' "$w" "$(ids $((w + 500)) 40)"; done > "$RUN/prompts.jsonl"
for w in $(seq 0 63); do printf '{"id":"w%02d","prompt_ids":[%s]}\n' "$w" "$(ids "$w" 48)"; done > "$RUN/windows.jsonl"
step() { local name=$1; shift; "$@" > "$RUN/$name.log" 2>&1 || { echo "FAIL: $name"; tail -5 "$RUN/$name.log"; exit 1; }; }

step teacher "$BIN" kld capture --model "$SRC" --prompts "$RUN/prompts.jsonl" --out "$RUN/teacher" --tokens 32 --top-k 10 \
    --label hy4-recipe --kv-quant off --ctx-size 8192
step capture "$BIN" kld capture --model "$SRC" --prompts "$RUN/windows.jsonl" --out "$RUN/cap" --tokens 1 --no-template \
    --layer-major
encode=("$BIN" convert exl3 --model "$SRC" --capture "$RUN/cap/calib" --out "$RUN/pack" --expert-codebook stq-pair1
    --experts-k 4 --drafter-k 4)
step encode "${encode[@]}"
(cd "$RUN/pack" && shasum -a 256 ./*.json ./*.safetensors) > "$RUN/pack.sha256"
first=$(ls "$RUN/pack" | grep '^model-layer-' | sed -n 1p)
rm "$RUN/pack/config.json" "$RUN/pack/model.safetensors.index.json" "$RUN/pack/$first" "$RUN/pack/model-globals.safetensors"
step resume "${encode[@]}" --resume
(cd "$RUN/pack" && shasum -a 256 -c --quiet "$RUN/pack.sha256") || { echo "FAIL: the resumed pack differs"; exit 1; }
k=$(python3 - "$RUN/pack/model-dfly.safetensors" <<'PY'
import json, struct, sys
raw = open(sys.argv[1], 'rb').read(1 << 20)
print(json.loads(raw[8:8 + struct.unpack('<Q', raw[:8])[0]])['__metadata__']['k'])
PY
)
[ "$k" = 4 ] || { echo "FAIL: model-dfly k $k, want 4"; exit 1; }
step repack "$BIN" repack "$RUN/pack"
grep -q 'is already pair-major' "$RUN/repack.log" || { echo "FAIL: the converted pack is not pair-major: a repack had work to do"; tail -3 "$RUN/repack.log"; exit 1; }
order=$(python3 - "$RUN/pack" <<'PY'
import glob, json, struct, sys
seen = set()
for f in sorted(glob.glob(sys.argv[1] + '/model-layer-*.safetensors')):
    raw = open(f, 'rb').read(1 << 20)
    head = json.loads(raw[8:8 + struct.unpack('<Q', raw[:8])[0]])
    assert any(k.endswith('.trellis_pm') for k in head), f
    assert not any(k.endswith('.trellis') for k in head), f
    seen.add(head['__metadata__']['tile_order'])
print(','.join(sorted(seen)))
PY
)
[ "$order" = pair_major ] || { echo "FAIL: layer shards stamp tile_order '$order', want pair_major"; exit 1; }
step compare "$BIN" kld compare --model "$RUN/pack" --fixture "$RUN/teacher" --kv-quant 8 --json "$RUN/kld.json"
positions=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["positions"])' "$RUN/kld.json")
[ "$positions" = 128 ] || { echo "FAIL: compare scored $positions positions, want 128"; exit 1; }
echo "PASS: teacher, capture, encode, resume to the same bytes, drafter K4, repack, compare over $positions positions"

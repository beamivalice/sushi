#!/bin/bash
# sushi-test: tier=live boot=own pack=default boots=2
# Vision prefill chunking: an image-bearing prompt must prefill in chunks like
# text does, instead of one whole-prompt forward (issue #197 — the unchunked
# width is what the memory guard bills, so long conversations + one screenshot
# hit a 400 no flag can fix).
#
# Two arms, same binary: SUSHI_VISION_CHUNKED=0 (old whole-prompt forward)
# vs default-on, both at --prefill-chunk 32 so chunk boundaries land
# INSIDE the image's placeholder span (the image splices at prompt start; the
# prompt is padded past nextChunkEnd's TAIL_MERGE_MAX so chunking actually
# engages). Asserts: [1] the on arm logs "[vision] chunked prefill" and the
# off arm doesn't (silent-no-op class); [2] BOTH arms name all four quadrant
# colors — a splice that restarts its row index at a chunk boundary re-reads
# the first image rows in every chunk, so the bottom quadrants (green/yellow)
# vanish from the answer. Byte-equality across arms is deliberately NOT
# asserted: chunk width changes GEMM shapes and a 4-bit checkpoint flips
# near-tie argmaxes across widths (same class as MTP verify — legit).
#
# The cap leaves room for the thought: a chunk width moves the thought's length, and a reply the cap cut after
# "Red, blue" reads as lost rows (the thought itself names all four quadrants).
#
# Requires a vision model; SKIPs without one.

set -u
. "$(dirname "$0")/lib/boot.sh"

MODEL=$(sushi_pack qwen4) || exit $?
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# A deterministic quadrant PNG (red TL, blue TR, green BL, yellow BR) — the
# answer depends on reading the WHOLE image, so losing the later vision rows
# is visible in the reply. Filler pads the prompt past TAIL_MERGE_MAX.
python3 - "$WORK" <<'PYEOF'
import base64, json, struct, sys, zlib

work = sys.argv[1]
W = H = 224
rows = b""
for y in range(H):
    row = b"\x00"
    for x in range(W):
        if x < W // 2 and y < H // 2:   px = (220, 30, 30)    # red TL
        elif x >= W // 2 and y < H // 2: px = (30, 30, 220)   # blue TR
        elif x < W // 2:                 px = (30, 200, 30)   # green BL
        else:                            px = (230, 220, 40)  # yellow BR
        row += bytes(px)
    rows += row
def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
png = (b"\x89PNG\r\n\x1a\n"
       + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
       + chunk(b"IDAT", zlib.compress(rows))
       + chunk(b"IEND", b""))
b64 = base64.b64encode(png).decode()

filler = " ".join(f"item{i} shelf {i % 7}." for i in range(120))
body = {
    "model": "sushi",
    "temperature": 0.0,
    "max_tokens": 512,
    "messages": [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
            {"type": "text", "text": "Notes: " + filler + "\nIgnoring the notes: name the color of each quadrant of the image, top-left first."},
        ]},
    ],
}
with open(f"{work}/req.json", "w") as f:
    json.dump(body, f)
PYEOF

run_arm() {
    local arm="$1" env_val="$2"
    SUSHI_VISION_CHUNKED="$env_val" sushi_boot "vision-$arm" "$MODEL" --prefill-chunk 32 --prefix-cache-entries 0 --log-level debug || exit $?
    sushi_require vision
    curl -sf -m 300 "$SUSHI_URL/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        --data-binary @"$WORK/req.json" > "$WORK/$arm.resp.json" \
        || { echo "[fail] request failed ($arm arm)"; exit 1; }
    cp "$SUSHI_LOG" "$WORK/$arm.log"
    sushi_stop 0
}

run_arm off 0
run_arm on 1

OFF_TXT=$(python3 -c "import json; print(json.load(open('$WORK/off.resp.json'))['choices'][0]['message']['content'])")
ON_TXT=$(python3 -c "import json; print(json.load(open('$WORK/on.resp.json'))['choices'][0]['message']['content'])")

for arm in off on; do
    FIN=$(python3 -c "import json; print(json.load(open('$WORK/$arm.resp.json'))['choices'][0]['finish_reason'])")
    if [ "$FIN" = length ]; then
        echo "[fail] $arm arm: the reply was cut by max_tokens, so its colors say nothing about the splice"
        exit 1
    fi
done

echo "off: $OFF_TXT"
echo "on:  $ON_TXT"

if ! grep -q '\[vision\] chunked prefill' "$WORK/on.log"; then
    echo "[fail] on arm never logged [vision] chunked prefill — vision still runs one whole-prompt forward"
    exit 1
fi
if grep -q '\[vision\] chunked prefill' "$WORK/off.log"; then
    echo "[fail] off arm logged [vision] chunked prefill — kill switch is a no-op"
    exit 1
fi
# Perceived-color set per arm. Checkpoint-agnostic bar: a model may call the
# dark-yellow quadrant "brown"/"orange" (gemma-4-qat does), but BOTH arms see
# the same image, so their color sets must match — the class bug (splice rows
# restarting at a chunk boundary) collapses the bottom quadrants onto the top
# colors, shrinking the on arm's set. The off arm must still read the top row
# (red + blue) and >= 3 distinct colors, or the model isn't reading the image
# at all and the set comparison proves nothing.
color_set() {
    echo "$1" | tr '[:upper:]' '[:lower:]' \
        | grep -oE 'red|blue|green|yellow|brown|orange|purple|gray|grey|white|black|cyan|magenta|gold|teal|pink' \
        | sort -u | tr '\n' ' '
}
OFF_SET=$(color_set "$OFF_TXT")
ON_SET=$(color_set "$ON_TXT")
echo "off colors: $OFF_SET"
echo "on colors:  $ON_SET"
case "$OFF_SET" in
    *red*) : ;;
    *) echo "[fail] off arm never names red — model is not reading the image"; exit 1 ;;
esac
case "$OFF_SET" in
    *blue*) : ;;
    *) echo "[fail] off arm never names blue — model is not reading the image"; exit 1 ;;
esac
if [ "$(echo "$OFF_SET" | wc -w)" -lt 3 ]; then
    echo "[fail] off arm names fewer than 3 colors — model is not reading the image"
    exit 1
fi
if [ "$ON_SET" != "$OFF_SET" ]; then
    echo "[fail] chunked arm perceives different colors than the unchunked arm — vision rows lost"
    exit 1
fi
echo "[pass] vision prefill chunks and the splice is chunk-exact across boundaries"

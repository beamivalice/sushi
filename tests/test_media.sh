#!/bin/bash
# sushi-test: tier=live boot=api
# Image and video input over HTTP, on whatever vision arch the pass booted. The towers' numerics are pinned by the
# per-arch fixture tests; this checks what only a running server shows:
#   [1] /v1/models advertises images, and video exactly where the arch serves it;
#   [2] the answer reads what only the pixels hold (the four quadrant colors) on chat completions, and the default
#       speculative mode (MTP, DFlash2 or DSpark) engages on that image request;
#   [3] a stream and a non-stream answer are the same bytes;
#   [4] a second turn recalls the image;
#   [5] /v1/messages reads the same image as a base64 block and as a data-URL source;
#   [6] a video_url of solid-color frames (only where video is served) names more than one color.
# What the model says is a checkpoint expectation: every color check accepts any answer that names what is there.
#   tests/run.sh --tier live --model <qwen4|glm5|dsv41|mimo2> --only test_media.sh
set -u
. "$(dirname "$0")/lib/boot.sh"
sushi_attach
sushi_require vision
BASE=$SUSHI_URL
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
pass=0; fail=0
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail+1)); fi; }
one() { sed 's/^[1-9][0-9]*$/1/'; }
hits() { grep -ciE "$2" <<< "$1" | one; }
colors() { local n=0 c; for c in red blue green yellow; do n=$((n + $(hits "$1" "$c"))); done; echo "$n"; }

# GLM's template has no thinking-off mode, so it thinks at low effort and needs room to answer.
case "$SUSHI_ARCH" in
    glm5_next)
        CHAT_EXTRA='{"reasoning_effort":"low","max_tokens":1024}'
        MSG_EXTRA='{"output_config":{"effort":"low"},"max_tokens":1024}' ;;
    *)
        CHAT_EXTRA='{"enable_thinking":false,"max_tokens":64}'
        MSG_EXTRA='{"thinking":{"type":"disabled"},"max_tokens":64}' ;;
esac
export SUSHI_NONCE CHAT_EXTRA MSG_EXTRA

cat > "$WORK/m.py" <<'PY'
import base64, json, os, struct, sys, zlib

nonce = os.environ["SUSHI_NONCE"]
QUESTION = nonce + " Name the color of each of the four quadrants of this image, top-left first. Colors only."
FOLLOWUP = nonce + " Which color was the top-left quadrant of the image? One word."
VIDEO_Q = nonce + " This video shows a sequence of solid colors. Name them in order."

def png(px, w, h):
    rows = b"".join(b"\x00" + b"".join(bytes(px(x, y)) for x in range(w)) for y in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))

def quadrants():
    # red TL, blue TR, green BL, yellow BR
    def px(x, y):
        if y < 128:
            return (220, 30, 30) if x < 128 else (30, 30, 220)
        return (30, 200, 30) if x < 128 else (230, 220, 40)
    return base64.b64encode(png(px, 256, 256)).decode()

def solid(rgb):
    return "data:image/png;base64," + base64.b64encode(png(lambda x, y: rgb, 224, 224)).decode()

def chat(stream, previous):
    url = "data:image/png;base64," + quadrants()
    msgs = [{"role": "user", "content": [{"type": "text", "text": QUESTION}, {"type": "image_url", "image_url": {"url": url}}]}]
    if previous:
        msgs += [{"role": "assistant", "content": previous}, {"role": "user", "content": FOLLOWUP}]
    req = {"model": "sushi", "temperature": 0, "stream": stream, "messages": msgs}
    req.update(json.loads(os.environ["CHAT_EXTRA"]))
    return req

def messages(source):
    if source == "url":
        src = {"type": "url", "url": "data:image/png;base64," + quadrants()}
    else:
        src = {"type": "base64", "media_type": "image/png", "data": quadrants()}
    req = {"model": "sushi", "temperature": 0, "messages": [{"role": "user", "content": [
        {"type": "text", "text": QUESTION}, {"type": "image", "source": src}]}]}
    req.update(json.loads(os.environ["MSG_EXTRA"]))
    return req

def video():
    cols = [(220, 30, 30)] * 2 + [(30, 200, 30)] * 2 + [(30, 30, 220)] * 2 + [(230, 220, 40)] * 2
    req = {"model": "sushi", "temperature": 0, "messages": [{"role": "user", "content": [
        {"type": "text", "text": VIDEO_Q},
        {"type": "video_url", "video_url": {"frames": [solid(c) for c in cols], "fps": 2}}]}]}
    req.update(json.loads(os.environ["CHAT_EXTRA"]))
    return req

def content(raw):
    return json.loads(raw)["choices"][0]["message"].get("content") or ""

def text_blocks(raw):
    return "".join(b.get("text", "") for b in json.loads(raw).get("content", []) if b.get("type") == "text")

def sse_content(raw):
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            for c in json.loads(line[6:]).get("choices", []):
                out.append(c.get("delta", {}).get("content") or "")
    return "".join(out)

def models(raw):
    d = json.loads(raw)["data"][0]
    print(("vision" in d.get("capabilities", [])) * 1, ("image" in d.get("input_modalities", [])) * 1,
          ("video" in d.get("input_modalities", [])) * 1)

cmd = sys.argv[1]
if cmd == "chat":
    print(json.dumps(chat(sys.argv[2] == "true", sys.argv[3] if len(sys.argv) > 3 else "")))
elif cmd == "messages":
    print(json.dumps(messages(sys.argv[2])))
elif cmd == "video":
    print(json.dumps(video()))
elif cmd == "models":
    models(sys.stdin.read())
else:
    print({"content": content, "text_blocks": text_blocks, "sse": sse_content}[cmd](sys.stdin.read()), end="")
PY
py() { python3 -I "$WORK/m.py" "$@"; }
post() { curl -s -m 1800 -o "$WORK/out" -w '%{http_code}' "$BASE$1" -H 'content-type: application/json' -d @-; }
spec_lines() { sushi_log_since | grep -cE '\[spec-stats\] mode=(mtp|dspark|dflash)' | one; }

echo "[1] images (and video where served) are advertised"
read -r cap_vision mod_image mod_video <<< "$(curl -s "$BASE/v1/models" | py models)"
check "capabilities name vision" "$cap_vision" "1"
check "input modalities name image" "$mod_image" "1"
# Video is served by the Qwen and GLM towers only (what sushi_require video asks).
if (sushi_require video > /dev/null); then want_video=1; else want_video=0; fi
check "video modality advertised exactly where served" "$mod_video" "$want_video"

echo "[2] chat completions reads the quadrant colors"
code=$(py chat false | post /v1/chat/completions)
check "image request answers 200" "$code" "200"
plain=$(py content < "$WORK/out"); echo "  $(echo "$plain" | tr '\n' ' ' | cut -c1-120)"
check "at least three quadrant colors named" "$([ "$(colors "$plain")" -ge 3 ] && echo 1 || echo 0)" "1"
if (sushi_require spec > /dev/null); then
    check "the default speculative mode engaged on the image request" "$([ "$(spec_lines)" -ge 1 ] && echo 1 || echo 0)" "1"
else
    echo "  skip speculative-mode row: the boot has no speculation"
fi
# M-RoPE is Qwen3-VL's position scheme; the other towers log nothing like it.
if [ "$SUSHI_ARCH" = qwen4_exp ]; then
    check "M-RoPE engaged on the image" "$(sushi_log_since | grep -c 'M-RoPE: 1 images' | one)" "1"
fi

echo "[3] stream == non-stream"
streamed=$(py chat true | curl -sN -m 1800 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @- | py sse)
check "same bytes" "$([ "$plain" = "$streamed" ] && echo 1 || echo 0)" "1"

echo "[4] a second turn recalls the image"
code=$(py chat false "$plain" | post /v1/chat/completions)
check "second turn answers 200" "$code" "200"
turn2=$(py content < "$WORK/out"); echo "  $(echo "$turn2" | tr '\n' ' ' | cut -c1-120)"
check "second turn names the top-left color" "$(hits "$turn2" 'red')" "1"

echo "[5] /v1/messages reads the image"
code=$(py messages base64 | post /v1/messages)
check "base64 image block answers 200" "$code" "200"
ra=$(py text_blocks < "$WORK/out"); echo "  $(echo "$ra" | tr '\n' ' ' | cut -c1-120)"
check "base64 image block: at least three colors" "$([ "$(colors "$ra")" -ge 3 ] && echo 1 || echo 0)" "1"
code=$(py messages url | post /v1/messages)
check "data-URL image source answers 200" "$code" "200"
ru=$(py text_blocks < "$WORK/out")
check "data-URL image source: at least three colors" "$([ "$(colors "$ru")" -ge 3 ] && echo 1 || echo 0)" "1"

echo "[6] video_url"
if [ "$want_video" = 1 ]; then
    code=$(py video | post /v1/chat/completions)
    check "video request answers 200" "$code" "200"
    rv=$(py content < "$WORK/out"); echo "  $(echo "$rv" | tr '\n' ' ' | cut -c1-120)"
    # One color is what a pipeline that threads a single frame into every temporal patch would say.
    check "at least two frame colors named" "$([ "$(colors "$rv")" -ge 2 ] && echo 1 || echo 0)" "1"
    check "the frames were decoded" "$(sushi_log_since | grep -c 'Decoded 8 frames' | one)" "1"
else
    echo "  skip: the boot does not serve video"
fi

check "server still healthy" "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health")" "200"
echo "passed $pass, failed $fail"
[ "$fail" -eq 0 ]

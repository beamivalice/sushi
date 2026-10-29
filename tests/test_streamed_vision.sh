#!/bin/bash
# sushi-test: tier=live boot=own pack=glm5 boots=2
# Vision under SSD expert streaming, live on the GLM pack (GLM_STREAM_VISION_BUDGET GiB of experts, default 32):
#   [1] a streamed boot without --vision names what the tower would cost, answers text and refuses an image with a
#       400 naming --vision;
#   [2] a streamed --vision boot bills the tower in the ssd budget (ledger, /props, the preflight weights rise by the
#       tower), advertises images, names the quadrant colors, answers stream == non-stream, answers text, recalls the
#       image across a second turn, reads a video_url of four solid colors, then unloads the model and cold-loads it
#       through /v1/load-model, which must bill the tower again and answer the same bytes.
# Each boot takes the GPU lock and runs under a private HOME (no saved model settings). Speculation and the prefix
# cache are off in both boots (--no-mtp --no-drafter --prefix-cache-entries 0), so the greedy bytes compare one cold
# forward. STREAM_VISION_BODY='{json}' merges into each request.
set -uo pipefail
. "$(dirname "$0")/lib/boot.sh"
MODEL=$(sushi_pack glm5) || exit $?
BUDGET="${GLM_STREAM_VISION_BUDGET:-32}"
EXTRA="${STREAM_VISION_BODY:-}"
[ -n "$EXTRA" ] || EXTRA='{"reasoning_effort":"low","max_tokens":1024}'
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
pass=0; fail=0
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail+1)); fi; }
COMMON=(--log-level debug --no-mtp --no-drafter --prefix-cache-entries 0 --ctx-size 16384)
boot_failed() { echo "  FAIL boot ($1)"; fail=$((fail+1)); }
one() { sed 's/^[1-9][0-9]*$/1/'; }

# Four flat quadrants: red TL, blue TR, green BL, yellow BR.
python3 - "$WORK/quadrants.png" <<'PY'
import struct, sys, zlib
W = H = 256
def px(x, y):
    if y < H // 2:
        return (220, 30, 30) if x < W // 2 else (30, 30, 220)
    return (30, 200, 30) if x < W // 2 else (230, 220, 40)
rows = b"".join(b"\x00" + b"".join(bytes(px(x, y)) for x in range(W)) for y in range(H))
def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
open(sys.argv[1], "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
                              + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
PY
QUESTION="Name the color of each of the four quadrants of this image, top-left first. Colors only."

body() { # $1 stream true|false, [$2 previous answer: adds a follow-up turn]
  python3 - "$WORK/quadrants.png" "$QUESTION" "$1" "$EXTRA" "${2:-}" <<'PY'
import base64, json, sys
path, question, stream, extra, previous = sys.argv[1], sys.argv[2], sys.argv[3] == "true", json.loads(sys.argv[4]), sys.argv[5]
url = "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()
messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": question}]}]
if previous:
    messages += [{"role": "assistant", "content": previous.split("\x1e", 1)[-1]},
                 {"role": "user", "content": "Which color was the top-left quadrant of the image? One word."}]
req = {"model": "sushi", "temperature": 0, "stream": stream, "messages": messages}
req.update(extra)
print(json.dumps(req))
PY
}
text_body() {
  python3 - "$EXTRA" <<'PY'
import json, sys
req = {"model": "sushi", "temperature": 0, "messages": [{"role": "user", "content": "What is 2+3? Answer with the number only."}]}
req.update(json.loads(sys.argv[1]))
print(json.dumps(req))
PY
}
video_body() { # eight solid frames at 2 fps: red, green, blue, yellow, two frames each
  python3 - "$EXTRA" <<'PY'
import base64, json, struct, sys, zlib
def png(rgb):
    W = H = 224
    rows = b"".join(b"\x00" + bytes(rgb) * W for _ in range(H))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(data).decode()
frames = [png(c) for c in [(220, 30, 30)] * 2 + [(30, 200, 30)] * 2 + [(30, 30, 220)] * 2 + [(230, 220, 40)] * 2]
req = {"model": "sushi", "temperature": 0, "stream": False,
       "messages": [{"role": "user", "content": [{"type": "video_url", "video_url": {"frames": frames, "fps": 2}},
                                                 {"type": "text", "text": "This video shows a sequence of solid colors. Name them in order."}]}]}
req.update(json.loads(sys.argv[1]))
print(json.dumps(req))
PY
}
post() { curl -s -m 1800 "$SUSHI_URL/v1/chat/completions" -H 'content-type: application/json' -d @-; }
content() { python3 -c "import sys,json; d=json.load(sys.stdin); m=d['choices'][0]['message']; print((m.get('reasoning_content') or '') + '\x1e' + (m.get('content') or ''), end='')"; }
stream_content() { python3 -c "
import sys, json
r, c = [], []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]':
        continue
    for ch in json.loads(line[6:]).get('choices', []):
        d = ch.get('delta', {})
        r.append(d.get('reasoning_content') or ''); c.append(d.get('content') or '')
print(''.join(r) + '\x1e' + ''.join(c), end='')"; }
answer() { printf '%s' "${1#*$'\x1e'}"; }
hits() { grep -ciE "$2" <<< "$1" | one; }
colors() { local n=0; for c in red blue green yellow; do n=$((n + $(hits "$(answer "$1")" "$c"))); done; echo "$n"; }
refusal() { # prints the HTTP status, keeps the body in $WORK/refusal.json
  body false | curl -s -m 600 -o "$WORK/refusal.json" -w '%{http_code}' "$SUSHI_URL/v1/chat/completions" -H 'content-type: application/json' -d @-
}
# The preflight's weights figure (GiB), from the Nth load in a log (default the last).
weights() { grep -o '\[preflight\] weights ~[0-9.]*' "$1" | sed 's/.*~//' | sed -n "${2:-\$}p"; }
tower_gib() { python3 -c "import sys,json; print(json.loads(sys.argv[1])['settings']['vision']['streamed_tower_bytes'] / 2**30)" "$1"; }
rose_by() { python3 -c "import sys; print(int(abs(float(sys.argv[2]) - float(sys.argv[1]) - float(sys.argv[3])) < 0.02))" "$1" "$2" "$3"; }


echo "[1] streamed, no --vision"
sushi_boot streamvis-off "$MODEL" "${COMMON[@]}" --ssd-budget-gb "$BUDGET" || exit $?
LOG1="$SUSHI_LOG"
grep -E '\[vision\]|\[expert-stream\] ssd budget' "$LOG1" | sed 's/^/  /'
check "off line names --vision and its cost" "$(grep -c '\[vision\] off under expert streaming; pass --vision' "$LOG1" | one)" "1"
check "/v1/models: no image modality" "$(hits "$(curl -s "$SUSHI_URL/v1/models")" '"image"')" "0"
check "image request is a 400" "$(refusal)" "400"
check "the 400 names --vision" "$(hits "$(cat "$WORK/refusal.json")" 'relaunch with --vision')" "1"
t=$(text_body | post | content)
check "text answers" "$(hits "$(answer "$t")" '5|five')" "1"
W_OFF=$(weights "$LOG1" 2>/dev/null)
sushi_stop 0

echo "[2] streamed, --vision"
sushi_boot streamvis-on "$MODEL" "${COMMON[@]}" --ssd-budget-gb "$BUDGET" --vision || exit $?
LOG2="$SUSHI_LOG"
U="$SUSHI_URL"
grep -E '\[vision\]|\[expert-stream\] ssd budget|\[preflight\] weights' "$LOG2" | sed 's/^/  /'
check "on line (--vision)" "$(grep -c '\[vision\] on (--vision): tower' "$LOG2" | one)" "1"
check "ledger bills a vision term" "$(grep -E '\[expert-stream\] ssd budget' "$LOG2" | grep -cvE 'vision 0\.00 GB' | one)" "1"
props=$(curl -s "$U/props")
check "/props settings.vision loaded" "$(python3 -c "import sys,json; v=json.loads(sys.argv[1])['settings']['vision']; print(int(v['loaded'] and v['source']=='--vision' and v['streamed_tower_bytes']>0 and v['streamed_encode_bytes']>0))" "$props")" "1"
TOWER=$(tower_gib "$props")
W_ON=$(weights "$LOG2")
[ -n "$W_OFF" ] && check "preflight weights rose by the tower ($W_OFF -> $W_ON GiB)" "$(rose_by "$W_OFF" "$W_ON" "$TOWER")" "1"
check "/v1/models: image modality" "$(hits "$(curl -s "$U/v1/models")" '"image"')" "1"
plain=$(body false | post | content); echo "  ${plain//$'\x1e'/ | }"
check "at least three quadrant colors named" "$([ "$(colors "$plain")" -ge 3 ] && echo 1 || echo 0)" "1"
streamed=$(body true | curl -sN -m 1800 "$U/v1/chat/completions" -H 'content-type: application/json' -d @- | stream_content)
check "stream == non-stream bytes" "$([ "$plain" = "$streamed" ] && echo 1 || echo 0)" "1"
t=$(text_body | post | content)
check "text answers" "$(hits "$(answer "$t")" '5|five')" "1"
h=$(body false "$plain" | post | content); echo "  turn 2: $(answer "$h")"
check "turn 2 recalls the image's top-left color" "$(hits "$(answer "$h")" 'red')" "1"
v=$(video_body | post | content); echo "  video: $(answer "$v")"
check "video: at least three frame colors named" "$([ "$(colors "$v")" -ge 3 ] && echo 1 || echo 0)" "1"
ID=$(curl -s "$U/v1/models" | python3 -c "import sys,json; print([m['id'] for m in json.load(sys.stdin)['data'] if m.get('loaded')][0])")
curl -s -m 600 -o /dev/null "$U/v1/unload-model" -H 'content-type: application/json' -d "{\"model\":\"$ID\"}"
check "unloaded" "$(curl -s "$U/v1/models" | python3 -c "import sys,json; print(int(not any(m.get('loaded') for m in json.load(sys.stdin)['data'])))")" "1"
check "reload: /v1/load-model 200" "$(curl -s -m 1800 -o /dev/null -w '%{http_code}' "$U/v1/load-model" -H 'content-type: application/json' -d "{\"model\":\"$ID\"}")" "200"
check "reload: a second [vision] on line" "$(grep -c '\[vision\] on (--vision): tower' "$LOG2")" "2"
check "reload: preflight weights equal the boot's" "$(weights "$LOG2" 2)" "$W_ON"
r=$(body false | post | content)
check "reload: image answered with the boot's bytes" "$([ "$r" = "$plain" ] && echo 1 || echo 0)" "1"
sushi_stop 0

echo "pass=$pass fail=$fail"
[ "$fail" = 0 ]

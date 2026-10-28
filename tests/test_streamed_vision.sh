#!/usr/bin/env bash
. "$(dirname "$0")/private_cache.sh"
# Vision under SSD expert streaming, live on one pack (GLM-5.3, MiMo or Qwen with a tower):
#   [1] a streamed boot without --vision names what the tower would cost, answers text and refuses an
#       image with a 400 naming --vision (SKIP_OFF=1 reuses this port's earlier [1] log);
#   [1b] NOVISION=1: a streamed --no-vision boot refuses an image naming the tower, not --vision;
#   [2] a streamed --vision boot bills the tower in the ssd budget (ledger, /props, the preflight
#       weights rise by the tower), advertises images, names the quadrant colors, answers
#       stream == non-stream, answers text, and recalls the image across a second turn;
#       VIDEO=1 also sends a video_url of four solid colors (GLM); RELOAD=1 then unloads the model and
#       cold-loads it through /v1/load-model, which must bill the tower again and answer the same bytes;
#   [3] RESIDENT=1: a resident boot answers the same image (and video) requests with the same bytes.
# Each boot takes the GPU lock, releases it when it stops, and runs under a private HOME (no saved
# model settings). Speculation and the prefix cache are off in every arm (--no-mtp --no-drafter
# --prefix-cache-entries 0), so the greedy bytes compare one cold forward.
#   STREAM_VISION_MODEL=<pack> STREAM_VISION_BUDGET=<GiB> [STREAM_VISION_BODY='{json}'] \
#     [SKIP_OFF=1] [NOVISION=1] [VIDEO=1] [RELOAD=1] [RESIDENT=1] bash tests/test_streamed_vision.sh [port]
# STREAM_VISION_BODY merges into each request (GLM: '{"reasoning_effort":"low","max_tokens":1024}').
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL="${STREAM_VISION_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/GLM-5.3-Flash-Sushi-2.4bpw}"
BUDGET="${STREAM_VISION_BUDGET:-32}"
EXTRA="${STREAM_VISION_BODY:-}"
[ -n "$EXTRA" ] || EXTRA='{"enable_thinking":false,"max_tokens":64}'
PORT="${1:-11447}"
BIN="${SUSHI_BIN:-$ROOT/zig-out/bin/sushi}"
RUNS="$HOME/.sushi/runs/streamed-vision"
OWNER="streamed-vision-$$"
mkdir -p "$RUNS"
[ -f "$MODEL/config.json" ] || { echo "SKIP: no model at $MODEL"; exit 0; }
[ -x "$BIN" ] || { echo "FAIL: $BIN missing (zig build -Doptimize=ReleaseFast)"; exit 1; }
WORK="$(mktemp -d)"
SPID=""
LOCKED=0
pass=0; fail=0
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail+1)); fi; }
stop_server() {
  [ -n "$SPID" ] && kill "$SPID" 2>/dev/null && wait "$SPID" 2>/dev/null; SPID=""
  [ "$LOCKED" = 1 ] && "$ROOT/scripts/gpu-lock.sh" release "$OWNER" >/dev/null 2>&1; LOCKED=0
}
trap 'stop_server; rm -rf "$WORK"' EXIT
U="http://127.0.0.1:$PORT"
COMMON=(--serve --host 127.0.0.1 --port "$PORT" --log-level debug --no-mtp --no-drafter --prefix-cache-entries 0 --ctx-size 16384)

boot() { # $1 log, rest: flags
  local log="$1"; shift
  "$ROOT/scripts/gpu-lock.sh" acquire "$OWNER" >/dev/null 2>&1; LOCKED=1
  mkdir -p "$WORK/home"
  HOME="$WORK/home" "$BIN" "${COMMON[@]}" --model "$MODEL" "$@" > "$log" 2>&1 &
  SPID=$!
  for _ in $(seq 1 900); do
    curl -s "$U/health" >/dev/null 2>&1 && grep -q "ready" "$log" && return 0
    kill -0 "$SPID" 2>/dev/null || { echo "server died"; tail -20 "$log"; SPID=""; return 1; }
    sleep 2
  done
  echo "server never became ready"; return 1
}
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
post() { curl -s -m 1800 "$U/v1/chat/completions" -H 'content-type: application/json' -d @-; }
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
  body false | curl -s -m 600 -o "$WORK/refusal.json" -w '%{http_code}' "$U/v1/chat/completions" -H 'content-type: application/json' -d @-
}
# The preflight's weights figure (GiB), from the Nth load in a log (default the last).
weights() { grep -o '\[preflight\] weights ~[0-9.]*' "$1" | sed 's/.*~//' | sed -n "${2:-\$}p"; }
tower_gib() { python3 -c "import sys,json; print(json.loads(sys.argv[1])['settings']['vision']['streamed_tower_bytes'] / 2**30)" "$1"; }
rose_by() { python3 -c "import sys; print(int(abs(float(sys.argv[2]) - float(sys.argv[1]) - float(sys.argv[3])) < 0.02))" "$1" "$2" "$3"; }

LOG1="$RUNS/off-$PORT.log"
if [ "${SKIP_OFF:-0}" = 1 ]; then echo "[1] skipped: weights from $LOG1"
else
echo "[1] streamed, no --vision"
if ! boot "$LOG1" --ssd-budget-gb "$BUDGET"; then boot_failed streamed; else
  grep -E '\[vision\]|\[expert-stream\] ssd budget' "$LOG1" | sed 's/^/  /'
  check "off line names --vision and its cost" "$(grep -c '\[vision\] off under expert streaming; pass --vision' "$LOG1" | one)" "1"
  check "/v1/models: no image modality" "$(hits "$(curl -s "$U/v1/models")" '"image"')" "0"
  check "image request is a 400" "$(refusal)" "400"
  check "the 400 names --vision" "$(hits "$(cat "$WORK/refusal.json")" 'relaunch with --vision')" "1"
  t=$(text_body | post | content)
  check "text answers" "$(hits "$(answer "$t")" '5|five')" "1"
fi
stop_server
fi
W_OFF=$(weights "$LOG1" 2>/dev/null)

if [ "${NOVISION:-0}" = 1 ]; then
  echo "[1b] streamed, --no-vision"
  LOGN="$RUNS/novision-$PORT.log"
  if ! boot "$LOGN" --ssd-budget-gb "$BUDGET" --no-vision; then boot_failed "streamed --no-vision"; else
    check "off line names --no-vision" "$(grep -c '\[vision\] off (--no-vision)' "$LOGN" | one)" "1"
    check "image request is a 400" "$(refusal)" "400"
    r=$(cat "$WORK/refusal.json")
    check "the 400 names the tower" "$(hits "$r" 'without its vision tower')" "1"
    check "the 400 does not ask for --vision" "$(hits "$r" 'relaunch with --vision')" "0"
  fi
  stop_server
fi

echo "[2] streamed, --vision"
LOG2="$RUNS/on-$PORT.log"
if ! boot "$LOG2" --ssd-budget-gb "$BUDGET" --vision; then boot_failed "streamed --vision"; else
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
  printf '%s' "$plain" > "$RUNS/on-$PORT.answer"
  check "at least three quadrant colors named" "$([ "$(colors "$plain")" -ge 3 ] && echo 1 || echo 0)" "1"
  streamed=$(body true | curl -sN -m 1800 "$U/v1/chat/completions" -H 'content-type: application/json' -d @- | stream_content)
  check "stream == non-stream bytes" "$([ "$plain" = "$streamed" ] && echo 1 || echo 0)" "1"
  t=$(text_body | post | content)
  check "text answers" "$(hits "$(answer "$t")" '5|five')" "1"
  h=$(body false "$plain" | post | content); echo "  turn 2: $(answer "$h")"
  check "turn 2 recalls the image's top-left color" "$(hits "$(answer "$h")" 'red')" "1"
  if [ "${VIDEO:-0}" = 1 ]; then
    v=$(video_body | post | content); echo "  video: $(answer "$v")"
    printf '%s' "$v" > "$RUNS/on-$PORT.video"
    check "video: at least three frame colors named" "$([ "$(colors "$v")" -ge 3 ] && echo 1 || echo 0)" "1"
  fi
  if [ "${RELOAD:-0}" = 1 ]; then
    ID=$(curl -s "$U/v1/models" | python3 -c "import sys,json; print([m['id'] for m in json.load(sys.stdin)['data'] if m.get('loaded')][0])")
    curl -s -m 600 -o /dev/null "$U/v1/unload-model" -H 'content-type: application/json' -d "{\"model\":\"$ID\"}"
    check "unloaded" "$(curl -s "$U/v1/models" | python3 -c "import sys,json; print(int(not any(m.get('loaded') for m in json.load(sys.stdin)['data'])))")" "1"
    check "reload: /v1/load-model 200" "$(curl -s -m 1800 -o /dev/null -w '%{http_code}' "$U/v1/load-model" -H 'content-type: application/json' -d "{\"model\":\"$ID\"}")" "200"
    check "reload: a second [vision] on line" "$(grep -c '\[vision\] on (--vision): tower' "$LOG2")" "2"
    check "reload: preflight weights equal the boot's" "$(weights "$LOG2" 2)" "$W_ON"
    r=$(body false | post | content)
    check "reload: image answered with the boot's bytes" "$([ "$r" = "$plain" ] && echo 1 || echo 0)" "1"
  fi
fi
stop_server

if [ "${RESIDENT:-0}" = 1 ]; then
  echo "[3] resident: the same requests, the same bytes"
  LOG3="$RUNS/resident-$PORT.log"
  if ! boot "$LOG3"; then boot_failed resident; else
    resident=$(body false | post | content); echo "  ${resident//$'\x1e'/ | }"
    check "image: streamed == resident bytes" "$([ "$resident" = "$(cat "$RUNS/on-$PORT.answer" 2>/dev/null)" ] && echo 1 || echo 0)" "1"
    if [ "${VIDEO:-0}" = 1 ]; then
      rv=$(video_body | post | content)
      check "video: streamed == resident bytes" "$([ "$rv" = "$(cat "$RUNS/on-$PORT.video" 2>/dev/null)" ] && echo 1 || echo 0)" "1"
    fi
  fi
  stop_server
fi

echo "pass=$pass fail=$fail (logs: $RUNS)"
[ "$fail" = 0 ]

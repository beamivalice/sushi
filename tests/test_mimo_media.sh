#!/bin/bash
. "$(dirname "$0")/private_cache.sh"
# MiMo-V2.6-Flash image input, live on the served pack. The tower's numerics
# are pinned by the hermetic and real-weight tests in src/mimo_vision.zig; this
# checks what only a running server shows:
#   [1] the load line + /v1/models advertise images (and not video);
#   [2] the answer reads what only the pixels hold (quadrant colors, sign text),
#       on chat/completions and on Anthropic /v1/messages;
#   [3] the same image twice reuses the prefix cache, a different one only its chat header;
#   [4] a stream and a non-stream answer are the same bytes;
#   [5] a second boot with --no-vision drops the tower's bytes and refuses an
#       image by name.
# What the model says is a checkpoint expectation: the color and text checks
# accept any answer that names what is there.
#   MIMO_MEDIA_MODEL=<pack> ./tests/test_mimo_media.sh [port]
set -u
MODEL="${MIMO_MEDIA_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/MiMo-V2.6-Flash-Sushi-2.3bpw}"
PORT="${1:-11431}"
BIN="${SUSHI_BIN:-./zig-out/bin/sushi}"
RUNS="$HOME/.sushi/runs/mimo-media"
mkdir -p "$RUNS"
[ -f "$MODEL/config.json" ] || { echo "SKIP: no model at $MODEL"; exit 0; }
[ -x "$BIN" ] || { echo "FAIL: $BIN missing (zig build -Doptimize=ReleaseFast)"; exit 1; }
SIGNS="tests/fixtures/street-name-signs.jpg"
HOUSE="tests/fixtures/house.jpeg"
for f in "$SIGNS" "$HOUSE"; do [ -f "$f" ] || { echo "SKIP: fixture $f missing"; exit 0; }; done
WORK="$(mktemp -d)"
SPID=""
pass=0; fail=0
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail+1)); fi; }
stop_server() { [ -n "$SPID" ] && kill "$SPID" 2>/dev/null && wait "$SPID" 2>/dev/null; SPID=""; }
trap 'stop_server; rm -rf "$WORK"' EXIT
U="http://127.0.0.1:$PORT"

boot() { # $1 log, rest: extra flags
  local log="$1"; shift
  "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level debug --prefix-cache-entries 4 "$@" > "$log" 2>&1 &
  SPID=$!
  for _ in $(seq 1 900); do
    curl -s "$U/health" >/dev/null 2>&1 && grep -q "ready" "$log" && return 0
    kill -0 "$SPID" 2>/dev/null || { echo "server died"; tail -20 "$log"; exit 1; }
    sleep 2
  done
  echo "server never became ready"; exit 1
}

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

body() { # $1 image, $2 question, $3 stream true|false
  python3 - "$1" "$2" "$3" <<'PY'
import base64, json, sys
path, question, stream = sys.argv[1], sys.argv[2], sys.argv[3] == "true"
mime = "image/png" if path.endswith(".png") else "image/jpeg"
url = f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()
print(json.dumps({"model": "sushi", "max_tokens": 64, "temperature": 0, "enable_thinking": False, "stream": stream,
                  "messages": [{"role": "user", "content": [
                      {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": question}]}]}))
PY
}
anthropic_body() { # $1 image, $2 question
  python3 - "$1" "$2" <<'PY'
import base64, json, sys
path, question = sys.argv[1], sys.argv[2]
data = base64.b64encode(open(path, "rb").read()).decode()
print(json.dumps({"model": "sushi", "max_tokens": 64, "temperature": 0, "thinking": {"type": "disabled"},
                  "messages": [{"role": "user", "content": [
                      {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                      {"type": "text", "text": question}]}]}))
PY
}
ask() { curl -s -m 900 "$U/v1/chat/completions" -H 'content-type: application/json' -d @- | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['prompt_tokens_details']['cached_tokens'], '|', d['choices'][0]['message']['content'].replace(chr(10),' '))"; }
ask_raw() { curl -s -m 900 "$U/v1/chat/completions" -H 'content-type: application/json' -d @- | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'], end='')"; }
ask_stream() { curl -sN -m 900 "$U/v1/chat/completions" -H 'content-type: application/json' -d @- | python3 -c "
import sys, json
out = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]':
        continue
    for c in json.loads(line[6:]).get('choices', []):
        out.append(c.get('delta', {}).get('content') or '')
print(''.join(out), end='')"; }
ask_anthropic() { curl -s -m 900 "$U/v1/messages" -H 'content-type: application/json' -d @- | python3 -c "import sys,json; d=json.load(sys.stdin); print(''.join(x.get('text','') for x in d.get('content',[]) if x.get('type')=='text').replace(chr(10),' '))"; }
hits() { grep -ciE "$2" <<< "$1" | sed 's/^[1-9][0-9]*$/1/'; }
active_bytes() { curl -s "$U/props" | python3 -c "import sys,json; print(json.load(sys.stdin)['memory']['active_bytes'])"; }
# A different image may share only the chat header before its placeholder, never rows at or after it.
before_image() { python3 -c "print(1 if int('$1') <= 8 else 0)"; }

LOG="$RUNS/server-$PORT.log"
echo "[boot] $MODEL"
boot "$LOG"

echo "[1] the tower loads and /v1/models advertises images"
check "MiMo-ViT load line" "$(grep -c 'Vision encoder: MiMo-ViT' "$LOG" | sed 's/^[1-9][0-9]*$/1/')" "1"
models=$(curl -s "$U/v1/models")
check "capabilities name vision" "$(hits "$models" '"vision"')" "1"
check "no video modality" "$(hits "$models" '"video"')" "0"
with_tower=$(active_bytes)

echo "[2] answers read the pixels"
rq=$(body "$WORK/quadrants.png" "Name the color of each of the four quadrants, top-left first. Colors only." false | ask); echo "  $rq"
colors=0; for c in red blue green yellow; do colors=$((colors + $(hits "$rq" "$c"))); done
check "at least three quadrant colors named" "$([ "$colors" -ge 3 ] && echo 1 || echo 0)" "1"
rs=$(body "$SIGNS" "What text is written on the green street signs? Answer with the words only." false | ask); echo "  $rs"
check "street-sign text read" "$(hits "$rs" 'gr[ae]y fox|waterfall')" "1"
check "new image: nothing reused past the chat header" "$(before_image "${rs%% |*}")" "1"
ra=$(anthropic_body "$SIGNS" "What text is written on the green street signs? Answer with the words only." | ask_anthropic); echo "  $ra"
check "Anthropic image block read" "$(hits "$ra" 'gr[ae]y fox|waterfall')" "1"

echo "[3] prefix cache keyed on the pixels"
rs2=$(body "$SIGNS" "What text is written on the green street signs? Answer with the words only." false | ask); echo "  $rs2"
check "same image: cached_tokens > 0" "$(python3 -c "print(1 if int('${rs2%% |*}') > 0 else 0)")" "1"
# Whichever tier this boot retains: at the defaults RAM retention is off, so the reuse comes off
# the SSD tier, and asserting only the RAM line hid that an image turn reused nothing at all.
check "a tier reports the reuse" "$(grep -cE 'hot-cache\] reused|disk-cache\] restored' "$LOG" | sed 's/^[1-9][0-9]*$/1/')" "1"
check "warm answer still reads the signs" "$(hits "$rs2" 'gr[ae]y fox|waterfall')" "1"
rh=$(body "$HOUSE" "What text is written on the green street signs? Answer with the words only." false | ask); echo "  $rh"
check "different image: nothing reused past the chat header" "$(before_image "${rh%% |*}")" "1"

echo "[4] stream == non-stream"
q="Describe this picture in one sentence."
plain=$(body "$HOUSE" "$q" false | ask_raw)
streamed=$(body "$HOUSE" "$q" true | ask_stream)
echo "  non-stream: $plain"
check "same bytes" "$([ "$plain" = "$streamed" ] && echo 1 || echo 0)" "1"

stop_server
echo "[5] --no-vision"
LOG2="$RUNS/server-$PORT-novision.log"
boot "$LOG2" --no-vision
check "no MiMo-ViT load line" "$(grep -c 'Vision encoder: MiMo-ViT' "$LOG2")" "0"
check "capabilities drop vision" "$(hits "$(curl -s "$U/v1/models")" '"vision"')" "0"
without_tower=$(active_bytes)
echo "  active_bytes with tower $with_tower, without $without_tower"
check "tower bytes gone (>= 1.3 GB)" "$(python3 -c "print(1 if $with_tower - $without_tower >= 1.3e9 else 0)")" "1"
code=$(body "$SIGNS" "What does the sign say?" false | curl -s -o "$WORK/refusal.json" -w '%{http_code}' -m 300 "$U/v1/chat/completions" -H 'content-type: application/json' -d @-)
check "image refused with 400" "$code" "400"
check "refusal names the tower" "$(hits "$(cat "$WORK/refusal.json")" 'vision tower')" "1"

echo "passed $pass, failed $fail"
[ "$fail" -eq 0 ]

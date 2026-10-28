#!/bin/bash
. "$(dirname "$0")/private_cache.sh"
# DeepSeek-V4.1-Flash image input, live on the served pack (streamed: the tower loads by default). The tower's
# numerics are pinned by the fixture tests in src/dsv41_vision.zig; this checks what only a running
# server shows:
#   [1] the tower loads with no flag, bills the ssd budget's `vision` term and /v1/models advertises images;
#   [2] the answer reads what only the pixels hold (quadrant colors), on chat/completions and /v1/messages;
#   [3] the image request decodes with DSpark, never plain ticks (`[spec-stats] mode=dspark`);
#   [4] a stream and a non-stream answer are the same bytes;
#   [5] under --no-vision the tower is skipped, billed 0, and an image is a 400 naming --no-vision.
# What the model says is a checkpoint expectation: the color check accepts any answer that names what is there.
# Hold the GPU lock around the run: scripts/gpu-lock.sh acquire <owner>.
#   DSV41_MEDIA_MODEL=<pack> DSV41_MEDIA_BUDGET=<GiB> ./tests/test_dsv41_media.sh [port]
set -u
MODEL="${DSV41_MEDIA_MODEL:?set DSV41_MEDIA_MODEL to a deepseek_v41 pack}"
BUDGET="${DSV41_MEDIA_BUDGET:-96}"
PORT="${1:-11433}"
BIN="${SUSHI_BIN:-./zig-out/bin/sushi}"
RUNS="$HOME/.sushi/runs/dsv41-media"
mkdir -p "$RUNS"
[ -f "$MODEL/config.json" ] || { echo "SKIP: no model at $MODEL"; exit 0; }
[ -x "$BIN" ] || { echo "FAIL: $BIN missing (zig build -Doptimize=ReleaseFast)"; exit 1; }
WORK="$(mktemp -d)"
SPID=""
pass=0; fail=0
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail+1)); fi; }
stop_server() { [ -n "$SPID" ] && kill "$SPID" 2>/dev/null && wait "$SPID" 2>/dev/null; SPID=""; }
trap 'stop_server; rm -rf "$WORK"' EXIT
U="http://127.0.0.1:$PORT"

boot() { # $1 log, rest: extra flags
  local log="$1"; shift
  "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --log-level debug --ssd-budget-gb "$BUDGET" "$@" > "$log" 2>&1 &
  SPID=$!
  for _ in $(seq 1 1800); do
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

body() { # $1 question, $2 stream true|false
  python3 - "$WORK/quadrants.png" "$1" "$2" <<'PY'
import base64, json, sys
path, question, stream = sys.argv[1], sys.argv[2], sys.argv[3] == "true"
url = "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()
print(json.dumps({"model": "sushi", "max_tokens": 48, "temperature": 0, "enable_thinking": False, "stream": stream,
                  "messages": [{"role": "user", "content": [
                      {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": question}]}]}))
PY
}
anthropic_body() {
  python3 - "$WORK/quadrants.png" "$1" <<'PY'
import base64, json, sys
data = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
print(json.dumps({"model": "sushi", "max_tokens": 48, "temperature": 0, "thinking": {"type": "disabled"},
                  "messages": [{"role": "user", "content": [
                      {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
                      {"type": "text", "text": sys.argv[2]}]}]}))
PY
}
ask() { curl -s -m 900 "$U/v1/chat/completions" -H 'content-type: application/json' -d @- | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'], end='')"; }
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
colors_named() { local n=0; for c in red blue green yellow; do n=$((n + $(hits "$1" "$c"))); done; echo $n; }
Q="Name the color of each of the four quadrants, top-left first. Colors only."

LOG="$RUNS/server-$PORT.log"
echo "[boot] $MODEL --ssd-budget-gb $BUDGET"
boot "$LOG"

echo "[1] the tower loads, is billed and is advertised"
check "tower load line" "$(grep -c 'Vision encoder: DeepSeek-V4.1 ViT' "$LOG" | sed 's/^[1-9][0-9]*$/1/')" "1"
check "[vision] on" "$(grep -c '^\[vision\] on' "$LOG" | sed 's/^[1-9][0-9]*$/1/')" "1"
ledger=$(grep -m1 'ssd budget .* GiB: trunk' "$LOG"); echo "  $ledger"
check "the budget's vision term is the tower, not 0.00" "$(hits "$ledger" 'vision 0\.5[0-9] GB')" "1"
check "capabilities name vision" "$(hits "$(curl -s "$U/v1/models")" '"vision"')" "1"

echo "[2] answers read the pixels"
rq=$(body "$Q" false | ask); echo "  $rq"
check "at least three quadrant colors named" "$([ "$(colors_named "$rq")" -ge 3 ] && echo 1 || echo 0)" "1"
ra=$(anthropic_body "$Q" | ask_anthropic); echo "  $ra"
check "Anthropic image block: at least three colors" "$([ "$(colors_named "$ra")" -ge 3 ] && echo 1 || echo 0)" "1"

echo "[3] DSpark serves the image request"
spec=$(grep '\[spec-stats\]' "$LOG" | tail -1); echo "  $spec"
check "mode=dspark" "$(hits "$spec" 'mode=dspark')" "1"

echo "[4] stream == non-stream"
plain=$(body "$Q" false | ask)
streamed=$(body "$Q" true | ask_stream)
check "same bytes" "$([ "$plain" = "$streamed" ] && echo 1 || echo 0)" "1"

stop_server
echo "[5] --no-vision"
LOG2="$RUNS/server-$PORT-novision.log"
boot "$LOG2" --no-vision
check "no tower load line" "$(grep -c 'Vision encoder: DeepSeek-V4.1 ViT' "$LOG2")" "0"
check "[vision] off (--no-vision)" "$(grep -c '^\[vision\] off (--no-vision)' "$LOG2" | sed 's/^[1-9][0-9]*$/1/')" "1"
check "the budget's vision term is 0.00" "$(hits "$(grep -m1 'ssd budget .* GiB: trunk' "$LOG2")" 'vision 0\.00 GB')" "1"
code=$(body "$Q" false | curl -s -o "$WORK/refusal.json" -w '%{http_code}' -m 300 "$U/v1/chat/completions" -H 'content-type: application/json' -d @-)
check "image refused with 400" "$code" "400"
check "refusal names --no-vision" "$(hits "$(cat "$WORK/refusal.json")" '\-\-no-vision')" "1"

echo "passed $pass, failed $fail"
[ "$fail" -eq 0 ]

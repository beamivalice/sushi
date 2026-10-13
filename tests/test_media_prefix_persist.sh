#!/bin/bash
. "$(dirname "$0")/private_cache.sh"
# Issue #47: an SSD prefix-cache record carries the images its rows were computed under, so a
# growing text -> image -> text conversation restores PAST its images, a changed image invalidates
# only what follows it, and both survive a restart. Runs on every served architecture because each
# keeps a different restore shape behind the same records (Qwen GDN checkpoints, MiMo's sliding
# ring, GLM's KDA state).
#
# Asserts CACHE invariants only. Nothing here asks the model to read the picture: the image sits
# early with ~3k tokens of text after it, so any reuse past PAST_BAR can only come from rows
# computed under that image, whatever the checkpoint answers.
set -u
MODEL="${MEDIA_PERSIST_MODEL:-${1:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen3.8-Flash-Next-Sushi-2.6bpw}}"
PORT="${2:-11461}"
# GLM's original template has no thinking-off mode, so the field a body carries is per model.
# Assigned in two steps: a brace inside ${:-} has to be escaped, and the backslash survives.
THINK_JSON="${MEDIA_PERSIST_THINK:-}"
[ -n "$THINK_JSON" ] || THINK_JSON='{"enable_thinking": false}'
MAXTOK="${MEDIA_PERSIST_MAXTOK:-16}"
PAST_BAR="${MEDIA_PERSIST_PAST_BAR:-1000}"
BIN="${SUSHI_BIN:-./zig-out/bin/sushi}"
TAG="$(basename "$MODEL")"
RUN="$HOME/.sushi/runs/media-prefix-persist"
LOG="$RUN/$TAG-$PORT.log"
CACHE="$RUN/cache-$TAG-$PORT"
mkdir -p "$RUN"
[ -f "$MODEL/config.json" ] || { echo "SKIP: no model at $MODEL"; exit 0; }
F1="tests/fixtures/street-name-signs.jpg"
F2="tests/fixtures/house.jpeg"
for f in "$F1" "$F2"; do [ -f "$f" ] || { echo "SKIP: fixture $f missing"; exit 0; }; done
rm -rf "$CACHE"; mkdir -p "$CACHE"
pass=0; fail=0
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass+1)); else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail+1)); fi; }

SPID=""
boot() { # $1 log suffix
  SUSHI_PREFIX_CACHE_DIR="$CACHE" "$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" \
    --log-level debug --prefix-cache-mem off --prefix-cache-disk 4GB --prefix-cache-entries 4 \
    --prefill-chunk 1024 > "$LOG$1" 2>&1 &
  SPID=$!
  for _ in $(seq 1 900); do
    curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && grep -q "ready" "$LOG$1" && return 0
    kill -0 $SPID 2>/dev/null || { echo "  server died; tail:"; tail -5 "$LOG$1"; return 1; }
    sleep 2
  done
  return 1
}
stop() { [ -n "$SPID" ] && { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; }; SPID=""; }
trap stop EXIT

U="http://127.0.0.1:$PORT"
# Prints "<cached_tokens> <prompt_tokens>", or "ERR <body>" so a refused body fails by NAME
# instead of as an empty string (GLM silently 400'd a thinking-off body for weeks of this script's
# Qwen-shaped ancestor).
ask() {
  curl -s -m 900 "$U/v1/chat/completions" -H 'content-type: application/json' -d @- | python3 -I -c '
import sys, json
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except Exception:
    print("unparseable-body:", raw[:200], file=sys.stderr)
    print("ERR ERR"); sys.exit()
if "usage" not in d:
    print("the-server-refused-this-body:", json.dumps(d)[:300], file=sys.stderr)
    print("ERR ERR"); sys.exit()
print(d["usage"]["prompt_tokens_details"]["cached_tokens"], d["usage"]["prompt_tokens"])'
}

turn_body() { # $1 image, $2 extra turns
  MP_THINK="$THINK_JSON" MP_MAXTOK="$MAXTOK" python3 -I - "$1" "$2" <<'PY'
import base64, json, os, sys

path, turns = sys.argv[1], int(sys.argv[2])
with open(path, "rb") as f:
    image_url = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
messages = [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": "Describe this picture in one sentence."},
    ]},
    {"role": "assistant", "content": "A street scene."},
    {"role": "user", "content": "Notes: " + "lorem ipsum dolor sit amet " * 420},
    {"role": "assistant", "content": "Noted."},
]
for i in range(turns):
    messages.append({"role": "user", "content": f"Turn {i}. Reply with ALPHA only."})
    if i + 1 < turns:
        messages.append({"role": "assistant", "content": "ALPHA"})
payload = {"model": "sushi", "max_tokens": int(os.environ["MP_MAXTOK"]), "temperature": 0,
           "messages": messages}
payload.update(json.loads(os.environ["MP_THINK"]))
print(json.dumps(payload))
PY
}

echo "== media-aware SSD prefix persistence: $TAG"
boot "" || { echo "FAIL: boot"; exit 1; }

echo "[1] cold image turn"
r1=($(turn_body "$F1" 1 | ask)); echo "  cached=${r1[0]} prompt=${r1[1]}"
check "cold turn answered" "$(echo "${r1[0]}" | grep -c ERR)" "0"
check "cold: nothing cached" "${r1[0]}" "0"

echo "[2] the same image, one turn appended"
r2=($(turn_body "$F1" 2 | ask)); echo "  cached=${r2[0]} prompt=${r2[1]}"
check "append restores PAST the image" "$(python3 -I -c "print(1 if int('${r2[0]}')>$PAST_BAR else 0)")" "1"

echo "[3] a different image under the same placeholders"
r3=($(turn_body "$F2" 2 | ask)); echo "  cached=${r3[0]} prompt=${r3[1]}"
check "a changed image invalidates its suffix" "$(python3 -I -c "print(1 if int('${r3[0]}')<=$PAST_BAR else 0)")" "1"

echo "[4] restart, same history"
stop
boot ".restart" || { echo "FAIL: restart boot"; exit 1; }
r4=($(turn_body "$F1" 2 | ask)); echo "  cached=${r4[0]} prompt=${r4[1]}"
check "the image prefix restores after a restart" "$(python3 -I -c "print(1 if int('${r4[0]}')>$PAST_BAR else 0)")" "1"
check "it came off the disk tier" "$(grep -c 'disk-cache\] restored' "$LOG.restart" | sed 's/^[1-9][0-9]*$/1/')" "1"

echo "pass=$pass fail=$fail"
[ "$fail" = "0" ] && echo "PASS: media prefix persistence ($TAG)" || { echo "FAIL: media prefix persistence ($TAG)"; grep -E "disk-cache|hot-cache" "$LOG.restart" 2>/dev/null | tail -10; }
[ "$fail" = "0" ]

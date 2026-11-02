#!/bin/bash
# sushi-test: tier=host
# A `--model` boot listens before its model is loaded: the page and the status routes answer during the load, a
# generation request sent then completes after it, and a load that fails ends the process non-zero by name. The model
# is the tiny DeepSeek fixture with a byte-level tokenizer made here (${SUSHI_MODELS_DIR}/test-fixtures/dsv41-dspark-tiny,
# EARLY_LISTEN_TINY_MODEL overrides; SKIPs without it); SUSHI_TEST_LOAD_DELAY_MS holds the load open after its
# refusals so the window is long enough to observe. Takes the GPU lock per boot; private HOME and prefix cache root.
# Usage: ./tests/test_early_listener.sh [port]
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BINARY:-$ROOT/zig-out/bin/sushi}"
LOCK="$ROOT/scripts/gpu-lock.sh"
FIXTURE="${EARLY_LISTEN_TINY_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/test-fixtures/dsv41-dspark-tiny}"
PORT="${1:-18771}"
BASE="http://127.0.0.1:$PORT"
HOLD_MS=8000
PASS=0
FAIL=0

check() {
    if [ "$2" = "1" ]; then PASS=$((PASS + 1)); echo "  PASS $1"; else FAIL=$((FAIL + 1)); echo "  FAIL $1"; fi
}
yes_if() { if eval "$1"; then echo 1; else echo 0; fi; }

[ -x "$BIN" ] || { echo "SKIP: no binary at $BIN (zig build -Doptimize=ReleaseFast)"; exit 3; }
[ -f "$FIXTURE/config.json" ] || { echo "SKIP: no tiny fixture at $FIXTURE"; exit 3; }
command -v python3 >/dev/null || { echo "SKIP: python3 not found"; exit 3; }
command -v jq >/dev/null || { echo "SKIP: jq not found"; exit 3; }
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "FAIL: port $PORT is busy; pass a free one"
    exit 1
fi

WORK="$(mktemp -d)"
owner="early-listen-$$"
SPID=""
HOLDER=""
cleanup() {
    [ -n "$SPID" ] && kill "$SPID" 2>/dev/null && wait "$SPID" 2>/dev/null
    [ -n "$HOLDER" ] && kill "$HOLDER" 2>/dev/null && wait "$HOLDER" 2>/dev/null
    "$LOCK" release "$owner" >/dev/null 2>&1
    rm -rf "$WORK"
}
trap cleanup EXIT

# tiny/: the fixture's weights linked, a byte tokenizer. broken/: the same with one trunk tensor renamed in its shard
# header and index, so the load fails after its refusals, at the weights.
make_model() {
    local dir="$1" break_it="$2"
    mkdir -p "$dir"
    for f in "$FIXTURE"/model-*.safetensors "$FIXTURE"/model.safetensors.index.json "$FIXTURE"/engram_token_map.json; do
        [ -e "$f" ] && ln -s "$f" "$dir/$(basename "$f")"
    done
    python3 -I - "$FIXTURE" "$dir" "$break_it" <<'PY'
import json, os, shutil, struct, sys
src, dst, break_it = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
cfg = json.load(open(src + "/config.json"))
json.dump(cfg, open(dst + "/config.json", "w"))
symbols = [chr(b) for b in range(0x21, 0x7F)] + ["Ġ", "Ċ"]
assert len(symbols) == cfg["text_config"]["vocab_size"] == 96
tok = {"model": {"type": "BPE", "vocab": {s: i for i, s in enumerate(symbols)}, "merges": []},
       "pre_tokenizer": {"type": "ByteLevel"}, "added_tokens": []}
json.dump(tok, open(dst + "/tokenizer.json", "w"))
json.dump({}, open(dst + "/tokenizer_config.json", "w"))
if break_it:
    old, new = "layers.0.attn_norm.weight", "layers.0.attn_norx.weight"
    index = json.load(open(src + "/model.safetensors.index.json"))
    shard = index["weight_map"].pop(old)
    index["weight_map"][new] = shard
    os.unlink(dst + "/model.safetensors.index.json")
    json.dump(index, open(dst + "/model.safetensors.index.json", "w"))
    os.unlink(dst + "/" + shard)
    raw = open(src + "/" + shard, "rb").read()
    n = struct.unpack("<Q", raw[:8])[0]
    header = raw[8:8 + n].replace(old.encode(), new.encode())
    assert len(header) == n
    open(dst + "/" + shard, "wb").write(raw[:8] + header + raw[8 + n:])
PY
}
make_model "$WORK/tiny" 0
make_model "$WORK/broken" 1
mkdir -p "$WORK/home"

LOG="$WORK/server.log"
boot() {
    local model="$1"
    : > "$LOG"
    HOME="$WORK/home" SUSHI_PREFIX_CACHE_DIR="$WORK/cache" SUSHI_TEST_LOAD_DELAY_MS="$HOLD_MS" \
        "$BIN" serve --model "$model" --host 127.0.0.1 --port "$PORT" --ssd-budget-gb 6 --log-file off > "$LOG" 2>&1 &
    SPID=$!
}
# Until the port answers anything at all: /health is a 503 while the model loads.
wait_listening() {
    for _ in $(seq 1 600); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' -m 2 "$BASE/health")" != "000" ] && return 0
        kill -0 "$SPID" 2>/dev/null || return 1
        sleep 0.05
    done
    return 1
}
ready_logged() { grep -q 'Model ready' "$LOG"; }
line_of() { grep -n -m1 -- "$1" "$LOG" | cut -d: -f1; }
code_of() { curl -s -o /dev/null -w '%{http_code}' -m 5 "$@"; }
wait_gone() {
    for _ in $(seq 1 600); do kill -0 "$SPID" 2>/dev/null || break; sleep 0.1; done
    if kill -0 "$SPID" 2>/dev/null; then RC=alive; return; fi
    wait "$SPID" 2>/dev/null
    RC=$?
    SPID=""
}

echo "── [1/3] a busy port still refuses before any load"
python3 -I -c 'import socket, sys, time
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", int(sys.argv[1]))); s.listen(1)
print("held", flush=True); time.sleep(60)' "$PORT" > "$WORK/holder.out" &
HOLDER=$!
for _ in $(seq 1 50); do grep -q held "$WORK/holder.out" 2>/dev/null && break; sleep 0.1; done
boot "$WORK/tiny"
wait_gone
check "exits non-zero (rc=$RC)" "$(yes_if '[ "$RC" != alive ] && [ "$RC" -ne 0 ]')"
check "names the busy port" "$(yes_if "grep -q 'port $PORT is already in use' '$LOG'")"
check "never listened and loaded nothing" "$(yes_if "! grep -qE 'Server listening|\\[preflight\\]|Model ready' '$LOG'")"
kill "$HOLDER" 2>/dev/null; wait "$HOLDER" 2>/dev/null; HOLDER=""

echo "── [2/3] the page and the status routes answer while the model loads; a request sent then completes after it"
"$LOCK" acquire "$owner" --pid $$ --max-min 20 >/dev/null || { echo "FAIL: could not take the GPU lock"; exit 1; }
boot "$WORK/tiny"
if ! wait_listening; then
    check "the server listens during the load" 0
    tail -20 "$LOG"
else
    early_ready=$(yes_if ready_logged)
    page=$(code_of "$BASE/")
    health_status=$(code_of "$BASE/health")
    health_body=$(curl -s -m 5 "$BASE/health")
    models=$(curl -s -m 5 "$BASE/v1/models")
    props=$(code_of "$BASE/props")
    settings=$(code_of "$BASE/v1/settings")
    metrics=$(code_of "$BASE/metrics.json")
    curl -s -m 120 "$BASE/v1/completions" -H 'Content-Type: application/json' \
        -d '{"model":"tiny","prompt":"abcdefghij abcdefghij","max_tokens":4,"temperature":0}' \
        -w '\n%{http_code}' > "$WORK/gen.out" 2>&1 &
    GEN=$!
    still_loading=$(yes_if '! ready_logged')
    check "the log has no Model ready when the port first answers" "$([ "$early_ready" = 0 ] && echo 1 || echo 0)"
    check "GET / serves the page during the load ($page)" "$(yes_if '[ "$page" = 200 ]')"
    check "/health is a 503 naming the load ($health_status $health_body)" "$(yes_if '[ "$health_status" = 503 ] && [ "$health_body" = "{\"status\":\"loading\"}" ]')"
    check "/v1/models lists the launch model as loading" "$(echo "$models" | jq -e '.data[0].id == "tiny" and .data[0].state == "loading"' >/dev/null && echo 1 || echo 0)"
    check "/props, /v1/settings and /metrics.json answer 200 ($props $settings $metrics)" "$(yes_if '[ "$props" = 200 ] && [ "$settings" = 200 ] && [ "$metrics" = 200 ]')"
    check "the generation request was sent before the load finished" "$still_loading"
    wait "$GEN"
    gen_code=$(tail -1 "$WORK/gen.out")
    check "it completes with 200 and a choice ($gen_code)" "$(yes_if "[ '$gen_code' = 200 ] && head -1 '$WORK/gen.out' | jq -e '.choices[0].text | type == \"string\"' >/dev/null")"
    check "and only after Model ready" "$(yes_if ready_logged)"
    check "/health is ok once the model is ready" "$(yes_if '[ "$(curl -s -m 5 "$BASE/health")" = "{\"status\":\"ok\"}" ]')"
    check "/v1/models lists it ready" "$(curl -s -m 5 "$BASE/v1/models" | jq -e '.data[0].state == "ready"' >/dev/null && echo 1 || echo 0)"
    listen=$(line_of 'Server listening'); ready=$(line_of 'Model ready'); chat=$(line_of 'chat in your browser: ')
    check "the log says Server listening, then Model ready, then the chat URL ($listen $ready $chat)" \
        "$(yes_if '[ -n "$listen" ] && [ -n "$ready" ] && [ -n "$chat" ] && [ "$listen" -lt "$ready" ] && [ "$ready" -lt "$chat" ]')"
    check "the chat URL is printed once" "$(yes_if '[ "$(grep -c "chat in your browser: " "$LOG")" = 1 ]')"
    kill "$SPID"
    wait_gone
    check "SIGTERM after the load exits 0 (rc=$RC)" "$(yes_if '[ "$RC" = 0 ]')"
fi
"$LOCK" release "$owner" >/dev/null 2>&1

echo "── [3/3] a load that fails after the listener is up ends the process non-zero by name"
"$LOCK" acquire "$owner" --pid $$ --max-min 20 >/dev/null || { echo "FAIL: could not take the GPU lock"; exit 1; }
boot "$WORK/broken"
if ! wait_listening; then
    check "the server listens during the failing load" 0
    tail -20 "$LOG"
else
    check "GET / answers while the failing model loads" "$(yes_if '[ "$(code_of "$BASE/")" = 200 ]')"
    gen=$(curl -s -m 120 "$BASE/v1/completions" -H 'Content-Type: application/json' \
        -d '{"model":"tiny","prompt":"abc","max_tokens":4}' -w '\n%{http_code}')
    gen_code=$(echo "$gen" | tail -1)
    check "a request sent during the load gets the load's error by name ($gen_code)" \
        "$(yes_if "[ '$gen_code' -ge 500 ] && echo '$gen' | head -1 | grep -q MissingWeight")"
    wait_gone
    check "the process exits non-zero (rc=$RC)" "$(yes_if '[ "$RC" != alive ] && [ "$RC" -ne 0 ]')"
    check "the log names the error" "$(yes_if "grep -q 'model load failed: MissingWeight' '$LOG'")"
    check "it never reported ready" "$(yes_if "! grep -qE 'Model ready|chat in your browser' '$LOG'")"
fi
"$LOCK" release "$owner" >/dev/null 2>&1

echo "early listener: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]

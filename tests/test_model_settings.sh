#!/usr/bin/env bash
. "$(dirname "$0")/private_cache.sh"
# Per-model settings (`~/.sushi/model-settings.json`, issue #269): a model's
# `ctx_size` / `kv_quant` / `mtp` / `mtp_acceptance` / `mtp_greedy_tail` follow the MODEL, apply on its
# load (boot AND cold load), and a second model cold-loaded in the same process
# keeps the globals and the launch flags. An explicit launch flag outranks the file, and an
# explicit --mtp refuses an SSD-streamed load (STREAM_MODEL, skipped when absent).
# The served packs cannot sit side by side under the resident cap, so the second
# model is loaded after the first is unloaded: precedence, not coexistence.
#
# Runs under a private HOME so the real settings file is never touched.
# NEEDS REAL MODELS: skips when the two defaults are absent.
#
# Usage: ./tests/test_model_settings.sh [port]
set -uo pipefail
PORT="${1:-11384}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/sushi"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

MODELS_ROOT="${MODELS_ROOT:-$HOME/.sushi/models}"
MODEL_A="${MODEL_A:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw}"
MODEL_B="${MODEL_B:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen3.8-Flash-Next-Sushi-4bpw}"
STREAM_MODEL="${STREAM_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen/Qwen3.8-Flash-Next}"
if [ ! -f "$MODEL_A/config.json" ] || [ ! -f "$MODEL_B/config.json" ]; then
    echo "SKIP: needs two local chat models (MODEL_A=$MODEL_A, MODEL_B=$MODEL_B)"
    exit 0
fi

PASS=0; FAIL=0
RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
check() {
    if [ "$2" = "1" ]; then PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $1"
    else FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $1"; fi
}

FAKE_HOME="$(mktemp -d)"
mkdir -p "$FAKE_HOME/.sushi"
SETTINGS="$FAKE_HOME/.sushi/model-settings.json"
LOG="$FAKE_HOME/server.log"
SRV=""
cleanup() {
    [ -n "$SRV" ] && kill "$SRV" 2>/dev/null
    rm -rf "$FAKE_HOME"
}
trap cleanup EXIT
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | grep -q LISTEN; then
    echo "port $PORT is already in use; stop that server or pass another port" >&2
    exit 1
fi

write_settings() { # write_settings <ctx> <kv>  — override for MODEL_A only
    cat >"$SETTINGS" <<JSON
{ "$MODEL_A/": { "ctx_size": $1, "kv_quant": "$2", "mtp": true, "mtp_acceptance": "typical", "mtp_greedy_tail": true }, "not-a-model": 1 }
JSON
}
write_settings 4096 8

boot() { # boot <extra flags...> — MODEL_A primary; no --ctx-size / --kv-quant unless passed
    HOME="$FAKE_HOME" "$BIN" --serve --model "$MODEL_A" --model-dir "$MODELS_ROOT" --port "$PORT" --log-file off "$@" >"$LOG" 2>&1 &
    SRV=$!
    UP=0
    for _ in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { UP=1; break; }
        kill -0 "$SRV" 2>/dev/null || break
        sleep 0.5
    done
    [ "$UP" = "1" ] || { echo "FAIL: server never became healthy"; tail -5 "$LOG"; exit 1; }
}
boot --no-mtp

row() { # row <model path> <ctx|kv|src> — context_length, meta.kv_quant or meta.kv_cache.source of the READY row
    curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "
import sys, json
want = sys.argv[1].rstrip('/')
for m in json.load(sys.stdin)['data']:
    if want.endswith('/' + m['id']):
        f = sys.argv[2]
        print(m['context_length'] if f == 'ctx' else m['meta']['kv_cache']['source'] if f == 'src' else m['meta'].get('kv_quant'))
        break
" "$1" "$2"
}
model_id() { # model_id <model path> — the /v1/models id of that path; an unknown id falls back to the default model
    curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "
import sys, json
want = sys.argv[1].rstrip('/')
print(next((m['id'] for m in json.load(sys.stdin)['data'] if want.endswith('/' + m['id'])), ''))
" "$1"
}
props_mtp_source() { # props_mtp_source <model path> — /props settings.mtp.source
    local id; id="$(model_id "$1")"
    curl -s "http://127.0.0.1:$PORT/props?model=$id" | python3 -c "import sys, json; print(json.load(sys.stdin)['settings']['mtp']['source'])"
}
post() { # post <route> <json>
    curl -s -o /dev/null -w '%{http_code}' --max-time 300 -X POST "http://127.0.0.1:$PORT/v1/$1" \
        -H 'Content-Type: application/json' -d "$2"
}
load() { # load <model path> — an unloaded model's pages come back to the OS lazily, so the load
    # preflight can refuse (503) for a few seconds after an unload; retry as its message says.
    local code
    for _ in $(seq 1 12); do
        code="$(post load-model "{\"model\":\"$1\"}")"
        [ "$code" = "503" ] || break
        sleep 5
    done
    echo "$code"
}

# [1] boot load honours the file where no flag was given; --no-mtp outranks its mtp: true
check "[1] boot: context_length 4096 from the file (got $(row "$MODEL_A" ctx))" "$([ "$(row "$MODEL_A" ctx)" = "4096" ] && echo 1 || echo 0)"
check "[1] boot: meta.kv_quant 8 from the file (got $(row "$MODEL_A" kv))" "$([ "$(row "$MODEL_A" kv)" = "8" ] && echo 1 || echo 0)"
check "[1] boot: kv_cache source model-settings.json (got $(row "$MODEL_A" src))" "$([ "$(row "$MODEL_A" src)" = "model-settings.json" ] && echo 1 || echo 0)"
check "[1] load log names the KV and ctx choices" "$(grep -q "\[kv-cache\] kv8 (model-settings.json); ctx 4096 (model-settings.json)" "$LOG" && echo 1 || echo 0)"
check "[1] load log: --no-mtp outranks mtp:true, acceptance from the file" \
    "$(grep -q "\[mtp\] off (--no-mtp); acceptance typical (model-settings.json)" "$LOG" && echo 1 || echo 0)"
check "[1] /props settings.mtp.source --no-mtp (got $(props_mtp_source "$MODEL_A"))" "$([ "$(props_mtp_source "$MODEL_A")" = "--no-mtp" ] && echo 1 || echo 0)"
check "[1] log names the override" "$(grep -q "\[model-settings\] .*ctx=4096 kv=8 mtp=on" "$LOG" && echo 1 || echo 0)"
check "[1] log names the MTP acceptance mode" "$(grep -q "\[model-settings\] .*accept=typical" "$LOG" && echo 1 || echo 0)"
check "[1] load log: greedy tail from the file" \
    "$(grep -q "acceptance typical (model-settings.json); greedy tail on (model-settings.json)" "$LOG" && echo 1 || echo 0)"

# [2] a second model keeps the globals, and its cold load carries the explicit --no-mtp
CODE="$(post unload-model "{\"model\":\"$MODEL_A\"}")"
check "[2] unload model A first -> 200 (got $CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"
CODE="$(load "$MODEL_B")"
check "[2] cold load of model B -> 200 (got $CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"
check "[2] model B keeps the kv8 default (got $(row "$MODEL_B" kv))" "$([ "$(row "$MODEL_B" kv)" = "8" ] && echo 1 || echo 0)"
check "[2] model B kv_cache source default (got $(row "$MODEL_B" src))" "$([ "$(row "$MODEL_B" src)" = "default" ] && echo 1 || echo 0)"
check "[2] cold-load log names the KV and ctx choices" "$(grep -q "\[kv-cache\] kv8 (default); ctx auto (default)" "$LOG" && echo 1 || echo 0)"
check "[2] cold-load log: --no-mtp, exact acceptance" "$(grep -q "\[mtp\] off (--no-mtp); acceptance exact (default)" "$LOG" && echo 1 || echo 0)"
check "[2] cold-load log: greedy tail off by default" "$(grep -q "acceptance exact (default); greedy tail off (default)" "$LOG" && echo 1 || echo 0)"
check "[2] /props settings.mtp.source --no-mtp for B (got $(props_mtp_source "$MODEL_B"))" "$([ "$(props_mtp_source "$MODEL_B")" = "--no-mtp" ] && echo 1 || echo 0)"

# [3] edit + unload + load applies the new values, no restart
write_settings 8192 4
CODE="$(post unload-model "{\"model\":\"$MODEL_B\"}")"
check "[3] unload model B -> 200 (got $CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"
CODE="$(load "$MODEL_A")"
check "[3] reload model A -> 200 (got $CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"
check "[3] model A now 8192 (got $(row "$MODEL_A" ctx))" "$([ "$(row "$MODEL_A" ctx)" = "8192" ] && echo 1 || echo 0)"
check "[3] model A now kv 4 (got $(row "$MODEL_A" kv))" "$([ "$(row "$MODEL_A" kv)" = "4" ] && echo 1 || echo 0)"
kill -0 "$SRV" 2>/dev/null; check "[3] server never restarted" "$([ $? = 0 ] && echo 1 || echo 0)"

# [4] a malformed file never stops a load
echo '{nope' >"$SETTINGS"
post unload-model "{\"model\":\"$MODEL_A\"}" >/dev/null
CODE="$(load "$MODEL_A")"
check "[4] malformed file: load -> 200 (got $CODE), defaults apply (kv source $(row "$MODEL_A" src))" \
    "$([ "$CODE" = "200" ] && [ "$(row "$MODEL_A" src)" = "default" ] && echo 1 || echo 0)"
check "[4] malformed file logged" "$(grep -q "\[model-settings\] .*malformed" "$LOG" && echo 1 || echo 0)"

# [5] ssd_budget_gb on a Sushi pack: the load streams its experts at that budget
cat >"$SETTINGS" <<JSON
{ "$MODEL_A/": { "ssd_budget_gb": 60 } }
JSON
post unload-model "{\"model\":\"$MODEL_A\"}" >/dev/null
CODE="$(load "$MODEL_A")"
check "[5] ssd_budget_gb on a Sushi pack: load -> 200 (got $CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"
check "[5] the setting is logged" "$(grep -q "\[model-settings\] .*ssd_budget_gb=60" "$LOG" && echo 1 || echo 0)"
check "[5] the load streams at the file's budget, never ignoring it" \
    "$(grep -q "\[expert-stream\] ssd budget 60 GiB from model-settings.json" "$LOG" && ! grep -q "ssd_budget_gb ignored" "$LOG" && echo 1 || echo 0)"

# [6] explicit launch flags outrank every competing key in the file
kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""
cat >"$SETTINGS" <<JSON
{ "$MODEL_A/": { "ctx_size": 4096, "kv_quant": "8", "mtp": false, "mtp_acceptance": "typical", "mtp_greedy_tail": false } }
JSON
boot --ctx-size 16384 --kv-quant 4 --mtp --mtp-tokenv3 0.9 --mtp-greedy-tail
check "[6] --ctx-size 16384 outranks ctx_size 4096 (got $(row "$MODEL_A" ctx))" "$([ "$(row "$MODEL_A" ctx)" = "16384" ] && echo 1 || echo 0)"
check "[6] --kv-quant 4 outranks kv_quant 8 (got $(row "$MODEL_A" kv), source $(row "$MODEL_A" src))" \
    "$([ "$(row "$MODEL_A" kv)" = "4" ] && [ "$(row "$MODEL_A" src)" = "--kv-quant" ] && echo 1 || echo 0)"
check "[6] load log names the flags" "$(grep -q "\[kv-cache\] kv4 (--kv-quant); ctx 16384 (--ctx-size)" "$LOG" && echo 1 || echo 0)"
check "[6] --mtp outranks mtp:false, --mtp-tokenv3 outranks typical" \
    "$(grep -q "\[mtp\] on (--mtp); acceptance tokenv3 (--mtp-tokenv3)" "$LOG" && echo 1 || echo 0)"
check "[6] /props settings.mtp.source --mtp (got $(props_mtp_source "$MODEL_A"))" "$([ "$(props_mtp_source "$MODEL_A")" = "--mtp" ] && echo 1 || echo 0)"
check "[6] --mtp-greedy-tail outranks mtp_greedy_tail:false" \
    "$(grep -q "acceptance tokenv3 (--mtp-tokenv3); greedy tail on (--mtp-greedy-tail)" "$LOG" && echo 1 || echo 0)"

# [7] no flag, no file: a served pack runs MTP by default; the file's mtp:false turns it off
kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""
echo '{}' >"$SETTINGS"
boot
props_mtp_default_on() { # props_mtp_default_on <model path> — /props settings.mtp.default_on
    local id; id="$(model_id "$1")"
    curl -s "http://127.0.0.1:$PORT/props?model=$id" | python3 -c "import sys, json; print(json.load(sys.stdin)['settings']['mtp']['default_on'])"
}
check "[7] load log: MTP on by default" "$(grep -q "\[mtp\] on (default)" "$LOG" && echo 1 || echo 0)"
check "[7] /props settings.mtp.default_on true, source default (got $(props_mtp_default_on "$MODEL_A") / $(props_mtp_source "$MODEL_A"))" \
    "$([ "$(props_mtp_default_on "$MODEL_A")" = "True" ] && [ "$(props_mtp_source "$MODEL_A")" = "default" ] && echo 1 || echo 0)"
cat >"$SETTINGS" <<JSON
{ "$MODEL_A/": { "mtp": false } }
JSON
post unload-model "{\"model\":\"$MODEL_A\"}" >/dev/null
CODE="$(load "$MODEL_A")"
check "[7] mtp:false in the file turns the default off (load $CODE, default_on $(props_mtp_default_on "$MODEL_A"))" \
    "$([ "$CODE" = "200" ] && [ "$(props_mtp_default_on "$MODEL_A")" = "False" ] && echo 1 || echo 0)"

# [8] an explicit --mtp refuses a streamed load
if [ -f "$STREAM_MODEL/config.json" ]; then
    kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""
    cat >"$SETTINGS" <<JSON
{ "$STREAM_MODEL/": { "ssd_budget_gb": 60 } }
JSON
    boot --mtp
    post unload-model "{\"model\":\"$MODEL_A\"}" >/dev/null
    BODY="$(curl -s --max-time 1800 -X POST "http://127.0.0.1:$PORT/v1/load-model" \
        -H 'Content-Type: application/json' -d "{\"model\":\"$STREAM_MODEL\"}")"
    check "[8] --mtp: the streamed load is refused (got $BODY)" \
        "$(echo "$BODY" | grep -q "expert_streaming_mtp_unsupported" && echo 1 || echo 0)"
else
    echo "  SKIP [8]: no streamed checkpoint at $STREAM_MODEL"
fi

if [ "$FAIL" -gt 0 ]; then
    echo "server log (loads and refusals):"
    grep -E "preflight|Insufficient|\[admission\]|\[registry\]|load failed|error" "$LOG" | tail -20
fi
echo "$PASS passed, $FAIL failed"
[ "$FAIL" = "0" ]

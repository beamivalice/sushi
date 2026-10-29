#!/bin/bash
# sushi-test: tier=host
. "$(dirname "$0")/private_cache.sh"
# Regression test: headless serve mode must honor the CLI's launch flags.
#
# `runHeadlessServe` builds its own ServerConfig literal, so a flag threaded
# only through the `--model` path never reached a headless request. This is
# the mode that matters: the Swift app ALWAYS launches headless
# (`--serve --model-dir`, no `--model`).
#
# Fully hermetic: an EMPTY --model-dir discovers zero models and never loads
# one, so the boot banner is reachable with no checkpoint on disk.
#
# Usage: ./tests/test_headless_spec_flags.sh [port]

set -u

PORT="${1:-11265}"
BINARY="${BINARY:-./zig-out/bin/sushi}"
PASS=0
FAIL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

check() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then
        PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $desc"
    else
        FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $desc"
    fi
}

if [ ! -x "$BINARY" ]; then
    echo "[fail] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi

EMPTY_DIR="$(mktemp -d)"
LOG="$(mktemp)"
SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    rm -rf "$EMPTY_DIR" "$LOG"
}
trap cleanup EXIT

require_free_port() {
    if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | grep -q LISTEN; then
        echo "port $PORT is already in use; stop that server or pass another port" >&2
        exit 1
    fi
}

# Boot headless over the empty dir, capture the banner, stop. Prints nothing;
# the caller greps "$LOG".
boot() {
    require_free_port
    : > "$LOG"
    "$BINARY" --serve --model-dir "$EMPTY_DIR" --port "$PORT" --log-file off "$@" > "$LOG" 2>&1 &
    local pid=$!
    SERVER_PID=$pid
    for _ in $(seq 1 60); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
        sleep 0.5
        kill -0 "$pid" 2>/dev/null || break
    done
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
        echo "  (server did not come up; log follows)"; cat "$LOG"; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; return 1
    }
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    return 0
}

echo "Headless launch flag plumbing (port $PORT)"

# `--max-tokens` used to feed only the offline `--prompt` mode; headless hand-rolls its own
# ServerConfig literal, which is how launch flags were eaten.
echo "[1/2] --max-tokens reaches the serve-mode omitted-field default"
if boot --max-tokens 4096; then
    grep -q "default max_tokens for omitted requests: 4096" "$LOG"
    check "--max-tokens 4096 reaches the headless request defaults" "$([ $? -eq 0 ] && echo 1 || echo 0)"
else
    check "boot with --max-tokens" 0
fi
if boot; then
    grep -q "default max_tokens for omitted requests" "$LOG"
    check "no flag announces no default (offline's 100 never leaks into serve)" "$([ $? -ne 0 ] && echo 1 || echo 0)"
else
    check "bare boot for the --max-tokens default" 0
fi

# Model resolution runs BEFORE dispatch, so with no default model an unknown
# path never reached the chain's 404 and came back 503 "No default model
# configured". A path's existence has nothing to do with model state, and a
# server that answers every unknown path with a non-404 looks like a catch-all
# to anything that maps endpoints by probing them — llmprobe scored every
# surface absent against a headless boot for exactly this reason (2026-07-25).
# Headless with an EMPTY model dir is the only place this is observable.
echo "[2/2] endpoint existence does not depend on a model being loaded"
require_free_port
: > "$LOG"
"$BINARY" --serve --model-dir "$EMPTY_DIR" --port "$PORT" --log-file off > "$LOG" 2>&1 &
HPID=$!
SERVER_PID=$HPID
for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 0.5
    kill -0 "$HPID" 2>/dev/null || break
done
post_code() {
    curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT$1" \
        -H 'Content-Type: application/json' -d '{}'
}
unknown_code=$(post_code /v1/__no_such_endpoint__)
known_code=$(post_code /v1/chat/completions)
media_code=$(post_code /v1/images/generations)
kill "$HPID" 2>/dev/null
wait "$HPID" 2>/dev/null
check "unknown endpoint is 404, not no-model (got $unknown_code)" \
    "$([ "$unknown_code" = "404" ] && echo 1 || echo 0)"
check "a served endpoint still reports 503 no-model (got $known_code)" \
    "$([ "$known_code" = "503" ] && echo 1 || echo 0)"
check "a removed media-generation endpoint is 404 (got $media_code)" \
    "$([ "$media_code" = "404" ] && echo 1 || echo 0)"

echo
echo "  passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]

#!/bin/bash
. "$(dirname "$0")/private_cache.sh"
# Runtime metrics: a server booted WITHOUT --metrics turns them on through `POST /metrics/enable`.
#
#  1. Off: GET /metrics and /metrics.json are 503 and the 503 body names the endpoint; GET /metrics/enable is 405.
#  2. A foreign Origin is refused (403) and nothing turns on.
#  3. POST /metrics/enable is 200 {"metrics":true,"already_enabled":false}; both feeds answer 200 at once.
#  4. A second POST is 200 with already_enabled true (idempotent).
#  5. Under --api-key --api-key-strict: no key and a wrong key are 401 and leave metrics off; the Bearer key enables.
#  6. --metrics at boot still works: the feeds are 200 and a POST reports already_enabled.
#
# Fully hermetic: an EMPTY --model-dir discovers zero models, so nothing loads. Do not run it beside another
# sushi server on the box without a free port.
#
# Usage: ./tests/test_metrics_enable.sh [port]

set -u

PORT="${1:-18853}"
BINARY="${BINARY:-./zig-out/bin/sushi}"
BASE="http://127.0.0.1:$PORT"
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
is() { [ "$1" = "$2" ] && echo 1 || echo 0; }

if [ ! -x "$BINARY" ]; then
    echo "[fail] $BINARY not found; build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | grep -q LISTEN; then
    echo "port $PORT is already in use; pass another port" >&2
    exit 1
fi

EMPTY_DIR="$(mktemp -d)"
WORK="$(mktemp -d)"
LOG="$WORK/server.log"
SPID=""
stop_server() {
    [ -n "$SPID" ] && kill "$SPID" 2>/dev/null && wait "$SPID" 2>/dev/null
    SPID=""
}
cleanup() {
    stop_server
    rm -rf "$EMPTY_DIR" "$WORK"
}
trap cleanup EXIT

boot() {
    stop_server
    : > "$LOG"
    HOME="$WORK" "$BINARY" --serve --model-dir "$EMPTY_DIR" --port "$PORT" --log-file off "$@" > "$LOG" 2>&1 &
    SPID=$!
    for _ in $(seq 1 60); do
        curl -sf "$BASE/health" >/dev/null 2>&1 && return 0
        kill -0 "$SPID" 2>/dev/null || break
        sleep 0.5
    done
    echo "  (server did not come up; log follows)"; cat "$LOG"
    return 1
}
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
enable() { curl -s -X POST "$BASE/metrics/enable" "$@"; }

echo "Runtime metrics (port $PORT)"

echo "[1/6] off by default"
if boot; then
    check "GET /metrics is 503" "$(is "$(code "$BASE/metrics")" 503)"
    check "GET /metrics.json is 503" "$(is "$(code "$BASE/metrics.json")" 503)"
    check "the 503 body names POST /metrics/enable" \
        "$(curl -s "$BASE/metrics.json" | grep -q 'POST /metrics/enable' && echo 1 || echo 0)"
    check "GET /metrics/enable is 405" "$(is "$(code "$BASE/metrics/enable")" 405)"
    check "the 405 did not enable anything" "$(is "$(code "$BASE/metrics.json")" 503)"

    echo "[2/6] a foreign Origin is refused"
    check "POST with Origin: http://evil.test is 403" \
        "$(is "$(code -X POST -H 'Origin: http://evil.test' "$BASE/metrics/enable")" 403)"
    check "and metrics stay off" "$(is "$(code "$BASE/metrics.json")" 503)"

    echo "[3/6] enable"
    BODY="$(enable)"
    check "POST /metrics/enable says enabled, not already" \
        "$(is "$BODY" '{"metrics":true,"already_enabled":false}')"
    check "GET /metrics is 200" "$(is "$(code "$BASE/metrics")" 200)"
    check "GET /metrics.json is 200 and parses" \
        "$(curl -s "$BASE/metrics.json" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert "counters" in d and "gauges" in d' 2>/dev/null && echo 1 || echo 0)"
    check "the server's own origin is accepted" \
        "$(is "$(code -X POST -H "Origin: $BASE" "$BASE/metrics/enable")" 200)"

    echo "[4/6] idempotent"
    check "a second POST says already_enabled" \
        "$(is "$(enable)" '{"metrics":true,"already_enabled":true}')"
    check "the boot log names the runtime enable once" \
        "$([ "$(grep -c 'ENABLED at runtime' "$LOG")" = 1 ] && echo 1 || echo 0)"
else
    check "boot without --metrics" 0
fi

echo "[5/6] --api-key --api-key-strict binds the route"
if boot --api-key enable-test-key --api-key-strict; then
    check "no key is 401" "$(is "$(code -X POST "$BASE/metrics/enable")" 401)"
    check "a wrong key is 401" "$(is "$(code -X POST -H 'Authorization: Bearer nope' "$BASE/metrics/enable")" 401)"
    check "metrics are still off" \
        "$(is "$(code -H 'Authorization: Bearer enable-test-key' "$BASE/metrics.json")" 503)"
    check "the Bearer key enables" \
        "$(is "$(enable -H 'Authorization: Bearer enable-test-key')" '{"metrics":true,"already_enabled":false}')"
    check "and the feed answers with the key" \
        "$(is "$(code -H 'Authorization: Bearer enable-test-key' "$BASE/metrics.json")" 200)"
else
    check "boot with --api-key" 0
fi

echo "[6/6] --metrics at boot is unchanged"
if boot --metrics; then
    check "GET /metrics.json is 200 from the start" "$(is "$(code "$BASE/metrics.json")" 200)"
    check "POST reports already_enabled" \
        "$(is "$(enable)" '{"metrics":true,"already_enabled":true}')"
else
    check "boot with --metrics" 0
fi

echo
echo "  passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]

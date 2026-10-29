#!/bin/bash
# sushi-test: tier=host
. "$(dirname "$0")/private_cache.sh"
# `--parent-pid <pid>`: a host that runs sushi as a guest engine names its own pid, and sushi must shut down once
# that process is gone. A crashed host otherwise leaves a sushi process holding GPU memory.
#
# Hermetic: headless over an EMPTY --model-dir loads no model, so no GPU lock is needed. The "parent" is a `sleep`.
#
# Usage: ./tests/test_parent_pid.sh [port]

set -u

PORT="${1:-21371}"
BINARY="${BINARY:-./zig-out/bin/sushi}"
PASS=0
FAIL=0

check() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then
        PASS=$((PASS + 1)); echo "  PASS $desc"
    else
        FAIL=$((FAIL + 1)); echo "  FAIL $desc"
    fi
}

if [ ! -x "$BINARY" ]; then
    echo "[fail] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[fail] port $PORT is busy — pass a free one"
    exit 1
fi

EMPTY_DIR="$(mktemp -d)"
LOG="$(mktemp)"
PIDS=()
cleanup() {
    for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; wait "$p" 2>/dev/null; done
    rm -rf "$EMPTY_DIR" "$LOG"
}
trap cleanup EXIT

# Waits up to $2 seconds for pid $1 to exit; sets RC to its exit status, or "alive".
wait_exit() {
    local pid="$1" secs="$2"
    for _ in $(seq 1 $((secs * 10))); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null
            RC=$?
            return
        fi
        sleep 0.1
    done
    RC=alive
}

serve() {
    "$BINARY" --serve --model-dir "$EMPTY_DIR" --host 127.0.0.1 --port "$PORT" --log-file off "$@" > "$LOG" 2>&1 &
    SUSHI=$!
    PIDS+=("$SUSHI")
}

up() {
    for _ in $(seq 1 60); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        kill -0 "$SUSHI" 2>/dev/null || return 1
        sleep 0.5
    done
    return 1
}

echo "--parent-pid watchdog (port $PORT)"

echo "[1/3] sushi serves while its parent lives and exits cleanly once it dies"
sleep 300 &
PARENT=$!
PIDS+=("$PARENT")
serve --parent-pid "$PARENT"
if up; then
    sleep 2
    check "still serving while the parent lives" "$(kill -0 "$SUSHI" 2>/dev/null && echo 1 || echo 0)"
    kill "$PARENT"
    wait "$PARENT" 2>/dev/null
    wait_exit "$SUSHI" 10
    check "exits within 10 s of the parent's death, status 0 (got $RC)" "$([ "$RC" = 0 ] && echo 1 || echo 0)"
    check "logs the reason" "$(grep -q "\[parent-pid\] $PARENT is gone" "$LOG" && echo 1 || echo 0)"
else
    check "server came up" 0
    sed 's/^/    /' "$LOG"
fi

echo "[2/3] a parent that is already gone stops the boot"
sleep 0 &
DEAD=$!
wait "$DEAD"
serve --parent-pid "$DEAD"
wait_exit "$SUSHI" 10
check "exits within 10 s (got $RC)" "$([ "$RC" != alive ] && echo 1 || echo 0)"
check "logs the reason" "$(grep -q "\[parent-pid\] $DEAD is gone" "$LOG" && echo 1 || echo 0)"

echo "[3/3] a pid that is not a pid is refused by name"
serve --parent-pid host
wait_exit "$SUSHI" 10
check "exits non-zero (got $RC)" "$([ "$RC" != alive ] && [ "$RC" != 0 ] && echo 1 || echo 0)"
check "names the flag" "$(grep -q -- "--parent-pid" "$LOG" && echo 1 || echo 0)"

echo ""
echo "$PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]

#!/bin/bash
# sushi-test: tier=live boot=api
# The Mac stays awake only while sushi has work in flight.
# Attaches to the shared api boot, where run.sh runs it last: it asserts the server is idle, so nothing else may be
# generating on that boot. The --no-prevent-sleep opt-out is a launch flag and has no arm here.

set -u

. "$(dirname "$0")/lib/boot.sh"
sushi_attach
BASE=$SUSHI_URL
PASS=0
FAIL=0
TOTAL=0

command -v pmset >/dev/null 2>&1 || { echo "SKIP $(basename "$0"): pmset not available (macOS only)"; exit 3; }

SERVER_PID=${SUSHI_PID:-$(lsof -ti "tcp:${SUSHI_URL##*:}" -sTCP:LISTEN 2>/dev/null | head -1)}
[ -n "$SERVER_PID" ] || { echo "FAIL: cannot find the server pid (SUSHI_PID unset, nothing listening on $SUSHI_URL)"; exit 1; }

# The type also matches powerd, and the name matches any OTHER sushi
# instance on the box (a live server generating concurrently would false-fail
# the idle arms), so match the assertion to OUR pid: pmset prints
# every assertion under a "pid NNN(process):" owner line.
NAME="sushi is generating"

run_test() {
    TOTAL=$((TOTAL + 1))
    if [ "$2" = PASS ]; then PASS=$((PASS + 1)); echo "  PASS: $1"
    else FAIL=$((FAIL + 1)); echo "  FAIL: $1 — $3"; fi
}

held() {
    pmset -g assertions 2>/dev/null \
        | awk -v pid="pid $SERVER_PID(" -v name="$NAME" \
              'index($0, pid) && index($0, name) { found = 1 } END { print found ? 1 : 0 }'
}

# wait_released <seconds>: polls until the assertion is gone; the earlier scripts' last requests may still be unwinding.
wait_released() {
    local i
    for i in $(seq 1 $(($1 * 2))); do
        [ "$(held)" = "0" ] && return 0
        sleep 0.5
    done
    return 1
}

generate() {
    curl -s -N -X POST "$BASE/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"sushi\",\"messages\":[{\"role\":\"user\",\"content\":\"$SUSHI_NONCE Write a long story about a lighthouse keeper.\"}],\"max_tokens\":800,\"stream\":true}" \
        -o /dev/null
}

echo "=== idle-sleep inhibition ($BASE, pid $SERVER_PID) ==="

wait_released 60 && run_test "idle server holds no assertion" PASS "" \
                 || run_test "idle server holds no assertion" FAIL "assertion still held 60s after the last script's request"

generate &
REQ=$!
SEEN=0
for _ in $(seq 1 1200); do
    [ "$(held)" = "1" ] && SEEN=1
    kill -0 $REQ 2>/dev/null || break
    [ "$SEEN" = "1" ] && break
    sleep 0.1
done
wait $REQ
[ "$SEEN" = "1" ] && run_test "assertion held during generation" PASS "" \
                  || run_test "assertion held during generation" FAIL "never held while a request was in flight"

wait_released 30 && run_test "assertion released once the server goes idle" PASS "" \
                 || run_test "assertion released once the server goes idle" FAIL "still held 30s after the last token"

# The boot lines are older than the attach point, so this reads the whole boot log; a log-order check avoids
# racing the short startup load.
ARM=$(grep -n -m1 'idle-sleep assertion held' "$SUSHI_LOG" | cut -d: -f1)
READY=$(grep -n -m1 'Model ready' "$SUSHI_LOG" | cut -d: -f1)
if [ -n "$ARM" ] && [ -n "$READY" ] && [ "$ARM" -lt "$READY" ]; then
    run_test "startup model load is covered" PASS ""
else
    run_test "startup model load is covered" FAIL "arm=${ARM:-none} ready=${READY:-none}"
fi

echo
echo "sleep-inhibit: $PASS/$TOTAL passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1

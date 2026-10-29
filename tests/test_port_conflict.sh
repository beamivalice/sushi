#!/usr/bin/env bash
# sushi-test: tier=host
. "$(dirname "$0")/private_cache.sh"
set -euo pipefail

BIN=${SUSHI_BIN:-zig-out/bin/sushi}
PORT=${PORT:-23817}
FAILED=""

err() { echo "FAIL: $*" >&2; FAILED=1; }
die() { echo "FAIL: $*" >&2; exit 1; }

[ -x "$BIN" ] || die "missing binary $BIN — build with .zig-toolchain/zig build -Doptimize=ReleaseFast"
command -v python3 >/dev/null || die "python3 not found"
command -v lsof >/dev/null || die "lsof not found"
[ "$PORT" -gt 20000 ] || die "PORT $PORT must be above 20000"
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    die "port $PORT is already busy; rerun with PORT=<free port above 20000>"
fi

WORK=$(mktemp -d)
HOLDER_PID=""
cleanup() {
    if [ -n "$HOLDER_PID" ] && kill -0 "$HOLDER_PID" 2>/dev/null; then
        kill "$HOLDER_PID" 2>/dev/null || true
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT
mkdir -p "$WORK/home" "$WORK/model"

wait_ready() {
    for _ in $(seq 1 50); do
        [ -e "$1" ] && return 0
        kill -0 "$HOLDER_PID" 2>/dev/null || die "port holder exited early: $(cat "$2")"
        sleep 0.1
    done
    die "port holder never became ready: $(cat "$2")"
}

start_listener() {
    rm -f "$WORK/ready"
    python3 -m http.server "$1" --bind 127.0.0.1 >"$WORK/holder.log" 2>&1 &
    HOLDER_PID=$!
    for _ in $(seq 1 50); do
        if lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; then
            return 0
        fi
        kill -0 "$HOLDER_PID" 2>/dev/null || die "listener on 127.0.0.1:$1 exited early: $(cat "$WORK/holder.log")"
        sleep 0.1
    done
    die "listener did not come up on 127.0.0.1:$1"
}

start_bound_socket() {
    rm -f "$WORK/ready"
    python3 -c "
import socket, sys, time
s = socket.socket()
s.bind(('127.0.0.1', int(sys.argv[1])))
open(sys.argv[2], 'w').close()
while True:
    time.sleep(1)
" "$1" "$WORK/ready" >"$WORK/holder.log" 2>&1 &
    HOLDER_PID=$!
    wait_ready "$WORK/ready" "$WORK/holder.log"
}

stop_holder() {
    if [ -n "$HOLDER_PID" ]; then
        kill "$HOLDER_PID" 2>/dev/null || true
        wait "$HOLDER_PID" 2>/dev/null || true
        HOLDER_PID=""
    fi
}

start_model_free_sushi() {
    HOME="$WORK/home" "$BIN" serve --host 127.0.0.1 --port "$1" </dev/null >"$WORK/first.log" 2>&1 &
    HOLDER_PID=$!
    for _ in $(seq 1 100); do
        if lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; then
            return 0
        fi
        kill -0 "$HOLDER_PID" 2>/dev/null || die "first sushi exited early: $(cat "$WORK/first.log")"
        sleep 0.1
    done
    die "first sushi never listened on $1"
}

expect_refusal() {
    local desc=$1
    shift
    HOME="$WORK/home" "$@" </dev/null >"$WORK/out.log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 100); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.1
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        err "$desc: still running after 10s — expected a refusal before serving"
        return 0
    fi
    local rc=0
    wait "$pid" || rc=$?
    if [ "$rc" -eq 0 ]; then
        err "$desc: exited 0 — expected non-zero"
        return 0
    fi
    if ! grep -q "port $PORT is already in use" "$WORK/out.log"; then
        err "$desc: missing 'port $PORT is already in use' in: $(cat "$WORK/out.log")"
        return 0
    fi
    if grep -q "Server listening" "$WORK/out.log"; then
        err "$desc: started serving instead of refusing"
        return 0
    fi
    echo "ok: $desc"
}

start_listener "$PORT"
expect_refusal "sushi serve default bind refuses a busy port" "$BIN" serve --port "$PORT"
expect_refusal "sushi run refuses a busy port before loading" "$BIN" run "$WORK/model" --port "$PORT"
expect_refusal "sushi serve --host 0.0.0.0 refuses a busy port" "$BIN" serve --host 0.0.0.0 --port "$PORT"
stop_holder

start_bound_socket "$PORT"
expect_refusal "sushi refuses a socket occupying its bind address without a listener" "$BIN" serve --port "$PORT"
stop_holder

start_model_free_sushi "$PORT"
expect_refusal "second sushi refuses a port the first sushi serves" "$BIN" serve --port "$PORT"
if ! kill -0 "$HOLDER_PID" 2>/dev/null; then
    err "first sushi died after the second's refusal"
fi
stop_holder

if [ -n "$FAILED" ]; then
    echo "FAIL: port conflict refusals" >&2
    exit 1
fi
echo "PASS: port conflict refusals"

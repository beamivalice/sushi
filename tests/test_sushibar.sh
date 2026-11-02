#!/bin/bash
# sushi-test: tier=host
# sushibar, the menu bar app: argument and session refusals by name, no MLX in its link line, and one short run
# under a private HOME on a private port: Start launches a headless sushi beside it, the pid files and /health agree,
# Stop ends only that server, a second sushibar defers to the first. The run drives sushibar's own code paths through
# SUSHIBAR_TEST_POLLS / SUSHIBAR_TEST_ACTIONS (docs/envvar.md), never the UI; it shows in the menu bar for ~20 s.
# Usage: ./tests/test_sushibar.sh [port]
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BAR="${SUSHIBAR_BIN:-$ROOT/zig-out/bin/sushibar}"
PORT="${1:-18801}"
PASS=0
FAIL=0

check() {
    if [ "$2" = "1" ]; then PASS=$((PASS + 1)); echo "  PASS $1"; else FAIL=$((FAIL + 1)); echo "  FAIL $1"; fi
}
yes_if() { if eval "$1"; then echo 1; else echo 0; fi; }

[ -x "$BAR" ] || { echo "FAIL: $BAR missing (zig build -Doptimize=ReleaseFast)"; exit 1; }

echo "== refusals =="
out=$("$BAR" --port abc 2>&1); rc=$?
check "--port abc exits non-zero and names --port" "$(yes_if '[ $rc -ne 0 ] && echo "$out" | grep -q -- "--port"')"
out=$("$BAR" --bogus 2>&1); rc=$?
check "an unknown argument exits non-zero and names it" "$(yes_if '[ $rc -ne 0 ] && echo "$out" | grep -q -- "--bogus"')"
out=$("$BAR" --port 2>&1); rc=$?
check "--port without a value exits non-zero" "$(yes_if '[ $rc -ne 0 ]')"
out=$("$BAR" --help 2>&1); rc=$?
check "--help lists --port and exits 0" "$(yes_if '[ $rc -eq 0 ] && echo "$out" | grep -q -- "--port"')"
out=$(SSH_CONNECTION="1.2.3.4 5 6.7.8.9 22" "$BAR" 2>&1); rc=$?
check "an ssh login exits non-zero saying there is no GUI session" "$(yes_if '[ $rc -ne 0 ] && echo "$out" | grep -qi "GUI session"')"

echo "== link line =="
stray=$(otool -L "$BAR" | tail -n +2 | grep -E 'libmlx|libwebp|libjinja|libc\+\+')
check "no libmlx, libmlxc, libwebp or libc++ is linked" "$(yes_if '[ -z "$stray" ]')"

echo "== run on port $PORT =="
SUSHI_BIN="$(dirname "$BAR")/sushi"
if [ ! -x "$SUSHI_BIN" ]; then echo "SKIP: no sushi beside sushibar"; [ "$FAIL" -eq 0 ] && exit 3 || exit 1; fi
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "FAIL: port $PORT is busy; pass a free one"
    exit 1
fi
WORK="$(mktemp -d)"
export HOME="$WORK/home"
export SUSHI_PREFIX_CACHE_DIR="$WORK/kv"
mkdir -p "$HOME"
LOG="$WORK/bar.err"
cleanup() {
    for f in "$HOME/.sushi/sushibar-server.pid" "$HOME/.sushi/sushibar.pid"; do
        [ -f "$f" ] && kill "$(cut -d' ' -f1 "$f")" 2>/dev/null
    done
    rm -rf "$WORK"
}
trap cleanup EXIT

SUSHIBAR_TEST_POLLS=16 SUSHIBAR_TEST_ACTIONS=start@2,stop@10 "$BAR" --port "$PORT" 2> "$LOG"
rc=$?
check "the terminal copy exits 0 after detaching" "$(yes_if '[ $rc -eq 0 ]')"
BARPID=$(cut -d' ' -f1 "$HOME/.sushi/sushibar.pid" 2>/dev/null)
check "the pid file names a live sushibar" "$(yes_if '[ -n "$BARPID" ] && kill -0 "$BARPID" 2>/dev/null')"
second=$("$BAR" --port "$PORT" 2>&1); rc=$?
check "a second sushibar says it is already running and exits 0" "$(yes_if '[ $rc -eq 0 ] && echo "$second" | grep -q "already running"')"

health=000
for _ in $(seq 1 40); do
    health=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
    [ "$health" = 200 ] && break
    sleep 0.5
done
check "Start launched a headless sushi: /health is 200" "$(yes_if '[ "$health" = 200 ]')"
SRVPID=$(cut -d' ' -f1 "$HOME/.sushi/sushibar-server.pid" 2>/dev/null)
check "the server pid file names a live sushi" "$(yes_if '[ -n "$SRVPID" ] && kill -0 "$SRVPID" 2>/dev/null')"

for _ in $(seq 1 60); do kill -0 "$BARPID" 2>/dev/null || break; sleep 0.5; done
check "sushibar quit after its polls" "$(yes_if '! kill -0 "$BARPID" 2>/dev/null')"
check "its pid file is gone" "$(yes_if '[ ! -f "$HOME/.sushi/sushibar.pid" ]')"
check "the icon is the outline offline and the colour one once the server answers" "$(yes_if 'grep -q "poll 1 .*icon=outline" "$LOG" && grep -q "Server: running, no model loaded.*icon=color" "$LOG" && tail -n 1 "$LOG" | grep -q "icon=outline"')"
check "the first poll reads not running with Start enabled" "$(yes_if 'grep -q "poll 1 title=\"\" status=\"Server: not running\" start=1 stop=0" "$LOG"')"
check "an idle server reads an icon-only title with Stop enabled" "$(yes_if 'grep -q "title=\"\" status=\"Server: running, no model loaded\" start=0 stop=1" "$LOG"')"
check "Stop ended the server it started" "$(yes_if '! kill -0 "$SRVPID" 2>/dev/null')"
check "the last poll reads not running again" "$(yes_if 'tail -n 1 "$LOG" | grep -q "poll 16 .*Server: not running"')"
check "a stopped server leaves no pid file entry alive" "$(yes_if '[ ! -f "$HOME/.sushi/sushibar-server.pid" ]')"

echo "== $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]

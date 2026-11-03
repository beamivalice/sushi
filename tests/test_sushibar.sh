#!/bin/bash
# sushi-test: tier=host
# sushibar, the menu bar app: argument and session refusals by name, no MLX in its link line, and runs under a
# private HOME on private ports (default 18801-18807): Start launches a headless sushi beside it, the pid files and
# /health agree, Stop ends only that server, a second sushibar defers; a sushi that sushibar did NOT start is found
# by the libproc scan, listed in the Port data, switched to, saved, and stopped only through the confirm path (never
# without the confirm seam); a saved port is used on relaunch and an explicit --port beats it; a port held by a
# non-sushi program blocks Start; a newer release (a fake GitHub on SUSHI_UPDATE_API) shows "Update to sushi X" and
# a Homebrew keg shows the brew command. The runs drive sushibar's own code paths through SUSHIBAR_TEST_POLLS /
# SUSHIBAR_TEST_ACTIONS (docs/envvar.md), never the UI; each shows in the menu bar for ~10 s. The update round trip
# itself (download, swap, relaunch) is tests/test_self_update.sh's and the unit tests'; here it stops at "available".
# Usage: ./tests/test_sushibar.sh [port]
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BAR="${SUSHIBAR_BIN:-$ROOT/zig-out/bin/sushibar}"
PORT="${1:-18801}"
FPORT=$((PORT + 1))
TPORT=$((PORT + 2))
UPORT=$((PORT + 5))
APORT=$((PORT + 6))
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

SUSHI_BIN="$(dirname "$BAR")/sushi"
if [ ! -x "$SUSHI_BIN" ]; then echo "SKIP: no sushi beside sushibar"; [ "$FAIL" -eq 0 ] && exit 3 || exit 1; fi
for p in "$PORT" "$FPORT" "$TPORT" "$UPORT" "$APORT"; do
    if lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "FAIL: port $p is busy; pass a free base port"
        exit 1
    fi
done

WORK="$(cd "$(mktemp -d)" && pwd -P)"
export HOME="$WORK/home"
export SUSHI_PREFIX_CACHE_DIR="$WORK/kv"
mkdir -p "$HOME"
LOG="$WORK/bar.err"
FPID=""
TPID=""
APID=""
cleanup() {
    for f in "$HOME/.sushi/sushibar-server.pid" "$HOME/.sushi/sushibar.pid"; do
        [ -f "$f" ] && kill "$(cut -d' ' -f1 "$f")" 2>/dev/null
    done
    for p in $FPID $TPID $APID; do kill "$p" 2>/dev/null; done
    rm -rf "$WORK"
}
trap cleanup EXIT

wait_health() {
    local port="$1" code=000
    for _ in $(seq 1 40); do
        code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" 2>/dev/null)
        [ "$code" = 200 ] && break
        sleep 0.5
    done
    echo "$code"
}

# run_bar <binary> <polls> <actions> [args...]: one detached sushibar run through the seams, waited for.
run_bar() {
    local bin="$1" polls="$2" actions="$3"
    shift 3
    : > "$LOG"
    SUSHIBAR_TEST_POLLS="$polls" SUSHIBAR_TEST_ACTIONS="$actions" "$bin" "$@" 2> "$LOG"
    BARRC=$?
    local pid
    pid=$(cut -d' ' -f1 "$HOME/.sushi/sushibar.pid" 2>/dev/null)
    for _ in $(seq 1 80); do [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
}

echo "== run on port $PORT =="
SUSHIBAR_TEST_POLLS=16 SUSHIBAR_TEST_ACTIONS=start@2,stop@10 "$BAR" --port "$PORT" 2> "$LOG"
rc=$?
check "the terminal copy exits 0 after detaching" "$(yes_if '[ $rc -eq 0 ]')"
BARPID=$(cut -d' ' -f1 "$HOME/.sushi/sushibar.pid" 2>/dev/null)
check "the pid file names a live sushibar" "$(yes_if '[ -n "$BARPID" ] && kill -0 "$BARPID" 2>/dev/null')"
second=$("$BAR" --port "$PORT" 2>&1); rc=$?
check "a second sushibar says it is already running and exits 0" "$(yes_if '[ $rc -eq 0 ] && echo "$second" | grep -q "already running"')"

health=$(wait_health "$PORT")
check "Start launched a headless sushi: /health is 200" "$(yes_if '[ "$health" = 200 ]')"
SRVPID=$(cut -d' ' -f1 "$HOME/.sushi/sushibar-server.pid" 2>/dev/null)
check "the server pid file names a live sushi" "$(yes_if '[ -n "$SRVPID" ] && kill -0 "$SRVPID" 2>/dev/null')"

for _ in $(seq 1 60); do kill -0 "$BARPID" 2>/dev/null || break; sleep 0.5; done
check "sushibar quit after its polls" "$(yes_if '! kill -0 "$BARPID" 2>/dev/null')"
check "its pid file is gone" "$(yes_if '[ ! -f "$HOME/.sushi/sushibar.pid" ]')"
check "the icon is the outline offline and the colour one once the server answers" "$(yes_if 'grep -q "poll 1 .*icon=outline" "$LOG" && grep -q "running, no model loaded.*icon=color" "$LOG" && tail -n 1 "$LOG" | grep -q "icon=outline"')"
check "the first poll reads not running with Start enabled, the port named" "$(yes_if 'grep -q "poll 1 title=\"\" status=\"Port '"$PORT"': not running\" start=1 stop=0" "$LOG"')"
check "an idle server reads an icon-only title with Stop enabled" "$(yes_if 'grep -q "title=\"\" status=\"Port '"$PORT"': running, no model loaded\" start=0 stop=1" "$LOG"')"
check "the header says Stopped with a Start toggle, then Running with a Stop toggle" "$(yes_if 'grep -q "poll 1 .*pill=\"Stopped\" line1=\"Not running\" line2=\"Port '"$PORT"'\" toggle=\"Start Server\"" "$LOG" && grep -q "pill=\"Running\" line1=\"No model loaded\".*toggle=\"Stop Server\"" "$LOG"')"
check "the Port data lists the running server, checked" "$(yes_if 'grep -q "ports=\"'"$PORT"' — no model\*;" "$LOG"')"
check "Stop ended the server it started" "$(yes_if '! kill -0 "$SRVPID" 2>/dev/null')"
check "the last poll reads not running again" "$(yes_if 'tail -n 1 "$LOG" | grep -q "poll 16 .*Port '"$PORT"': not running"')"
check "a stopped server leaves no pid file entry alive" "$(yes_if '[ ! -f "$HOME/.sushi/sushibar-server.pid" ]')"

echo "== a sushi sushibar did not start, on port $FPORT =="
start_foreign() {
    "$SUSHI_BIN" --serve --port "$FPORT" > /dev/null 2>&1 &
    FPID=$!
    [ "$(wait_health "$FPORT")" = 200 ]
}
start_foreign
check "a headless sushi started outside sushibar answers on $FPORT" "$(yes_if 'kill -0 "$FPID" 2>/dev/null')"
run_bar "$BAR" 10 "confirm@1,port$FPORT@2,stop@6" --port "$PORT"
check "the Port data lists the foreign sushi found by the scan" "$(yes_if 'grep -q "ports=\".*'"$FPORT"' — no model;" "$LOG"')"
check "choosing it switches the watched port and names it in the status" "$(yes_if 'grep -q "poll [0-9]* .*Port '"$FPORT"': running, no model loaded\" start=0 stop=1" "$LOG"')"
check "a sushi sushibar did not start offers Stop with the confirm ellipsis" "$(yes_if 'grep -q "toggle=\"Stop Server…\"" "$LOG"')"
check "the choice is saved" "$(yes_if 'grep -q "\"port\":'"$FPORT"'" "$HOME/.sushi/sushibar.json"')"
for _ in $(seq 1 20); do kill -0 "$FPID" 2>/dev/null || break; sleep 0.5; done
check "Stop with the confirm answered ended that sushi" "$(yes_if '! kill -0 "$FPID" 2>/dev/null')"
wait "$FPID" 2>/dev/null

start_foreign
run_bar "$BAR" 6 "stop@2" --port "$FPORT"
check "Stop without a confirm leaves a sushi sushibar did not start running" "$(yes_if 'kill -0 "$FPID" 2>/dev/null')"
check "that status offered Stop" "$(yes_if 'grep -q "poll 1 .*Port '"$FPORT"': running.*stop=1" "$LOG"')"
kill "$FPID" 2>/dev/null; wait "$FPID" 2>/dev/null

echo "== the Port field: an invalid number is refused in place, a valid one switches =="
run_bar "$BAR" 8 "type99999@2,type$FPORT@5" --port "$PORT"
check "an invalid port turns the field red and keeps the watched port" "$(yes_if 'grep -q "poll [2-4] .*Port '"$PORT"':.*field_invalid=1" "$LOG"')"
check "a valid port typed in the field switches the watched port" "$(yes_if 'grep -q "poll [6-8] .*Port '"$FPORT"':.*field_invalid=0" "$LOG"')"

echo "== the chosen port persists, --port beats it =="
run_bar "$BAR" 2 ""
check "a saved port is used on relaunch" "$(yes_if 'grep -q "poll 1 .*Port '"$FPORT"':" "$LOG"')"
run_bar "$BAR" 2 "" --port "$PORT"
check "an explicit --port beats the saved one" "$(yes_if 'grep -q "poll 1 .*Port '"$PORT"':" "$LOG"')"

echo "== a port held by another program =="
mkdir -p "$WORK/empty"
(cd "$WORK/empty" && exec python3 -m http.server "$TPORT" --bind 127.0.0.1 > /dev/null 2>&1) &
TPID=$!
for _ in $(seq 1 40); do curl -s -o /dev/null "http://127.0.0.1:$TPORT/" && break; sleep 0.25; done
run_bar "$BAR" 3 "" --port "$TPORT"
check "the status says the port is in use and Start is disabled" "$(yes_if 'grep -q "poll 2 .*Port '"$TPORT"': in use by another program\" start=0 stop=0" "$LOG"')"
kill "$TPID" 2>/dev/null; wait "$TPID" 2>/dev/null

echo "== a newer pack revision on the loaded model =="
SPORT=$((PORT + 3))
cat > "$WORK/stub_models.py" <<'EOF'
import http.server, json, sys

PACK = {"current": "1.0", "latest": "2.0", "command": "sushi pull deepseek-v4.1-flash"}

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith("/metrics.json"):
            body = {}
        elif self.path.startswith("/v1/models"):
            body = {"data": [{"id": "stub-pack", "state": "ready", "meta": {"pack_update": PACK}}]}
        else:
            self.send_response(404); self.end_headers(); return
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
EOF
python3 "$WORK/stub_models.py" "$SPORT" > /dev/null 2>&1 &
TPID=$!
for _ in $(seq 1 40); do curl -s -o /dev/null "http://127.0.0.1:$SPORT/v1/models" && break; sleep 0.25; done
run_bar "$BAR" 3 "" --port "$SPORT"
check "the header card names the pack update under the model" "$(yes_if 'grep -q "poll [23] .*pack=\"Pack 2.0 available (you have 1.0)\"" "$LOG"')"
check "the menu offers to copy the pull command" "$(yes_if 'grep -q "packcopy=\"Copy “sushi pull deepseek-v4.1-flash”\"" "$LOG"')"
kill "$TPID" 2>/dev/null; wait "$TPID" 2>/dev/null

echo "== version and update =="
CUR="$("$SUSHI_BIN" --version 2>/dev/null | head -1 | awk '{print $2}')"
stage_install() {
    local dir="$1"
    mkdir -p "$dir/lib"
    cp "$SUSHI_BIN" "$dir/sushi"
    cp "$BAR" "$dir/sushibar"
    cp "$ROOT"/lib/mlx/lib/*.dylib "$ROOT"/lib/mlx/lib/mlx.metallib "$dir/lib/"
    install_name_tool -change @rpath/libmlxc.dylib @executable_path/lib/libmlxc.dylib "$dir/sushi" 2>/dev/null
    install_name_tool -change @rpath/libmlx.dylib @loader_path/libmlx.dylib "$dir/lib/libmlxc.dylib" 2>/dev/null
    install_name_tool -add_rpath @loader_path "$dir/lib/libmlx.dylib" 2>/dev/null
    for f in "$dir/lib/"*.dylib "$dir/sushi" "$dir/sushibar"; do codesign --force --sign - "$f" 2>/dev/null; done
}
mkdir -p "$WORK/www/repos/beamivalice/sushi"
cat > "$WORK/www/repos/beamivalice/sushi/releases" <<EOF
[{"tag_name":"v99.0.0","html_url":"https://example.invalid/releases/v99.0.0","draft":false,"prerelease":false,
  "assets":[{"name":"sushi-bin-macos-arm64.tar.gz","browser_download_url":"http://127.0.0.1:$APORT/dl/a"},
            {"name":"sushi-bin-macos-arm64.tar.gz.sha256","browser_download_url":"http://127.0.0.1:$APORT/dl/b"}]}]
EOF
python3 -m http.server "$APORT" --bind 127.0.0.1 --directory "$WORK/www" > /dev/null 2>&1 &
APID=$!
for _ in $(seq 1 40); do curl -sf "http://127.0.0.1:$APORT/repos/beamivalice/sushi/releases" > /dev/null && break; sleep 0.25; done
export SUSHI_UPDATE_API="http://127.0.0.1:$APORT"

INST="$WORK/inst/sushi-macos-arm64"
stage_install "$INST"
run_bar "$INST/sushibar" 8 "check@2" --port "$UPORT"
check "the menu shows the sushi beside sushibar by version" "$(yes_if 'grep -q "version=\"'"$CUR"'" "$LOG"')"
check "a newer release shows the update item with its mark" "$(yes_if 'grep -q "update=\"Update to sushi 99.0.0\"" "$LOG"')"
check "the button carries the dot while an update waits" "$(yes_if 'grep -q "title=\"•\"" "$LOG"')"
check "no server on the port: the CLI path is chosen (no brew command shown)" "$(yes_if '! grep -q "brew upgrade" "$LOG"')"

KEG="$WORK/Cellar/sushi/1.0/libexec"
stage_install "$KEG"
run_bar "$KEG/sushibar" 6 "check@2" --port "$UPORT"
check "a Homebrew keg shows the brew command instead of an update" "$(yes_if 'grep -q "update=\"Copy “brew upgrade sushi”\"" "$LOG"')"

echo "== $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]

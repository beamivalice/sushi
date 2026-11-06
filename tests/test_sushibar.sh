#!/bin/bash
# sushi-test: tier=host
# sushibar, the menu bar app: argument and session refusals by name, no MLX in its link line, and runs under a
# private HOME on private ports (default 18801-18812): Start launches a headless sushi beside it and loads nothing, the pid files and
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
HPORT=$((PORT + 7))
SPORT=$((PORT + 8))
LPORT=$((PORT + 11))
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
for p in "$PORT" "$FPORT" "$TPORT" "$UPORT" "$APORT" "$HPORT" "$SPORT" "$LPORT"; do
    if lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "FAIL: port $p is busy; pass a free base port"
        exit 1
    fi
done

WORK="$(cd "$(mktemp -d)" && pwd -P)"
export HOME="$WORK/home"
export SUSHI_PREFIX_CACHE_DIR="$WORK/kv"
mkdir -p "$HOME/.sushi"
# A private default port, so no run here ever lists or falls back to a real server's port.
seed_default() { echo '{"server":{"port":'"$((PORT + 9))"'}}' > "$HOME/.sushi/model-settings.json"; }
seed_default
LOG="$WORK/bar.err"
FPID=""
TPID=""
APID=""
HPID=""
SPID=""
cleanup() {
    for f in "$HOME"/.sushi/sushibar-server-*.pid "$HOME/.sushi/sushibar.pid"; do
        [ -f "$f" ] && kill "$(cut -d' ' -f1 "$f")" 2>/dev/null
    done
    for p in $FPID $TPID $APID $HPID $SPID; do kill "$p" 2>/dev/null; done
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
SRVPID=$(cut -d' ' -f1 "$HOME/.sushi/sushibar-server-$PORT.pid" 2>/dev/null)
check "the server pid file names a live sushi" "$(yes_if '[ -n "$SRVPID" ] && kill -0 "$SRVPID" 2>/dev/null')"

for _ in $(seq 1 60); do kill -0 "$BARPID" 2>/dev/null || break; sleep 0.5; done
check "sushibar quit after its polls" "$(yes_if '! kill -0 "$BARPID" 2>/dev/null')"
check "its pid file is gone" "$(yes_if '[ ! -f "$HOME/.sushi/sushibar.pid" ]')"
check "the icon is the outline offline and the colour one once the server answers" "$(yes_if 'grep -q "poll 1 .*icon=outline" "$LOG" && grep -q "running, no model loaded.*icon=color" "$LOG" && tail -n 1 "$LOG" | grep -q "icon=outline"')"
check "the first poll reads not running with Start enabled, the port named" "$(yes_if 'grep -q "poll 1 title=\"\" status=\"Port '"$PORT"': not running\" start=1 stop=0" "$LOG"')"
check "an idle server reads an icon-only title with Stop enabled" "$(yes_if 'grep -q "title=\"\" status=\"Port '"$PORT"': running, no model loaded\" start=0 stop=1" "$LOG"')"
check "the header says Stopped with a Start toggle, then No model pointing at the Models page with a Stop toggle" "$(yes_if 'grep -q "poll 1 .*pill=\"Stopped\" line1=\"Ready to start · port '"$PORT"'\" line2=\"No model to load yet\" toggle=\"Start Server\"" "$LOG" && grep -q "pill=\"No model\" line1=\"Open Models to load one\".*toggle=\"Stop Server\"" "$LOG"')"
check "Load is off with its reason before the server answers and with no model to load" "$(yes_if 'grep -q "poll 1 .*load=\"Load\" load_on=0 why=\"Start the server to load a model\"" "$LOG" && grep -q "pill=\"No model\".*load=\"Load\" load_on=0 why=\"No model to load yet: pick one in Models\"" "$LOG"')"
check "the Port data lists the running server, checked" "$(yes_if 'grep -q "ports=\".*'"$PORT"' — no model\*;" "$LOG"')"
check "Stop ended the server it started" "$(yes_if '! kill -0 "$SRVPID" 2>/dev/null')"
check "the last poll reads not running again" "$(yes_if 'tail -n 1 "$LOG" | grep -q "poll 16 .*Port '"$PORT"': not running"')"
check "a stopped server leaves no pid file entry alive" "$(yes_if '[ ! -f "$HOME/.sushi/sushibar-server-'"$PORT"'.pid" ]')"
check "the stat tiles read the web UI's words" "$(yes_if 'grep -q "stats=\"GEN TOK/S|PROMPT TOK/S|GPU MHZ\"" "$LOG"')"
check "nothing the menu prints uses a retired word (decode, prefill, reply, prefix cache, guess)" "$(yes_if '! grep "^sushibar: poll " "$LOG" | grep -qiE "decode|prefill|repl|prefix cache|guess"')"

echo "== a sushi sushibar did not start, on port $FPORT =="
start_foreign() {
    "$SUSHI_BIN" --serve --port "$FPORT" > /dev/null 2>&1 &
    FPID=$!
    [ "$(wait_health "$FPORT")" = 200 ]
}
start_foreign
check "a headless sushi started outside sushibar answers on $FPORT" "$(yes_if 'kill -0 "$FPID" 2>/dev/null')"
run_bar "$BAR" 2 "" --port "$PORT"
check "a sushi nobody added is found by the scan but not listed in the Port data" "$(yes_if '! grep -q "ports=\".*'"$FPORT"' —" "$LOG"')"
echo '{"port":'"$PORT"',"ports":['"$FPORT"']}' > "$HOME/.sushi/sushibar.json"
run_bar "$BAR" 11 "confirm@1,models$FPORT@2,port$FPORT@4,stop@8" --port "$PORT"
check "an added port is listed with the state the scan finds, its model asked when the submenu opens" "$(yes_if 'grep -q "ports=\".*'"$FPORT"' — no model;" "$LOG"')"
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

echo "== a server that wants its API key even from loopback =="
"$SUSHI_BIN" --serve --port "$FPORT" --api-key sb-test-key --api-key-strict > /dev/null 2>&1 &
FPID=$!
for _ in $(seq 1 40); do [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$FPORT/metrics.json")" = 401 ] && break; sleep 0.5; done
code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$FPORT/metrics.json")
check "the strict server refuses a request without its key" "$(yes_if '[ "$code" = 401 ]')"
run_bar "$BAR" 3 "" --port "$FPORT"
check "sushibar sends the key the server's argv gives it and reads the server as running" "$(yes_if 'grep -q "poll [23] .*Port '"$FPORT"': running, no model loaded\" .*badge=0" "$LOG" && ! grep -q "Key needed" "$LOG"')"
kill "$FPID" 2>/dev/null; wait "$FPID" 2>/dev/null
SB_TEST_KEY=sb-env-key "$SUSHI_BIN" --serve --port "$FPORT" --api-key-env SB_TEST_KEY --api-key-strict > /dev/null 2>&1 &
FPID=$!
for _ in $(seq 1 40); do [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$FPORT/metrics.json")" = 401 ] && break; sleep 0.5; done
run_bar "$BAR" 3 "" --port "$FPORT"
check "a key from --api-key-env is read from the server's own environment" "$(yes_if 'grep -q "poll [23] .*Port '"$FPORT"': running, no model loaded\"" "$LOG" && ! grep -q "Key needed" "$LOG"')"
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

echo "== added ports are kept, removable, and the default port is neither =="
# The default is a private port too: the watch falls back to it, and a port nothing of this run owns is never polled.
DPORT=$((PORT + 9))
APORT2=$((PORT + 10))
rm -f "$HOME/.sushi/sushibar.json"
echo '{"server":{"port":'"$DPORT"'}}' > "$HOME/.sushi/model-settings.json"
run_bar "$BAR" 9 "port$APORT2@2,remove$DPORT@4,remove$APORT2@6" --port "$PORT"
check "the default port is listed first and has no remove button" "$(yes_if 'grep -q "poll 1 .*ports=\"'"$DPORT"' — not running;" "$LOG" && ! grep -q "removable=\"[^\"]*'"$DPORT"'" "$LOG"')"
check "a port switched to is added, checked and removable" "$(yes_if 'grep -q "poll [3-4] .*ports=\"[^\"]*'"$APORT2"' — not running\*;.*removable=\"[^\"]*'"$APORT2"'" "$LOG"')"
check "removing the default port does nothing" "$(yes_if 'grep -q "poll [5-6] .*ports=\"'"$DPORT"' — not running;'"$APORT2"' — not running\*;" "$LOG"')"
check "removing the watched port forgets it and watches the default" "$(yes_if 'grep -q "poll [7-9] .*Port '"$DPORT"':.*ports=\"'"$DPORT"' — not running\*;" "$LOG" && ! grep -q "poll [7-9] .*'"$APORT2"' — not running" "$LOG"')"
check "the saved file keeps the watched port and no added one" "$(yes_if 'grep -q "\"port\":'"$DPORT"',\"ports\":\[\]" "$HOME/.sushi/sushibar.json"')"
start_foreign
run_bar "$BAR" 8 "remove$FPORT@4" --port "$FPORT"
check "a watched port that was never added, with a sushi on it, has a remove button" "$(yes_if 'grep -q "poll [1-3] .*Port '"$FPORT"':.*removable=\"[^\"]*'"$FPORT"'" "$LOG"')"
check "removing it moves the watch to the default and the sushi still listening there is no longer listed" "$(yes_if 'grep -q "poll [6-8] .*Port '"$DPORT"':.*ports=\"'"$DPORT"' — not running\*;\"" "$LOG" && ! grep -q "poll [6-8] .*ports=\"[^\"]*'"$FPORT"' —" "$LOG"')"
kill "$FPID" 2>/dev/null; wait "$FPID" 2>/dev/null
echo '{"port":'"$FPORT"'}' > "$HOME/.sushi/sushibar.json"
run_bar "$BAR" 2 ""
check "a file from before added ports still sets the watched port" "$(yes_if 'grep -q "poll 1 .*Port '"$FPORT"':" "$LOG"')"
rm -f "$HOME/.sushi/sushibar.json"
echo '{"server":{"port":'"$APORT2"'}}' > "$HOME/.sushi/model-settings.json"
run_bar "$BAR" 2 ""
check "server.port in the settings file is the default port: watched at first, listed first, not removable" "$(yes_if 'grep -q "poll 1 .*Port '"$APORT2"':.*ports=\"'"$APORT2"' — not running\*;" "$LOG" && ! grep -q "removable=\"[^\"]*'"$APORT2"'" "$LOG"')"
seed_default
rm -f "$HOME/.sushi/sushibar.json"

echo "== a port held by another program =="
mkdir -p "$WORK/empty"
(cd "$WORK/empty" && exec python3 -m http.server "$TPORT" --bind 127.0.0.1 > /dev/null 2>&1) &
TPID=$!
for _ in $(seq 1 40); do curl -s -o /dev/null "http://127.0.0.1:$TPORT/" && break; sleep 0.25; done
run_bar "$BAR" 3 "" --port "$TPORT"
check "the status says the port is in use and Start is disabled" "$(yes_if 'grep -q "poll 2 .*Port '"$TPORT"': in use by another program\" start=0 stop=0" "$LOG"')"
check "the header says why in an orange pill, and the icon carries a badge" "$(yes_if 'grep -q "poll 2 .*badge=1.*" "$LOG" && grep -q "poll 2 .*pill=\"Port in use\" line1=\"Another program holds this port\"" "$LOG"')"
kill "$TPID" 2>/dev/null; wait "$TPID" 2>/dev/null

echo "== Listen on LAN binds the servers started from now on =="
rm -f "$HOME/.sushi/sushibar.json"
run_bar "$BAR" 8 "lan@2,start@3" --port "$LPORT"
check "the toggle flips its state and notes it applies to the next Start" "$(yes_if 'grep -q "poll 1 .*lan=0" "$LOG" && grep -q "poll [3-8] .*lan=1" "$LOG" && grep -q "note=\"Next Start listens on the LAN" "$LOG"')"
check "the saved file keeps the LAN switch" "$(yes_if 'grep -q "\"lan\":true" "$HOME/.sushi/sushibar.json"')"
SRVPID=$(cut -d' ' -f1 "$HOME/.sushi/sushibar-server-$LPORT.pid" 2>/dev/null)
check "the started sushi carries --host 0.0.0.0" "$(yes_if '[ -n "$SRVPID" ] && ps -ww -o command= -p "$SRVPID" | grep -q -- "--host 0.0.0.0"')"
LAN_OK=0
for _ in $(seq 1 40); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$LPORT/health" 2>/dev/null)" = 200 ] && { LAN_OK=1; break; }
    sleep 0.5
done
check "a LAN-bound sushi still answers on loopback" "$(yes_if '[ "$LAN_OK" = 1 ]')"
run_bar "$BAR" 3 "" --port "$LPORT"
check "the header names the LAN bind" "$(yes_if 'grep -q "line2=\"Port '"$LPORT"' · LAN" "$LOG"')"
check "the LAN switch persists over relaunch" "$(yes_if 'grep -q "poll [123] .*lan=1" "$LOG"')"
kill "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null
rm -f "$HOME/.sushi/sushibar-server-$LPORT.pid"
run_bar "$BAR" 2 "lan@1" --port "$LPORT"
check "toggling LAN off leaves it out of the saved file" "$(yes_if '! grep -q "\"lan\"" "$HOME/.sushi/sushibar.json"')"
rm -f "$HOME/.sushi/sushibar.json"

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
check "a newer release shows the Update button, named in its tooltip" "$(yes_if 'grep -q "button=\"Update\" tip=\"Update to sushi 99.0.0\"" "$LOG"')"
check "the button carries the dot while an update waits" "$(yes_if 'grep -q "title=\"•\"" "$LOG"')"
check "no server on the port: the CLI path is chosen (the button is not a brew command)" "$(yes_if 'grep -q "button=\"Update\"" "$LOG" && ! grep -q "brew upgrade" "$LOG"')"

KEG="$WORK/Cellar/sushi/1.0/libexec"
stage_install "$KEG"
run_bar "$KEG/sushibar" 6 "check@2" --port "$UPORT"
check "a Homebrew keg shows one Upgrade button that copies the brew command" "$(yes_if 'grep -q "button=\"Upgrade\" tip=\"Copy “brew upgrade sushi”\"" "$LOG"')"

echo "== update failures show where the menu shows them =="
FAKE="$WORK/fake"
mkdir -p "$FAKE"
cp "$BAR" "$FAKE/sushibar"
long=$(printf 'x%.0s' $(seq 1 300))
cat > "$FAKE/sushi" <<EOF
#!/bin/sh
case "\$1" in
  --version) echo "sushi 1.0.0"; exit 0;;
  update)
    echo "── sushi 1.0.0 update, 2026-11-03 14:57 UTC"
    if [ "\${FAKE_MODE:-}" = "avail" ] && [ "\${2:-}" = "--check" ]; then echo "sushi 9.0.0 is available (this is 1.0.0): x"; exit 0; fi
    if [ "\${FAKE_MODE:-}" = "current" ] && [ "\${2:-}" = "--check" ]; then echo "sushi 1.0.0 is up to date"; exit 0; fi
    echo "update: disk full while unpacking $long"
    echo "trailing log line"
    exit 1;;
  --serve) exit 1;;
esac
exit 1
EOF
chmod +x "$FAKE/sushi"
rm -f "$HOME/.sushi/update-check.json"
run_bar "$FAKE/sushibar" 5 "check@2" --port "$UPORT"
check "a failed manual check shows its update: line in the note item, cut with an ellipsis" "$(yes_if 'grep -q "poll [3-5] .*note=\"update: disk full while unpacking x*…\" problem=1" "$LOG"')"
check "that note is not the trailing log line" "$(yes_if '! grep -q "note=\"trailing" "$LOG"')"
check "the Update button reads Checking… in the poll of the click and Retry once the check failed" "$(yes_if 'grep -q "poll 2 .*button=\"Checking…\"" "$LOG" && grep -q "poll [3-5] .*button=\"Retry\"" "$LOG"')"
FAKE_MODE=current run_bar "$FAKE/sushibar" 5 "check@2" --port "$UPORT"
check "the Update button reads Up to date once a check found nothing newer" "$(yes_if 'grep -q "poll [3-5] .*button=\"Up to date\"" "$LOG"')"
FAKE_MODE=avail run_bar "$FAKE/sushibar" 10 "check@2,update@5" --port "$UPORT"
check "the Update button reads Update once a check found a release, and still after a failed update" "$(yes_if 'grep -q "poll [3-4] .*button=\"Update\"" "$LOG" && grep -q "poll [7-9] .*button=\"Update\"" "$LOG"')"
check "a failed update leaves the Update button and shows why in an orange note" "$(yes_if 'grep -q "poll [7-9] .*note=\"Update failed: update: disk full while [a-z]*…\" problem=1.*button=\"Update\"" "$LOG"')"

echo "== a server that fails to start says so =="
run_bar "$FAKE/sushibar" 6 "start@2" --port "$UPORT"
check "a server that exits at once shows an orange note with its exit status" "$(yes_if 'grep -q "note=\"Port '"$UPORT"': server did not start (exit 1)\" problem=1" "$LOG"')"

echo "== a started server stays tracked after Start on another port =="
run_bar "$BAR" 14 "start@2,port$FPORT@5,start@8" --port "$PORT"
check "each port keeps its own pid record" "$(yes_if '[ -f "$HOME/.sushi/sushibar-server-'"$PORT"'.pid" ] && [ -f "$HOME/.sushi/sushibar-server-'"$FPORT"'.pid" ]')"
P1=$(cut -d' ' -f1 "$HOME/.sushi/sushibar-server-$PORT.pid" 2>/dev/null)
P2=$(cut -d' ' -f1 "$HOME/.sushi/sushibar-server-$FPORT.pid" 2>/dev/null)
check "both servers are alive" "$(yes_if '[ -n "$P1" ] && [ -n "$P2" ] && kill -0 "$P1" 2>/dev/null && kill -0 "$P2" 2>/dev/null')"
run_bar "$BAR" 3 "" --port "$PORT"
check "back on the first port, its server is sushibar's own (Stop without the confirm ellipsis)" "$(yes_if 'grep -q "poll 2 .*Port '"$PORT"': .*toggle=\"Stop Server\"" "$LOG"')"
kill "$P1" "$P2" 2>/dev/null
rm -f "$HOME"/.sushi/sushibar-server-*.pid

echo "== Start brings the server up empty, Load and Unload act on the last model, and a server exit is reported =="
LM="$WORK/lm"
mkdir -p "$LM"
cat > "$LM/fake.c" <<'EOF'
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <unistd.h>
/* A stand-in for `sushi --serve --port N [--host A] [--api-key K]`: answers the feeds sushibar reads, lists
   FAKE_READY as the ready model, appends the body of every POST /v1/load-model (and "unload " and the body of every
   /v1/unload-model) to FAKE_LOG, and lists the model last loaded, until unloaded. With a key it answers 401 to a
   request without it. FAKE_VERSION_FILE holds what --version prints, FAKE_CHECK what `update --check` prints;
   FAKE_SERVER_VERSION is the version /props reports, after FAKE_PROPS_DELAY seconds; POST /v1/update is accepted
   and changes nothing. FAKE_LINGER: a SIGTERM closes the port at once and exits that many seconds later.
   FAKE_LOAD_DELAY: a POST /v1/load-model answers that many seconds late. */
static int fd, linger;
static void on_term(int sig) { (void)sig; close(fd); sleep(linger); _exit(0); }
int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--version")) {
        char v[32] = "1.0.0";
        FILE *f = getenv("FAKE_VERSION_FILE") ? fopen(getenv("FAKE_VERSION_FILE"), "r") : 0;
        if (f) { if (fscanf(f, "%31s", v) != 1) strcpy(v, "1.0.0"); fclose(f); }
        printf("sushi %s\n", v);
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "update") && getenv("FAKE_CHECK")) { puts(getenv("FAKE_CHECK")); return 0; }
    int port = 0;
    const char *host = "127.0.0.1", *key = 0;
    for (int i = 1; i + 1 < argc; i++) {
        if (!strcmp(argv[i], "--port")) port = atoi(argv[i + 1]);
        if (!strcmp(argv[i], "--host")) host = argv[i + 1];
        if (!strcmp(argv[i], "--api-key")) key = argv[i + 1];
    }
    if (!port) return 1;
    fd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    if (inet_pton(AF_INET, host, &a.sin_addr) != 1) return 1;
    if (bind(fd, (void *)&a, sizeof a) || listen(fd, 16)) return 1;
    signal(SIGCHLD, SIG_IGN);
    if (getenv("FAKE_LINGER")) { linger = atoi(getenv("FAKE_LINGER")); signal(SIGTERM, on_term); }
    char bearer[160] = "";
    if (key) snprintf(bearer, sizeof bearer, "Bearer %s", key);
    const char *server_version = getenv("FAKE_SERVER_VERSION");
    int props_delay = getenv("FAKE_PROPS_DELAY") ? atoi(getenv("FAKE_PROPS_DELAY")) : 0;
    int load_delay = getenv("FAKE_LOAD_DELAY") ? atoi(getenv("FAKE_LOAD_DELAY")) : 0;
    const char *ready = getenv("FAKE_READY"), *log = getenv("FAKE_LOG");
    static char cur[640];
    if (ready) snprintf(cur, sizeof cur, "%s", ready);
    for (;;) {
        int c = accept(fd, 0, 0);
        if (c < 0) continue;
        char req[8192];
        ssize_t n = 0, got;
        req[0] = 0;
        while ((got = read(c, req + n, sizeof req - 1 - n)) > 0) {
            n += got;
            req[n] = 0;
            char *end = strstr(req, "\r\n\r\n"), *cl = strcasestr(req, "content-length:");
            if (!end || n >= (end + 4 - req) + (cl ? atoi(cl + 15) : 0)) break;
        }
        char body[2048] = "{}";
        const char *status = "200 OK";
        int child = 0;
        if (key && !strstr(req, bearer)) status = "401 Unauthorized";
        else if (!strncmp(req, "POST /v1/update", 15)) status = "202 Accepted";
        else if (!strncmp(req, "GET /props", 10) && server_version) {
            snprintf(body, sizeof body, "{\"update\":{\"current\":\"%s\"}}", server_version);
            /* A slow /props answers from a child, so the polls around it are not held up. */
            if (props_delay > 0) {
                pid_t k = fork();
                if (k > 0) { close(c); continue; }
                child = k == 0;
                sleep(props_delay);
            }
        } else if (!strncmp(req, "GET /v1/models", 14))
            snprintf(body, sizeof body, cur[0] ? "{\"data\":[{\"id\":\"%s\",\"state\":\"ready\"}]}" : "{\"data\":[]}", cur);
        if (load_delay > 0 && !strncmp(req, "POST /v1/load-model", 19)) {
            pid_t k = fork();
            if (k > 0) { close(c); continue; }
            child = k == 0;
            sleep(load_delay);
        }
        int is_load = !strncmp(req, "POST /v1/load-model", 19), is_unload = !strncmp(req, "POST /v1/unload-model", 21);
        if ((is_load || is_unload) && status[0] == '2') {
            char *end = strstr(req, "\r\n\r\n"), *m = end ? strstr(end, "\"model\":\"") : 0;
            cur[0] = 0;
            if (is_load && m) { snprintf(cur, sizeof cur, "%s", m + 9); char *q = strchr(cur, '"'); if (q) *q = 0; }
            FILE *f = log ? fopen(log, "a") : 0;
            if (f && end) { fprintf(f, "%s%s\n", is_unload ? "unload " : "", end + 4); fclose(f); }
        }
        char head[192];
        int hl = snprintf(head, sizeof head, "HTTP/1.1 %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n", status, strlen(body));
        write(c, head, hl);
        write(c, body, strlen(body));
        close(c);
        if (child) _exit(0);
    }
}
EOF
if command -v cc > /dev/null 2>&1 && cc -o "$LM/sushi" "$LM/fake.c" 2> /dev/null; then
    cp "$BAR" "$LM/sushibar"
    stop_fakes() {
        for f in "$HOME"/.sushi/sushibar-server-*.pid; do [ -f "$f" ] && kill "$(cut -d' ' -f1 "$f")" 2> /dev/null; done
        rm -f "$HOME"/.sushi/sushibar-server-*.pid
        sleep 0.5
    }
    rm -f "$HOME/.sushi/sushibar.json"
    FAKE_READY=Learned-Pack FAKE_LOG="$WORK/load.log" run_bar "$LM/sushibar" 8 "start@2" --port "$UPORT"
    check "the model the watched server has loaded is remembered in sushibar.json" "$(yes_if 'grep -q "\"model\":\"Learned-Pack\"" "$HOME/.sushi/sushibar.json"')"
    check "a server that has a model asked for nothing at start" "$(yes_if '[ ! -s "$WORK/load.log" ]')"
    stop_fakes
    : > "$WORK/load.log"
    FAKE_LOG="$WORK/load.log" run_bar "$LM/sushibar" 10 "start@2" --port "$UPORT"
    check "the header names the last model under Ready to start while the server is stopped" "$(yes_if 'grep -q "poll 1 .*line1=\"Ready to start · port '"$UPORT"'\" line2=\"Learned-Pack\"" "$LOG"')"
    check "Start brings the server up empty: nothing is asked to load, the card reads Ready to load and Load is on" "$(yes_if '[ ! -s "$WORK/load.log" ] && grep -q "pill=\"No model\" line1=\"Ready to load · port '"$UPORT"'\" line2=\"Learned-Pack\".*load=\"Load\" load_on=1" "$LOG"')"
    stop_fakes
    : > "$WORK/load.log"
    FAKE_LOG="$WORK/load.log" run_bar "$LM/sushibar" 8 "start@2,load@2" --port "$UPORT"
    check "a Load in the tick of the Start loads nothing: the server has not answered" "$(yes_if '[ ! -s "$WORK/load.log" ]')"
    stop_fakes
    : > "$WORK/load.log"
    FAKE_LOG="$WORK/load.log" run_bar "$LM/sushibar" 20 "start@2,load@7,unload@13" --port "$UPORT"
    check "Load asks the answering server for the last model as the web page does, as the default" "$(yes_if 'grep -q "\"model\":\"Learned-Pack\",\"default\":true" "$WORK/load.log"')"
    check "the button then reads Unload with the model in its place" "$(yes_if 'grep -q "pill=\"Running\" .*load=\"Unload\" load_on=1" "$LOG"')"
    check "Unload names the model and leaves the server running" "$(yes_if 'grep -q "^unload {\"model\":\"Learned-Pack\"}" "$WORK/load.log" && grep -q "poll 20 .*status=\"Port '"$UPORT"': running, no model loaded\".*load=\"Load\" load_on=1" "$LOG"')"
    stop_fakes
    : > "$WORK/load.log"
    echo '{"port":'"$UPORT"',"ports":[]}' > "$HOME/.sushi/sushibar.json"
    FAKE_LOG="$WORK/load.log" run_bar "$LM/sushibar" 8 "start@2" --port "$UPORT"
    check "with no remembered model nothing is asked and the header points at the Models page" "$(yes_if '[ ! -s "$WORK/load.log" ] && grep -q "pill=\"No model\" line1=\"Open Models to load one\".*models_link=1" "$LOG"')"
    stop_fakes

    : > "$LOG"
    SUSHIBAR_TEST_POLLS=14 SUSHIBAR_TEST_ACTIONS="start@2" FAKE_READY=Crash-Pack "$LM/sushibar" --port "$UPORT" 2> "$LOG"
    BP=$(cut -d' ' -f1 "$HOME/.sushi/sushibar.pid" 2> /dev/null)
    [ "$(wait_health "$UPORT")" = 200 ] && sleep 3
    kill -9 "$(cut -d' ' -f1 "$HOME/.sushi/sushibar-server-$UPORT.pid" 2> /dev/null)" 2> /dev/null
    for _ in $(seq 1 60); do kill -0 "$BP" 2> /dev/null || break; sleep 0.5; done
    check "a server sushibar started that dies after answering says so with the signal, in an orange note" "$(yes_if 'grep -q "note=\"Port '"$UPORT"': server exited (signal 9)\" problem=1" "$LOG"')"
    check "and the icon carries a badge until a server answers again" "$(yes_if 'grep -q "note=\"Port '"$UPORT"': server exited (signal 9)\" problem=1.*badge=1" "$LOG"')"
    check "that exit is not then reported as a stop once the poll sees the port gone" "$(yes_if '! grep -q "server stopped at" "$LOG" && tail -n 1 "$LOG" | grep -q "note=\"Port '"$UPORT"': server exited (signal 9)\""')"
    rm -f "$HOME"/.sushi/sushibar-server-*.pid

    "$LM/sushi" --serve --port "$FPORT" > /dev/null 2>&1 &
    FPID=$!
    wait_health "$FPORT" > /dev/null
    : > "$LOG"
    SUSHIBAR_TEST_POLLS=12 "$LM/sushibar" --port "$FPORT" 2> "$LOG"
    BP=$(cut -d' ' -f1 "$HOME/.sushi/sushibar.pid" 2> /dev/null)
    sleep 4
    kill -9 "$FPID" 2> /dev/null
    wait "$FPID" 2> /dev/null
    for _ in $(seq 1 60); do kill -0 "$BP" 2> /dev/null || break; sleep 0.5; done
    check "a server sushibar did not start that goes away is reported with the time" "$(yes_if 'grep -qE "note=\"Port '"$FPORT"': server stopped at [0-9]{2}:[0-9]{2}\" problem=1" "$LOG"')"

    echo "== Stop on a server that closes its port before it exits =="
    FAKE_LINGER=4 run_bar "$LM/sushibar" 13 "start@2,stop@5" --port "$UPORT"
    check "the menu reads Stopping… until the server has exited, never Starting" "$(yes_if 'grep -q "poll [6-8] .*toggle=\"Stopping…\"" "$LOG" && ! grep -Eq "poll ([6-9]|1[0-3]) .*(pill=\"Starting\"|toggle=\"Starting…\")" "$LOG" && tail -n 1 "$LOG" | grep -q "pill=\"Stopped\".*toggle=\"Start Server\""')"
    check "and its exit is no news" "$(yes_if '! grep -q "note=\"Port '"$UPORT"': server" "$LOG"')"
    stop_fakes
    "$LM/sushi" --serve --port "$FPORT" > /dev/null 2>&1 &
    FPID=$!
    wait_health "$FPORT" > /dev/null
    FAKE_LINGER=4 run_bar "$LM/sushibar" 9 "start@2,stop@5,port$FPORT@6" --port "$UPORT"
    check "a Stop on one port does not read Stopping… on the port switched to" "$(yes_if 'grep -q "poll [7-9] .*Port '"$FPORT"': running.*toggle=\"Stop Server…\"" "$LOG" && ! grep -q "Port '"$FPORT"': .*toggle=\"Stopping…\"" "$LOG"')"
    kill "$FPID" 2> /dev/null; wait "$FPID" 2> /dev/null
    stop_fakes

    echo "== a server started before sushibar relaunched itself =="
    echo 1.0.0 > "$WORK/fake-version"
    : > "$LOG"
    SUSHIBAR_TEST_POLLS=12 SUSHIBAR_TEST_ACTIONS="start@2,check@11" FAKE_VERSION_FILE="$WORK/fake-version" "$LM/sushibar" --port "$UPORT" 2> "$LOG"
    BP=$(cut -d' ' -f1 "$HOME/.sushi/sushibar.pid" 2> /dev/null)
    wait_health "$UPORT" > /dev/null
    echo 1.0.1 > "$WORK/fake-version"
    for _ in $(seq 1 80); do [ "$(grep -c "poll 1 " "$LOG")" -ge 2 ] && break; sleep 0.25; done
    SRV=$(cut -d' ' -f1 "$HOME/.sushi/sushibar-server-$UPORT.pid" 2> /dev/null)
    sleep 2
    kill -9 "$SRV" 2> /dev/null
    sleep 2
    zombie=$(ps -o stat= -p "$SRV" 2> /dev/null | tr -d ' ')
    for _ in $(seq 1 60); do kill -0 "$BP" 2> /dev/null || break; sleep 0.5; done
    check "sushibar relaunched itself on the new version" "$(yes_if '[ "$(grep -c "poll 1 " "$LOG")" = 2 ]')"
    check "a server it started before the relaunch is reaped when it dies, not left a zombie" "$(yes_if '[ -n "$SRV" ] && [ -z "$zombie" ]')"
    check "and its exit is reported as such" "$(yes_if 'awk "/poll 1 /{n++} n>=2" "$LOG" | grep -q "note=\"Port '"$UPORT"': server exited (signal 9)\""')"
    rm -f "$HOME"/.sushi/sushibar-server-*.pid

    echo "== one model operation at a time, and a problem note outlives a check =="
    echo '{"port":'"$UPORT"',"ports":['"$FPORT"'],"model":"Pack-One"}' > "$HOME/.sushi/sushibar.json"
    "$LM/sushi" --serve --port "$FPORT" > /dev/null 2>&1 &
    FPID=$!
    wait_health "$FPORT" > /dev/null
    : > "$WORK/load.log"
    FAKE_LOAD_DELAY=6 FAKE_LOG="$WORK/load.log" run_bar "$LM/sushibar" 16 "start@2,load@6,port$FPORT@7,load@9" --port "$UPORT"
    check "while a load runs on another port, Load is off and says which model and port" "$(yes_if 'grep -Eq "poll (9|10|11) .*Port '"$FPORT"':.*load=\"Load\" load_on=0 why=\"Loading Pack-One on port '"$UPORT"'…\"" "$LOG"')"
    check "a click on it starts nothing: the model was asked for once" "$(yes_if '[ "$(grep -c "Pack-One" "$WORK/load.log")" = 1 ]')"
    check "and Load is on again once that load has ended" "$(yes_if 'tail -n 1 "$LOG" | grep -q "Port '"$FPORT"':.*load=\"Load\" load_on=1"')"
    kill "$FPID" 2> /dev/null; wait "$FPID" 2> /dev/null
    stop_fakes

    echo "== a started server that answers while another port is watched, and a note a check must not wipe =="
    echo '{"port":'"$UPORT"',"ports":[]}' > "$HOME/.sushi/sushibar.json"
    : > "$LOG"
    FAKE_CHECK="sushi 1.0.0 is up to date" SUSHIBAR_TEST_POLLS=19 SUSHIBAR_TEST_ACTIONS="start@2,port$FPORT@2,check@14" "$LM/sushibar" --port "$UPORT" 2> "$LOG"
    BP=$(cut -d' ' -f1 "$HOME/.sushi/sushibar.pid" 2> /dev/null)
    [ "$(wait_health "$UPORT")" = 200 ] && sleep 4
    kill -9 "$(cut -d' ' -f1 "$HOME/.sushi/sushibar-server-$UPORT.pid" 2> /dev/null)" 2> /dev/null
    for _ in $(seq 1 60); do kill -0 "$BP" 2> /dev/null || break; sleep 0.5; done
    check "a server that answered, then died while another port was watched, exited (it did not fail to start)" "$(yes_if 'grep -q "Port '"$FPORT"':.*note=\"Port '"$UPORT"': server exited (signal 9)\" problem=1" "$LOG"')"
    check "a check that finds sushi up to date leaves that note" "$(yes_if 'tail -n 1 "$LOG" | grep -q "button=\"Up to date\".*" && tail -n 1 "$LOG" | grep -q "note=\"Port '"$UPORT"': server exited (signal 9)\" problem=1"')"
    rm -f "$HOME"/.sushi/sushibar-server-*.pid

    echo "== a long model id is remembered whole, one past the bound is refused with a note =="
    LONG=$(printf 'Long-Pack-%.0s' $(seq 1 20))
    TOOLONG=$(printf 'Too-Long-Pack-%.0s' $(seq 1 45))
    rm -f "$HOME/.sushi/sushibar.json"
    FAKE_READY="$LONG" run_bar "$LM/sushibar" 8 "start@2" --port "$UPORT"
    check "a 200-byte id is saved whole" "$(yes_if 'grep -q "\"model\":\"'"$LONG"'\"" "$HOME/.sushi/sushibar.json"')"
    stop_fakes
    : > "$WORK/load.log"
    FAKE_LOG="$WORK/load.log" run_bar "$LM/sushibar" 10 "start@2,load@6" --port "$UPORT"
    check "and Load asks for exactly that id" "$(yes_if 'grep -q "\"model\":\"'"$LONG"'\",\"default\":true" "$WORK/load.log"')"
    stop_fakes
    rm -f "$HOME/.sushi/sushibar.json"
    FAKE_READY="$TOOLONG" run_bar "$LM/sushibar" 8 "start@2" --port "$UPORT"
    check "an id past the bound is not saved cut" "$(yes_if '! grep -q "model" "$HOME/.sushi/sushibar.json"')"
    check "and the note says so" "$(yes_if 'grep -q "note=\"Port '"$UPORT"': the model id is over" "$LOG"')"
    stop_fakes
    rm -f "$HOME/.sushi/sushibar.json"

    echo "== the props of a port no longer watched stay with it =="
    FAKE_SERVER_VERSION=9.9.9 FAKE_PROPS_DELAY=3 "$LM/sushi" --serve --port "$FPORT" > /dev/null 2>&1 &
    FPID=$!
    wait_health "$FPORT" > /dev/null
    run_bar "$LM/sushibar" 6 "port$UPORT@2" --port "$FPORT"
    check "a /props answer that lands after a port switch is not shown for the new port" "$(yes_if 'grep -q "poll [3-6] .*Port '"$UPORT"': not running" "$LOG" && ! grep -q "Port '"$UPORT"': .*version=\"[^\"]*server 9.9.9" "$LOG"')"
    kill "$FPID" 2> /dev/null; wait "$FPID" 2> /dev/null

    echo "== an Update asked before the server's version is known =="
    FAKE_SERVER_VERSION=1.0.0 FAKE_PROPS_DELAY=3 "$LM/sushi" --serve --port "$FPORT" > /dev/null 2>&1 &
    FPID=$!
    wait_health "$FPORT" > /dev/null
    FAKE_CHECK="sushi 9.0.0 is available (this is 1.0.0): x" run_bar "$LM/sushibar" 16 "check@1,update@2" --port "$FPORT"
    check "the update goes to the server, which accepts it and stays on its version" "$(yes_if 'grep -q "button=\"Updating…\"" "$LOG"')"
    check "and it is not reported as updated" "$(yes_if '! grep -q "note=\"Updated" "$LOG"')"
    kill "$FPID" 2> /dev/null; wait "$FPID" 2> /dev/null
    echo '{"port":'"$UPORT"',"ports":[]}' > "$HOME/.sushi/sushibar.json"

    echo "== a server bound to a non-loopback address that wants its key =="
    IP=$(ifconfig | awk '/inet / && $2 !~ /^127\./ {print $2; exit}')
    fw=$(/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate 2> /dev/null | grep -c "State = 0")
    if [ -n "$IP" ] && [ "$fw" = 1 ]; then
        "$LM/sushi" --serve --port "$FPORT" --host "$IP" --api-key sb-test-key > /dev/null 2>&1 &
        FPID=$!
        for _ in $(seq 1 20); do [ "$(curl -s -o /dev/null -w '%{http_code}' "http://$IP:$FPORT/health")" = 401 ] && break; sleep 0.25; done
        run_bar "$LM/sushibar" 3 "" --port "$FPORT"
        check "it reads running from the first poll, with its key, never Not answering" "$(yes_if 'grep -q "poll 1 .*Port '"$FPORT"': running, no model loaded" "$LOG" && ! grep -q "not answering" "$LOG"')"
        kill "$FPID" 2> /dev/null; wait "$FPID" 2> /dev/null
    else
        echo "  SKIP no non-loopback IPv4 address, or the application firewall is on"
    fi
else
    echo "  SKIP no C compiler"
fi

echo "== a server that wants an API key =="
cat > "$WORK/stub_key.py" <<'EOF'
import http.server, sys

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(401); self.send_header("Content-Length", "0"); self.end_headers()

http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
EOF
python3 "$WORK/stub_key.py" "$TPORT" > /dev/null 2>&1 &
TPID=$!
for _ in $(seq 1 40); do curl -s -o /dev/null "http://127.0.0.1:$TPORT/" && break; sleep 0.25; done
run_bar "$BAR" 3 "" --port "$TPORT"
check "a 401 reads Key needed with its cause in the header, a badge and no words in the title" "$(yes_if 'grep -q "poll 2 title=\"\" .*pill=\"Key needed\" line1=\"The server requires an API key\".*badge=1" "$LOG"')"
kill "$TPID" 2>/dev/null; wait "$TPID" 2>/dev/null

echo "== two sushibars launched at once leave one =="
DUAL="$WORK/dual"
mkdir -p "$DUAL"
cp "$BAR" "$DUAL/sushibar"
single=1
for trial in 1 2 3; do
    export HOME="$WORK/home-dual-$trial"
    mkdir -p "$HOME"
    ( SUSHIBAR_TEST_POLLS=6 "$DUAL/sushibar" --port "$UPORT" 2> /dev/null & SUSHIBAR_TEST_POLLS=6 "$DUAL/sushibar" --port "$UPORT" 2> /dev/null & wait )
    sleep 1.5
    n=$(pgrep -f "^$DUAL/sushibar" | wc -l | tr -d ' ')
    [ "$n" = 1 ] || single=0
    for _ in $(seq 1 30); do [ -n "$(pgrep -f "^$DUAL/sushibar")" ] || break; sleep 0.5; done
done
export HOME="$WORK/home"
check "three simultaneous double launches each left exactly one sushibar" "$single"

if command -v cc > /dev/null 2>&1; then
echo "== a hung sushi elsewhere, and a hung one on the watched port =="
mkdir -p "$WORK/hung"
cat > "$WORK/hung/hang.c" <<'EOF'
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
int main(int argc, char **argv) {
    int fd = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET;
    a.sin_port = htons(atoi(argv[1]));
    a.sin_addr.s_addr = htonl(0x7f000001);
    if (bind(fd, (void *)&a, sizeof a) || listen(fd, 64)) return 1;
    for (;;) {
        if (accept(fd, 0, 0) >= 0 && argc > 2) {
            int log = open(argv[2], O_WRONLY | O_CREAT | O_APPEND, 0600);
            if (log >= 0) { write(log, "x", 1); close(log); }
        }
    }
}
EOF
if cc -o "$WORK/hung/sushi" "$WORK/hung/hang.c" 2> /dev/null; then
    "$WORK/hung/sushi" "$HPORT" "$WORK/hung/accepts" &
    HPID=$!
    sleep 0.5
    cat > "$WORK/stub.py" <<'EOF'
import http.server, json, socketserver, sys, time
hits = {}
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        path = self.path.split("?")[0]
        hits[path] = hits.get(path, 0) + 1
        now = int(time.time() * 1000)
        if path == "/metrics.json": body = {"gauges": {}, "sessions": [], "requests": [], "history": {"fields": ["t_ms", "gen_tokens"], "samples": []}}
        elif path == "/v1/models": body = {"data": [{"id": "Stub", "state": "ready"}]}
        elif path == "/hits": body = hits
        else:
            self.send_response(404); self.end_headers(); return
        data = json.dumps(body).encode()
        self.send_response(200); self.send_header("content-type", "application/json"); self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)
class S(socketserver.ThreadingMixIn, http.server.HTTPServer): daemon_threads = True
S(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
EOF
    python3 "$WORK/stub.py" "$SPORT" &
    SPID=$!
    for _ in $(seq 1 40); do curl -s -o /dev/null "http://127.0.0.1:$SPORT/hits" && break; sleep 0.25; done
    # Another sushibar on the box (an older build polls every listener) would be counted as ours: measure it first.
    : > "$WORK/hung/accepts"
    sleep 6
    noise=$(wc -c < "$WORK/hung/accepts" | tr -d ' ')
    : > "$WORK/hung/accepts"
    run_bar "$BAR" 12 "" --port "$SPORT"
    if [ "$noise" = 0 ]; then
        check "with no Port submenu opened, a sushi on another port is never contacted" "$(yes_if '[ ! -s "$WORK/hung/accepts" ]')"
    else
        echo "  SKIP another program on this box polls sushi listeners; the never-contacted check needs a quiet one"
    fi
    check "the watched port is still polled about once a second" "$(yes_if '[ "$(curl -s "http://127.0.0.1:'"$SPORT"'/hits" | python3 -c "import sys,json; print(json.load(sys.stdin).get(\"/metrics.json\",0))")" -ge 10 ]')"
    curl -s "http://127.0.0.1:$SPORT/hits" > /dev/null
    : > "$WORK/hung/accepts"
    before=$(curl -s "http://127.0.0.1:$SPORT/hits" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("/metrics.json",0))')
    echo '{"port":'"$SPORT"',"ports":['"$HPORT"']}' > "$HOME/.sushi/sushibar.json"
    run_bar "$BAR" 12 "models$HPORT@4" --port "$SPORT"
    after=$(curl -s "http://127.0.0.1:$SPORT/hits" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("/metrics.json",0))')
    check "opening the submenu asks the hung sushi once" "$(yes_if '[ -s "$WORK/hung/accepts" ]')"
    check "that hung sushi never slows the watched port's polls" "$(yes_if '[ $((after - before)) -ge 10 ]')"
    check "its label reads not answering, never no model" "$(yes_if 'grep -q "ports=\".*'"$HPORT"' — not answering;" "$LOG" && ! grep -q "'"$HPORT"' — no model" "$LOG"')"
    run_bar "$BAR" 10 "" --port "$HPORT"
    check "a hung sushi on the watched port reads Not answering, with a badge on the icon" "$(yes_if 'grep -q "poll 10 .*Port '"$HPORT"': not answering.*pill=\"Not answering\".*badge=1" "$LOG"')"
    check "and Stop is offered with the confirm ellipsis" "$(yes_if 'grep -q "poll 10 .*stop=1.*toggle=\"Stop Server…\"" "$LOG"')"
    kill "$HPID" "$SPID" 2>/dev/null
    wait "$HPID" "$SPID" 2>/dev/null
else
    echo "  SKIP no C compiler"
fi
fi

echo "== Homebrew: an upgrade replaces the keg under a running sushibar =="
BREW="$WORK/brew"
mkdir -p "$BREW/bin" "$WORK/kegsrc"
# A keg's sushi is a program that names itself `sushi` from its first instruction, as the real one does.
cat > "$WORK/kegsrc/sushi.c" <<'EOF'
#include <stdio.h>
#include <string.h>
#include <unistd.h>
int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--version")) { printf("sushi %s\n", VERSION); return 0; }
    if (argc > 1 && !strcmp(argv[1], "update")) { printf("sushi %s is up to date (latest release %s)\n", VERSION, VERSION); return 0; }
    if (argc > 1 && !strcmp(argv[1], "--serve")) { argv[0] = REAL; execv(REAL, argv); }
    return 1;
}
EOF
mk_keg() {
    local v="$1" dir="$BREW/Cellar/sushi/$1/libexec"
    mkdir -p "$dir"
    cc -o "$dir/sushi" "$WORK/kegsrc/sushi.c" -DVERSION="\"$v\"" -DREAL="\"$SUSHI_BIN\"" 2> /dev/null
    cp "$BAR" "$dir/sushibar"
}
if command -v cc > /dev/null 2>&1; then
    mk_keg 1.0.0
    ln -s ../Cellar/sushi/1.0.0/libexec/sushibar "$BREW/bin/sushibar"
    : > "$LOG"
    SUSHIBAR_TEST_POLLS=16 SUSHIBAR_TEST_ACTIONS="check@6,start@9" "$BREW/bin/sushibar" --port "$UPORT" 2> "$LOG"
    BPID=$(cut -d' ' -f1 "$HOME/.sushi/sushibar.pid" 2>/dev/null)
    sleep 3
    mk_keg 1.1.0
    ln -sfn ../Cellar/sushi/1.1.0/libexec/sushibar "$BREW/bin/sushibar"
    rm -rf "$BREW/Cellar/sushi/1.0.0"
    for _ in $(seq 1 80); do [ -n "$BPID" ] && kill -0 "$BPID" 2>/dev/null || break; sleep 0.5; done
    check "after the upgrade deleted the old keg, Start still starts a server" "$(yes_if '[ "$(wait_health "$UPORT")" = 200 ]')"
    check "the version moved to the new keg's" "$(yes_if 'grep -q "version=\"1.1.0" "$LOG"')"
    check "no step reported that it could not run sushi" "$(yes_if '! grep -q "could not run sushi" "$LOG"')"
    check "sushibar relaunched itself into the new keg on the same pid" "$(yes_if '[ "$(grep -c "poll 1 " "$LOG")" = 2 ]')"
    for f in "$HOME"/.sushi/sushibar-server-*.pid; do [ -f "$f" ] && kill "$(cut -d' ' -f1 "$f")" 2> /dev/null; done
else
    echo "  SKIP no C compiler"
fi

echo "== $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]

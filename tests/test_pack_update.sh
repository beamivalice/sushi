#!/usr/bin/env bash
# sushi-test: tier=host
# Pack-revision update notice, hermetic: a stub Hugging Face (SUSHI_HF_ENDPOINT, the test-only hook) on a private port
# serves the repo listing and `config.json` for a stamped (2.0), a legacy and a missing remote. `sushi list` under a
# private HOME with a stub pack folder named like the DeepSeek repo prints the update line against the 2.0 remote and
# nothing against a legacy (1.0) remote, a 404 or a stub that is down. The server's check and its /v1/models row are
# pinned by the pack_update.zig and models_api.zig unit tests (a headless server cannot load stub weights).
# Usage: ./tests/test_pack_update.sh [port]
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN=${SUSHI_BIN:-$ROOT/zig-out/bin/sushi}
PORT="${1:-18861}"
[ -x "$BIN" ] || { echo "FAIL: missing binary $BIN" >&2; exit 1; }
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then echo "FAIL: port $PORT is busy" >&2; exit 1; fi

WORK="$(cd "$(mktemp -d)" && pwd -P)"
SPID=""
trap '[ -n "$SPID" ] && kill "$SPID" 2>/dev/null; rm -rf "$WORK"' EXIT
export HOME="$WORK/home"
export SUSHI_PREFIX_CACHE_DIR="$WORK/kv"
export SUSHI_HF_ENDPOINT="http://127.0.0.1:$PORT"
unset HF_TOKEN HF_TOKEN_PATH HF_HOME
FAILED=""
PACK=DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw
UPDATE_LINE="update available: 2.0 (you have 1.0) — sushi pull deepseek-v4.1-flash"

mkdir -p "$HOME/.sushi/models/beamster/$PACK"
echo '{"model_type":"deepseek_v41"}' > "$HOME/.sushi/models/beamster/$PACK/config.json"
: > "$HOME/.sushi/models/beamster/$PACK/model.safetensors"

cat > "$WORK/stub_hf.py" <<'EOF'
import http.server, json, sys

MODE = sys.argv[2]

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def reply(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
    def do_GET(self):
        if self.path.startswith("/api/models/"):
            return self.reply(200, {"siblings": [{"size": 1000}]})
        if self.path.endswith("/resolve/main/config.json"):
            if MODE == "v2":
                return self.reply(200, {"model_type": "deepseek_v41", "pack_revision": {"id": "2.0", "unstamped": []}})
            if MODE == "legacy":
                return self.reply(200, {"model_type": "deepseek_v41"})
        self.reply(404, {})

http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
EOF

serve() {
    python3 "$WORK/stub_hf.py" "$PORT" "$1" > /dev/null 2>&1 &
    SPID=$!
    for _ in $(seq 1 40); do curl -s -o /dev/null "http://127.0.0.1:$PORT/api/models/x" && break; sleep 0.25; done
}
stop() { kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; SPID=""; }
check() {
    if [ "$2" = "1" ]; then echo "  PASS $1"; else echo "  FAIL $1"; FAILED=1; fi
}
count() { grep -c "update available" <<<"$1"; }

serve v2
out=$("$BIN" list 2>&1)
check "a 2.0 remote against a legacy pack prints the update line" "$([ "$(grep -c -F "  $UPDATE_LINE" <<<"$out")" = 1 ] && echo 1 || echo 0)"
check "the line sits directly under the pack's row" "$(grep -A1 "deepseek-v4.1-flash:1.75bpw" <<<"$out" | tail -1 | grep -qF "$UPDATE_LINE" && echo 1 || echo 0)"
check "only that pack is flagged" "$([ "$(count "$out")" = 1 ] && echo 1 || echo 0)"
echo "  list output:"; sed 's/^/    /' <<<"$out"
stop

serve legacy
out=$("$BIN" list 2>&1)
check "a legacy remote prints no update line" "$([ "$(count "$out")" = 0 ] && echo 1 || echo 0)"
stop

serve missing
out=$("$BIN" list 2>&1)
check "a 404 remote prints no update line" "$([ "$(count "$out")" = 0 ] && echo 1 || echo 0)"
stop

out=$("$BIN" list 2>&1)
check "a stub that is down prints no update line" "$([ "$(count "$out")" = 0 ] && echo 1 || echo 0)"

serve v2
echo '{"model_type":"deepseek_v41","pack_revision":{"id":"2.0","unstamped":[]}}' > "$HOME/.sushi/models/beamster/$PACK/config.json"
out=$("$BIN" list 2>&1)
check "a pack already at 2.0 prints no update line" "$([ "$(count "$out")" = 0 ] && echo 1 || echo 0)"
stop

[ -z "$FAILED" ] && echo "PASS: pack update notice"
exit ${FAILED:+1}

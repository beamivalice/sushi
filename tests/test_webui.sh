#!/bin/bash
. "$(dirname "$0")/private_cache.sh"
# The browser chat page: `GET /` and `GET /chat` serve the page embedded from
# src/webui/index.html, every other method on those paths is a 405, the API
# routes beside it answer as before, and `--api-key` leaves the page open while
# the API it calls stays behind the key.
#
# Fully hermetic: an EMPTY --model-dir discovers zero models, so nothing loads.
#
# Usage: ./tests/test_webui.sh [port]

set -u

PORT="${1:-18851}"
BINARY="${BINARY:-./zig-out/bin/sushi}"
PAGE="src/webui/index.html"
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
    echo "[fail] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
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

echo "Chat page (port $PORT)"

echo "[1/5] the page and its neighbours"
if boot; then
    for path in / /chat; do
        curl -s -D "$WORK/headers" -o "$WORK/body" "$BASE$path"
        check "GET $path is 200" "$(grep -q '^HTTP/1.1 200' "$WORK/headers" && echo 1 || echo 0)"
        check "GET $path is text/html; charset=utf-8" \
            "$(grep -qi '^Content-Type: text/html; charset=utf-8' "$WORK/headers" && echo 1 || echo 0)"
        check "GET $path body is the embedded page" "$(cmp -s "$WORK/body" "$PAGE" && echo 1 || echo 0)"
    done
    check "tools pack is available to the local page" \
        "$(curl -s -X POST "$BASE/v1/tools" -H "Origin: $BASE" -H 'Content-Type: application/json' -d '{}' | grep -q 'web_search' && echo 1 || echo 0)"
    check "tools reject another page origin" \
        "$(is "$(code -X POST "$BASE/v1/tools" -H 'Origin: http://evil.test' -d '{}')" 403)"
    check "tools require a page origin" \
        "$(is "$(code -X POST "$BASE/v1/tools" -d '{}')" 403)"
    check "file tools refuse paths outside the server folder" \
        "$(curl -s -X POST "$BASE/v1/tools" -H "Origin: $BASE" -H 'Content-Type: application/json' -d '{"name":"read_file","arguments":"{\"path\":\"../outside\"}"}' | grep -q 'refused' && echo 1 || echo 0)"
    mkdir -p "$WORK/picked/child"
    echo 'chosen folder fixture' > "$WORK/picked/note.txt"
    curl -s -X POST "$BASE/v1/tools" -H "Origin: $BASE" -H 'Content-Type: application/json' \
        -d "{\"browse\":true,\"directory\":\"$WORK/picked\"}" > "$WORK/folders.json"
    check "folder picker lists directories in the chosen folder" \
        "$(python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["directories"] == ["child"]; assert x["parent"]' "$WORK/folders.json" 2>/dev/null && echo 1 || echo 0)"
    python3 - "$WORK" <<'PYDATA'
import json, pathlib, sys
work = pathlib.Path(sys.argv[1])
for filename, path in [("read.json", "note.txt"), ("outside.json", "../outside")]:
    (work / filename).write_text(json.dumps({"directory": str(work / "picked"), "name": "read_file", "arguments": json.dumps({"path": path})}))
# Write calls from a chat with its pencil off and on; a file keeps the nested JSON quoting out of bash.
for filename, path, on in [("write_off.json", "off.md", False), ("write_on.json", "on.md", True)]:
    (work / filename).write_text(json.dumps({
        "directory": str(work / "picked"), "name": "write_file", "write": on,
        "arguments": json.dumps({"path": path, "content": "x\n"})}))
PYDATA
    check "tools read from the chosen folder" \
        "$(curl -s -X POST "$BASE/v1/tools" -H "Origin: $BASE" -H 'Content-Type: application/json' \
        --data-binary @"$WORK/read.json" | grep -q 'chosen folder fixture' && echo 1 || echo 0)"
    check "selected folder still confines file reads" \
        "$(curl -s -X POST "$BASE/v1/tools" -H "Origin: $BASE" -H 'Content-Type: application/json' \
        --data-binary @"$WORK/outside.json" | grep -q 'refused' && echo 1 || echo 0)"
    check "choosing a folder does not change the server default" \
        "$(curl -s -X POST "$BASE/v1/tools" -H "Origin: $BASE" -d '{}' | python3 -c 'import json,os,sys; assert json.load(sys.stdin)["root"] == os.path.realpath(".")' && echo 1 || echo 0)"
    check "invalid selected directory is rejected" \
        "$(is "$(code -X POST "$BASE/v1/tools" -H "Origin: $BASE" -d '{"directory":"/sushi-folder-does-not-exist"}')" 400)"
    check "without --edit a new chat starts with editing off" \
        "$(curl -s -X POST "$BASE/v1/tools" -H "Origin: $BASE" -H 'Content-Type: application/json' -d '{}' \
            | grep -q '"edit_default":false' && echo 1 || echo 0)"
    check "a write call is refused while the chat's editing is off" \
        "$(curl -s -X POST "$BASE/v1/tools" -H "Origin: $BASE" -H 'Content-Type: application/json' \
        --data-binary @"$WORK/write_off.json" | grep -q 'writing files is off' && echo 1 || echo 0)"
    check "a refused write leaves no file behind" \
        "$(is "$(find "$WORK/picked" -name 'off.md' | wc -l | tr -d ' ')" 0)"
    check "a chat that turns editing on writes without any launch flag" \
        "$(curl -s -X POST "$BASE/v1/tools" -H "Origin: $BASE" -H 'Content-Type: application/json' \
        --data-binary @"$WORK/write_on.json" | grep -q 'wrote 2 bytes' && [ -f "$WORK/picked/on.md" ] && echo 1 || echo 0)"
    check "POST / is 405" "$(is "$(code -X POST "$BASE/" -d '{}')" 405)"
    check "GET /health still answers ok" "$(is "$(curl -s "$BASE/health")" '{"status":"ok"}')"
    check "GET /v1/models still lists" "$(is "$(curl -s "$BASE/v1/models")" '{"object":"list","data":[]}')"
    check "an unknown path is still 404" "$(is "$(code "$BASE/index.html")" 404)"
    check "a served API route still reports 503 with no model" \
        "$(is "$(code -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{}')" 503)"
    check "the boot banner prints the page URL once" \
        "$(is "$(grep -c "chat in your browser: http://127.0.0.1:$PORT/" "$LOG")" 1)"
else
    check "boot" 0
fi

echo "[2/5] --edit on: new chats start with editing on, and writes stay in the folder"
if boot --edit on; then
    python3 - "$WORK" <<'PYDATA'
import json, pathlib, sys
work = pathlib.Path(sys.argv[1])
picked = str(work / "picked")
calls = {
    "listing.json": {"write": True},
    "write_note.json": {"directory": picked, "write": True, "name": "write_file",
                        "arguments": json.dumps({"path": "written.md", "content": "first line\n"})},
    "edit_note.json": {"directory": picked, "write": True, "name": "edit_file",
                       "arguments": json.dumps({"path": "written.md", "old_string": "first", "new_string": "second"})},
    "write_outside.json": {"directory": picked, "write": True, "name": "write_file",
                           "arguments": json.dumps({"path": "../escaped.md", "content": "x\n"})},
    "write_secret.json": {"directory": picked, "write": True, "name": "write_file",
                          "arguments": json.dumps({"path": "server.key", "content": "x\n"})},
    "edit_missing.json": {"directory": picked, "write": True, "name": "edit_file",
                          "arguments": json.dumps({"path": "gone.md", "old_string": "a", "new_string": "b"})},
}
for filename, body in calls.items():
    (work / filename).write_text(json.dumps(body))
PYDATA
    tools_post() { curl -s -X POST "$BASE/v1/tools" -H "Origin: $BASE" -H 'Content-Type: application/json' --data-binary @"$WORK/$1"; }
    check "the listing says new chats start with editing on and offers both write tools" \
        "$(tools_post listing.json | python3 -c 'import json,sys; x=json.load(sys.stdin); names=[t["function"]["name"] for t in x["tools"]]; assert x["edit_default"] is True and "write_file" in names and "edit_file" in names' 2>/dev/null && echo 1 || echo 0)"
    check "write_file creates the file in the chosen folder" \
        "$(tools_post write_note.json | grep -q 'wrote 11 bytes' && [ "$(cat "$WORK/picked/written.md" 2>/dev/null)" = "first line" ] && echo 1 || echo 0)"
    check "edit_file changes that file in place" \
        "$(tools_post edit_note.json | grep -q 'edited written.md: 1 replacement' && [ "$(cat "$WORK/picked/written.md" 2>/dev/null)" = "second line" ] && echo 1 || echo 0)"
    check "a write outside the chosen folder is refused and nothing appears there" \
        "$(tools_post write_outside.json | grep -q 'refused' && [ ! -e "$WORK/escaped.md" ] && echo 1 || echo 0)"
    check "a secret-named file is refused and not created" \
        "$(tools_post write_secret.json | grep -q 'secrets' && [ ! -e "$WORK/picked/server.key" ] && echo 1 || echo 0)"
    check "editing a file that is not there names it" \
        "$(tools_post edit_missing.json | grep -q 'no such file' && echo 1 || echo 0)"
else
    check "boot with --edit on" 0
fi

echo "[3/5] --api-key --api-key-strict: page open, API behind the key"
if boot --api-key webui-test-key --api-key-strict; then
    check "tools require the configured API key" \
        "$(is "$(code -X POST "$BASE/v1/tools" -H "Origin: $BASE" -d '{}')" 401)"
    check "tools accept the configured API key" \
        "$(is "$(code -X POST "$BASE/v1/tools" -H "Origin: $BASE" -H 'Authorization: Bearer webui-test-key' -d '{}')" 200)"
    check "GET / needs no key" "$(is "$(code "$BASE/")" 200)"
    check "GET /chat needs no key" "$(is "$(code "$BASE/chat")" 200)"
    check "GET /v1/models without the key is 401" "$(is "$(code "$BASE/v1/models")" 401)"
    check "GET /v1/models with the Bearer key is 200" \
        "$(is "$(code -H 'Authorization: Bearer webui-test-key' "$BASE/v1/models")" 200)"
    check "POST / without the key is 401" "$(is "$(code -X POST "$BASE/" -d '{}')" 401)"
else
    check "boot with --api-key" 0
fi

echo "[4/5] the page stays self-contained"
check "no external http(s) fetch or CDN in the page" \
    "$(grep -Eq '(src|href)="https?://|@import|fetch\("https?://' "$PAGE" && echo 0 || echo 1)"
check "the page is under 200 KB" "$([ "$(wc -c < "$PAGE")" -lt 204800 ] && echo 1 || echo 0)"

echo "[5/5] the page's own functions (node): effort menu, tool loop, SSD prefix-cache meter"
if command -v node >/dev/null 2>&1; then
    for t in effort tools prefix_cache; do
        check "tests/test_webui_$t.cjs passes" "$(node "tests/test_webui_$t.cjs" >/dev/null 2>&1 && echo 1 || echo 0)"
    done
    check "the prefix-cache meter test names its pass" \
        "$(node tests/test_webui_prefix_cache.cjs 2>&1 | grep -q 'prefix-cache meter: passed' && echo 1 || echo 0)"
else
    echo "  (node not installed: skipped)"
fi

echo
echo "  passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]

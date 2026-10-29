#!/bin/bash
# sushi-test: tier=host
. "$(dirname "$0")/private_cache.sh"
# Self-update end to end, offline. A local HTTP server stands in for GitHub (SUSHI_UPDATE_API, the test-only hook)
# and serves releases staged from this tree the way release.yml packages them: the current build, and the same
# tree built with the next patch version. Covers `sushi update --check`, a running sushi refused, a corrupt tarball
# refused on SHA-256, a new build that does not run swapped back out, the update and the version after,
# `--rollback`, a source build refused, the daily-check switches, and the chat-page path (POST /v1/update, the
# server replaced in place by the updater and relaunched on the same pid and argv), failure included, and a
# Homebrew keg leaving every update to `brew upgrade sushi`. No model loads, so no GPU lock.
#
# Usage: ./tests/test_self_update.sh [port]
# NEW_BINARY=<path> reuses a build of this tree made with -Dversion=<the next patch version>.

set -u
cd "$(dirname "$0")/.."

PORT="${1:-18861}"
API_PORT=$((PORT + 1))
BINARY="${BINARY:-./zig-out/bin/sushi}"
ZIG="${ZIG:-./.zig-toolchain/zig}"
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
has() { grep -q -- "$2" "$1" && echo 1 || echo 0; }

if [ ! -x "$BINARY" ]; then
    echo "[fail] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi

CUR="$("$BINARY" --version 2>/dev/null | head -1 | awk '{print $2}')"
IFS=. read -r MAJ MIN PAT <<< "${CUR%%[-+]*}"
NEXT="$MAJ.$MIN.$((PAT + 1))"

# Physical path: the updater names the install by its real path.
WORK="$(cd "$(mktemp -d)" && pwd -P)"
EMPTY="$WORK/models"
WWW="$WORK/www"
INST="$WORK/inst/sushi-macos-arm64"
export HOME="$WORK/home"
export SUSHI_UPDATE_API="http://127.0.0.1:$API_PORT"
mkdir -p "$EMPTY" "$WWW/repos/beamivalice/sushi" "$WWW/dl" "$HOME"
SPID=""
API_PID=""
stop_server() {
    [ -n "$SPID" ] && kill "$SPID" 2>/dev/null && wait "$SPID" 2>/dev/null
    SPID=""
}
cleanup() {
    stop_server
    [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null && wait "$API_PID" 2>/dev/null
    rm -rf "$WORK"
}
trap cleanup EXIT

echo "Self-update (sushi $CUR -> $NEXT, port $PORT, fake API on $API_PORT)"

NEW="${NEW_BINARY:-}"
if [ -z "$NEW" ]; then
    echo "  building this tree as $NEXT (one ReleaseFast build)..."
    "$ZIG" build -Doptimize=ReleaseFast -Dversion="$NEXT" -p "$WORK/next" > "$WORK/build.log" 2>&1 \
        || { cat "$WORK/build.log"; echo "[fail] build"; exit 1; }
    NEW="$WORK/next/bin/sushi"
fi

# The release.yml packaging, ad-hoc signed: `stage <binary> <dir> [nolib]`.
stage() {
    local bin="$1" dir="$2"
    mkdir -p "$dir/lib"
    cp "$bin" "$dir/sushi"
    cp LICENSE LICENSE-APACHE-2.0 NOTICE "$dir/"
    cp lib/mlx/lib/*.dylib lib/mlx/lib/mlx.metallib "$dir/lib/"
    install_name_tool -change @rpath/libmlxc.dylib @executable_path/lib/libmlxc.dylib "$dir/sushi" 2>/dev/null
    install_name_tool -change @rpath/libmlx.dylib @loader_path/libmlx.dylib "$dir/lib/libmlxc.dylib" 2>/dev/null
    install_name_tool -add_rpath @loader_path "$dir/lib/libmlx.dylib" 2>/dev/null
    for f in "$dir/lib/"*.dylib "$dir/sushi"; do codesign --force --sign - "$f" 2>/dev/null; done
    "$dir/sushi" --guest-manifest > "$dir/guest.json" 2>/dev/null
    [ "${3:-}" = nolib ] && rm -rf "$dir/lib"
    return 0
}
# `pack <name> <binary> [nolib]`: a release tarball and its .sha256 under $WORK/rel/<name>.
pack() {
    local out="$WORK/rel/$1"
    mkdir -p "$out/src"
    stage "$2" "$out/src/sushi-macos-arm64" "${3:-}"
    tar -czf "$out/sushi-bin-macos-arm64.tar.gz" -C "$out/src" sushi-macos-arm64
    (cd "$out" && shasum -a 256 sushi-bin-macos-arm64.tar.gz > sushi-bin-macos-arm64.tar.gz.sha256)
    rm -rf "$out/src"
}
# `publish <name>`: the fake GitHub now lists v$NEXT with that release's assets.
publish() {
    cp "$WORK/rel/$1/sushi-bin-macos-arm64.tar.gz" "$WORK/rel/$1/sushi-bin-macos-arm64.tar.gz.sha256" "$WWW/dl/"
    cat > "$WWW/repos/beamivalice/sushi/releases" <<EOF
[{"tag_name":"v$NEXT","html_url":"https://github.com/beamivalice/sushi/releases/tag/v$NEXT","draft":false,"prerelease":false,
  "assets":[{"name":"sushi-bin-macos-arm64.tar.gz","browser_download_url":"$SUSHI_UPDATE_API/dl/sushi-bin-macos-arm64.tar.gz"},
            {"name":"sushi-bin-macos-arm64.tar.gz.sha256","browser_download_url":"$SUSHI_UPDATE_API/dl/sushi-bin-macos-arm64.tar.gz.sha256"}]},
 {"tag_name":"v99.0.0","draft":true,"prerelease":false,"assets":[]},
 {"tag_name":"v$CUR","html_url":"https://github.com/beamivalice/sushi/releases/tag/v$CUR","draft":false,"prerelease":false,"assets":[]}]
EOF
}
installed() { "$INST/sushi" --version 2>/dev/null | head -1 | awk '{print $2}'; }
upd() { "$INST/sushi" update "$@" > "$WORK/upd.log" 2>&1; }

echo "  staging the releases..."
stage "$BINARY" "$INST"
pack good "$NEW"
pack broken "$NEW" nolib
mkdir -p "$WORK/rel/corrupt"
cp "$WORK/rel/good/"* "$WORK/rel/corrupt/"
printf 'corrupt' | dd of="$WORK/rel/corrupt/sushi-bin-macos-arm64.tar.gz" bs=1 seek=4096 conv=notrunc 2>/dev/null
publish good
python3 -m http.server "$API_PORT" --bind 127.0.0.1 --directory "$WWW" > "$WORK/api.log" 2>&1 &
API_PID=$!
for _ in $(seq 1 40); do curl -sf "$SUSHI_UPDATE_API/repos/beamivalice/sushi/releases" >/dev/null && break; sleep 0.25; done

boot() {
    stop_server
    : > "$WORK/server.log"
    # SIGHUP ignored, as under nohup: the update must keep it so.
    (trap '' HUP; exec "$INST/sushi" serve --model-dir "$EMPTY" --port "$PORT" --log-file off "$@") > "$WORK/server.log" 2>&1 &
    SPID=$!
    for _ in $(seq 1 60); do
        curl -sf "$BASE/health" >/dev/null 2>&1 && return 0
        kill -0 "$SPID" 2>/dev/null || break
        sleep 0.5
    done
    echo "  (server did not come up; log follows)"; cat "$WORK/server.log"
    return 1
}
prop() { curl -s "$BASE/props" | python3 -c "import json,sys; v=json.load(sys.stdin).get('update',{}).get('$1'); print('null' if v is None else str(v).lower() if isinstance(v,bool) else v)" 2>/dev/null; }
# `wait_for <field> <value> [seconds]`: poll /props across the relaunch.
wait_for() {
    for _ in $(seq 1 $((${3:-180} * 2))); do
        [ "$(prop "$1")" = "$2" ] && return 0
        sleep 0.5
    done
    return 1
}
post_update() { curl -s -o "$WORK/post.json" -w '%{http_code}' -X POST "$BASE/v1/update" "$@"; }

echo "[1/7] check, and the refusals that leave the install alone"
upd --check
check "--check names $NEXT" "$(has "$WORK/upd.log" "sushi $NEXT is available (this is $CUR)")"
check "--check recorded the answer" "$(has "$HOME/.sushi/update-check.json" "\"latest\":\"$NEXT\"")"
if boot; then
    upd
    check "a running sushi from the install refuses the update" "$(has "$WORK/upd.log" "(pid $SPID) is running from")"
    stop_server
fi
publish corrupt
upd
check "a corrupt tarball is refused on its SHA-256" "$(has "$WORK/upd.log" "SHA-256 mismatch")"
check "... and the install still runs $CUR" "$(is "$(installed)" "$CUR")"
publish broken
upd
check "a new build that does not run is swapped back out" "$(has "$WORK/upd.log" "the new build does not run; the old install is back")"
check "... and the install still runs $CUR" "$(is "$(installed)" "$CUR")"
check "... with no .previous made" "$([ ! -e "$INST.previous" ] && echo 1 || echo 0)"
"$BINARY" update > "$WORK/src.log" 2>&1
check "a source build refuses by name" "$(has "$WORK/src.log" "built from source: git pull and rebuild")"

echo "[2/7] update and rollback"
publish good
upd
check "sushi update installs $NEXT" "$(has "$WORK/upd.log" "updated $CUR -> $NEXT")"
check "the install now runs $NEXT" "$(is "$(installed)" "$NEXT")"
check "the old install is kept as .previous" "$(is "$("$INST.previous/sushi" --version 2>/dev/null | head -1)" "sushi $CUR")"
check "the download folder is cleaned up" "$([ ! -e "$WORK/inst/.sushi-macos-arm64.update" ] && echo 1 || echo 0)"
upd
check "a second run finds nothing newer" "$(has "$WORK/upd.log" "sushi $NEXT is up to date")"
upd --rollback
check "--rollback swaps the previous install back" "$(has "$WORK/upd.log" "rolled back $NEXT -> $CUR")"
check "the install runs $CUR again" "$(is "$(installed)" "$CUR")"
check "... and keeps $NEXT as .previous" "$(is "$("$INST.previous/sushi" --version 2>/dev/null | head -1)" "sushi $NEXT")"
check "every step reached ~/.sushi/logs/update.log" "$(has "$HOME/.sushi/logs/update.log" "rolled back")"

echo "[3/7] the daily check's switches"
if boot --no-update-check; then
    check "--no-update-check turns it off by name" "$(has "$WORK/server.log" "\[update\] daily check off (--no-update-check)")"
fi
stop_server
rm -f "$HOME/.sushi/update-check.json"
API_HITS=$(wc -l < "$WORK/api.log")
"$BINARY" serve --model-dir "$EMPTY" --port "$PORT" --log-file off > "$WORK/src-serve.log" 2>&1 &
SPID=$!
for _ in $(seq 1 60); do curl -sf "$BASE/health" >/dev/null 2>&1 && break; sleep 0.5; done
sleep 2
# A source build stays silent and never reaches the release API, so it leaves no cache behind either.
check "a source build logs no update line" "$([ "$(grep -c '\[update\]\|is available' "$WORK/src-serve.log")" = 0 ] && echo 1 || echo 0)"
check "a source build never asks the release API" "$(is "$(wc -l < "$WORK/api.log")" "$API_HITS")"
check "a source build writes no check cache" "$([ ! -e "$HOME/.sushi/update-check.json" ] && echo 1 || echo 0)"
stop_server

echo "[4/7] the chat page's guard"
rm -f "$HOME/.sushi/update-check.json"
if boot; then
    check "the daily check is on by default" "$(has "$WORK/server.log" "\[update\] daily check on (default)")"
    wait_for available true 30
    check "/props reports $NEXT as available" "$(is "$(prop latest)" "$NEXT")"
    check "... with the release page" "$(is "$(prop url)" "https://github.com/beamivalice/sushi/releases/tag/v$NEXT")"
    check "... and no command for a release install" "$(is "$(prop command)" null)"
    check "the check logged one line" "$(is "$(grep -c "sushi $NEXT is available: run \`sushi update\`" "$WORK/server.log")" 1)"
    check "GET /v1/update is 405" "$(is "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/v1/update")" 405)"
    check "a POST without an Origin is 403" "$(is "$(post_update)" 403)"
    check "a POST from another origin is 403" "$(is "$(post_update -H 'Origin: http://evil.test')" 403)"
    check "a POST from another port is 403" "$(is "$(post_update -H "Origin: http://127.0.0.1:$API_PORT")" 403)"
    check "the refusals left the server up" "$(is "$(curl -s "$BASE/health")" '{"status":"ok"}')"
    stop_server
fi
if boot --api-key self-update-key; then
    wait_for available true 30
    check "under --api-key a keyless POST from loopback is 401" "$(is "$(post_update -H "Origin: $BASE")" 401)"
    stop_server
fi

echo "[5/7] update and restart from the chat page"
if boot; then
    wait_for available true 30
    PID_BEFORE="$SPID"
    check "POST /v1/update with the page's Origin is 202" "$(is "$(post_update -H "Origin: $BASE")" 202)"
    check "... naming both versions" "$(has "$WORK/post.json" "\"from\":\"$CUR\",\"to\":\"$NEXT\"")"
    wait_for current "$NEXT" 180
    check "the server came back as $NEXT" "$(is "$(prop current)" "$NEXT")"
    check "... on the same pid (replaced in place, never forked)" "$(kill -0 "$PID_BEFORE" 2>/dev/null && echo 1 || echo 0)"
    check "... with no update error" "$(is "$(prop error)" null)"
    check "... and nothing newer to offer" "$(is "$(prop available)" false)"
    check "the updater's steps reached the server's stderr" "$(has "$WORK/server.log" "updated $CUR -> $NEXT")"
    check "the relaunch reused the same argv" "$(has "$WORK/server.log" "relaunching $INST/sushi serve --model-dir $EMPTY --port $PORT --log-file off")"
    kill -HUP "$PID_BEFORE" 2>/dev/null
    sleep 1
    check "an ignored SIGHUP stayed ignored through both execs" "$(curl -sf "$BASE/health" >/dev/null && echo 1 || echo 0)"
    stop_server
fi

echo "[6/7] a failed update from the chat page relaunches the old install"
upd --rollback
publish corrupt
if boot; then
    wait_for available true 30
    check "POST /v1/update is 202" "$(is "$(post_update -H "Origin: $BASE")" 202)"
    for _ in $(seq 1 360); do
        e="$(prop error)"
        [ -n "$e" ] && [ "$e" != null ] && break
        sleep 0.5
    done
    check "/props names the failure" "$(prop error | grep -q 'SHA-256 mismatch' && echo 1 || echo 0)"
    check "the old install ($CUR) is serving again" "$(is "$(prop current)" "$CUR")"
    check "the install on disk is still $CUR" "$(is "$(installed)" "$CUR")"
    stop_server
fi

echo "[7/7] an updater that cannot start ends the server with a named non-zero exit"
if boot; then
    wait_for available true 30
    chmod -x "$INST/sushi"
    check "POST /v1/update is 202" "$(is "$(post_update -H "Origin: $BASE")" 202)"
    wait "$SPID"
    CODE=$?
    SPID=""
    check "the server exits 1 instead of stopping cleanly" "$(is "$CODE" 1)"
    check "... naming why" "$(has "$WORK/server.log" "\[update\] cannot run $INST/sushi")"
    chmod +x "$INST/sushi"
fi

echo "[brew] a Homebrew keg leaves every update to brew"
KEG="$WORK/brew/Cellar/sushi/$CUR/libexec"
BREW_SAYS="sushi was installed with Homebrew; run: brew upgrade sushi"
stage "$BINARY" "$KEG"
publish good
"$KEG/sushi" update > "$WORK/brew.log" 2>&1
CODE=$?
check "sushi update exits 0 naming brew upgrade" "$([ "$CODE" = 0 ] && grep -q "$BREW_SAYS" "$WORK/brew.log" && echo 1 || echo 0)"
"$KEG/sushi" update --rollback > "$WORK/brew.log" 2>&1
CODE=$?
check "... and so does --rollback" "$([ "$CODE" = 0 ] && grep -q "$BREW_SAYS" "$WORK/brew.log" && echo 1 || echo 0)"
check "... leaving the keg as it was" "$(is "$("$KEG/sushi" --version 2>/dev/null | head -1) $(ls -A "$WORK/brew/Cellar/sushi/$CUR")" "sushi $CUR libexec")"
"$KEG/sushi" update --check > "$WORK/brew.log" 2>&1
check "--check names $NEXT and brew upgrade" "$(has "$WORK/brew.log" "sushi $NEXT is available (this is $CUR): run \`brew upgrade sushi\`")"
if INST="$KEG" boot; then
    wait_for available true 30
    check "/props names the brew command for the chat page" "$(is "$(prop command)" "brew upgrade sushi")"
    check "the daily check's line names it too" "$(has "$WORK/server.log" "sushi $NEXT is available: run \`brew upgrade sushi\`")"
    check "POST /v1/update from the page is 409" "$(is "$(post_update -H "Origin: $BASE")" 409)"
    check "... naming brew upgrade" "$(has "$WORK/post.json" "$BREW_SAYS")"
    stop_server
fi

# Opt-in, with a model (two loads per section, each under the GPU lock): SELF_UPDATE_MODEL=<pack dir>.
if [ -n "${SELF_UPDATE_MODEL:-}" ]; then
    echo "[model 1/2] the SSD prefix cache written before a page update restores after it"
    publish good
    PROMPT="$(python3 -c 'print(" ".join(f"Line {i}: the quick brown fox jumps over the lazy dog." for i in range(300)))')"
    BODY="$(python3 -c 'import json,sys; print(json.dumps({"messages":[{"role":"user","content":sys.argv[1] + " Say OK."}],"max_tokens":4,"temperature":0}))' "$PROMPT")"
    cached() { curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY" | python3 -c "import json,sys; print(json.load(sys.stdin)['usage']['prompt_tokens_details']['cached_tokens'])" 2>/dev/null; }
    scripts/gpu-lock.sh acquire self-update
    stop_server
    (trap '' HUP; exec "$INST/sushi" serve --model "$SELF_UPDATE_MODEL" --port "$PORT" --prefix-cache-disk 4GB --ctx-size 8192 --log-file off) > "$WORK/model.log" 2>&1 &
    SPID=$!
    for _ in $(seq 1 1200); do curl -sf "$BASE/health" >/dev/null 2>&1 && break; sleep 0.5; done
    wait_for available true 30
    check "a cold prompt reads nothing from cache" "$(is "$(cached)" 0)"
    sleep 5
    check "POST /v1/update is 202" "$(is "$(post_update -H "Origin: $BASE")" 202)"
    wait_for current "$NEXT" 900
    check "the model server came back as $NEXT" "$(is "$(prop current)" "$NEXT")"
    CACHED="$(cached)"
    check "the same prompt restores from the SSD tier ($CACHED tokens)" "$([ "${CACHED:-0}" -gt 0 ] 2>/dev/null && echo 1 || echo 0)"
    stop_server
    scripts/gpu-lock.sh release self-update

    echo "[model 2/2] /update in sushi run on a real TTY restarts the chat on the same model"
    upd --rollback
    scripts/gpu-lock.sh acquire self-update
    python3 - "$INST" "$SELF_UPDATE_MODEL" "$PORT" "$NEXT" "$WORK/repl.txt" <<'EOF'
import os, pty, select, subprocess, sys, time
inst, model, port, nxt, out = sys.argv[1:6]
master, slave = pty.openpty()
child = subprocess.Popen([inst + "/sushi", "run", model, "--port", port, "--ctx-size", "4096", "--log-file", "off"],
                         stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
os.close(slave)
buf, state, mark, deadline = bytearray(), 0, 0, time.monotonic() + 1800
while time.monotonic() < deadline and child.poll() is None:
    if select.select([master], [], [], 0.25)[0]:
        try:
            data = os.read(master, 65536)
        except OSError:
            break
        buf.extend(data)
    if state == 0 and b"chat is live" in buf:
        os.write(master, b"/update\n")
        state, mark = 1, len(buf)
    elif state == 1 and b"chat is live" in buf[mark:]:
        os.write(master, b"/bye\n")
        state = 2
try:
    child.wait(timeout=30)
except subprocess.TimeoutExpired:
    child.kill()
open(out, "wb").write(buf)
sys.exit(0 if state == 2 and child.returncode == 0 and b"updated " in buf[mark:] else 1)
EOF
    REPL_OK=$?
    scripts/gpu-lock.sh release self-update
    check "/update updated, restarted the chat, and /bye exited 0" "$(is "$REPL_OK" 0)"
    check "... announcing $NEXT" "$(has "$WORK/repl.txt" "updating to sushi $NEXT")"
    check "the install now runs $NEXT" "$(is "$(installed)" "$NEXT")"
fi

echo
echo "  passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]

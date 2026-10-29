#!/bin/bash
# sushi-test: tier=hermetic
# Repo-hygiene gate: the prefix cache's SSD tier is on by default, so a test script that boots a server
# without its own cache root would read, write and sweep the real ~/.sushi/kv-cache.
#   [1] tests/lib/boot.sh boots every server on a private HOME and cache root under the run dir, whatever
#       the caller exported, and the user's kv-cache stays untouched
#   [2] tests/private_cache.sh (scripts not yet on the helper) gives a script a private root, wipes only a
#       root it made, and leaves a caller-set root alone
#   [3] every script that names a sushi binary boots through the helper, or still isolates the cache itself;
#       the scan runs on tests/ and on a fixture dir with one script that boots outside both
# Hermetic, no model, no GPU: the server is a stub.
set -u
cd "$(dirname "$0")/.." || exit 1
PASS=0
FAIL=0
check() { if [ "$2" = "1" ]; then PASS=$((PASS + 1)); echo "  PASS $1"; else FAIL=$((FAIL + 1)); echo "  FAIL $1"; fi; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "[1/3] the boot helper"
mkdir -p "$WORK/home/.sushi/kv-cache" "$WORK/pack" "$WORK/mine"
echo sentinel > "$WORK/home/.sushi/kv-cache/user-entry"
echo keep > "$WORK/mine/entry"
cat > "$WORK/stub" <<'PY'
#!/usr/bin/env python3
import os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
print("home: " + os.environ["HOME"], flush=True)
print("cache: " + os.environ["SUSHI_PREFIX_CACHE_DIR"], flush=True)
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
    def log_message(self, *a):
        pass
srv = HTTPServer(("127.0.0.1", int(sys.argv[sys.argv.index("--port") + 1])), H)
print("Model ready.", flush=True)
srv.serve_forever()
PY
chmod +x "$WORK/stub"
cat > "$WORK/boots.sh" <<EOF
#!/bin/bash
. "$PWD/tests/lib/boot.sh"
sushi_boot probe "$WORK/pack" > /dev/null || exit 1
cp "\$SUSHI_LOG" "$WORK/probe.log"
sushi_stop 0
echo "\$SUSHI_RUN_DIR"
EOF
run_dir="$(env HOME="$WORK/home" SUSHI_BIN="$WORK/stub" SUSHI_PREFIX_CACHE_DIR="$WORK/mine" GPU_LOCK_DIR="$WORK/lock.d" \
    GPU_LOCK_POLL_S=0.2 /bin/bash "$WORK/boots.sh")"
root="$(sed -n 's/^cache: //p' "$WORK/probe.log")"
srv_home="$(sed -n 's/^home: //p' "$WORK/probe.log")"
check "a boot gets a cache root under the run dir, not the caller's" "$([ "$root" = "$run_dir/probe.kv-cache" ] && echo 1 || echo 0)"
check "a boot gets a private HOME under the run dir" "$([ "$srv_home" = "$run_dir/probe.home" ] && echo 1 || echo 0)"
check "the run dir defaults under HOME/.sushi/runs" "$([[ "$run_dir" == "$WORK/home/.sushi/runs/"* ]] && echo 1 || echo 0)"
check "the user's kv-cache is untouched" "$([ "$(cat "$WORK/home/.sushi/kv-cache/user-entry")" = sentinel ] && echo 1 || echo 0)"
check "a caller-set root is not written or wiped" "$([ "$(cat "$WORK/mine/entry")" = keep ] && [ "$(ls "$WORK/mine" | wc -l | tr -d ' ')" = 1 ] && echo 1 || echo 0)"

echo "[2/3] private_cache.sh"
cat > "$WORK/probe.sh" <<EOF
#!/bin/bash
. "$PWD/tests/private_cache.sh"
echo "\$SUSHI_PREFIX_CACHE_DIR"
EOF
root="$(env -u SUSHI_PREFIX_CACHE_DIR HOME="$WORK/home" bash "$WORK/probe.sh")"
check "a script without a root gets one under HOME/.sushi/runs" "$([ "$root" = "$WORK/home/.sushi/runs/test-kv-cache/probe" ] && [ -d "$root" ] && echo 1 || echo 0)"
echo stale > "$root/stale"
env -u SUSHI_PREFIX_CACHE_DIR HOME="$WORK/home" bash "$WORK/probe.sh" > /dev/null
check "a root the helper made is wiped for the next run" "$([ ! -e "$root/stale" ] && echo 1 || echo 0)"
got="$(SUSHI_PREFIX_CACHE_DIR="$WORK/mine" HOME="$WORK/home" bash "$WORK/probe.sh")"
check "a caller-set root is kept and not wiped" "$([ "$got" = "$WORK/mine" ] && [ "$(cat "$WORK/mine/entry")" = keep ] && echo 1 || echo 0)"

echo "[3/3] every script that boots a binary isolates its cache"
# Scripts that name a sushi binary but neither use the helper nor isolate the cache; lib/ and run.sh are the helper.
unguarded_in() {
    local dir="$1" f out=""
    for f in $(find "$dir" -type f -name '*.sh' ! -path "$dir/lib/*" ! -name run.sh ! -name private_cache.sh ! -name test_private_cache.sh | sort); do
        grep -Eq '\$BINARY|\$BIN\b|\$\{BINARY|\$\{BIN\b|SUSHI_BIN|SUSHI_BINARY|zig-out/bin/sushi' "$f" || continue
        grep -q 'lib/boot.sh\|private_cache.sh\|SUSHI_PREFIX_CACHE_DIR' "$f" || out="$out ${f#"$dir"/}"
    done
    echo "$out"
}
unguarded="$(unguarded_in tests)"
check "no script boots at the real cache root (${unguarded:-none})" "$([ -z "$unguarded" ] && echo 1 || echo 0)"
mkdir -p "$WORK/fx/qwen4"
printf '#!/bin/bash\n. "$(dirname "$0")/lib/boot.sh"\nsushi_boot x "$P"\n' > "$WORK/fx/test_helper.sh"
printf '#!/bin/bash\n. "$(dirname "$0")/private_cache.sh"\n"$BINARY" --serve\n' > "$WORK/fx/test_legacy.sh"
printf '#!/bin/bash\n"$SUSHI_BIN" --serve --model "$P"\n' > "$WORK/fx/qwen4/test_outside.sh"
check "a fixture script that boots outside the helper is the only one flagged" "$([ "$(unguarded_in "$WORK/fx" | tr -d ' ')" = "qwen4/test_outside.sh" ] && echo 1 || echo 0)"

echo
echo "  passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]

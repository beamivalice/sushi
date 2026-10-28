#!/bin/bash
# The prefix cache's default tiers: no RAM retention, an SSD tier sized per model.
#
#   [1] defaults: one `Allocating ... SSD` line naming the sizing terms and the bound, no RAM line,
#       `/props settings.prefix_cache` = ram off + a disk budget, a growing chat reuses its prefix
#       (cached_tokens > 0), one `[disk-cache] usage` line per turn, and `/props` reports bytes in use
#   [2] `--prefix-cache-mem 1GB --prefix-cache-disk 3GB`: both `Allocating` lines, the SSD one naming the flag
#   [3] `--prefix-cache-disk off`: no SSD line, `/props` disk budget 0, no usage line
#
# Private cache dir per boot (SUSHI_PREFIX_CACHE_DIR) and HOME. `GPU_LOCK_OWNER=<name>` takes the GPU
# lock around each boot (scripts/gpu-lock.sh), the way a live run on a shared box must.
#
# Usage: SUSHI_MODELS_DIR=<dir> ./tests/test_prefix_cache_ssd_default.sh [model_dir] [port]

set -u

MODEL="${1:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw}"
PORT="${2:-19321}"
BINARY="${BINARY:-./zig-out/bin/sushi}"
BASE="http://127.0.0.1:$PORT"
LOCK="$(dirname "$0")/../scripts/gpu-lock.sh"
PASS=0
FAIL=0
check() { if [ "$2" = "1" ]; then PASS=$((PASS + 1)); echo "  PASS $1"; else FAIL=$((FAIL + 1)); echo "  FAIL $1"; fi; }
has() { grep -q -- "$1" "$LOG" && echo 1 || echo 0; }

[ -d "$MODEL" ] || { echo "SKIP: model dir not found: $MODEL"; exit 0; }
[ -x "$BINARY" ] || { echo "fail: build sushi first"; exit 1; }
command -v jq >/dev/null || { echo "needs jq"; exit 1; }
WORK="$(mktemp -d)"
LOG="$WORK/server.log"
SPID=""
LOCKED=""
stop_server() {
    [ -n "$SPID" ] && kill "$SPID" 2>/dev/null && wait "$SPID" 2>/dev/null
    SPID=""
    [ -n "$LOCKED" ] && "$LOCK" release "$GPU_LOCK_OWNER" >/dev/null 2>&1
    LOCKED=""
}
trap 'stop_server; rm -rf "$WORK"' EXIT

boot() {
    stop_server
    if [ -n "${GPU_LOCK_OWNER:-}" ]; then "$LOCK" acquire "$GPU_LOCK_OWNER" >/dev/null && LOCKED=1; fi
    rm -rf "$WORK/kv"
    : > "$LOG"
    HOME="$WORK" SUSHI_PREFIX_CACHE_DIR="$WORK/kv" "$BINARY" --model "$MODEL" --serve --port "$PORT" --log-level info "$@" > "$LOG" 2>&1 &
    SPID=$!
    for _ in $(seq 1 1500); do
        curl -sf "$BASE/health" >/dev/null 2>&1 && return 0
        kill -0 "$SPID" 2>/dev/null || break
        sleep 1
    done
    echo "  (server did not come up)"; tail -20 "$LOG"
    return 1
}
ask() { # prompt text -> cached_tokens
    jq -n --arg p "$1" '{model:"x",temperature:0,max_tokens:8,messages:[{role:"user",content:$p}]}' |
        curl -s -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @- |
        jq -r '.usage.prompt_tokens_details.cached_tokens // 0'
}
props() { curl -s "$BASE/props" | jq -r "$1"; }
words() { python3 -c "print(' '.join('word%d' % i for i in range($1)))"; }

echo "Prefix cache default tiers: $(basename "$MODEL") (port $PORT)"

echo "[1/3] defaults"
if boot; then
    check "one Allocating SSD line" "$([ "$(grep -c '^Allocating .* GB SSD for the prefix cache' "$LOG")" = 1 ] && echo 1 || echo 0)"
    check "the SSD line names the sizing terms and the bound" "$(has 'entries x .* tokens x .* KB + 2 GB; bound: ')"
    check "no RAM Allocating line" "$([ "$(grep -c 'GB RAM for the prefix cache' "$LOG")" = 0 ] && echo 1 || echo 0)"
    check "/props: RAM retention off" "$([ "$(props .settings.prefix_cache.ram_enabled)" = false ] && echo 1 || echo 0)"
    check "/props: a disk budget" "$([ "$(props .settings.prefix_cache.disk_bytes)" -gt 0 ] && echo 1 || echo 0)"
    ask "$(words 3000)" >/dev/null
    sleep 5
    cached=$(ask "$(words 3000) and then some more words")
    sleep 5
    check "a growing chat reuses its prefix (cached_tokens=$cached)" "$([ "$cached" -gt 1000 ] && echo 1 || echo 0)"
    check "a [disk-cache] usage line per turn" "$([ "$(grep -c '^\[disk-cache\] usage ' "$LOG")" -ge 2 ] && echo 1 || echo 0)"
    check "/props reports bytes in use" "$([ "$(props .settings.prefix_cache.disk_used_bytes)" -gt 0 ] && echo 1 || echo 0)"
else
    check "boot at defaults" 0
fi

echo "[2/3] --prefix-cache-mem 1GB --prefix-cache-disk 3GB"
if boot --prefix-cache-mem 1GB --prefix-cache-disk 3GB; then
    check "RAM line" "$(has 'Allocating 1.0 GB RAM for the prefix cache (--prefix-cache-mem)')"
    check "SSD line names the flag" "$(has 'Allocating 3.0 GB SSD for the prefix cache (--prefix-cache-disk)')"
    check "/props: RAM on, 3 GB disk" "$([ "$(props .settings.prefix_cache.ram_enabled)" = true ] && [ "$(props .settings.prefix_cache.disk_bytes)" = 3221225472 ] && echo 1 || echo 0)"
else
    check "boot with both flags" 0
fi

echo "[3/3] --prefix-cache-disk off"
if boot --prefix-cache-disk off; then
    check "no SSD Allocating line" "$([ "$(grep -c 'GB SSD for the prefix cache' "$LOG")" = 0 ] && echo 1 || echo 0)"
    check "/props: no disk budget" "$([ "$(props .settings.prefix_cache.disk_bytes)" = 0 ] && echo 1 || echo 0)"
    ask "$(words 100)" >/dev/null
    check "no usage line" "$([ "$(grep -c '^\[disk-cache\] usage ' "$LOG")" = 0 ] && echo 1 || echo 0)"
else
    check "boot with the SSD tier off" 0
fi

echo
echo "  passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]

#!/bin/bash
# sushi-test: tier=live boot=own pack=default boots=3
# The prefix cache's default tiers: no RAM retention, an SSD tier sized per model.
#
#   [1] defaults: one `Allocating ... SSD` line naming the sizing terms and the bound, no RAM line,
#       `/props settings.prefix_cache` = ram off + a disk budget, a growing chat reuses its prefix
#       (cached_tokens > 0), one `[disk-cache] usage` line per turn, and `/props` reports bytes in use
#   [2] `--prefix-cache-mem 1GB --prefix-cache-disk 3GB`: both `Allocating` lines, the SSD one naming the flag
#   [3] `--prefix-cache-disk off`: no SSD line, `/props` disk budget 0, no usage line
#
# The Allocating lines are boot-time lines and each arm changes the cache flags, so each arm is its own boot
# (private HOME and cache root per boot).

set -u
. "$(dirname "$0")/lib/boot.sh"
PASS=0
FAIL=0
check() { if [ "$2" = "1" ]; then PASS=$((PASS + 1)); echo "  PASS $1"; else FAIL=$((FAIL + 1)); echo "  FAIL $1"; fi; }
has() { grep -q -- "$1" "$LOG" && echo 1 || echo 0; }

command -v jq >/dev/null || { echo "needs jq"; exit 1; }
MODEL=$(sushi_pack qwen4) || exit $?

boot() { # label, flags...
    local label="$1"; shift
    sushi_stop 0
    sushi_boot "$label" "$MODEL" --log-level info "$@" || return 1
    LOG="$SUSHI_LOG"
    BASE="$SUSHI_URL"
}
ask() { # prompt text -> cached_tokens
    jq -n --arg p "$1" '{model:"x",temperature:0,max_tokens:8,messages:[{role:"user",content:$p}]}' |
        curl -s -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @- |
        jq -r '.usage.prompt_tokens_details.cached_tokens // 0'
}
props() { curl -s "$BASE/props" | jq -r "$1"; }
words() { python3 -c "print(' '.join('word%d' % i for i in range($1)))"; }

echo "Prefix cache default tiers: $(basename "$MODEL")"

echo "[1/3] defaults"
if boot ssd-default; then
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
if boot ssd-both --prefix-cache-mem 1GB --prefix-cache-disk 3GB; then
    check "RAM line" "$(has 'Allocating 1.0 GB RAM for the prefix cache (--prefix-cache-mem)')"
    check "SSD line names the flag" "$(has 'Allocating 3.0 GB SSD for the prefix cache (--prefix-cache-disk)')"
    check "/props: RAM on, 3 GB disk" "$([ "$(props .settings.prefix_cache.ram_enabled)" = true ] && [ "$(props .settings.prefix_cache.disk_bytes)" = 3221225472 ] && echo 1 || echo 0)"
else
    check "boot with both flags" 0
fi

echo "[3/3] --prefix-cache-disk off"
if boot ssd-off --prefix-cache-disk off; then
    check "no SSD Allocating line" "$([ "$(grep -c 'GB SSD for the prefix cache' "$LOG")" = 0 ] && echo 1 || echo 0)"
    check "/props: no disk budget" "$([ "$(props .settings.prefix_cache.disk_bytes)" = 0 ] && echo 1 || echo 0)"
    ask "$(words 100)" >/dev/null
    check "no usage line" "$([ "$(grep -c '^\[disk-cache\] usage ' "$LOG")" = 0 ] && echo 1 || echo 0)"
else
    check "boot with the SSD tier off" 0
fi

echo
sushi_stop 0
echo "  passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]

#!/bin/bash
# sushi-test: tier=live boot=own pack=default boots=2
# A model whose SSD prefix cache the load declined (the cache volume has no room above the 4 GB reserve) gets it back.
#
#   [1] the volume frees up: the boot prints the OFF line and serves with no tier; a settings save that leaves the
#       ask unnamed on a still-full volume prints nothing; once space is back the idle loop re-arms within its minute
#       (`[disk-cache] re-armed (free space returned)`), `/props` reports a disk budget, and a growing chat then
#       reuses its prefix from the SSD tier
#   [2] a setting changes: a boot on a full volume, then `POST /v1/settings` naming `prefix_cache_disk_gb` re-arms at
#       once (`re-armed (SSD-cache setting changed)`: a named size is the budget, the volume's room does not bound it)
#
# The cache root is a symlink into a 6 GB APFS sparse image, filled by `F_PREALLOCATE` (blocks reserved, nothing
# written, so no real disk is used) and emptied by deleting the file. Private HOME and cache root per boot.

set -u
. "$(dirname "$0")/lib/boot.sh"
PASS=0
FAIL=0
check() { if [ "$2" = "1" ]; then PASS=$((PASS + 1)); echo "  PASS $1"; else FAIL=$((FAIL + 1)); echo "  FAIL $1"; fi; }
yes_if() { if eval "$1"; then echo 1; else echo 0; fi; }

command -v jq >/dev/null || { echo "needs jq"; exit 1; }
command -v hdiutil >/dev/null || { echo "SKIP: needs hdiutil"; exit 3; }
MODEL=$(sushi_pack qwen4) || exit $?

: "${SUSHI_RUN_DIR:=$HOME/.sushi/runs/tests-$(date +%Y%m%d-%H%M%S)-$$}"
export SUSHI_RUN_DIR
mkdir -p "$SUSHI_RUN_DIR"
IMG="$SUSHI_RUN_DIR/rearm.sparseimage"
VOL="$SUSHI_RUN_DIR/rearm.vol"
FILL="$VOL/fill"
PREALLOC="$(mktemp -d)/prealloc.py"
cat > "$PREALLOC" << 'EOF'
import fcntl, os, struct, sys
F_PREALLOCATE, F_ALLOCATEALL, F_PEOFPOSMODE = 42, 4, 3
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT, 0o600)
fcntl.fcntl(fd, F_PREALLOCATE, struct.pack("Iiqqq", F_ALLOCATEALL, F_PEOFPOSMODE, 0, int(sys.argv[2]) << 30, 0))
os.close(fd)
EOF

cleanup() {
    sushi_stop 0
    hdiutil detach "$VOL" -force > /dev/null 2>&1
    rm -f "$IMG"
    rm -rf "$(dirname "$PREALLOC")"
}
trap cleanup EXIT

hdiutil create -size 6g -type SPARSE -fs APFS -volname rearm "$IMG" > /dev/null 2>&1 || { echo "cannot create the volume image"; exit 1; }
mkdir -p "$VOL"
hdiutil attach "$IMG" -mountpoint "$VOL" -nobrowse > /dev/null 2>&1 || { echo "cannot attach the volume image"; exit 1; }

fill() { python3 -I "$PREALLOC" "$FILL" "$1"; }
free_gb() { df -g "$VOL" | awk 'NR == 2 { print $4 }'; }

boot() { # label -> the cache root of the label is a directory of the small volume
    mkdir -p "$VOL/$1"
    ln -sfn "$VOL/$1" "$SUSHI_RUN_DIR/$1.kv-cache"
    sushi_boot "$1" "$MODEL" --log-level info || return 1
    LOG="$SUSHI_LOG"
    BASE="$SUSHI_URL"
}
ask() { # prompt text -> cached_tokens
    jq -n --arg p "$1" '{model:"x",temperature:0,max_tokens:8,messages:[{role:"user",content:$p}]}' |
        curl -s -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @- |
        jq -r '.usage.prompt_tokens_details.cached_tokens // 0'
}
props() { curl -s "$BASE/props" | jq -r "$1"; }
words() { python3 -I -c "print(' '.join('word%d' % i for i in range($1)))"; }
save_gb() { curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/v1/settings" -H 'Content-Type: application/json' -d "{\"server\":{\"prefix_cache_disk_gb\":$1}}"; }
wait_for() { # pattern, seconds
    local n=0
    while [ "$n" -lt "$2" ]; do
        grep -q -- "$1" "$LOG" && return 0
        sleep 2
        n=$((n + 2))
    done
    return 1
}
rearmed() { grep -c '\[disk-cache\] re-armed' "$LOG"; }
reuse() { # a growing chat: the second turn must come from the tier
    ask "$(words 3000)" > /dev/null
    sleep 5
    local cached
    cached=$(ask "$(words 3000) and then some more words")
    sleep 5
    echo "$cached"
}

echo "Prefix cache re-arm: $(basename "$MODEL")"

echo "[1/2] space returns"
fill 4
check "the volume starts under the reserve ($(free_gb) GB free)" "$(yes_if '[ "$(free_gb)" -lt 4 ]')"
if boot rearm-space; then
    check "the boot declines the tier on the full volume" "$(yes_if 'grep -q "SSD prefix cache OFF" "$LOG"')"
    check "/props: no disk budget" "$(yes_if '[ "$(props .settings.prefix_cache.disk_bytes)" = 0 ]')"
    check "a save of the default size on the still-full volume is accepted" "$(yes_if '[ "$(save_gb null)" = 200 ]')"
    sleep 8
    check "a check that fails prints nothing: no re-armed line" "$(yes_if '[ "$(rearmed)" = 0 ]')"
    rm -f "$FILL"
    check "the volume has room again ($(free_gb) GB free)" "$(yes_if '[ "$(free_gb)" -gt 4 ]')"
    check "the idle loop re-arms within its minute" "$(yes_if 'wait_for "re-armed (free space returned)" 150')"
    check "one re-armed line" "$(yes_if '[ "$(rearmed)" = 1 ]')"
    check "/props: a disk budget" "$(yes_if '[ "$(props .settings.prefix_cache.disk_bytes)" -gt 0 ]')"
    cached=$(reuse)
    check "a growing chat reuses its prefix from the tier (cached_tokens=$cached)" "$(yes_if '[ "$cached" -gt 1000 ]')"
    check "a [disk-cache] usage line is printed" "$(yes_if '[ "$(grep -c "^\[disk-cache\] usage " "$LOG")" -ge 1 ]')"
    check "/props reports bytes in use" "$(yes_if '[ "$(props .settings.prefix_cache.disk_used_bytes)" -gt 0 ]')"
else
    check "boot on a full volume" 0
fi
sushi_stop 0

echo "[2/2] a setting changes"
fill 4
if boot rearm-setting; then
    check "the boot declines the tier on the full volume" "$(yes_if 'grep -q "SSD prefix cache OFF" "$LOG"')"
    check "no re-armed line at boot" "$(yes_if '[ "$(rearmed)" = 0 ]')"
    check "a named size is accepted" "$(yes_if '[ "$(save_gb 1)" = 200 ]')"
    check "the save re-arms within seconds" "$(yes_if 'wait_for "re-armed (SSD-cache setting changed)" 20')"
    check "/props: the named 1 GB budget" "$(yes_if '[ "$(props .settings.prefix_cache.disk_bytes)" = 1073741824 ]')"
    rm -f "$FILL"
    cached=$(reuse)
    check "a growing chat reuses its prefix from the tier (cached_tokens=$cached)" "$(yes_if '[ "$cached" -gt 1000 ]')"
else
    check "boot on a full volume" 0
fi

echo
sushi_stop 0
echo "  passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]

#!/bin/bash
# sushi-test: tier=hermetic
# Repo-hygiene gate: the prefix cache's SSD tier is on by default, so a test script that boots a server
# without its own cache root would read, write and sweep the real ~/.sushi/kv-cache.
#   [1] tests/private_cache.sh gives a script a private root under HOME, wipes only a root it made,
#       and leaves a caller-set root (and a sentinel beside it) untouched
#   [2] every tests/*.sh that names a sushi binary sources it or sets SUSHI_PREFIX_CACHE_DIR itself
# Hermetic, no model, no GPU.
set -u
cd "$(dirname "$0")/.." || exit 1
PASS=0
FAIL=0
check() { if [ "$2" = "1" ]; then PASS=$((PASS + 1)); echo "  PASS $1"; else FAIL=$((FAIL + 1)); echo "  FAIL $1"; fi; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "[1/2] the helper"
mkdir -p "$WORK/home/.sushi/kv-cache"
echo sentinel > "$WORK/home/.sushi/kv-cache/user-entry"
cat > "$WORK/probe.sh" <<EOF
#!/bin/bash
. "$PWD/tests/private_cache.sh"
echo "\$SUSHI_PREFIX_CACHE_DIR"
EOF
root="$(env -u SUSHI_PREFIX_CACHE_DIR HOME="$WORK/home" bash "$WORK/probe.sh")"
check "a script without a root gets one under HOME/.sushi/runs" "$([ "$root" = "$WORK/home/.sushi/runs/test-kv-cache/probe" ] && [ -d "$root" ] && echo 1 || echo 0)"
check "the user's kv-cache is untouched" "$([ "$(cat "$WORK/home/.sushi/kv-cache/user-entry")" = sentinel ] && echo 1 || echo 0)"
echo stale > "$root/stale"
env -u SUSHI_PREFIX_CACHE_DIR HOME="$WORK/home" bash "$WORK/probe.sh" > /dev/null
check "a root the helper made is wiped for the next run" "$([ ! -e "$root/stale" ] && echo 1 || echo 0)"
mkdir -p "$WORK/mine"
echo keep > "$WORK/mine/entry"
got="$(SUSHI_PREFIX_CACHE_DIR="$WORK/mine" HOME="$WORK/home" bash "$WORK/probe.sh")"
check "a caller-set root is kept and not wiped" "$([ "$got" = "$WORK/mine" ] && [ "$(cat "$WORK/mine/entry")" = keep ] && echo 1 || echo 0)"

echo "[2/2] every script that boots a binary isolates its cache"
unguarded=""
for f in tests/*.sh; do
    case "$f" in tests/private_cache.sh | tests/test_private_cache.sh) continue ;; esac
    grep -Eq '\$BINARY|\$BIN\b|\$\{BINARY|\$\{BIN\b|SUSHI_BIN|SUSHI_BINARY|zig-out/bin/sushi' "$f" || continue
    grep -q 'private_cache.sh\|SUSHI_PREFIX_CACHE_DIR' "$f" || unguarded="$unguarded $f"
done
check "no script boots at the real cache root (${unguarded:-none})" "$([ -z "$unguarded" ] && echo 1 || echo 0)"

echo
echo "  passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]

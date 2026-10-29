#!/bin/bash
# sushi-test: tier=hermetic
# test_build_mlx_pins.sh — build-mlx.sh refuses a submodule checkout that is not the
# revision the superproject pins (a pull without `git submodule update`), before building.
#
# Usage: ./tests/test_build_mlx_pins.sh

set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PASS=0
FAIL=0
ok()   { PASS=$((PASS + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }

g() { git -c user.name=t -c user.email=t@t -c protocol.file.allow=always "$@"; }

for m in mlx mlxc; do
  g init -q "$TMP/$m"
  touch "$TMP/$m/CMakeLists.txt"
  g -C "$TMP/$m" add CMakeLists.txt
  g -C "$TMP/$m" commit -qm old
  echo next > "$TMP/$m/NEXT"
  g -C "$TMP/$m" add NEXT
  g -C "$TMP/$m" commit -qm new
done

SUPER="$TMP/super"
g init -q "$SUPER"
mkdir -p "$SUPER/scripts" "$SUPER/patches" "$SUPER/lib"
cp "$ROOT/scripts/build-mlx.sh" "$SUPER/scripts/"
cp "$ROOT/patches/mlxc-gather-qmm-global-scale.patch" "$SUPER/patches/"
g -C "$SUPER" submodule add -q "$TMP/mlx" lib/mlx-src
g -C "$SUPER" submodule add -q "$TMP/mlxc" lib/mlxc-src
g -C "$SUPER" add -A
g -C "$SUPER" commit -qm pin

echo "== stale submodule refused =="
g -C "$SUPER/lib/mlx-src" checkout -q HEAD~1
out="$(cd "$SUPER" && bash scripts/build-mlx.sh 2>&1)"; rc=$?
if [ $rc -ne 0 ] && echo "$out" | grep -q "git submodule update --init"; then
  ok "stale lib/mlx-src refused with the update command"
else
  fail "stale lib/mlx-src not refused (rc=$rc): $(echo "$out" | head -3)"
fi
if echo "$out" | grep -q "cmake\|SDK"; then fail "refusal came after the build started"; else ok "refused before building"; fi

echo "== a staged pin bump is the pin =="
g -C "$SUPER" add lib/mlx-src
out="$(cd "$SUPER" && bash scripts/build-mlx.sh 2>&1)"
if echo "$out" | grep -q "pins"; then fail "staged bump refused: $(echo "$out" | head -1)"; else ok "staged bump passes the pin check"; fi

echo "== $PASS pass, $FAIL fail =="
[ "$FAIL" -eq 0 ]

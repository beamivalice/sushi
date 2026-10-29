#!/bin/bash
# sushi-test: tier=host
. "$(dirname "$0")/private_cache.sh"
# test_serving_deps.sh — the served binary's runtime dependency and refusal contract.
#
# This engine serves mimo_v2 / qwen4_exp / EXL3 on our self-built MLX, and
# nothing else. A stray `linkSystemLibrary(...)` re-links silently and can make
# the binary unlaunchable without a staged dylib no served path ever calls.
#
# Checks:
#   1. zig-out/bin/sushi exists (build it first)
#   2. otool -L lists no non-system dylib beyond the staged MLX and libwebp
#   3. the staged MLX is still linked (the contract is "no stray deps", not "no deps")
#   4. an unknown flag is REJECTED BY NAME, never silently eaten — a script that
#      passes one must fail loudly, not serve under different settings (the
#      flag-eater rule, docs/server-lifecycle.md)
#   5. a checkpoint in an unsupported file format is refused by name before load
#
# No model, no server, seconds.
#
# Usage: ./tests/test_serving_deps.sh

set -u
cd "$(dirname "$0")/.." || exit 1

BIN="zig-out/bin/sushi"
PASS=0
FAIL=0

ok()   { PASS=$((PASS + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }

echo "== binary =="
if [ -x "$BIN" ]; then
  ok "$BIN exists"
else
  echo "  FAIL: $BIN missing — run: ./.zig-toolchain/zig build -Doptimize=ReleaseFast"
  exit 1
fi

echo "== linked libraries =="
LINKED=$(otool -L "$BIN" | tail -n +2)
echo "$LINKED" | sed 's/^/    /'

STRAY=$(echo "$LINKED" | awk '{print $1}' | grep -v -E '^(/System/|/usr/lib/)' | grep -v -E '/libmlxc\.dylib$|/libwebp[.0-9]*\.dylib$')
if [ -n "$STRAY" ]; then
  fail "unexpected non-system dylib(s): $(echo "$STRAY" | tr '\n' ' ')"
else
  ok "no non-system dylib beyond libmlxc and libwebp"
fi

if echo "$LINKED" | grep -q "libmlxc"; then
  ok "libmlxc still linked (self-built MLX)"
else
  fail "libmlxc missing — the served path needs the staged MLX"
fi

echo "== unknown flags rejected by name =="
check_flag() {
  # "$@" is the flag as a launch script would pass it.
  OUT=$("$BIN" "$@" 2>&1)
  RC=$?
  if [ "$RC" -eq 0 ]; then
    fail "$1 exited 0 — the flag was silently eaten"
  elif echo "$OUT" | grep -q -- "$1"; then
    ok "$1 refused by name"
  else
    fail "$1 exited $RC but never named itself: $(echo "$OUT" | head -1)"
  fi
}
check_flag --no-such-flag 8
check_flag --engine mlx
# Removed launch flags: each must stay refused, not quietly accepted again.
check_flag --no-safety
check_flag --warmup-eager
check_flag --no-warmup-eager
check_flag --kv-attn-mode auto
check_flag --decode-attn-quant
check_flag --no-decode-attn-quant
check_flag --mtp-depth 3
check_flag --mtp-history-window 0
check_flag --draft-block-size 4
check_flag --tokenize-cache-entries 4
check_flag --prefill-trace
check_flag --hf-overrides '{}'
check_flag --mtp-cascade 0.95
check_flag --no-prefix-cache-ram
check_flag --fast
check_flag --pld
check_flag --no-pld
check_flag --pld-draft-len 5
check_flag --pld-key-len 3
check_flag --ssm-checkpoint-stride 256
check_flag --expert-cache-gb 60
check_flag --embedding-max-length 512
check_flag --logit-bias-file bias.json

echo "== unsupported checkpoint format refused by name =="
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
printf 'GGUF' > "$TMP/model.gguf"
OUT=$("$BIN" --model "$TMP/model.gguf" 2>&1)
RC=$?
if [ "$RC" -ne 0 ] && echo "$OUT" | grep -q "unsupported model format"; then
  ok "unsupported format refused by name (exit $RC)"
else
  fail "unsupported format not refused by name (exit $RC): $(echo "$OUT" | tail -1)"
fi

echo
echo "PASS: $PASS  FAIL: $FAIL"
[ "$FAIL" -eq 0 ]

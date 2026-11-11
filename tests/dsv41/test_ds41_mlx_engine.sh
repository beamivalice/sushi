#!/bin/bash
# sushi-test: tier=live boot=own pack=dsv41 boots=2
# `--ds41-mlx-engine` moves DeepSeek-V4.1's decode from Sushi Cooler to MLX at the load and changes no byte.
# Boot 1 (default) records a greedy reply and the decode-side load line; boot 2 (`--ds41-mlx-engine`) must say MLX
# decodes because the flag was given, report `ds41_cooler` locked by the flag on /v1/settings, and answer with the
# same bytes. A Mac without matrix units runs both boots on MLX; the line then names the hardware, and the script says so.
# COOLER_BOOT=default|mlx runs one boot only, keeping its reply in COOLER_REF (a directory the caller names), and the
# second one compares: a caller that wants a notice around each load runs the two apart.
set -u
. "$(dirname "$0")/../lib/boot.sh"
RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
MODEL=$(sushi_pack dsv41) || exit $?
WORK=${COOLER_REF:-$(mktemp -d)}
mkdir -p "$WORK"
[ -n "${COOLER_REF:-}" ] || trap 'rm -rf "$WORK"' EXIT
fail() { echo -e "${RED}FAIL${NC} $*"; exit 1; }

PAYLOAD='{"model":"sushi","messages":[{"role":"user","content":"Name the three primary colors, in one short sentence."}],"max_tokens":48,"temperature":0.0,"stream":false}'
post() { curl -s -m 300 -X POST -H "Content-Type: application/json" -d @- "$SUSHI_URL/v1/chat/completions"; }
# The reasoning text counts: a short budget can end inside the thinking, and an empty answer would compare equal for free.
text_of() { python3 -c "import sys, json; m = json.load(sys.stdin)['choices'][0]['message']; print((m.get('reasoning_content') or '') + '|' + (m.get('content') or ''))"; }
FLAGS=$(sushi_profile_flags exact dsv41) || exit $?

BOOT=${COOLER_BOOT:-both}
if [ "$BOOT" = both ] || [ "$BOOT" = default ]; then
echo "== boot 1: the default =="
# shellcheck disable=SC2086
sushi_boot cooler-default "$MODEL" $FLAGS || exit $?
LINE1=$(grep -a "\[dsv41\] decode forwards" "$SUSHI_LOG" | tail -1)
[ -n "$LINE1" ] || fail "the default boot logged no decode-side line"
echo "$PAYLOAD" | post | text_of > "$WORK/default.txt"
[ "$(wc -c < "$WORK/default.txt")" -gt 40 ] || fail "the default boot answered next to nothing: $(cat "$WORK/default.txt")"
FLAG1=$(curl -s "$SUSHI_URL/v1/settings" | python3 -c "import sys,json; d=json.load(sys.stdin); print([m['settings']['ds41_cooler']['flag'] for m in d['models'] if 'ds41_cooler' in m['settings']][0])")
[ "$FLAG1" = None ] || fail "the default boot reports ds41_cooler locked by '$FLAG1'"
# The engine's own counter: a boot that decodes on Sushi Cooler has dispatched work, a boot that does not has none.
dispatches() { grep -a "\[engine\] since load" "$SUSHI_LOG" | tail -1 | sed -E 's/.*since load: ([0-9]+) dispatches.*/\1/'; }
case "$LINE1" in
    *"on Sushi Cooler"*) [ "$(dispatches)" -gt 0 ] 2> /dev/null || fail "the default boot says Sushi Cooler decodes but the engine dispatched nothing" ;;
esac
echo "  default engine dispatches: $(dispatches)"
sushi_stop 0
echo "  default: $LINE1"
echo "$LINE1" > "$WORK/default.line"
fi

if [ "$BOOT" = both ] || [ "$BOOT" = mlx ]; then
echo "== boot 2: --ds41-mlx-engine =="
# shellcheck disable=SC2086
sushi_boot cooler-mlx "$MODEL" $FLAGS --ds41-mlx-engine || exit $?
LINE2=$(grep -a "\[dsv41\] decode forwards" "$SUSHI_LOG" | tail -1)
case "$LINE2" in
    *"on MLX because --ds41-mlx-engine was given (Sushi Cooler is off)"*) ;;
    *"Sushi Cooler is proven on a GPU with matrix units only"* | *"Sushi Cooler has no quantized_matmul"*)
        echo "  NOTE: this Mac or pack cannot run Sushi Cooler; the flag has nothing to move: $LINE2" ;;
    *) fail "boot 2 line does not say the flag moved decode to MLX: '$LINE2'" ;;
esac
echo "$PAYLOAD" | post | text_of > "$WORK/mlx.txt"
FLAG2=$(curl -s "$SUSHI_URL/v1/settings" | python3 -c "import sys,json; d=json.load(sys.stdin); m=[m for m in d['models'] if 'ds41_cooler' in m['settings']][0]['settings']['ds41_cooler']; print(m['flag'], m['locked'])")
[ "$FLAG2" = "--ds41-mlx-engine True" ] || fail "boot 2 reports ds41_cooler as '$FLAG2'"
D2=$(grep -a "\[engine\] since load" "$SUSHI_LOG" | tail -1 | sed -E 's/.*since load: ([0-9]+) dispatches.*/\1/')
case "$D2" in "" | 0) ;; *) fail "the MLX boot ran $D2 engine dispatches" ;; esac
echo "  mlx engine dispatches: ${D2:-none logged}"
sushi_stop 0
echo "$LINE2" > "$WORK/mlx.line"
fi

[ "$BOOT" != default ] || { echo "PASS (boot 1 only): reply kept in $WORK"; exit 0; }
[ -s "$WORK/default.txt" ] || fail "no default reply to compare with (run COOLER_BOOT=default first)"
if cmp -s "$WORK/default.txt" "$WORK/mlx.txt"; then
    echo -e "${GREEN}PASS${NC} the same greedy reply on Sushi Cooler and on MLX ($(wc -c < "$WORK/default.txt" | tr -d ' ') bytes); md5 $(md5 -q "$WORK/default.txt")"
else
    echo -e "${RED}FAIL${NC} the replies differ:"; diff "$WORK/default.txt" "$WORK/mlx.txt" | sed 's/^/    /'; exit 1
fi

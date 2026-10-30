#!/bin/bash
# sushi-test: tier=host
# `sushi bench --model`: the bench in this process on a tiny DeepSeek fixture with a byte-level tokenizer made here.
# A rung subset and a short decode run through the serve path with no listener; a repeat of the 4k prompt restores
# from the prefix cache and the table reports it. The tiny fixture is ${SUSHI_MODELS_DIR}/test-fixtures/dsv41-dspark-tiny
# (BENCH_TINY_MODEL overrides); SKIPs without it.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BINARY:-$ROOT/zig-out/bin/sushi}"
LOCK="$ROOT/scripts/gpu-lock.sh"
FIXTURE="${BENCH_TINY_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/test-fixtures/dsv41-dspark-tiny}"
PASS=0
FAIL=0

check() {
    if [ "$2" = "1" ]; then PASS=$((PASS + 1)); echo "  PASS $1"; else FAIL=$((FAIL + 1)); echo "  FAIL $1"; fi
}
has() { grep -q -- "$2" "$1" && echo 1 || echo 0; }

[ -x "$BIN" ] || { echo "SKIP: no binary at $BIN (zig build -Doptimize=ReleaseFast)"; exit 3; }
[ -f "$FIXTURE/config.json" ] || { echo "SKIP: no tiny fixture at $FIXTURE"; exit 3; }
command -v python3 >/dev/null || { echo "SKIP: python3 not found"; exit 3; }

WORK="$(mktemp -d)"
owner="bench-inprocess-$$"
have_lock=0
cleanup() {
    [ "$have_lock" -eq 1 ] && "$LOCK" release "$owner" >/dev/null 2>&1
    rm -rf "$WORK"
}
trap cleanup EXIT

# The fixture's weights, linked; a config whose context holds the 4k rung; a tokenizer with one token a byte of the corpus.
MODEL="$WORK/tiny"
mkdir -p "$MODEL" "$WORK/home"
for f in "$FIXTURE"/model-*.safetensors "$FIXTURE"/model.safetensors.index.json "$FIXTURE"/engram_token_map.json; do
    [ -e "$f" ] && ln -s "$f" "$MODEL/$(basename "$f")"
done
python3 -I - "$FIXTURE/config.json" "$MODEL" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
text = cfg["text_config"]
text["max_position_embeddings"] = 16384
text["rope_scaling"]["original_max_position_embeddings"] = 4096
text["rope_scaling"]["factor"] = 4
json.dump(cfg, open(sys.argv[2] + "/config.json", "w"))
# GPT-2's byte-to-unicode map for the printable ASCII bytes, the space and the newline: 96 ids, the fixture's vocabulary.
symbols = [chr(b) for b in range(0x21, 0x7F)] + ["Ġ", "Ċ"]
assert len(symbols) == text["vocab_size"] == 96
tok = {"model": {"type": "BPE", "vocab": {s: i for i, s in enumerate(symbols)}, "merges": []},
       "pre_tokenizer": {"type": "ByteLevel"}, "added_tokens": []}
json.dump(tok, open(sys.argv[2] + "/tokenizer.json", "w"))
json.dump({}, open(sys.argv[2] + "/tokenizer_config.json", "w"))
PY

OUT="$WORK/run"
echo "── sushi bench --model: a rung subset on the tiny fixture"
"$LOCK" acquire "$owner" >/dev/null || { echo "FAIL: could not take the GPU lock"; exit 1; }
have_lock=1
"$BIN" bench --model "$MODEL" --rungs 4k --tokens 8 --code-only --ssd-budget-gb 8 --out "$OUT" \
    > "$WORK/stdout.txt" 2> "$WORK/stderr.txt"
rc=$?
"$LOCK" release "$owner" >/dev/null 2>&1
have_lock=0
[ "$rc" -ne 0 ] && { echo "--- stderr tail"; tail -20 "$WORK/stderr.txt"; }
sed 's/^/  | /' "$WORK/stdout.txt"
check "exits 0" "$([ "$rc" -eq 0 ] && echo 1 || echo 0)"
check "the table's header names its columns" "$(has "$WORK/stdout.txt" 'decode tok/s')"
check "the table has a code row at 4k" "$(grep -qE '^code +4k ' "$WORK/stdout.txt" && echo 1 || echo 0)"
check "the prose arm did not run" "$([ "$(grep -c '^prose' "$WORK/stdout.txt")" -eq 0 ] && echo 1 || echo 0)"
restored="$(sed -n 's/.*\[bench\] code 4096 repeat: restored \([0-9]*\),.*/\1/p' "$WORK/stderr.txt" | head -1)"
check "a repeat of the 4k prompt reports a restore (restored ${restored:-none})" "$([ "${restored:-0}" -ge 4000 ] && echo 1 || echo 0)"
check "bench.json holds the 3 samples of the rung" "$(python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); print(1 if len(d["samples"])==3 and d["url"] is None else 0)' "$OUT/bench.json" 2>/dev/null || echo 0)"
check "the run's server.log is in --out and names no listener" "$([ -f "$OUT/server.log" ] && ! grep -q 'Server listening' "$OUT/server.log" && echo 1 || echo 0)"
check "the private prefix cache stays with --out" "$([ -d "$OUT/cache" ] && echo 1 || echo 0)"

echo "── the default --out is a fresh run directory and its cache is removed"
"$LOCK" acquire "$owner" >/dev/null || { echo "FAIL: could not take the GPU lock"; exit 1; }
have_lock=1
HOME="$WORK/home" "$BIN" bench --model "$MODEL" --rungs 4k --tokens 8 --code-only --ssd-budget-gb 8 \
    > "$WORK/stdout2.txt" 2> "$WORK/stderr2.txt"
rc=$?
"$LOCK" release "$owner" >/dev/null 2>&1
have_lock=0
run2="$(ls -d "$WORK"/home/.sushi/runs/bench-* 2>/dev/null | head -1)"
check "exits 0 without --out" "$([ "$rc" -eq 0 ] && echo 1 || echo 0)"
check "bench.json lands in ~/.sushi/runs/bench-<stamp>" "$([ -n "$run2" ] && [ -f "$run2/bench.json" ] && echo 1 || echo 0)"
check "the cache directory is gone" "$([ -n "$run2" ] && [ ! -e "$run2/cache" ] && echo 1 || echo 0)"

echo "── refusals happen before any load"
refuse() {
    local desc="$1"
    shift
    "$BIN" "$@" > "$WORK/r.out" 2> "$WORK/r.err"
    local code=$?
    check "$desc" "$([ "$code" -ne 0 ] && echo 1 || echo 0)"
}
refuse "a rung the ladder cannot land on" bench --model "$MODEL" --rungs 6k
refuse "both arm flags" bench --model "$MODEL" --code-only --prose-only
refuse "a serve flag that does not exist" bench --model "$MODEL" --no-such-flag
refuse "--port with no server to bind" bench --model "$MODEL" --port 9
refuse "--log in process mode" bench --model "$MODEL" --log x
refuse "no server at the --url" bench --url 127.0.0.1:1 --out "$WORK/u" --log "$WORK/u.log" --rungs 4k
check "the url mode still says no server answers" "$(has "$WORK/r.err" 'no server answers')"

echo
echo "bench in-process: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]

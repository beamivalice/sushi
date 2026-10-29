#!/bin/bash
. "$(dirname "$0")/../../private_cache.sh"
# test_mimo_batched_equivalence.sh — MiMo decodes concurrent plain slots as rows of one forward
# (`forwardMimoBatchedDecode`), and every row must be that slot's own decode tick. STRICT: each
# answer of a concurrent group equals the same request answered alone, byte for byte (greedy,
# thinking off, MTP and prompt lookup off since a speculating slot decodes serial, prefix cache
# off). Prompts sit below the 128 window, past it, past the 640-row ring cap and past the packed
# global arm's 4096 keys; groups of 2, 3 and 4 batch, and a group of 5 sends one slot serial by
# name (`row_cap`). The engagement line must appear. A second boot with MTP on (the default) answers
# each prompt alone and in crowded groups of 3 and 4 (MTP streams that decode as batched rows): every
# answer must be the serial bytes of the first boot.
#
# Env: SUSHI_MODELS_DIR (default $HOME/.sushi/models), MIMO_MODEL, PORT (default 19089), BINARY, MIMO_BATCH_OUTPUT_DIR.

set -uo pipefail

MODEL="${MIMO_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/MiMo-V2.6-Flash-Sushi-2.3bpw}"
PORT="${PORT:-19089}"
BIN="${BINARY:-./zig-out/bin/sushi}"
BASE="http://127.0.0.1:$PORT"

[ -d "$MODEL" ] || { echo "SKIP: model dir not found: $MODEL"; exit 0; }
[ -x "$BIN" ]   || { echo "fail: build sushi first"; exit 1; }
command -v jq >/dev/null || { echo "needs jq"; exit 1; }
curl -sf --max-time 2 "$BASE/health" >/dev/null 2>&1 && { echo "fail: port $PORT is busy"; exit 1; }

WORK="${MIMO_BATCH_OUTPUT_DIR:-$(mktemp -d)}"
mkdir -p "$WORK"
LOG="$WORK/server.log"
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    [ -n "${MIMO_BATCH_OUTPUT_DIR:-}" ] || rm -rf "$WORK"
}
trap cleanup EXIT

# Request bodies first: a malformed one fails here, before any model loads.
python3 - "$WORK" <<'PY' || { echo "fail: request bodies"; exit 1; }
import json, sys
work = sys.argv[1]
words = "the ring keeps a window of rows and compacts when the slack fills while global layers keep every key".split()
def filler(n):
    return " ".join(words[i % len(words)] for i in range(n))
asks = [
    "Name three primary colors, one per line.",
    filler(130) + "\nSummarize the text above in two sentences.",
    filler(700) + "\nList five words from the text above.",
    filler(4300) + "\nWhat is the text above about? One sentence.",
]
for i, a in enumerate(asks):
    body = {"messages": [{"role": "user", "content": a}], "max_tokens": 96, "temperature": 0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    s = json.dumps(body)
    json.loads(s)
    open(f"{work}/p{i}.json", "w").write(s)
PY

"$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --kv-quant 8 --no-mtp \
    --max-concurrent 5 --prefix-cache-entries 0 --log-level info > "$LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 900); do
    curl -sf --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"' && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "fail: server died:"; tail -20 "$LOG"; exit 1; }
    sleep 1
done

EC=0
fail() { echo "FAIL: $*"; EC=1; }
ask() { curl -sf --max-time 600 -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @"$1" | jq -e -j '.choices[0].message.content | select(type == "string")'; }

for i in 0 1 2 3; do
    ask "$WORK/p$i.json" > "$WORK/solo$i.txt" || { echo "fail: solo $i"; tail -20 "$LOG"; exit 1; }
done
group() { # label prompt indices...
    local label=$1; shift
    local pids=() k=0
    for i in "$@"; do
        ask "$WORK/p$i.json" > "$WORK/$label-$k.txt" &
        pids+=($!)
        k=$((k + 1))
    done
    if [[ "$label" == m3 || "$label" == m4 ]]; then
        local running
        : > "$WORK/$label-widths.txt"
        while true; do
            running=0
            for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && running=1; done
            [ "$running" = 1 ] || break
            curl -sf --max-time 2 "$BASE/metrics.json" | jq -r '.gauges.batched_group_size' >> "$WORK/$label-widths.txt"
            sleep 0.05
        done
        grep -qx "${#pids[@]}" "$WORK/$label-widths.txt" || fail "$label: expected batch width never engaged"
    fi
    for pid in "${pids[@]}"; do wait "$pid" || fail "$label: request failed"; done
    k=0
    for i in "$@"; do
        cmp -s "$WORK/solo$i.txt" "$WORK/$label-$k.txt" || fail "$label: slot $k (prompt $i) differs from its solo answer"
        k=$((k + 1))
    done
}
group g2 0 1
group g3 0 1 2
group g4 0 1 2 3
group g5 1 1 1 1 1

grep -q '\[batched\] mimo batched decode engaged' "$LOG" || fail "no batched MiMo decode engaged"
grep -q 'slot serial: row_cap' "$LOG" || fail "a fifth slot did not decode serial by name"
grep '\[batched\]' "$LOG" | sort | uniq -c

curl -sf --max-time 600 -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(jq '.stream = true' "$WORK/p1.json")" > "$WORK/stream.sse" || fail "stream request failed"
python3 - "$WORK" <<'PYCODE' || fail "stream differs from non-stream"
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
chunks = []
for line in (p / "stream.sse").read_text().splitlines():
    if line.startswith("data: ") and line[6:] != "[DONE]":
        choices = json.loads(line[6:]).get("choices", [])
        if choices:
            chunks.append(choices[0].get("delta", {}).get("content") or "")
assert "".join(chunks).encode() == (p / "solo1.txt").read_bytes()
PYCODE
# MTP on (the default): a crowded group of 3 or 4 MTP streams decodes as plain batched rows and its heads
# resume afterwards; every answer is still the serial answer above.
kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null
LOG="$WORK/server-mtp.log"
"$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --kv-quant 8 --mtp --metrics \
    --max-concurrent 5 --prefix-cache-entries 0 --log-level info > "$LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 900); do
    curl -sf --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"' && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "fail: MTP server died:"; tail -20 "$LOG"; exit 1; }
    sleep 1
done
for i in 0 1 2 3; do
    ask "$WORK/p$i.json" > "$WORK/mtp-solo$i.txt" || { echo "fail: MTP solo $i"; tail -20 "$LOG"; exit 1; }
    cmp -s "$WORK/solo$i.txt" "$WORK/mtp-solo$i.txt" || fail "MTP solo: prompt $i differs from its serial answer"
done
group m3 1 1 1
group m4 1 1 1 1
group mixed-m3 0 1 2
group mixed-m4 0 1 2 3
grep -q '\[spec-stats\] mode=mtp' "$LOG" || fail "no MTP rounds ran"
grep -q '\[batched\] mimo batched decode engaged' "$LOG" || fail "crowded MTP streams did not decode as batched rows"
grep '\[batched\]' "$LOG" | sort | uniq -c

[ $EC = 0 ] && echo "PASS"
exit $EC

#!/bin/bash
# sushi-test: tier=live boot=exact
# test_mimo_batched_equivalence.sh — MiMo decodes concurrent plain slots as rows of one forward
# (`forwardMimoBatchedDecode`), and every row must be that slot's own decode tick. STRICT: each
# answer of a concurrent group equals the same request answered alone, byte for byte (greedy,
# thinking off, MTP off per request since a speculating slot decodes serial, prefix cache off). Prompts sit
# below the 128 window, past it, past the 640-row ring cap and past the packed global arm's 4096 keys; groups of
# 2, 3 and 4 batch. The engagement line must appear. Then the same prompts with MTP on (the boot's default,
# fixed depth) answer alone and in crowded groups of 3 and 4 (MTP streams that decode as batched rows): every
# answer must be the serial bytes. A group of five sending one slot serial by name (`row_cap`) needs five slots;
# the exact boot serves four, and the row-cap bill is a unit test.

set -uo pipefail
. "$(dirname "$0")/../lib/boot.sh"
sushi_attach
sushi_require batching
sushi_require mtp
BASE="$SUSHI_URL"
command -v jq >/dev/null || { echo "needs jq"; exit 1; }

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Request bodies first: a malformed one fails here, before any model loads.
python3 - "$WORK" "$SUSHI_NONCE" <<'PY' || { echo "fail: request bodies"; exit 1; }
import json, sys
work, nonce = sys.argv[1:3]
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
    for arm, mtp in (("", False), ("m", True)):
        body = {"messages": [{"role": "user", "content": nonce + " " + a}], "max_tokens": 96, "temperature": 0,
                "stream": False, "enable_mtp": mtp, "chat_template_kwargs": {"enable_thinking": False}}
        s = json.dumps(body)
        json.loads(s)
        open(f"{work}/{arm}p{i}.json", "w").write(s)
PY

EC=0
fail() { echo "FAIL: $*"; EC=1; }
ask() { curl -sf --max-time 600 -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @"$1" | jq -e -j '.choices[0].message.content | select(type == "string")'; }

for i in 0 1 2 3; do
    ask "$WORK/p$i.json" > "$WORK/solo$i.txt" || { echo "fail: solo $i"; sushi_log_since | tail -20; exit 1; }
done
group() { # label prompt-prefix prompt indices...
    local label=$1 pre=$2; shift 2
    local pids=() k=0
    for i in "$@"; do
        ask "$WORK/${pre}p$i.json" > "$WORK/$label-$k.txt" &
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
group g2 "" 0 1
group g3 "" 0 1 2
group g4 "" 0 1 2 3

# Logged once per process, so an earlier script on this boot may have logged it: read the whole log.
grep -q '\[batched\] mimo batched decode engaged' "$SUSHI_LOG" || fail "no batched MiMo decode engaged"

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
# MTP on: a crowded group of 3 or 4 MTP streams decodes as plain batched rows and its heads resume afterwards.
mark=$(sushi_log_since | wc -c)
for i in 0 1 2 3; do
    ask "$WORK/mp$i.json" > "$WORK/mtp-solo$i.txt" || { echo "fail: MTP solo $i"; sushi_log_since | tail -20; exit 1; }
    cmp -s "$WORK/solo$i.txt" "$WORK/mtp-solo$i.txt" || fail "MTP solo: prompt $i differs from its serial answer"
done
group m3 m 1 1 1
group m4 m 1 1 1 1
group mixed-m3 m 0 1 2
group mixed-m4 m 0 1 2 3
MTP_LOG=$(sushi_log_since | tail -c +"$((mark + 1))")
grep -q '\[spec-stats\] mode=mtp' <<< "$MTP_LOG" || fail "no MTP rounds ran"

[ $EC = 0 ] && echo "PASS"
exit $EC

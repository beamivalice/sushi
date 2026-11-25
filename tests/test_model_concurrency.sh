#!/bin/bash
# sushi-test: tier=live boot=own pack=qwen4 boots=1
# A model's max_concurrent and max_kv_tokens hold, wait and refuse on a real pack: with max_concurrent 2 a third
# request waits for a slot; two requests whose reservations (prompt + max_tokens) pass max_kv_tokens together run one
# after the other; a request over the cap alone is refused 400 naming the cap, the same bytes streamed or not. MTP is
# off: two MTP slots share one head and serialize on their own, which would pass the first check without the cap.
. "$(dirname "$0")/lib/boot.sh"

PACK=${SUSHI_CONCURRENCY_PACK:-$(sushi_pack qwen4)} || exit $?
CAP=6000
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
python3 -I -c 'import json, sys; print(json.dumps({sys.argv[1]: {"max_concurrent": 2, "max_kv_tokens": int(sys.argv[2])}}))' \
    "$PACK" "$CAP" > "$WORK/settings.json"
SUSHI_BOOT_SETTINGS="$WORK/settings.json" sushi_boot concurrency "$PACK" --no-mtp --prefix-cache-disk off --log-level debug || exit $?

fail=0
check() { if [ "$1" = 1 ]; then echo "PASS: $2"; else echo "FAIL: $2 ($3)"; fail=1; fi; }

props=$(curl -sf "$SUSHI_URL/props")
# A load that decodes one row at a time holds every request to 1 whatever the cap, so it cannot test the caps.
served=$(python3 -I -c 'import json, sys; print(json.loads(sys.argv[1])["settings"]["max_concurrent"])' "$props")
if [ "$served" -lt 2 ]; then
    echo "SKIP: this load serves $served request at a time ($(grep -m1 '\[concurrency\]' "$SUSHI_LOG" || echo 'no [concurrency] line')); the caps need a load that batches"
    sushi_stop
    exit 3
fi
check "$(python3 -I -c 'import json, sys; s = json.loads(sys.argv[1])["settings"]; print(int(s["max_concurrent"] == 2 and s["max_kv_tokens"] == int(sys.argv[2])))' "$props" "$CAP")" \
    "/props reports the model's max_concurrent 2 and max_kv_tokens $CAP" "$(echo "$props" | head -c 300)"

# Each request decodes exactly max_tokens (greedy, ignore_eos), stamping when its first and last bytes arrived.
run() {
    local name="$1" max="$2"
    python3 -I - "$SUSHI_URL" "$max" > "$WORK/$name.json" <<'PY'
import json, sys, time, urllib.request
url, max_tokens = sys.argv[1], int(sys.argv[2])
body = json.dumps({"prompt": "Count upward from one, one number per line.", "max_tokens": max_tokens,
                   "temperature": 0, "ignore_eos": True, "stream": True}).encode()
req = urllib.request.Request(url + "/v1/completions", body, {"Content-Type": "application/json"})
t0 = time.time(); first = None
with urllib.request.urlopen(req, timeout=900) as r:
    for line in r:
        if first is None and line.startswith(b"data: {"):
            first = time.time()
print(json.dumps({"first": first - t0, "end": time.time() - t0}))
PY
}

# 1. max_concurrent 2: three requests of 1,000 tokens (3 x ~1,010 reserved, under the cap): the third starts after one ends.
pids=()
for n in a b c; do run "c-$n" 1000 & pids+=($!); sleep 0.3; done
wait "${pids[@]}"
third=$(python3 -I -c '
import json, sys
r = [json.load(open(f"{sys.argv[1]}/c-{n}.json")) for n in "abc"]
print(int(r[2]["first"] + 0.3 * 2 >= min(r[0]["end"], r[1]["end"] + 0.3) - 0.5))' "$WORK")
check "$third" "with max_concurrent 2 the third request starts only after one of the first two ends" "$(cat "$WORK"/c-*.json | tr '\n' ' ')"
check "$(grep -c '\[queue\].*max_concurrent' "$SUSHI_LOG" | awk '{print ($1 > 0)}')" "the queue line names max_concurrent" "no [queue] line names it"

# 2. max_kv_tokens: two requests of 3,400 tokens (2 x ~3,410 > 6,000): the second waits for the first.
run k-a 3400 & pa=$!; sleep 0.3; run k-b 3400 & pb=$!; wait "$pa" "$pb"
second=$(python3 -I -c '
import json, sys
a, b = (json.load(open(f"{sys.argv[1]}/k-{n}.json")) for n in "ab")
print(int(b["first"] + 0.3 >= a["end"] - 0.5))' "$WORK")
check "$second" "two requests over max_kv_tokens together run one after the other" "$(cat "$WORK"/k-*.json | tr '\n' ' ')"
check "$(grep -c '\[queue\].*max_kv_tokens' "$SUSHI_LOG" | awk '{print ($1 > 0)}')" "the queue line names max_kv_tokens" "no [queue] line names it"

# 3. A request over the cap alone is refused 400 naming the cap, streamed or not, the same bytes.
over() { curl -s -o "$WORK/over-$1.body" -w '%{http_code}' "$SUSHI_URL/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"hi\",\"max_tokens\":$((CAP + 100)),\"stream\":$1}"; }
code_n=$(over false); code_s=$(over true)
check "$([ "$code_n" = 400 ] && grep -q max_kv_tokens "$WORK/over-false.body" && echo 1 || echo 0)" \
    "a request over max_kv_tokens alone is a 400 naming the cap" "$code_n $(head -c 200 "$WORK/over-false.body")"
check "$([ "$code_s" = 400 ] && cmp -s "$WORK/over-false.body" "$WORK/over-true.body" && echo 1 || echo 0)" \
    "the streamed refusal is the same bytes" "$code_s $(head -c 200 "$WORK/over-true.body")"

sushi_stop
exit $fail

#!/bin/bash
# sushi-test: tier=live boot=own pack=default boots=2
# Qwen decode-end restore points (docs/engine-prefix-cache.md#decode-end-points): a turn that MTP ends on an
# end-of-turn token leaves a restore point right behind its reply, and the next turn (turn 1 + its reply + a
# follow-up) restores there, not at turn 1's prompt less 31. One boot per tier:
#
#  ram: RAM tier only (`--prefix-cache-disk 0`).
#  ssd: SSD tier only (`--prefix-cache-mem 0`): the point rides the record and turn 2 restores from the SSD.
#
#  A. A plain reply (thinking off): turn 1 logs `decode-end restore point at P1 + C1`, and turn 2 restores exactly
#     that many tokens (RAM: with nothing lost).
#  B. A thinking turn with tools: whatever the model answers, a point it takes is where turn 2 restores, and a turn
#     that takes none names why.
#
# Greedy; the reply's bytes are the model's choice, so B asserts the invariant, not the answer.

. "$(dirname "$0")/../lib/boot.sh"
MODEL=$(sushi_pack qwen4) || exit $?
FAIL=0

check() { # <description> <condition exit status> [detail]
    if [ "$2" -eq 0 ]; then
        echo "  PASS: $1"
    else
        echo "  FAIL: $1${3:+ ($3)}"
        FAIL=1
    fi
}

# two_turns <plain|tools> <ram|ssd>: runs turn 1 and turn 2 of a case and prints
# `P1 C1 FINISH POINT REUSED LOST REASON` from the usage and the log slice of each turn.
two_turns() {
    python3 -I - "$SUSHI_URL" "$SUSHI_LOG" "$1" "$2" "${SUSHI_NONCE:-de}" <<'PY'
import json, re, sys, time, urllib.request
url, log, case, tier, nonce = sys.argv[1:6]
para = ("The kingdom of Avalon was beset by trials. Each season brought new challenges to its people, "
        "but the king remained steadfast. ") * 40
tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file from the repository",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]

def log_size():
    with open(log, "rb") as f:
        return len(f.read())

def log_from(off):
    with open(log, "rb") as f:
        return f.read()[off:].decode(errors="replace")

def post(msgs, extra):
    body = {"model": "sushi", "messages": msgs, "max_tokens": 400, "temperature": 0, "stream": False}
    body.update(extra)
    off = log_size()
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    return out, off

if case == "plain":
    extra = {"enable_thinking": False}
    m1 = [{"role": "system", "content": tier + nonce + " You are a helpful assistant.\n\nBackground:\n" + para},
          {"role": "user", "content": "Count from 1 to 30, separated by commas, on one line."}]
    follow = {"role": "user", "content": "Now count from 30 down to 1 the same way."}
else:
    extra = {"tools": tools, "reasoning_effort": "low"}
    m1 = [{"role": "system", "content": tier + nonce + " You are a coding agent. Read files before you answer.\n\nNotes:\n" + para},
          {"role": "user", "content": "What does src/main.zig do? Read it first."}]
    follow = {"role": "user", "content": "Thanks. Summarize that in one sentence."}

r1, off1 = post(m1, extra)
p1, c1 = r1["usage"]["prompt_tokens"], r1["usage"]["completion_tokens"]
# The SSD record of the whole turn lands after the response: wait for it, or turn 2 restores the prefill's.
if tier == "ssd":
    for _ in range(120):
        if f"complete on disk: {p1 + c1} tokens" in log_from(off1):
            break
        time.sleep(0.25)
log1 = log_from(off1)
choice = r1["choices"][0]
msg = choice["message"]
point = re.search(r"\[prefix\] decode-end restore point at (\d+)", log1)
none = re.search(r"\[prefix\] no decode-end restore point: (\S+)", log1)
assistant = {"role": "assistant", "content": msg.get("content") or ""}
if msg.get("reasoning_content"):
    assistant["reasoning_content"] = msg["reasoning_content"]
if msg.get("tool_calls"):
    assistant["tool_calls"] = msg["tool_calls"]
    tail = {"role": "tool", "tool_call_id": msg["tool_calls"][0]["id"], "content": "pub fn main() void {}\n"}
else:
    tail = follow
r2, off2 = post(m1 + [assistant, tail], extra)
log2 = log_from(off2)
ram = re.search(r"\[hot-cache\] reused (\d+)/\d+ tokens \(matched \d+; entry \d+/\d+; lost (\d+)", log2)
ssd = re.search(r"\[prefix\] reused (\d+) of \d+ prompt tokens from SSD", log2)
reused, lost = (ram.group(1), ram.group(2)) if ram else ((ssd.group(1), "-") if ssd else (0, -1))
print(p1, c1, choice["finish_reason"], point.group(1) if point else "none", reused, lost, none.group(1) if none else "-")
PY
}

run_tier() { # <ram|ssd>
    local tier=$1 P1 C1 FIN POINT REUSED LOST WHY
    echo
    echo "--- $tier, A: a plain reply ---"
    read -r P1 C1 FIN POINT REUSED LOST WHY < <(two_turns plain "$tier")
    echo "  turn 1: prompt $P1 completion $C1 finish $FIN point $POINT ($WHY); turn 2 reused $REUSED, lost $LOST"
    check "turn 1 ends on its end-of-turn token" "$([ "$FIN" = stop ]; echo $?)" "finish_reason $FIN"
    check "turn 1 takes a decode-end point right behind its reply" "$([ "$POINT" = $((P1 + C1)) ]; echo $?)" "point $POINT, reason $WHY"
    check "turn 2 restores at the decode-end point" "$([ "$REUSED" = "$POINT" ]; echo $?)" "reused $REUSED"
    [ "$tier" = ram ] && check "turn 2 loses no matched token" "$([ "$LOST" = 0 ]; echo $?)" "lost $LOST"

    echo "--- $tier, B: a thinking turn with tools ---"
    read -r P1 C1 FIN POINT REUSED LOST WHY < <(two_turns tools "$tier")
    echo "  turn 1: prompt $P1 completion $C1 finish $FIN point $POINT ($WHY); turn 2 reused $REUSED, lost $LOST"
    if [ "$POINT" != none ]; then
        check "a point taken sits right behind the reply" "$([ "$POINT" = $((P1 + C1)) ]; echo $?)" "point $POINT"
        check "turn 2 restores at it" "$([ "$REUSED" = "$POINT" ]; echo $?)" "reused $REUSED"
    else
        check "a turn that takes no point names a reason" "$([ "$WHY" != - ]; echo $?)"
    fi
}

echo "=== Qwen decode-end restore points ==="
sushi_boot decode-end-ram "$MODEL" --prefix-cache-entries 4 --prefix-cache-mem 4GB --prefix-cache-disk 0 \
    --log-level debug || exit $?
sushi_require mtp
run_tier ram
sushi_stop

sushi_boot decode-end-ssd "$MODEL" --prefix-cache-entries 4 --prefix-cache-mem 0 --prefix-cache-disk 8GB \
    --log-level debug || exit $?
run_tier ssd
sushi_stop

echo
if [ "$FAIL" -ne 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL TESTS PASSED"

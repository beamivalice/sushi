#!/bin/bash
# sushi-test: tier=live boot=exact
# Prefill-side interleaving: a decode stream that shares the box with a cold prefill keeps its bytes.
#   [1] the stream alone and the same stream beside a ~6000-word cold prefill produce the same greedy output
#       (interleaving reorders ticks, never math)
#   [2] the log shows the yield, `[interleave] engaged`, so [1] did not compare two serial runs
# Needs a second slot: a boot that does not batch (streamed DeepSeek) SKIPs.
set -u
. "$(dirname "$0")/lib/boot.sh"
sushi_attach
sushi_require batching

python3 - "$SUSHI_URL" "$SUSHI_NONCE" "$SUSHI_ARCH" <<'PY' || exit 1
import hashlib, json, sys, threading, time, urllib.request

base, nonce, arch = sys.argv[1:]
# GLM-5.3's template has no thinking-off turn; its low effort thinks briefly, and thought tokens count as decode.
think = {"reasoning_effort": "low"} if arch == "glm5_next" else {"enable_thinking": False}


def stream(payload, stamps, parts):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                for ch in json.loads(line[6:]).get("choices", []):
                    delta = ch.get("delta", {})
                    d = delta.get("content") or delta.get("reasoning_content")
                    if d:
                        stamps.append(time.monotonic())
                        parts.append(d)


a = {"model": "sushi", "temperature": 0.0, "max_tokens": 500, "stream": True, "enable_mtp": False, **think,
     "messages": [{"role": "user", "content": f"{nonce} Count from 1 to 200, one number per line."}]}
words = " ".join(f"w{i}" for i in range(6000))
b = {"model": "sushi", "temperature": 0.0, "max_tokens": 4, "stream": True, "enable_mtp": False, **think,
     "messages": [{"role": "user", "content": f"{nonce} Summarize in one word. {words}"}]}

alone_st, alone_tx = [], []
stream(a, alone_st, alone_tx)

a_st, a_tx = [], []
ta = threading.Thread(target=stream, args=(a, a_st, a_tx))
ta.start()
deadline = time.monotonic() + 30
while len(a_st) < 20 and ta.is_alive() and time.monotonic() < deadline:
    time.sleep(0.01)
if len(a_st) < 20:
    sys.exit("FAIL: the decoder did not produce 20 chunks before the competing prefill")
tb = threading.Thread(target=stream, args=(b, [], []))
tb.start()
tb.join()
ta.join()

sha = lambda parts: hashlib.sha256("".join(parts).encode()).hexdigest()[:12]
if sha(alone_tx) != sha(a_tx):
    sys.exit(f"FAIL: the stream's greedy output changed beside a cold prefill ({sha(alone_tx)} vs {sha(a_tx)})")
print("PASS: the stream's output is byte-identical alone and beside a cold prefill")
PY

if sushi_log_since | grep -q '\[interleave\] engaged'; then
    echo "PASS: [interleave] engaged"
else
    echo "FAIL: no [interleave] engaged line: the prefill never yielded to the decode stream"
    exit 1
fi

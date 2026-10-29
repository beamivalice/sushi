#!/bin/bash
# sushi-test: tier=live boot=knobs
# Think penalty (`--think-penalty`) live contract on the knobs boot (qwen4_exp).
#
#   1. ENGAGEMENT: the load logs `[think-penalty] lambda L (--think-penalty)`, every thinking
#      request with lambda > 0 logs `think penalty L:`, and a request's `think_penalty: 0` arms
#      nothing.
#   2. MTP == SERIAL: greedy reasoning + answer bytes with MTP equal `enable_mtp:false`, on a thought the
#      cap cuts open (xhigh) and one that closes before its answer (low).
#   3. STREAM == NON-STREAM on every request.
set -u
. "$(dirname "$0")/lib/boot.sh"
sushi_attach
sushi_require mtp
LAMBDA="${SUSHI_TEST_THINK_PENALTY:-2}"
ARTIFACTS=$(mktemp -d)
trap 'rm -rf "$ARTIFACTS"' EXIT

run_requests() { # $1 = arm name, $2 = enable_mtp (True|False)
    python3 - "$SUSHI_URL" "$ARTIFACTS/$1.json" "$2" "$SUSHI_NONCE" <<'PY'
import json, sys, urllib.request

base, out, mtp, nonce = sys.argv[1], sys.argv[2], sys.argv[3] == "True", sys.argv[4]
REQS = [
    ("closes", {"reasoning_effort": "low", "max_tokens": 900, "messages": [{"role": "user", "content":
        nonce + " A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"}]}),
    ("open", {"reasoning_effort": "xhigh", "max_tokens": 500, "messages": [{"role": "user", "content":
        nonce + " How many positive integers n below 100 make n^2 + n + 41 composite? Check your work."}]}),
    ("off", {"reasoning_effort": "low", "max_tokens": 300, "think_penalty": 0, "messages": [{"role": "user", "content":
        nonce + " Name three prime numbers above 50."}]}),
]

def post(body):
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return r.read().decode()

res = {}
for name, body in REQS:
    body = dict(body, model="default", temperature=0, enable_mtp=mtp)
    m = json.loads(post(dict(body, stream=False)))["choices"][0]["message"]
    plain = (m.get("reasoning_content") or "") + "\x01" + (m.get("content") or "")
    think, text = [], []
    for line in post(dict(body, stream=True)).splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        for ch in json.loads(line[6:]).get("choices", []):
            d = ch.get("delta", {})
            think.append(d.get("reasoning_content") or "")
            text.append(d.get("content") or "")
    res[name] = {"plain": plain, "stream": "".join(think) + "\x01" + "".join(text)}
json.dump(res, open(out, "w"))
PY
}

# Each arm's log slice is what its requests wrote.
for arm in mtp serial; do
    before=$(sushi_log_since | wc -c)
    if [ "$arm" = mtp ]; then enable=True; else enable=False; fi
    run_requests "$arm" "$enable" || { echo "FAIL: requests ($arm)"; exit 1; }
    sushi_log_since | tail -c +"$((before + 1))" > "$ARTIFACTS/server-$arm.log"
done
# The load line is logged once at boot, before this script attached.
cp "$SUSHI_LOG" "$ARTIFACTS/server-boot.log"

python3 - "$ARTIFACTS" "$LAMBDA" <<'PY'
import json, re, sys

art, lam = sys.argv[1], sys.argv[2]
fails = []
mtp = json.load(open(f"{art}/mtp.json"))
serial = json.load(open(f"{art}/serial.json"))
if not re.search(r"\[think-penalty\] lambda %s \(--think-penalty\)" % re.escape(lam), open(f"{art}/server-boot.log").read()):
    fails.append(f"no load line for lambda {lam}")
for arm, res in (("mtp", mtp), ("serial", serial)):
    log = open(f"{art}/server-{arm}.log").read()
    armed = len(re.findall(r"think penalty [0-9.]+: markers lowered", log))
    if armed != 4:
        fails.append(f"{arm}: {armed} armed requests, want 4 (think_penalty 0 arms none)")
    for name, r in res.items():
        if r["plain"] != r["stream"]:
            fails.append(f"{arm}/{name}: stream != non-stream")
for name in mtp:
    if mtp[name]["plain"] != serial[name]["plain"]:
        a, b = mtp[name]["plain"], serial[name]["plain"]
        i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        fails.append(f"{name}: MTP != serial at char {i}: {a[max(0, i - 40):i + 40]!r} vs {b[max(0, i - 40):i + 40]!r}")
if "\x01" not in mtp["closes"]["plain"] or not mtp["closes"]["plain"].split("\x01", 1)[1].strip():
    print("note: the low-effort thought did not close inside its cap; the after-span gate went unexercised")
if not re.search(r"\[spec-stats\] mode=mtp", open(f"{art}/server-mtp.log").read()):
    fails.append("mtp: no [spec-stats] mode=mtp line")
for f in fails:
    print("FAIL:", f)
print("PASS" if not fails else f"{len(fails)} failure(s)")
sys.exit(1 if fails else 0)
PY

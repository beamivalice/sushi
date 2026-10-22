#!/bin/bash
. "$(dirname "$0")/private_cache.sh"
# Qwen3.8-Flash-Next + DFlash2 drafter: engagement and byte-equality with serial decoding.
#
# Contract pinned here (one pack per run; the owner's gate is all three Sushi Qwen packs):
#   1. ENGAGEMENT (with --drafter): the boot logs `DFlash drafter ready`, every bank request logs
#      `[spec-stats] mode=dflash` with attempts > 0, and the bank accepts drafts (avg_per_round > 0).
#   2. EQUIVALENCE: greedy output of every bank prompt, with the drafter, equals the output of a
#      `--no-drafter --no-mtp` boot byte for byte (non-stream), and the streamed answer equals the
#      non-stream one.
#   3. OPT-OUT: a request with enable_drafter:false logs no mode=dflash line.
#
#   4. OPT-IN: a drafter beside the pack is NOT loaded without --drafter: the boot decodes with MTP.
#   6. IMAGE: a request with an image runs DFlash rounds and equals serial decoding byte for byte (the target's verify
#      carries the M-RoPE positions; the drafter reads the sequence).
#   5. SAMPLED (temperature 1.0, top_p 0.95, top_k 20, what agent clients send): every bank request runs DFlash rounds
#      and answers; the per-draft acceptance is printed, because a greedy-only bar never sees the runtime gate act.
#
# Usage: DFLASH2_QWEN_MODEL=<pack dir> [DFLASH2_QWEN_DRAFTER=<drafter dir>] ./tests/test_dflash2_qwen.sh [port]
# Boot flags: SUSHI_TEST_EXTRA_ARGS (e.g. a pinned block size knob while measuring).

set -u
MODELS="${SUSHI_MODELS_DIR:-$HOME/.sushi/models}"
MODEL="${DFLASH2_QWEN_MODEL:-$MODELS/Qwen3.8-Flash-Next-Sushi-2.6bpw}"
DRAFTER="${DFLASH2_QWEN_DRAFTER:-$MODELS/incoai/Qwen3.8-Flash-Next-DFlash2}"
PORT="${1:-11331}"
BIN="${SUSHI_BINARY:-./zig-out/bin/sushi}"
EXTRA_ARGS="${SUSHI_TEST_EXTRA_ARGS:-}"
ARTIFACTS="${DFLASH2_TEST_OUTPUT_DIR:-$(mktemp -d)}"
mkdir -p "$ARTIFACTS"
LOG=""
SERVER_PID=""
PASS=0
FAIL=0
ok() { echo "PASS $1"; PASS=$((PASS + 1)); }
bad() { echo "FAIL $1"; shift; for line in "$@"; do echo "  $line"; done; FAIL=$((FAIL + 1)); }

for d in "$MODEL" "$DRAFTER"; do
    [ -f "$d/config.json" ] || { echo "SKIP: $d/config.json not found"; exit 0; }
done
[ -f "$DRAFTER/model.safetensors" ] || { echo "SKIP: $DRAFTER/model.safetensors not found"; exit 0; }

stop_server() {
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap stop_server EXIT
trap 'exit 130' INT TERM

start_server() { # $1 = boot name, rest = flags
    local name=$1
    shift
    LOG="$ARTIFACTS/server-$name.log"
    # --prefix-cache-entries 0: a warm restore re-runs the hybrid recurrence in another block size.
    # shellcheck disable=SC2086
    "$BIN" --model "${BOOT_MODEL:-$MODEL}" --serve --port "$PORT" --prefix-cache-entries 0 --log-level info $EXTRA_ARGS "$@" >"$LOG" 2>&1 &
    SERVER_PID=$!
    local ready_secs=$(( 300 + $(du -sm "$MODEL" 2>/dev/null | awk '{print $1}') / 100 ))
    for _ in $(seq 1 $((ready_secs / 3))); do
        grep -q "Model ready (loaded on inference thread)" "$LOG" && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 3
    done
    echo "FAIL: $name server did not become ready"; tail -20 "$LOG"; exit 1
}

# One bank run: prints a JSON list of {"id","text"}; every request is greedy.
bank() { # $1 = out file, $2 = extra JSON fields for each request body
    python3 - "$PORT" "$1" "${2:-}" <<'PYBANK'
import base64, json, struct, sys, urllib.request, zlib
port, out, extra = sys.argv[1], sys.argv[2], json.loads(sys.argv[3] or "{}")


def png(w, h):
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    rows = []
    for y in range(h):
        row = b"".join(bytes(((x // 28) * 37 % 256, (y // 28) * 53 % 256, 128)) * 28 for x in range(0, w, 28))
        rows.append(b"\x00" + row[: 3 * w])
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b"")


IMAGE = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png(224, 224)).decode()}},
         {"type": "text", "text": "Describe this image in one sentence, then count from 1 to 30."}]
CODE = "def total_by_sku(rows):\n    totals = {}\n    for row in rows:\n        totals[row['sku']] = totals.get(row['sku'], 0) + int(row['qty'])\n    return totals\n"
bank = [
    ("prose", "Write a short story about a robot learning to paint.", False, 200),
    ("code", "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.", False, 240),
    ("copy", "Return this code unchanged except rename `totals` to `acc`:\n```python\n" + CODE + "```", False, 160),
    ("math", "A train leaves at 9:40 and travels 150 km at 60 km/h. When does it arrive? Show the steps.", False, 200),
    ("json", "Give a JSON object describing three fruits with name, color and calories. Output only the JSON.", False, 160),
    ("think", "Why is the sky blue? Answer in two sentences.", True, 400),
    ("image", IMAGE, False, 240),
]
rows = []
for name, prompt, think, max_tokens in bank:
    body = {"model": "default", "stream": False, "temperature": 0, "max_tokens": max_tokens,
            "enable_thinking": think, "messages": [{"role": "user", "content": prompt}]}
    body.update(extra)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        m = json.load(resp)["choices"][0]["message"]
    rows.append({"id": name, "text": (m.get("reasoning_content") or "") + "\x01" + (m.get("content") or "")})
json.dump(rows, open(out, "w"))
PYBANK
}

stream_matches() { # whole-answer equality of one streamed and one non-stream request
    python3 - "$PORT" <<'PYSTREAM'
import json, sys, urllib.request
port = sys.argv[1]
body = {"model": "default", "temperature": 0, "max_tokens": 120, "enable_thinking": False,
        "messages": [{"role": "user", "content": "List the numbers from 1 to 20, one per line."}]}
def call(stream):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps({**body, "stream": stream}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        if not stream:
            return json.load(resp)["choices"][0]["message"].get("content") or ""
        out = []
        for raw in resp:
            line = raw.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                for c in json.loads(line[6:]).get("choices", []):
                    out.append(c.get("delta", {}).get("content") or "")
        return "".join(out)
sys.exit(0 if call(False) == call(True) else 1)
PYSTREAM
}

echo "== boot 0: drafter beside the pack, no --drafter"
BESIDE="$ARTIFACTS/pack-with-drafter"
mkdir -p "$BESIDE"
for f in "$MODEL"/*; do ln -sfn "$f" "$BESIDE/$(basename "$f")"; done
ln -sfn "$DRAFTER" "$BESIDE/dflash2"
BOOT_MODEL="$BESIDE" start_server beside
grep -q "decodes with MTP (DFlash2: --drafter <path>)" "$LOG" && ! grep -q "DFlash drafter ready" "$LOG" &&
    ok "a drafter beside the pack is ignored: MTP decodes" || bad "a drafter beside the pack is ignored" "$(grep -E '\[spec\]|DFlash' "$LOG" | head -3)"
stop_server

echo "== boot 1: drafter ($MODEL)"
start_server dflash --drafter "$DRAFTER"
grep -q "DFlash drafter ready" "$LOG" && ok "boot: DFlash drafter ready ($(grep -o 'DFlash drafter ready[^.]*' "$LOG" | head -1))" ||
    bad "boot: DFlash drafter ready" "$(grep -i -E 'dflash|drafter' "$LOG" | head -5)"
bank "$ARTIFACTS/dflash.json" || bad "bank with the drafter" "request failed"
N_STATS=$(grep -c "mode=dflash" "$LOG")
if [ "$N_STATS" -eq 7 ]; then ok "every bank request ran DFlash rounds ($N_STATS stats lines)"; else bad "seven mode=dflash stats lines" "saw $N_STATS"; fi
ATTEMPTS_BAD=$(grep "mode=dflash" "$LOG" | grep -c "attempts=0 ")
[ "$ATTEMPTS_BAD" -eq 0 ] && ok "no request ran zero rounds" || bad "no request ran zero rounds" "$ATTEMPTS_BAD with attempts=0"
grep "mode=dflash" "$LOG" | sed -n 's/.*attempts=\([0-9]*\).*avg_per_round=\([0-9.]*\).*per_draft_pct=\([0-9.]*\).*/  attempts=\1 avg_per_round=\2 per_draft_pct=\3/p'
AVG_OK=$(grep "mode=dflash" "$LOG" | sed -n 's/.*avg_per_round=\([0-9.]*\).*/\1/p' | python3 -c 'import sys; print("yes" if any(float(x) > 0 for x in sys.stdin.read().split()) else "no")')
[ "$AVG_OK" = yes ] && ok "the drafter's tokens are accepted (avg_per_round > 0 somewhere)" || bad "the drafter's tokens are accepted" "every request accepted 0 drafts"
stream_matches && ok "stream == non-stream with the drafter" || bad "stream == non-stream with the drafter"
BEFORE=$N_STATS
bank "$ARTIFACTS/optout.json" '{"enable_drafter": false}' >/dev/null 2>&1
AFTER=$(grep -c "mode=dflash" "$LOG")
# stream_matches ran two dflash requests (2 stats lines); the opt-out bank adds none.
[ "$AFTER" -eq $((BEFORE + 2)) ] && ok "enable_drafter:false runs no DFlash rounds" || bad "enable_drafter:false runs no DFlash rounds" "stats lines $BEFORE -> $AFTER (expected +2 from the stream pair)"
SAMPLED_BEFORE=$AFTER
bank "$ARTIFACTS/sampled.json" '{"temperature": 1.0, "top_p": 0.95, "top_k": 20}' || bad "sampled bank with the drafter" "request failed"
SAMPLED_AFTER=$(grep -c "mode=dflash" "$LOG")
[ "$SAMPLED_AFTER" -eq $((SAMPLED_BEFORE + 7)) ] && ok "every sampled bank request ran DFlash rounds" ||
    bad "every sampled bank request ran DFlash rounds" "stats lines $SAMPLED_BEFORE -> $SAMPLED_AFTER (expected +7)"
echo "sampled (T1.0 top_p 0.95 top_k 20):"
tail -n 7 <(grep "mode=dflash" "$LOG") | sed -n 's/.*attempts=\([0-9]*\).*avg_per_round=\([0-9.]*\).*per_draft_pct=\([0-9.]*\).*runtime_disabled=\([a-z]*\).*/  attempts=\1 avg_per_round=\2 per_draft_pct=\3 runtime_disabled=\4/p'
python3 - "$ARTIFACTS/sampled.json" <<'PYSAMPLED'
import json, sys
rows = json.load(open(sys.argv[1]))
sys.exit(0 if all(len(r["text"].replace("\x01", "").strip()) > 0 for r in rows) else 1)
PYSAMPLED
[ $? -eq 0 ] && ok "every sampled answer is non-empty" || bad "every sampled answer is non-empty"
stop_server

echo "== boot 2: serial (--no-drafter --no-mtp)"
start_server serial --no-drafter --no-mtp
bank "$ARTIFACTS/serial.json" || bad "bank without the drafter" "request failed"
stop_server

python3 - "$ARTIFACTS/dflash.json" "$ARTIFACTS/serial.json" <<'PYCMP'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
rc = 0
for x, y in zip(a, b):
    if x["text"] == y["text"]:
        print("PASS greedy bytes equal:", x["id"], len(x["text"]))
    else:
        n = next((i for i, (p, q) in enumerate(zip(x["text"], y["text"])) if p != q), min(len(x["text"]), len(y["text"])))
        print("FAIL greedy bytes differ:", x["id"], "first difference at char", n)
        print("  dflash:", repr(x["text"][max(0, n - 40):n + 40]))
        print("  serial:", repr(y["text"][max(0, n - 40):n + 40]))
        rc = 1
sys.exit(rc)
PYCMP
if [ $? -eq 0 ]; then ok "greedy DFlash2 == greedy serial, byte for byte, all bank prompts"; else bad "greedy DFlash2 == greedy serial"; fi

echo "artifacts: $ARTIFACTS"
echo "passed=$PASS failed=$FAIL"
[ "$FAIL" -eq 0 ]

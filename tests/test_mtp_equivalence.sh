#!/bin/bash
. "$(dirname "$0")/private_cache.sh"
# MTP (native multi-token-prediction head) correctness + engagement test, for both served
# archs: Qwen's head and MiMo-V2.6's three heads (`model_type` from config.json picks the
# markers, the engagement lines and whether reasoning is part of the compared answer).
#
# Contract pinned here:
#   1. ENGAGEMENT — with an MTP sidecar present, every request on
#      /v1/chat/completions (stream + non-stream) and /v1/messages
#      (non-stream) runs speculative rounds: the server log shows
#      `[spec-stats] mode=mtp` with attempts > 0 per request. This is the
#      anti-dispatch-hole check: output-equality alone can't see a silent
#      fallback to regular decode (the drafter shipped exactly that bug).
#   1b. ACCEPTANCE FLOOR — at least one request must report
#      avg_per_round >= 0.5 (accepted tokens per attempt). A structurally
#      broken head (e.g. the delta-encoded-norms trap: sidecar built without
#      the +1 fold-in) still "engages" but accepts ~0 per round before the
#      runtime gate silently falls back to regular decode — equivalence and
#      engagement checks both pass in that state. avg_per_round is the
#      depth-independent floor: healthy measures ~0.7 at depth 1 and ~0.75+
#      at the depth-3 default even on creative temp-0 content where the
#      chained per_draft_pct legitimately dilutes to ~25%.
#   2. EQUIVALENCE — full output bytes must match at temp=0.
#      Every divergence fails, with the serial top-two gap reported.
#   3. PROMPT LOOKUP — a copy task (return a file with one rename) runs
#      lookup rounds (`lookup=R/..` with R > 0) and still matches the
#      --no-mtp bytes, stream and non-stream; `--no-mtp-lookup` runs none
#      and matches too; a seeded sampled copy is the same bytes streamed.
#
# Usage: MTP_TEST_MODEL=<model-dir> ./tests/test_mtp_equivalence.sh [port]
# Default model: ${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw. A standalone
# sidecar (every mtp.sidecar_rel_paths location) or a sharded/monolithic
# checkpoint carrying one of mtp_marker_keys works.
# Calibrated auto-depth surfaces can additionally pin their live dispatch arm:
#   MTP_EXPECT_AUTO_PROFILE=g17_nax_q4_gs64 MTP_EXPECT_AUTO_DEPTH=8 \
#     MTP_TEST_MODEL=<model-dir> ./tests/test_mtp_equivalence.sh
#
# MoE trunks (35B-A3B, qwen4_exp) keep MTP default-OFF per request; set
# MTP_FORCE_ENABLE=1 to inject "enable_mtp":true into every request body so
# engagement + acceptance-floor checks exercise the MoE head arm.
#
# mimo_v2 (MTP_TEST_MODEL=<MiMo pack>): a pack without its heads FAILS, never skips. Its
# thinking is on by default, so reasoning + content is the compared answer. It checks the
# qk-192 fused prefill engagement instead of Qwen's hd-256 and GDN lines, the head-count
# depth cap instead of the chunk-B extension, and adds a pinned `--mtp-min-depth 3 --mtp-max-depth 3` boot that
# must be byte-identical to --no-mtp. Its copy task runs prompt-lookup rounds as Qwen's does.

set -u
MODEL="${MTP_TEST_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw}"
PORT="${1:-11313}"
BIN="${SUSHI_BINARY:-./zig-out/bin/sushi}"
EXTRA_ARGS="${SUSHI_TEST_EXTRA_ARGS:-}"
MAX_TOKENS=120
ARCH=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("model_type", ""))' "$MODEL/config.json" 2>/dev/null)
# MiMo answers inside a think block by default: compare the reasoning too.
REASONING=0
[ "$ARCH" = mimo_v2 ] && REASONING=1
export REASONING
PROMPT="Write a short story about a robot learning to paint."
EXPECT_AUTO_PROFILE="${MTP_EXPECT_AUTO_PROFILE:-}"
EXPECT_AUTO_DEPTH="${MTP_EXPECT_AUTO_DEPTH:-}"
if { [ -n "$EXPECT_AUTO_PROFILE" ] && [ -z "$EXPECT_AUTO_DEPTH" ]; } ||
    { [ -z "$EXPECT_AUTO_PROFILE" ] && [ -n "$EXPECT_AUTO_DEPTH" ]; }; then
    echo "ERROR: MTP_EXPECT_AUTO_PROFILE and MTP_EXPECT_AUTO_DEPTH must be set together"
    exit 2
fi
# Injected into every request body; empty by default (server defaults apply).
OPTIN=""
if [ "${MTP_FORCE_ENABLE:-0}" = "1" ]; then
    OPTIN='"enable_mtp":true,'
fi

checkpoint_has_mtp_head() {
    [ -f "$MODEL/mtp/weights.safetensors" ] ||
        [ -f "$MODEL/mtp.safetensors" ] ||
        [ -f "$MODEL/model-mtp.safetensors" ] ||
        [ -f "$MODEL/optiq/mtp.safetensors" ] ||
        python3 - "$MODEL" <<'PY'
import json
import pathlib
import sys

model = pathlib.Path(sys.argv[1])
markers = {
    "mtp.fc.weight",
    "language_model.mtp.fc.weight",
    "mtp.eh_proj.weight",
    "language_model.mtp.eh_proj.weight",
    "mtp.fc_hidden.weight",
    "language_model.mtp.fc_hidden.weight",
    "model.mtp.layers.0.eh_proj.weight",
}

try:
    weight_map = json.loads((model / "model.safetensors.index.json").read_text()).get("weight_map", {})
    if markers.intersection(weight_map):
        raise SystemExit(0)
except (OSError, ValueError, AttributeError):
    pass

try:
    with (model / "model.safetensors").open("rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        if header_len > 64 * 1024 * 1024:
            raise ValueError("oversized safetensors header")
        header = json.loads(f.read(header_len))
    raise SystemExit(0 if markers.intersection(header) else 1)
except (OSError, ValueError, AttributeError):
    raise SystemExit(1)
PY
}

if [ ! -d "$MODEL" ] || ! checkpoint_has_mtp_head; then
    if [ -n "$EXPECT_AUTO_PROFILE" ] || [ "$ARCH" = mimo_v2 ]; then
        echo "FAIL: required MTP checkpoint not detected at $MODEL"
        exit 1
    fi
    echo "SKIP: model with MTP head not found at $MODEL"
    exit 0
fi

PASS=0
FAIL=0
ARTIFACTS="${MTP_TEST_OUTPUT_DIR:-$(mktemp -d)}"
mkdir -p "$ARTIFACTS"
LOG="$ARTIFACTS/mtp_equiv_server.log"
BOOT=0
SERVER_PID=""
exec 3>&1 4>&2
exec >"$ARTIFACTS/suite.log" 2>&1
finish_suite() {
    local rc=$?
    if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi
    if [ "$rc" -ne 0 ]; then cat "$ARTIFACTS/suite.log" >&4; fi
}
trap finish_suite EXIT
trap 'exit 130' INT TERM
python3 "$(dirname "$0")/test_mtp_equivalence_strict.py" || exit 1

start_server() { # $1 = extra flags
    BOOT=$((BOOT+1))
    # --prefix-cache-entries 0: byte-stable greedy on a HYBRID needs the
    # prefix cache off (docs/engine-prefix-cache.md) — a warm restore re-runs the recurrence in
    # a different block size and legitimately flips near-tie argmaxes inside
    # the byte-compared prefix (the char-~116 drift noted above was this).
    # --no-drafter: a pack shipping its own drafter/ would otherwise outrank
    # the MTP head and this script would measure DFlash.
    # shellcheck disable=SC2086
    "$BIN" --model "$MODEL" --serve --port "$PORT" --no-drafter --prefix-cache-entries 0 --log-level info $EXTRA_ARGS $1 >"$LOG" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 120); do
        curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
        sleep 1
    done
    # /health answers as soon as the socket binds — the model is still
    # loading behind it. Wait for the load's own ready line, timeout scaled
    # to the checkpoint size.
    local model_mb ready_secs
    model_mb=$(du -sm "$MODEL" 2>/dev/null | awk '{print $1}')
    ready_secs=$(( 300 + ${model_mb:-0} / 100 ))
    for _ in $(seq 1 $((ready_secs / 3)) ); do
        grep -q "Model ready (loaded on inference thread)" "$LOG" && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 3
    done
    echo "FAIL: server did not become ready"; cat "$LOG" | tail -20; exit 1
}

stop_server() {
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
    cp "$LOG" "$LOG.boot$BOOT"
}

chat_nonstream() {
    curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
        $OPTIN\"model\":\"default\",\"stream\":false,\"temperature\":0,\"max_tokens\":$MAX_TOKENS,
        \"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}]}" |
        python3 -c "
import json, os, sys
m = json.load(sys.stdin)['choices'][0]['message']
if os.environ['REASONING'] == '1':
    print((m.get('reasoning_content') or '') + '\x01' + (m.get('content') or ''), end='')
else:
    print(m['content'], end='')"
}

chat_stream() {
    curl -sN "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
        $OPTIN\"model\":\"default\",\"stream\":true,\"temperature\":0,\"max_tokens\":$MAX_TOKENS,
        \"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}]}" |
        python3 -c "
import json, sys
import os
out = []
think = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith('data: ') or line == 'data: [DONE]': continue
    try: d = json.loads(line[6:])
    except Exception: continue
    for c in d.get('choices', []):
        out.append(c.get('delta', {}).get('content') or '')
        think.append(c.get('delta', {}).get('reasoning_content') or '')
if os.environ['REASONING'] == '1':
    print(''.join(think) + '\x01', end='')
print(''.join(out), end='')"
}

messages_nonstream() {
    curl -s "http://127.0.0.1:$PORT/v1/messages" -H 'Content-Type: application/json' -d "{
        $OPTIN\"model\":\"default\",\"stream\":false,\"max_tokens\":$MAX_TOKENS,\"temperature\":0,
        \"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}]}" |
        python3 -c "
import json, os, sys
blocks = json.load(sys.stdin)['content']
if os.environ['REASONING'] == '1':
    print(''.join(b.get('thinking', '') for b in blocks if b.get('type') == 'thinking') + '\x01', end='')
print(''.join(b.get('text','') for b in blocks), end='')"
}

# A copy task: return a file with one rename. Built as JSON by python (the
# prompt is multi-line code); thinking off so the answer is the copy.
COPY_MAX_TOKENS=480
COPY_PROMPT_FILE="$ARTIFACTS/copy_prompt.txt"
cat >"$COPY_PROMPT_FILE" <<'COPYEOF'
Rename the function `load_rows` to `read_rows` everywhere in the file below (its definition and every call). Output the complete updated file only, with no commentary.

```python
import csv
from collections import defaultdict


def load_rows(path):
    with open(path, newline="") as fh:
        return [row for row in csv.DictReader(fh)]


def total_by_sku(rows):
    totals = defaultdict(int)
    for row in rows:
        totals[row["sku"]] += int(row["qty"])
    return dict(totals)


def low_stock(rows, threshold=5):
    return sorted(sku for sku, qty in total_by_sku(rows).items() if qty < threshold)


def summarize(path, threshold=5):
    rows = load_rows(path)
    totals = total_by_sku(rows)
    lines = [f"{sku}: {qty}" for sku, qty in sorted(totals.items())]
    lines.append(f"low stock: {', '.join(low_stock(rows, threshold)) or 'none'}")
    return "\n".join(lines)


def merge(paths):
    rows = []
    for path in paths:
        rows.extend(load_rows(path))
    return total_by_sku(rows)


if __name__ == "__main__":
    import sys
    print(summarize(sys.argv[1]))
```
COPYEOF
COPY_PROMPT="$(cat "$COPY_PROMPT_FILE")"

copy_request() { # $1 stream true|false, $2 sampling JSON fields (empty = greedy)
    python3 - "$PORT" "$1" "$COPY_MAX_TOKENS" "$COPY_PROMPT_FILE" "${2:-}" <<'PYCOPY'
import json, sys, urllib.request
port, stream, max_tokens, prompt_file, sampling = sys.argv[1], sys.argv[2] == "true", int(sys.argv[3]), sys.argv[4], sys.argv[5]
body = {"model": "default", "stream": stream, "max_tokens": max_tokens, "enable_thinking": False,
        "messages": [{"role": "user", "content": open(prompt_file).read()}]}
body.update(json.loads(sampling) if sampling else {"temperature": 0})
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=900) as resp:
    if not stream:
        print(json.load(resp)["choices"][0]["message"]["content"], end="")
        sys.exit(0)
    out = []
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            d = json.loads(line[6:])
        except ValueError:
            continue
        for c in d.get("choices", []):
            out.append(c.get("delta", {}).get("content") or "")
    print("".join(out), end="")
PYCOPY
}

SEEDED='{"temperature":0.6,"top_p":0.95,"top_k":20,"seed":1234}'

lookup_rounds_max() { # largest lookup round count on any [spec-stats] line of this boot
    grep -o 'lookup=[0-9]*/' "$LOG" | tr -dc '0-9\n' | sort -n | tail -1
}

# Every mismatch fails and reports its serial top-two gap.
tie_gap_at_divergence() { # $1 expected-file, $2 actual-file → prints gap or "none"
    python3 - "$1" "$2" "$PORT" "$MAX_TOKENS" "$PROMPT" "${GAP_EXTRA:-}" <<'PYEOF'
import json, sys, urllib.request
expf, actf, port, max_tokens, prompt = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
extra = json.loads(sys.argv[6]) if sys.argv[6] else {}
exp = open(expf).read()
act = open(actf).read()
n = min(len(exp), len(act))
i = next((k for k in range(n) if exp[k] != act[k]), n)
body = {"model": "default", "stream": False, "temperature": 0, "max_tokens": max_tokens,
        "enable_mtp": False, "logprobs": True, "top_logprobs": 2,
        "messages": [{"role": "user", "content": prompt}]}
body.update(extra)
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
resp = json.load(urllib.request.urlopen(req, timeout=600))
entries = (resp["choices"][0].get("logprobs") or {}).get("content") or []
pos = 0
for e in entries:
    tok = e.get("token") or ""
    if pos + len(tok) > i:
        tops = e.get("top_logprobs") or []
        if len(tops) >= 2:
            print(f"{abs(tops[0]['logprob'] - tops[1]['logprob']):.4f}")
        else:
            print("none")
        sys.exit(0)
    pos += len(tok)
print("none")
PYEOF
}

check() { # $1 name, $2 expected-prefix-file, $3 actual-file, $4 expected new mtp engagements (log delta)
    local name="$1" expf="$2" actf="$3" want_engage="$4"
    if [ ! -s "$actf" ]; then
        echo "FAIL [$name]: empty output"; FAIL=$((FAIL+1)); return
    fi
    if [ "$want_engage" = "yes" ]; then
        local stats
        stats=$(grep -c "\[spec-stats\] mode=mtp" "$LOG")
        if [ "$stats" -lt "$ENGAGE_BASE" ] || [ "$stats" -eq "$ENGAGE_BASE" ]; then
            echo "FAIL [$name]: no new '[spec-stats] mode=mtp' log line (engagement hole!)"
            FAIL=$((FAIL+1)); return
        fi
        ENGAGE_BASE=$stats
    fi
    if ! cmp -s "$expf" "$actf"; then
        local gap
        gap=$(tie_gap_at_divergence "$expf" "$actf")
        echo "FAIL [$name]: full output bytes differ from no-mtp baseline (top-2 gap at divergence: ${gap:-unreadable})"
        python3 - "$expf" "$actf" <<'PYBYTES'
import pathlib, sys
for label, path in zip(("expected", "actual"), sys.argv[1:]):
    print(f"  {label}: {pathlib.Path(path).read_bytes()[:80]!r}...")
PYBYTES
        FAIL=$((FAIL+1)); return
    fi
    PASS=$((PASS+1))
}

echo "── baseline server (--no-mtp) ──"
start_server "--no-mtp"
chat_nonstream > "$ARTIFACTS/mtp_base_chat.txt"
messages_nonstream > "$ARTIFACTS/mtp_base_msg.txt"
copy_request false > "$ARTIFACTS/mtp_base_copy.txt"
if grep -q "mode=mtp" "$LOG"; then
    echo "FAIL: --no-mtp server ran MTP rounds"; FAIL=$((FAIL+1))
else
    echo "PASS [no-mtp baseline clean]"; PASS=$((PASS+1))
fi
stop_server

echo "── MTP server (default-on) ──"
start_server ""
# The qwen4_exp head is the checkpoint's own layer and logs its own line; MiMo's heads log theirs.
HEAD_READY="MTP head ready\|\[qwen4\] MTP head loaded"
[ "$ARCH" = mimo_v2 ] && HEAD_READY="\[mimo-mtp\] [0-9]* heads loaded"
if ! grep -q "$HEAD_READY" "$LOG"; then
    echo "FAIL: server did not auto-load the MTP sidecar"; tail -5 "$LOG"; FAIL=$((FAIL+1))
else
    echo "PASS [mtp auto-load]"; PASS=$((PASS+1))
fi
if [ -n "$EXPECT_AUTO_PROFILE" ]; then
    EXPECT_READY="MTP head ready (depth=$EXPECT_AUTO_DEPTH, profile=$EXPECT_AUTO_PROFILE)."
    if grep -Fq "$EXPECT_READY" "$LOG"; then
        echo "PASS [auto profile fingerprint] ($EXPECT_AUTO_PROFILE, depth=$EXPECT_AUTO_DEPTH)"; PASS=$((PASS+1))
    else
        echo "FAIL [auto profile fingerprint]: expected '$EXPECT_READY'"
        grep "MTP head ready" "$LOG" | tail -1
        FAIL=$((FAIL+1))
    fi
fi
ENGAGE_BASE=0
chat_nonstream > "$ARTIFACTS/mtp_on_chat.txt"
check "chat non-stream" "$ARTIFACTS/mtp_base_chat.txt" "$ARTIFACTS/mtp_on_chat.txt" yes
chat_stream > "$ARTIFACTS/mtp_on_chat_stream.txt"
check "chat stream" "$ARTIFACTS/mtp_base_chat.txt" "$ARTIFACTS/mtp_on_chat_stream.txt" yes
messages_nonstream > "$ARTIFACTS/mtp_on_msg.txt"
check "messages non-stream" "$ARTIFACTS/mtp_base_msg.txt" "$ARTIFACTS/mtp_on_msg.txt" yes
copy_request false > "$ARTIFACTS/mtp_on_copy.txt"
PROMPT="$COPY_PROMPT" MAX_TOKENS=$COPY_MAX_TOKENS GAP_EXTRA='{"enable_thinking":false}' \
    check "copy non-stream (lookup on)" "$ARTIFACTS/mtp_base_copy.txt" "$ARTIFACTS/mtp_on_copy.txt" yes
copy_request true > "$ARTIFACTS/mtp_on_copy_stream.txt"
PROMPT="$COPY_PROMPT" MAX_TOKENS=$COPY_MAX_TOKENS GAP_EXTRA='{"enable_thinking":false}' \
    check "copy stream (lookup on)" "$ARTIFACTS/mtp_base_copy.txt" "$ARTIFACTS/mtp_on_copy_stream.txt" yes
LOOKUP_ROUNDS=$(lookup_rounds_max)
if [ "${LOOKUP_ROUNDS:-0}" -gt 0 ] && grep -q "\[mtp\] prompt-lookup drafts engaged" "$LOG"; then
    echo "PASS [prompt lookup engages on the copy task] (lookup rounds=$LOOKUP_ROUNDS)"; PASS=$((PASS+1))
else
    echo "FAIL [prompt lookup engagement]: lookup rounds=${LOOKUP_ROUNDS:-none} on a copy task"; FAIL=$((FAIL+1))
fi
# Acceptance floor: a broken head engages but accepts ~0 tokens per round.
# avg_per_round is depth-independent (per_draft_pct divides by depth and
# legitimately dilutes on chained creative drafts at the depth-3 default).
BEST_ACCEPT=$(grep -o 'avg_per_round=[0-9.]*' "$LOG" | cut -d= -f2 | sort -n | tail -1)
if python3 -c "import sys; sys.exit(0 if float('${BEST_ACCEPT:-0}') >= 0.5 else 1)"; then
    echo "PASS [acceptance floor] (best avg_per_round=${BEST_ACCEPT})"; PASS=$((PASS+1))
else
    echo "FAIL [acceptance floor]: best avg_per_round=${BEST_ACCEPT:-none} < 0.5 — head is drafting garbage"
    FAIL=$((FAIL+1))
fi
# Fused-kernel engagement (anti-silent-no-op, kv-quant class): every qwen
# 3.5/3.6 checkpoint is hd 256 with GDN layers, so both fusions must fire on
# the verify widths this server just ran. MiMo's global and sliding layers both
# prefill through the fused qk-192 kernel. Output equality alone is blind to a
# decline gate quietly routing everything back to the composed chain.
ENGAGE_LINES=("\[attn\] fused QK-norm+RoPE (hd-256) engaged" "\[gdn\] packed prework engaged")
[ "$ARCH" = mimo_v2 ] && ENGAGE_LINES=("\[attn-pd\] engaged" "\[attn-pd\] sliding band engaged")
for ENGAGE_LINE in "${ENGAGE_LINES[@]}"; do
    if grep -q "$ENGAGE_LINE" "$LOG"; then
        echo "PASS [engaged: $ENGAGE_LINE]"; PASS=$((PASS+1))
    else
        echo "FAIL [not engaged: $ENGAGE_LINE] — fused path silently declined"; FAIL=$((FAIL+1))
    fi
done
# Per-request opt-out must fall back to regular decode.
ENGAGE_PRE=$(grep -c "\[spec-stats\] mode=mtp" "$LOG")
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
    \"model\":\"default\",\"stream\":false,\"temperature\":0,\"max_tokens\":24,\"enable_mtp\":false,
    \"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" >/dev/null
ENGAGE_POST=$(grep -c "\[spec-stats\] mode=mtp" "$LOG")
if [ "$ENGAGE_PRE" -eq "$ENGAGE_POST" ]; then
    echo "PASS [enable_mtp:false opt-out]"; PASS=$((PASS+1))
else
    echo "FAIL [enable_mtp:false opt-out]: MTP ran despite per-request disable"; FAIL=$((FAIL+1))
fi

stop_server

echo "── MTP server, prompt lookup off (--no-mtp-lookup) ──"
# The EV checks need MTP rounds on an echo; with lookup on, an echo is served
# by lookup rounds instead. The echo runs first: a copy before it would seed the
# EV surface at the depth cap, where a round has no chunk B to extend into.
start_server "--no-mtp-lookup"
ENGAGE_BASE=0
# EV-controller engagement (dispatch-hole lesson: output equality can't see a
# silent fallback). An ECHO workload is the max-confidence case: past the
# ~10-round warmup the chain confidence clears any tau, so chunk-B extension
# must fire (ext_rounds > 0 in [spec-stats]) under the adaptive default.
ECHO_PROMPT="Repeat the following code block back EXACTLY as written, no commentary: def gcd(a, b):\\n    while b:\\n        a, b = b, a % b\\n    return a\\n\\ndef fib(n, memo={}):\\n    if n in memo: return memo[n]\\n    if n < 2: return n\\n    memo[n] = fib(n-1, memo) + fib(n-2, memo)\\n    return memo[n]\\n\\ndef reverse_string(s):\\n    out = ''\\n    for ch in s:\\n        out = ch + out\\n    return out"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
    $OPTIN\"model\":\"default\",\"stream\":false,\"temperature\":0,\"max_tokens\":160,
    \"messages\":[{\"role\":\"user\",\"content\":\"$ECHO_PROMPT\"}]}" >/dev/null
EXT=$(grep -o 'ext_rounds=[0-9]*' "$LOG" | tail -1 | cut -d= -f2)
if [ "$ARCH" = mimo_v2 ]; then
    # MiMo drafts at most one token per head: the echo runs at the three-head cap.
    ECHO_DEPTH=$(grep -o '\[spec-stats\] mode=mtp.*' "$LOG" | tail -1 | grep -o ' depth=[0-9]*' | grep -o '[0-9]*')
    if [ "${ECHO_DEPTH:-0}" = "3" ]; then
        echo "PASS [MiMo depth at its head count on echo] (depth=$ECHO_DEPTH)"; PASS=$((PASS+1))
    else
        echo "FAIL [MiMo depth cap]: depth=${ECHO_DEPTH:-none} on echo, want 3 (one draft per head)"
        FAIL=$((FAIL+1))
    fi
elif [ "${EXT:-0}" -gt 0 ]; then
    echo "PASS [EV chunk-B extension engages on echo] (ext_rounds=$EXT)"; PASS=$((PASS+1))
else
    echo "FAIL [EV chunk-B extension]: ext_rounds=${EXT:-none} on a max-confidence echo — extension path never fired"
    FAIL=$((FAIL+1))
fi
if [ -n "$EXPECT_AUTO_DEPTH" ]; then
    AUTO_STATS=$(grep -o '\[spec-stats\] mode=mtp.*' "$LOG" | tail -1)
    AUTO_DEPTH=$(echo "$AUTO_STATS" | grep -o ' depth=[0-9]*' | grep -o '[0-9]*')
    if [ "${AUTO_DEPTH:-0}" = "$EXPECT_AUTO_DEPTH" ]; then
        echo "PASS [auto depth realized on echo] (depth=$AUTO_DEPTH)"; PASS=$((PASS+1))
    else
        echo "FAIL [auto depth realized on echo]: depth=${AUTO_DEPTH:-none}, expected $EXPECT_AUTO_DEPTH"
        FAIL=$((FAIL+1))
    fi
fi
ENGAGE_BASE=$(grep -c "\[spec-stats\] mode=mtp" "$LOG")
copy_request false > "$ARTIFACTS/mtp_nolookup_copy.txt"
PROMPT="$COPY_PROMPT" MAX_TOKENS=$COPY_MAX_TOKENS GAP_EXTRA='{"enable_thinking":false}' \
    check "copy non-stream (lookup off)" "$ARTIFACTS/mtp_base_copy.txt" "$ARTIFACTS/mtp_nolookup_copy.txt" yes
if [ "$(lookup_rounds_max)" = "0" ] && ! grep -q "prompt-lookup drafts engaged" "$LOG"; then
    echo "PASS [--no-mtp-lookup runs no lookup round]"; PASS=$((PASS+1))
else
    echo "FAIL [--no-mtp-lookup]: lookup rounds ran"; FAIL=$((FAIL+1))
fi
stop_server

echo "── fixed-depth server (SUSHI_MTP_ADAPTIVE=0) ──"
# The env kill switch must fully revert: legacy cap 3 (not the adaptive auto
# cap) and zero chunk-B extensions on the same echo workload.
BOOT=$((BOOT+1))
# SUSHI_MTP_COST_TABLE=0: the lookup gate and the plan read no measured round
# times, so a seeded sampled request takes the same rounds streamed or not.
SUSHI_MTP_ADAPTIVE=0 SUSHI_MTP_COST_TABLE=0 "$BIN" --model "$MODEL" --serve --port "$PORT" --no-drafter --prefix-cache-entries 0 --log-level info $EXTRA_ARGS >"$LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 120); do
    curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
done
# Same readiness rule as start_server: /health is up before the load ends.
MODEL_MB=$(du -sm "$MODEL" 2>/dev/null | awk '{print $1}')
READY_SECS=$(( 300 + ${MODEL_MB:-0} / 100 ))
for _ in $(seq 1 $((READY_SECS / 3)) ); do
    grep -q "Model ready (loaded on inference thread)" "$LOG" && break
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 3
done
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
    $OPTIN\"model\":\"default\",\"stream\":false,\"temperature\":0,\"max_tokens\":160,
    \"messages\":[{\"role\":\"user\",\"content\":\"$ECHO_PROMPT\"}]}" >/dev/null
FIXED_STATS=$(grep -o '\[spec-stats\] mode=mtp.*' "$LOG" | tail -1)
FIXED_EXT=$(echo "$FIXED_STATS" | grep -o 'ext_rounds=[0-9]*' | cut -d= -f2)
FIXED_DEPTH=$(echo "$FIXED_STATS" | grep -o ' depth=[0-9]*' | grep -o '[0-9]*')
if [ "${FIXED_EXT:-1}" = "0" ] && [ "${FIXED_DEPTH:-0}" = "3" ]; then
    echo "PASS [SUSHI_MTP_ADAPTIVE=0 reverts to fixed depth 3, no extension]"; PASS=$((PASS+1))
else
    echo "FAIL [adaptive kill switch]: depth=${FIXED_DEPTH:-none} ext_rounds=${FIXED_EXT:-none} (want depth=3 ext_rounds=0)"
    FAIL=$((FAIL+1))
fi
copy_request false "$SEEDED" > "$ARTIFACTS/mtp_seeded_copy.txt"
copy_request true "$SEEDED" > "$ARTIFACTS/mtp_seeded_copy_stream.txt"
if [ -s "$ARTIFACTS/mtp_seeded_copy.txt" ] && cmp -s "$ARTIFACTS/mtp_seeded_copy.txt" "$ARTIFACTS/mtp_seeded_copy_stream.txt"; then
    echo "PASS [seeded sampled copy: stream == non-stream]"; PASS=$((PASS+1))
else
    echo "FAIL [seeded sampled copy]: stream and non-stream bytes differ"; FAIL=$((FAIL+1))
fi
if [ "$(lookup_rounds_max)" -gt 0 ] 2>/dev/null; then
    echo "PASS [prompt lookup engages on the seeded copy]"; PASS=$((PASS+1))
else
    echo "FAIL [prompt lookup on the seeded copy]: no lookup round"; FAIL=$((FAIL+1))
fi
stop_server

if [ "$ARCH" = mimo_v2 ]; then
    echo "── pinned-depth server (--mtp-min-depth 3 --mtp-max-depth 3) ──"
    # Every MiMo verify row keeps its decode tick's arithmetic, so a pinned depth-3 round
    # must reproduce --no-mtp byte for byte.
    start_server "--mtp-min-depth 3 --mtp-max-depth 3"
    ENGAGE_BASE=$(grep -c "\[spec-stats\] mode=mtp" "$LOG")
    chat_nonstream > "$ARTIFACTS/mtp_forced_chat.txt"
    check "forced depth 3, chat non-stream" "$ARTIFACTS/mtp_base_chat.txt" "$ARTIFACTS/mtp_forced_chat.txt" yes
    messages_nonstream > "$ARTIFACTS/mtp_forced_msg.txt"
    check "forced depth 3, messages non-stream" "$ARTIFACTS/mtp_base_msg.txt" "$ARTIFACTS/mtp_forced_msg.txt" yes
    stop_server
fi

echo
printf '{"passed":%s,"failed":%s,"acquittals":0}\n' "$PASS" "$FAIL" >"$ARTIFACTS/result.json"
echo "RESULT: $PASS passed, $FAIL failed, 0 acquittals"
[ "$FAIL" -eq 0 ] || exit 1

#!/bin/bash
. "$(dirname "$0")/private_cache.sh"
# test_tool_choice_forced.sh — a forced tool_choice is honoured on every surface.
#
# `required` / Anthropic `any` / a named function used to reach only the generic
# fallback render, so a template that renders tools itself (Qwen3.8, MiMo) never
# saw it. The call is now committed at decode, after the closed thought. Per pack
# (both served archs), on /v1/chat/completions, /v1/messages and /v1/responses,
# stream and non-stream, thinking on (tiny reasoning budget) and off, greedy:
#
#   1. a named choice yields a call to THAT function, even one the question does
#      not need; `required`/`any` yields at least one call
#   2. every call names a declared function and its arguments are a JSON object
#   3. stream and non-stream deliver the same calls, content and reasoning
#   4. no tool or think markup leaks into content or reasoning
#   5. the log shows one `[tool-choice] forced call` line per forced request and
#      no prompt-only fallback
#   6. `none` yields no call; a named choice for an undeclared function is a 400
#
# Model choices (which function `required` picks, what the thought says) are
# printed, never asserted.
#
# Usage: ./tests/test_tool_choice_forced.sh [pack_dir ...]
# Env: SUSHI_MODELS_DIR (default $HOME/.sushi/models), QWEN_MODEL, MIMO_MODEL,
#      PORT (default 19141), BINARY, GPU_LOCK_OWNER. Owns one GPU-lock run per pack, restores QoS,
#      and keeps each server log under ~/.sushi/runs/tool-choice-forced/.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

MODELS_DIR="${SUSHI_MODELS_DIR:-$HOME/.sushi/models}"
if [ $# -gt 0 ]; then
    PACKS=("$@")
else
    PACKS=("${QWEN_MODEL:-$MODELS_DIR/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw}" "${MIMO_MODEL:-$MODELS_DIR/MiMo-V2.6-Flash-Sushi-2.3bpw}")
fi
PORT="${PORT:-19141}"
BIN="${BINARY:-./zig-out/bin/sushi}"
BASE="http://127.0.0.1:$PORT"
LOCK_OWNER="${GPU_LOCK_OWNER:-tool-choice-forced}"
RUNS="$HOME/.sushi/runs/tool-choice-forced"

[ -x "$BIN" ] || { echo "fail: build sushi first (zig build -Doptimize=ReleaseFast)"; exit 1; }
curl -sf --max-time 2 "$BASE/health" >/dev/null 2>&1 && { echo "fail: port $PORT is busy"; exit 1; }

SERVER_PID=""
LOCKED=0
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null
    SERVER_PID=""
    [ "$LOCKED" = 1 ] && scripts/gpu-lock.sh release "$LOCK_OWNER" >/dev/null
    LOCKED=0
}
trap cleanup EXIT

TOTAL_FAIL=0
for PACK in "${PACKS[@]}"; do
    if [ ! -d "$PACK" ]; then echo "SKIP: pack not found: $PACK"; continue; fi
    NAME="$(basename "$PACK")"
    OUT="$RUNS/$NAME-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$OUT"
    echo "[tool-choice] === $NAME (commit $(git rev-parse --short HEAD), binary $(stat -f %Sm "$BIN")) ==="

    scripts/gpu-lock.sh acquire "$LOCK_OWNER" >/dev/null || { echo "fail: GPU lock"; exit 1; }
    LOCKED=1
    taskpolicy -a "$BIN" --model "$PACK" --serve --host 127.0.0.1 --port "$PORT" --ctx-size 32768 \
        --prefix-cache-entries 0 --max-concurrent 2 --log-level info > "$OUT/server.log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 900); do
        curl -sf --max-time 2 "$BASE/health" 2>/dev/null | grep -q '"ok"' && break
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "fail: server died:"; tail -20 "$OUT/server.log"; exit 1; }
        sleep 1
    done

    python3 - "$BASE" "$OUT/server.log" <<'PY'
import json, sys, urllib.request, urllib.error, threading

BASE, LOG = sys.argv[1], sys.argv[2]
BUDGET = 256
TOOLS = [
    ("get_time", "Current local time in a city", {"city": {"type": "string"}}),
    ("get_weather", "Current weather in a city", {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}),
]
DECLARED = {n for n, _, _ in TOOLS}
QUESTION = "What time is it in Tokyo right now?"
LEAKS = ("<tool_call>", "</tool_call>", "<function=", "</think>", "<think>")
passed = failed = 0
forced_requests = 0

def check(label, ok, detail=""):
    global passed, failed
    if ok: passed += 1; print(f"  PASS {label}")
    else: failed += 1; print(f"  FAIL {label} {detail}")

def post(path, body, stream):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            raw = r.read().decode()
            return r.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

def sse(raw):
    for line in raw.splitlines():
        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
            try: yield json.loads(line[6:])
            except ValueError: pass

# ── per-surface request builders and reply readers: reply = (calls[(name, args_obj)], content, reasoning)

def chat_body(choice, thinking, stream):
    b = {"model": "x", "messages": [{"role": "user", "content": QUESTION}], "temperature": 0, "max_tokens": 2048,
         "tools": [{"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": p}}} for n, d, p in TOOLS],
         "tool_choice": choice, "enable_thinking": thinking, "stream": stream}
    if thinking: b["reasoning_budget_tokens"] = BUDGET
    return b

def chat_read(raw, stream):
    if not stream:
        m = json.loads(raw)["choices"][0]["message"]
        calls = [(c["function"]["name"], c["function"]["arguments"]) for c in (m.get("tool_calls") or [])]
        return calls, m.get("content") or "", m.get("reasoning_content") or ""
    calls, content, reasoning = {}, "", ""
    for ev in sse(raw):
        for c in ev.get("choices", []):
            d = c.get("delta") or {}
            content += d.get("content") or ""
            reasoning += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), ["", ""])
                f = tc.get("function") or {}
                slot[0] += f.get("name") or ""
                slot[1] += f.get("arguments") or ""
    return [tuple(calls[k]) for k in sorted(calls)], content, reasoning

def chat_choice(kind):
    return {"none": "none", "required": "required", "undeclared": {"type": "function", "function": {"name": "nope"}}}.get(
        kind, {"type": "function", "function": {"name": kind}})

def msgs_body(choice, thinking, stream):
    b = {"model": "x", "max_tokens": 2048, "temperature": 0, "stream": stream,
         "messages": [{"role": "user", "content": QUESTION}],
         "tools": [{"name": n, "description": d, "input_schema": {"type": "object", "properties": p}} for n, d, p in TOOLS],
         "tool_choice": choice,
         "thinking": {"type": "enabled", "budget_tokens": BUDGET} if thinking else {"type": "disabled"}}
    return b

def msgs_read(raw, stream):
    if not stream:
        d = json.loads(raw)
        blocks = d.get("content") or []
        calls = [(b["name"], json.dumps(b.get("input"))) for b in blocks if b.get("type") == "tool_use"]
        return calls, "".join(b.get("text", "") for b in blocks if b.get("type") == "text"), \
            "".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
    blocks = {}
    for ev in sse(raw):
        t = ev.get("type")
        if t == "content_block_start":
            blocks[ev["index"]] = dict(ev["content_block"], _json="")
        elif t == "content_block_delta":
            b, d = blocks[ev["index"]], ev["delta"]
            if d.get("type") == "text_delta": b["text"] = b.get("text", "") + d["text"]
            if d.get("type") == "thinking_delta": b["thinking"] = b.get("thinking", "") + d["thinking"]
            if d.get("type") == "input_json_delta": b["_json"] += d["partial_json"]
    ordered = [blocks[k] for k in sorted(blocks)]
    calls = [(b["name"], json.dumps(json.loads(b["_json"] or "{}"))) for b in ordered if b.get("type") == "tool_use"]
    return calls, "".join(b.get("text", "") for b in ordered if b.get("type") == "text"), \
        "".join(b.get("thinking", "") for b in ordered if b.get("type") == "thinking")

def msgs_choice(kind):
    return {"none": {"type": "none"}, "required": {"type": "any"}, "undeclared": {"type": "tool", "name": "nope"}}.get(
        kind, {"type": "tool", "name": kind})

def resp_body(choice, thinking, stream):
    b = {"model": "x", "input": QUESTION, "temperature": 0, "max_output_tokens": 2048, "stream": stream,
         "tools": [{"type": "function", "name": n, "description": d, "parameters": {"type": "object", "properties": p}} for n, d, p in TOOLS],
         "tool_choice": choice, "reasoning": {"effort": "low" if thinking else "none"}}
    if thinking: b["reasoning_budget_tokens"] = BUDGET
    return b

def resp_output(output):
    calls, content, reasoning = [], "", ""
    for it in output:
        if it.get("type") == "function_call": calls.append((it["name"], it["arguments"]))
        if it.get("type") == "message":
            content += "".join(p.get("text", "") for p in it.get("content") or [])
        if it.get("type") == "reasoning":
            reasoning += "".join(p.get("text", "") for p in (it.get("summary") or []) + (it.get("content") or []))
    return calls, content, reasoning

def resp_read(raw, stream):
    if not stream: return resp_output(json.loads(raw).get("output") or [])
    for ev in sse(raw):
        if ev.get("type") == "response.completed": return resp_output(ev["response"].get("output") or [])
    return [], "", ""

def resp_choice(kind):
    return {"none": "none", "required": "required", "undeclared": {"type": "function", "name": "nope"}}.get(
        kind, {"type": "function", "name": kind})

SURFACES = [
    ("chat", "/v1/chat/completions", chat_body, chat_read, chat_choice),
    ("messages", "/v1/messages", msgs_body, msgs_read, msgs_choice),
    ("responses", "/v1/responses", resp_body, resp_read, resp_choice),
]

def args_object(a):
    try: return isinstance(json.loads(a), dict)
    except ValueError: return False

for surface, path, body, read, wire in SURFACES:
    print(f"[tool-choice] --- {surface}")
    for kind in ("get_weather", "required"):
        for thinking in (True, False):
            replies = {}
            for stream in (False, True):
                forced_requests += 1
                status, raw = post(path, body(wire(kind), thinking, stream), stream)
                tag = f"{surface} {kind} thinking={thinking} stream={stream}"
                if status != 200:
                    check(f"{tag}: 200", False, f"(got {status}: {raw[:200]})"); continue
                calls, content, reasoning = read(raw, stream)
                replies[stream] = (calls, content, reasoning)
                print(f"    {tag}: calls={[c[0] for c in calls]} reasoning={len(reasoning)}c content={len(content)}c")
                if kind == "required":
                    check(f"{tag}: at least one call", len(calls) >= 1)
                else:
                    check(f"{tag}: the first call is the named function", len(calls) >= 1 and calls[0][0] == kind, f"(calls {calls})")
                check(f"{tag}: every call names a declared function", all(n in DECLARED for n, _ in calls), f"(calls {calls})")
                check(f"{tag}: every call's arguments are a JSON object", all(args_object(a) for _, a in calls), f"(calls {calls})")
                check(f"{tag}: no markup leaks into content or reasoning",
                      not any(m in content or m in reasoning for m in LEAKS), f"(content {content[:120]!r})")
            if len(replies) == 2:
                check(f"{surface} {kind} thinking={thinking}: stream and non-stream deliver the same reply",
                      replies[False] == replies[True], f"({replies[False]} vs {replies[True]})")

    status, raw = post(path, body(wire("none"), False, False), False)
    calls = read(raw, False)[0] if status == 200 else None
    check(f"{surface} none: 200 and no call", status == 200 and calls == [], f"(status {status}, calls {calls})")
    status, raw = post(path, body(wire("undeclared"), False, False), False)
    check(f"{surface} undeclared named function: 400", status == 400, f"(got {status})")

# Two at once: a forced call beside an auto turn. The forced slot decodes serial
# until its call is committed; both must finish, the forced one with its call.
print("[tool-choice] --- concurrent forced + auto")
results = {}
def run(key, choice):
    results[key] = post("/v1/chat/completions", chat_body(choice, False, False), False)
threads = [threading.Thread(target=run, args=("forced", chat_choice("get_weather"))), threading.Thread(target=run, args=("auto", "auto"))]
for t in threads: t.start()
for t in threads: t.join()
forced_requests += 1
for key, (status, raw) in results.items():
    check(f"concurrent {key}: 200", status == 200, f"(got {status})")
if results.get("forced", (0,))[0] == 200:
    calls = chat_read(results["forced"][1], False)[0]
    check("concurrent forced: the first call is the named function", len(calls) >= 1 and calls[0][0] == "get_weather", f"(calls {calls})")

log = open(LOG, errors="replace").read()
armed = log.count("[tool-choice] forced call ")
check(f"log: one arming line per forced request ({armed} for {forced_requests})", armed == forced_requests)
check("log: no prompt-only fallback", "forced call is prompt-only" not in log)
print(f"[tool-choice] {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
PY
    RC=$?
    cleanup
    [ "$RC" = 0 ] || TOTAL_FAIL=$((TOTAL_FAIL + 1))
    echo "[tool-choice] log: $OUT/server.log"
done
exit "$TOTAL_FAIL"

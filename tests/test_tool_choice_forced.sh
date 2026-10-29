#!/bin/bash
# sushi-test: tier=live boot=api
# test_tool_choice_forced.sh — a forced tool_choice is honoured on every surface.
#
# `required` / Anthropic `any` / a named function used to reach only the generic
# fallback render, so a template that renders tools itself (Qwen3.8, MiMo) never
# saw it. The call is now committed at decode, after the closed thought. On the
# pass's model, on /v1/chat/completions, /v1/messages and /v1/responses,
# stream and non-stream, thinking on (tiny reasoning budget) and off, greedy:
#
#   1. a named choice yields a call to THAT function, even one the question does
#      not need; `required`/`any` yields at least one call
#   2. every call names a declared function and its arguments are a JSON object
#   3. stream and non-stream deliver the same calls, content and reasoning (the
#      second request restores the first's cached prefix, which is not bit-exact on
#      a hybrid: with thinking on only the call names are compared)
#   4. no tool or think markup leaks into content or reasoning
#   5. the log shows one `[tool-choice] forced call` line per forced request and
#      no prompt-only fallback
#   6. `none` yields no call; a named choice for an undeclared function is a 400
#
# Model choices (which function `required` picks, what the thought says) are
# printed, never asserted.
#
# GLM cannot turn thinking off (the server answers 400), so its thinking-off arms are not sent.
#
# Attaches to the shared api boot: tests/run.sh --only test_tool_choice_forced.sh --model <m>.

set -uo pipefail
. "$(dirname "$0")/lib/boot.sh"
sushi_attach

python3 - "$SUSHI_URL" "$SUSHI_NONCE" "$SUSHI_ARCH" "$(dirname "$0")/lib" <<'PY'
import json, sys, urllib.request, urllib.error, threading

BASE, NONCE, ARCH = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, sys.argv[4])
import sushi_test

BUDGET = 256
# GLM refuses thinking off with a 400, so every arm that would send it sends thinking on.
THINKING_ARMS = (True,) if ARCH == "glm5_next" else (True, False)
OFF = ARCH != "glm5_next"
TOOLS = [
    ("get_time", "Current local time in a city", {"city": {"type": "string"}}),
    ("get_weather", "Current weather in a city", {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}),
]
DECLARED = {n for n, _, _ in TOOLS}
QUESTION = f"{NONCE} What time is it in Tokyo right now?"
LEAKS = ("<tool_call>", "</tool_call>", "<function=", "</think>", "<think>")
passed = failed = 0
forced_requests = 0

def prompt_only():
    """True once any forced request fell back to the prompt instruction alone: this template has no call
    markup to commit after the thought, so the call itself is the checkpoint's choice."""
    return "forced call is prompt-only" in sushi_test.log_since()

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
        for thinking in THINKING_ARMS:
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
                if prompt_only():
                    print("      (prompt-only force on this template: whether it calls is the checkpoint's choice)")
                elif kind == "required":
                    check(f"{tag}: at least one call", len(calls) >= 1)
                else:
                    check(f"{tag}: the first call is the named function", len(calls) >= 1 and calls[0][0] == kind, f"(calls {calls})")
                check(f"{tag}: every call names a declared function", all(n in DECLARED for n, _ in calls), f"(calls {calls})")
                check(f"{tag}: every call's arguments are a JSON object", all(args_object(a) for _, a in calls), f"(calls {calls})")
                check(f"{tag}: no markup leaks into content or reasoning",
                      not any(m in content or m in reasoning for m in LEAKS), f"(content {content[:120]!r})")
            if len(replies) == 2:
                same = replies[False] == replies[True] if not thinking else \
                    [n for n, _ in replies[False][0]] == [n for n, _ in replies[True][0]]
                check(f"{surface} {kind} thinking={thinking}: stream and non-stream deliver the same reply",
                      same, f"({replies[False]} vs {replies[True]})")

    status, raw = post(path, body(wire("none"), not OFF, False), False)
    calls = read(raw, False)[0] if status == 200 else None
    check(f"{surface} none: 200 and no call", status == 200 and calls == [], f"(status {status}, calls {calls})")
    status, raw = post(path, body(wire("undeclared"), not OFF, False), False)
    check(f"{surface} undeclared named function: 400", status == 400, f"(got {status})")

# Two at once: a forced call beside an auto turn. The forced slot decodes serial
# until its call is committed; both must finish, the forced one with its call.
print("[tool-choice] --- concurrent forced + auto")
results = {}
def run(key, choice):
    results[key] = post("/v1/chat/completions", chat_body(choice, not OFF, False), False)
threads = [threading.Thread(target=run, args=("forced", chat_choice("get_weather"))), threading.Thread(target=run, args=("auto", "auto"))]
for t in threads: t.start()
for t in threads: t.join()
forced_requests += 1
for key, (status, raw) in results.items():
    check(f"concurrent {key}: 200", status == 200, f"(got {status})")
if results.get("forced", (0,))[0] == 200 and not prompt_only():
    calls = chat_read(results["forced"][1], False)[0]
    check("concurrent forced: the first call is the named function", len(calls) >= 1 and calls[0][0] == "get_weather", f"(calls {calls})")

log = sushi_test.log_since()
armed = log.count("[tool-choice] forced call ")
fallbacks = log.count("forced call is prompt-only")
check(f"log: one arming line per forced request ({armed} for {forced_requests})", armed == forced_requests)
if ARCH in ("qwen4_exp", "mimo_v2"):
    check("log: no prompt-only fallback", fallbacks == 0)
else:
    print(f"[tool-choice] {fallbacks} prompt-only fallbacks on {ARCH}")
print(f"[tool-choice] {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
PY

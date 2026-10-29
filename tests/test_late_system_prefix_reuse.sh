#!/bin/bash
# sushi-test: tier=live boot=api
# test_late_system_prefix_reuse.sh — a mid-conversation system turn keeps the prefix cache.
#
# Codex sends a new `developer` turn mid-input and Claude Code sends hook output as
# a `system` message between turns. /v1/responses and /v1/messages used to fold every
# such turn into the leading system message, so on MiMo (whose template places a late
# system turn) turn N's prompt stopped being a prefix of turn N+1's and every turn
# prefilled from scratch. Per surface, greedy, thinking off (GLM refuses that: a small budget):
#
#   1. turn 1 answers; turn 2 = turn 1 + its reply + a NEW late system/developer turn
#      + a user turn
#   2. MiMo and Qwen3.8: turn 2 reads at least 90% of turn 1's prompt from the cache
#      (`input_tokens_details.cached_tokens`, `cache_read_input_tokens`)
#   3. Qwen3.8 (the stock template refuses a late system turn; the late-system adapter renders
#      it in place): the same 90% bar, which assumes a pack that ships the stock template
#
# The other archs only have to answer both turns. The reuse bar reads the boot's SSD prefix tier.
. "$(dirname "$0")/lib/boot.sh"; sushi_attach
set -uo pipefail

python3 - "$SUSHI_URL" "${SUSHI_ARCH:-}" "$SUSHI_NONCE" <<'PY'
import json, sys, urllib.request

BASE, ARCH, NONCE = sys.argv[1], sys.argv[2], sys.argv[3]
places_late_system = ARCH in ("mimo_v2", "qwen4_exp")
passed = failed = 0

if ARCH == "glm5_next":
    R_THINK = {"reasoning": {"effort": "low"}, "reasoning_budget_tokens": 8}
    M_THINK = {"thinking": {"type": "enabled", "budget_tokens": 8}}
    TOKENS = 64
else:
    R_THINK = {"reasoning": {"effort": "none"}}
    M_THINK = {"thinking": {"type": "disabled"}}
    TOKENS = 16

def check(label, ok, detail=""):
    global passed, failed
    if ok: passed += 1; print(f"  PASS {label}")
    else: failed += 1; print(f"  FAIL {label} {detail}")

def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())

def long_system(topic):
    return NONCE + " " + (f"You are a careful assistant for the {topic} project. Answer in one word when asked. ") * 40

# /v1/responses, Codex-style: the whole input every turn, a new developer turn mid-conversation.
instructions = long_system("responses")
turn1 = [{"role": "developer", "content": "Environment: repository at /repo."},
         {"role": "user", "content": f"{NONCE} Reply with the single word: ready."}]
r1 = post("/v1/responses", dict({"model": "x", "instructions": instructions, "input": turn1, "temperature": 0,
                                 "max_output_tokens": TOKENS}, **R_THINK))
reply = "".join(p.get("text", "") for it in r1.get("output", []) if it.get("type") == "message" for p in it.get("content") or []) or "ready"
turn2 = turn1 + [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": reply}]},
                 {"role": "developer", "content": "Approval mode: never ask."},
                 {"role": "user", "content": f"{NONCE} Reply with the single word: done."}]
r2 = post("/v1/responses", dict({"model": "x", "instructions": instructions, "input": turn2, "temperature": 0,
                                 "max_output_tokens": TOKENS}, **R_THINK))
p1, c2 = r1["usage"]["input_tokens"], r2["usage"]["input_tokens_details"]["cached_tokens"]
print(f"    responses: turn 1 prompt {p1}, turn 2 prompt {r2['usage']['input_tokens']} cached {c2}")
check("responses: both turns answer", r1.get("status") in ("completed", "incomplete") and r2.get("status") in ("completed", "incomplete"))
if places_late_system:
    check(f"responses: turn 2 reuses turn 1's prompt ({c2} cached of {p1})", c2 >= 0.9 * p1)

# /v1/messages, Claude Code-style: hook output arrives as a system message between turns.
system = long_system("messages")
m_turn1 = [{"role": "user", "content": f"{NONCE} Reply with the single word: ready."}]
m1 = post("/v1/messages", dict({"model": "x", "system": system, "messages": m_turn1, "max_tokens": TOKENS, "temperature": 0}, **M_THINK))
m_reply = "".join(b.get("text", "") for b in m1.get("content", []) if b.get("type") == "text") or "ready"
m_turn2 = m_turn1 + [{"role": "assistant", "content": m_reply},
                     {"role": "system", "content": "Hook: session started."},
                     {"role": "user", "content": f"{NONCE} Reply with the single word: done."}]
m2 = post("/v1/messages", dict({"model": "x", "system": system, "messages": m_turn2, "max_tokens": TOKENS, "temperature": 0}, **M_THINK))
mp1, mc2 = m1["usage"]["input_tokens"], m2["usage"]["cache_read_input_tokens"]
print(f"    messages: turn 1 prompt {mp1}, turn 2 prompt {m2['usage']['input_tokens']} cached {mc2}")
check("messages: both turns answer", m1.get("type") == "message" and m2.get("type") == "message")
if places_late_system:
    check(f"messages: turn 2 reuses turn 1's prompt ({mc2} cached of {mp1})", mc2 >= 0.9 * mp1)

print(f"[late-system] {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
PY

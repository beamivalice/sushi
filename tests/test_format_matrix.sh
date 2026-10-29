#!/bin/bash
# sushi-test: tier=live boot=api
# test_format_matrix.sh — format-correctness matrix of the served architecture.
#
# Live layer of the format test suite (the hermetic layer is
# src/format_corpus_test.zig — run `zig build test -Dtest-filter="format corpus"`).
# Attaches to the api boot of one model pass (qwen4_exp, mimo_v2, glm5_next or deepseek_v41) and asserts:
#
#   1. Thinking OFF baseline: answer in content, no control-tag leak (GLM has no off mode: a 400)
#   2. Thinking ON truncated: reasoning_content only, content empty/clean
#      (pins the Qwen template-opened-thinking leak at the HTTP level)
#   3. Thinking ON full round: answer in content, reasoning separated,
#      usage.completion_tokens_details.reasoning_tokens > 0
#   4. Stream + tools + thinking: answer arrives as CONTENT deltas, no leak
#      (pins the answer-misfiled-as-reasoning stream bug)
#   5. Tool-call fidelity, non-stream: write tool, path byte-exact
#   6. Tool-call fidelity, stream: same via accumulated delta.tool_calls
#   7. Omitted max_tokens: long answer must NOT truncate at the old 256 default
#   8. /v1/messages stream + tools + thinking: tool_use block, no leak
#   9-11. Truncated thinking on /v1/messages, /v1/responses and the chat stream
#   12-14. Thinking streams: split in live chunks, with tools, and after a tool result
#
# A thought the token cap cuts open is a legal turn on every row: the rows then assert the split (reasoning kept,
# nothing in content), never the answer. Thinking rows ask `reasoning_effort: low`, since the split is the subject.
# On any FAIL the slice's raw-output dump lines ("raw generated text before tool parse") are echoed for harvesting
# into src/format_corpus_test.zig.
#
# Usage: tests/run.sh --only test_format_matrix.sh --model <qwen4|mimo2|glm5|dsv41>

set -u

. "$(dirname "$0")/lib/boot.sh"
sushi_attach

PASS=0
FAIL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Only GLM-5.3 refuses thinking off (checkThinkingSupport in src/server.zig): its thinking-off row is the 400.
GLM=0
[ "${SUSHI_ARCH:-}" = glm5_next ] && GLM=1

THINKS=$(curl -sf "$SUSHI_URL/v1/models" | python3 -I -c 'import json,sys
d = json.load(sys.stdin)
print(1 if any("reasoning" in m.get("capabilities", []) for m in d["data"]) else 0)')

check() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then
        PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $desc"
    else
        FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $desc"
    fi
}

jget() { # jget <json> <python-expr over r>
    python3 -I -c "import json,sys; r=json.loads(sys.argv[1]); print($2)" "$1" 2>/dev/null
}

# chk <desc> <python-expr over r>: PASS when the expression is truthy over the last response, $R; a raise fails.
chk() {
    check "$1" "$(jget "$R" "1 if ($2) else 0")"
}

# One request, normalized: ask <chat|messages|responses> <0|1 stream> <json body>. Prints one JSON object with
# http, finish, content, reasoning, calls[{name,args}], usage (chat) or text/thinking (messages, responses: text only).
HELPER=$(cat <<'PY'
import json, sys, urllib.error, urllib.request

base, mode, stream, body = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4]
path = {"chat": "/v1/chat/completions", "messages": "/v1/messages", "responses": "/v1/responses"}[mode]
payload = json.loads(body)
if stream:
    payload["stream"] = True
req = urllib.request.Request(base + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
try:
    resp = urllib.request.urlopen(req, timeout=1800)
except urllib.error.HTTPError as e:
    resp = e
out = {"http": resp.getcode()}


def sse(resp):
    for line in resp:
        line = line.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        p = line[5:].strip()
        if p == "[DONE]":
            yield None
            return
        try:
            yield json.loads(p)
        except ValueError:
            pass


if out["http"] != 200:
    out["error"] = resp.read().decode("utf-8", "replace")
elif mode == "responses":
    d = json.load(resp)
    out.update(finish=d.get("status"), usage=d.get("usage") or {},
               text="".join(c.get("text", "") for it in d.get("output", []) if it.get("type") == "message" for c in it.get("content", [])))
elif mode == "chat" and not stream:
    d = json.load(resp)
    ch = d["choices"][0]
    m = ch["message"]
    out.update(finish=ch.get("finish_reason"), content=m.get("content") or "", reasoning=m.get("reasoning_content") or "",
               calls=[{"name": t["function"]["name"], "args": t["function"]["arguments"]} for t in m.get("tool_calls") or []],
               usage=d.get("usage") or {})
elif mode == "chat":
    calls, c, r, fin, cn, rn = {}, "", "", None, 0, 0
    for o in sse(resp):
        if o is None:
            break
        ch = (o.get("choices") or [{}])[0]
        fin = ch.get("finish_reason") or fin
        dl = ch.get("delta", {})
        c += dl.get("content") or ""
        r += dl.get("reasoning_content") or ""
        cn += 1 if dl.get("content") else 0
        rn += 1 if dl.get("reasoning_content") else 0
        for tc in dl.get("tool_calls") or []:
            e = calls.setdefault(tc.get("index", 0), {"name": "", "args": ""})
            f = tc.get("function", {})
            if f.get("name"):
                e["name"] = f["name"]
            e["args"] += f.get("arguments") or ""
    out.update(finish=fin, content=c, reasoning=r, content_chunks=cn, reasoning_chunks=rn, calls=[calls[k] for k in sorted(calls)])
elif not stream:
    d = json.load(resp)
    blocks = d.get("content", [])
    out.update(finish=d.get("stop_reason"), usage=d.get("usage") or {},
               text="".join(b.get("text", "") for b in blocks if b["type"] == "text"),
               thinking="".join(b.get("thinking", "") for b in blocks if b["type"] == "thinking"),
               calls=[{"name": b["name"], "args": json.dumps(b.get("input", {}))} for b in blocks if b["type"] == "tool_use"])
else:
    blocks, fin = {}, None
    for ev in sse(resp):
        if ev is None:
            break
        t = ev.get("type")
        if t == "content_block_start":
            cb = ev["content_block"]
            blocks[ev["index"]] = {"type": cb["type"], "name": cb.get("name"), "text": "", "json": ""}
        elif t == "content_block_delta":
            d, b = ev["delta"], blocks.get(ev["index"])
            if b is not None:
                b["text"] += d.get("text") or d.get("thinking") or ""
                b["json"] += d.get("partial_json", "")
        elif t == "message_delta":
            fin = (ev.get("delta") or {}).get("stop_reason") or fin
    out.update(finish=fin,
               text="".join(b["text"] for b in blocks.values() if b["type"] == "text"),
               thinking="".join(b["text"] for b in blocks.values() if b["type"] == "thinking"),
               calls=[{"name": b["name"], "args": b["json"]} for b in blocks.values() if b["type"] == "tool_use"])
print(json.dumps(out))
PY
)
ask() { python3 -I -c "$HELPER" "$SUSHI_URL" "$1" "$2" "$3"; }

# no_leak <text> — 0 (clean) unless a control tag from any template family
# appears; prints the offending tag on leak.
no_leak() {
    python3 -I - "$1" <<'PY'
import sys
text = sys.argv[1]
tags = ["<think>", "</think>", "<tool_call", "<arg_key>", "<arg_value>", "DSML"]
for t in tags:
    if t in text:
        print(f"  leaked tag: {t}", file=sys.stderr)
        sys.exit(1)
sys.exit(0)
PY
}

# chk_clean <desc> <field>: no control tag in that field of $R
chk_clean() {
    check "$1" "$(no_leak "$(jget "$R" "r['$2']")" && echo 1 || echo 0)"
}

user() { printf '{"role":"user","content":"%s %s"}' "$SUSHI_NONCE" "$1"; }

PROMPT=$(user 'What is 17*23? Be brief.')
WRITE_PROMPT=$(user 'Use the write tool to create a file named exactly report_v2.html containing <h1>Report</h1>. Do not ask questions, call the tool now.')
BASH_TOOL='{"type":"function","function":{"name":"bash","description":"Run a shell command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}'
WRITE_TOOL='{"type":"function","function":{"name":"write","description":"Write a file to disk","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"},"content":{"type":"string","description":"File content"}},"required":["path","content"]}}}'
SHELL_TOOL='{"type":"function","function":{"name":"shell","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}'
ANTH_WRITE_TOOL='{"name":"write","description":"Write a file to disk","input_schema":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}'
THINK='"enable_thinking":true,"reasoning_effort":"low"'

# Validate a write tool call (r['calls'][0]): name == write, args a JSON object, path byte-exact.
WRITE_NAME="bool(r['calls']) and r['calls'][0]['name']=='write'"
WRITE_JSON="isinstance(json.loads(r['calls'][0]['args']),dict)"
WRITE_PATH="json.loads(r['calls'][0]['args']).get('path')=='report_v2.html'"

echo -e "${YELLOW}=== [${SUSHI_ARCH:-?}] format matrix ===${NC}"

# ── 1. Thinking OFF baseline ──
R=$(ask chat 0 "{\"model\":\"x\",\"messages\":[$PROMPT],\"max_tokens\":200,\"temperature\":0,\"enable_thinking\":false}")
if [ "$GLM" = 1 ]; then
    chk "1. thinking-off: refused with a 400 naming the efforts GLM serves" "r['http']==400 and 'low, high, max' in r.get('error','')"
else
    chk "1. thinking-off: answer (391) in content" "'391' in r['content']"
    chk_clean "1. thinking-off: no control-tag leak" content
fi

if [ "$THINKS" = 1 ]; then
    # ── 2. Thinking ON, truncated (pins bug 1 at HTTP level) ──
    # A short thinker can close the thought inside the cap; only a cut thought must leave content empty.
    R=$(ask chat 0 "{\"model\":\"x\",\"messages\":[$PROMPT],\"max_tokens\":120,\"temperature\":0,$THINK}")
    chk "2. truncated thinking: content empty when the cap cut the thought" "r['finish']!='length' or not r['content']"
    chk "2. truncated thinking: reasoning_content non-empty" "r['reasoning']"
    chk "2. truncated thinking: usage reasoning_tokens > 0" "r['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0)>0"

    # ── 3. Thinking ON, full round ──
    R=$(ask chat 0 "{\"model\":\"x\",\"messages\":[$PROMPT],\"max_tokens\":3000,\"temperature\":0,$THINK}")
    chk "3. full round: answer (391) in content" "'391' in r['content'] or r['finish']=='length'"
    chk_clean "3. full round: no control-tag leak in content" content
    chk "3. full round: reasoning separated (more than 50 chars)" "len(r['reasoning'])>50"
    chk "3. full round: usage reasoning_tokens > 0" "r['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0)>0"
fi

# ── 4. Stream + tools (+thinking where supported): answer must arrive as
#      CONTENT deltas (pins the answer-misfiled-as-reasoning stream bug) ──
T4=""
[ "$THINKS" = 1 ] && T4=",$THINK"
R=$(ask chat 1 "{\"model\":\"x\",\"messages\":[$PROMPT],\"max_tokens\":3000,\"temperature\":0,\"tools\":[$BASH_TOOL]$T4}")
chk "4. stream+tools: answer (391) in CONTENT deltas" "'391' in r['content'] or r['finish']=='length'"
chk_clean "4. stream+tools: no control-tag leak" content
[ "$THINKS" != 1 ] || chk "4. stream+tools: reasoning still separated" "r['reasoning']"

# ── 5. Tool-call fidelity, non-stream (pins the brace-swallow bug live) ──
R=$(ask chat 0 "{\"model\":\"x\",\"messages\":[$WRITE_PROMPT],\"max_tokens\":2000,\"temperature\":0,\"reasoning_effort\":\"low\",\"tools\":[$WRITE_TOOL]}")
chk "5. non-stream tool call: name == write" "$WRITE_NAME"
chk "5. non-stream tool call: args are valid JSON" "$WRITE_NAME and $WRITE_JSON"
chk "5. non-stream tool call: path byte-exact report_v2.html" "$WRITE_NAME and $WRITE_PATH"

# ── 6. Tool-call fidelity, stream (accumulated delta.tool_calls) ──
R=$(ask chat 1 "{\"model\":\"x\",\"messages\":[$WRITE_PROMPT],\"max_tokens\":2000,\"temperature\":0,\"reasoning_effort\":\"low\",\"tools\":[$WRITE_TOOL]}")
chk "6. stream tool call: name == write" "$WRITE_NAME"
chk "6. stream tool call: args are valid JSON" "$WRITE_NAME and $WRITE_JSON"
chk "6. stream tool call: path byte-exact report_v2.html" "$WRITE_NAME and $WRITE_PATH"

# ── 7. Omitted max_tokens must not truncate at the old 256 default ──
R=$(ask chat 0 "{\"model\":\"x\",\"messages\":[$(user 'List the numbers from 1 to 200, one per line. No other text.')],\"temperature\":0,\"reasoning_effort\":\"low\"}")
FINISH=$(jget "$R" "r['finish']")
CTOK=$(jget "$R" "r['usage'].get('completion_tokens',0)")
check "7. omitted max_tokens: finish_reason != length (got $FINISH @ $CTOK tok)" "$([ "$FINISH" != "length" ] && echo 1 || echo 0)"
check "7. omitted max_tokens: completion_tokens != 256" "$([ "${CTOK:-0}" != "256" ] && echo 1 || echo 0)"

# ── 8. /v1/messages stream + tools (+thinking): the Claude Code surface.
#      Tool call must arrive as a tool_use block (name + byte-exact path),
#      and no text delta may carry a raw control tag.
A8=""
[ "$THINKS" = 1 ] && A8=',"thinking":{"type":"enabled","budget_tokens":2000}'
R=$(ask messages 1 "{\"model\":\"x\",\"max_tokens\":4000,\"temperature\":0,\"tools\":[$ANTH_WRITE_TOOL],\"messages\":[$WRITE_PROMPT]$A8}")
chk "8. /v1/messages stream: tool_use block name == write" "$WRITE_NAME"
chk "8. /v1/messages stream: args valid + path byte-exact" "$WRITE_NAME and $WRITE_JSON and $WRITE_PATH"
chk_clean "8. /v1/messages stream: no control-tag leak in deltas" text

if [ "$THINKS" = 1 ]; then
    # ── 9. /v1/messages: truncated thinking lands in the thinking block ──
    R=$(ask messages 0 "{\"model\":\"x\",\"max_tokens\":120,\"messages\":[$PROMPT],\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":4000}}")
    chk "9. /v1/messages: truncated thinking not in the text block" "r['finish']!='max_tokens' or not r['text']"
    chk "9. /v1/messages: thinking block non-empty" "r['thinking']"

    # ── 10. /v1/responses: truncated thinking is no message text, and usage reports the reasoning ──
    R=$(ask responses 0 "{\"model\":\"x\",\"input\":\"$SUSHI_NONCE What is 17*23? Be brief.\",\"max_output_tokens\":120,\"reasoning\":{\"effort\":\"low\"}}")
    chk "10. /v1/responses: truncated thinking not in message text" "r['finish']!='incomplete' or not r['text']"
    chk "10. /v1/responses: usage output_tokens_details.reasoning_tokens > 0" "r['usage'].get('output_tokens_details',{}).get('reasoning_tokens',0)>0"

    # ── 11. Stream + thinking, truncated: deltas split, nothing in content ──
    R=$(ask chat 1 "{\"model\":\"x\",\"messages\":[$PROMPT],\"max_tokens\":120,\"temperature\":0,$THINK}")
    chk "11. stream truncated thinking: reasoning deltas arrive" "r['reasoning']"
    chk "11. stream truncated thinking: no content deltas when the cap cut the thought" "r['finish']!='length' or not r['content']"
fi

if [ "$THINKS" = 1 ]; then
    # ── 12. Stream + thinking, no tools: a turn is EITHER a streamed think block OR a direct answer as content,
    #      never the whole answer filed as reasoning with empty content ──
    R=$(ask chat 1 "{\"model\":\"x\",\"messages\":[$(user 'What is 17 * 23? Just the number after thinking.')],\"max_tokens\":1500,\"temperature\":0,$THINK}")
    if [ "$(jget "$R" "1 if r['reasoning'] else 0")" = 1 ]; then
        chk "12. stream thinking: reasoning_content populated (>50)" "len(r['reasoning'])>50"
        chk "12. stream thinking: reasoning arrives in more than one chunk (live streaming)" "r['reasoning_chunks']>1"
    else
        chk "12. stream no-think turn: answer arrives as CONTENT, reasoning stays empty" "r['content'] and r['reasoning_chunks']==0"
    fi
    chk "12. stream thinking: no leaked think tags in reasoning" "not any(t in r['reasoning'] for t in ('<think','</think'))"
    chk_clean "12. stream thinking: no leaked tags in content" content

    # ── 13. Stream + thinking + tools ──
    R=$(ask chat 1 "{\"model\":\"x\",\"messages\":[$(user 'What time is it? Use the shell tool.')],\"max_tokens\":2000,\"temperature\":0,\"tools\":[$SHELL_TOOL],$THINK}")
    chk "13. stream thinking+tools: at least one tool_call (or the thought was cut)" "r['calls'] or (r['finish']=='length' and r['reasoning'])"
    chk_clean "13. stream thinking+tools: no tag or tool-call leak in content" content
    chk "13. stream thinking+tools: no tag leak in reasoning" "not any(t in r['reasoning'] for t in ('<think','</think'))"

    # ── 14. Tool round-trip with thinking (turn 2 after the tool result): the turn is well-formed and not empty ──
    RT_TAIL='{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"shell","arguments":"{\"command\":\"date\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"Fri Apr 17 08:15:00 PDT 2026"}'
    RT_BODY="\"model\":\"x\",\"messages\":[$(user 'What time is it? Use the shell tool.'),$RT_TAIL],\"max_tokens\":2000,\"temperature\":0,\"tools\":[$SHELL_TOOL],$THINK"
    R=$(ask chat 0 "{$RT_BODY}")
    chk "14. round-trip non-stream: content populated (>5), a further call, or a cut thought" "len(r['content'])>5 or r['calls'] or (r['finish']=='length' and r['reasoning'])"
    R=$(ask chat 1 "{$RT_BODY}")
    chk "14. round-trip stream: content populated (>5), a further call, or a cut thought" "len(r['content'])>5 or r['calls'] or (r['finish']=='length' and r['reasoning'])"
    chk_clean "14. round-trip stream: no tag leak in content" content
    chk "14. round-trip stream: no tag leak in reasoning" "not any(t in r['reasoning'] for t in ('<think','</think'))"
fi

echo
echo "Results: $PASS passed, $FAIL failed"
if [ "$FAIL" -gt 0 ]; then
    echo -e "  ${YELLOW}--- raw model output dumps (corpus harvesting) ---${NC}"
    sushi_log_since | grep -a "raw generated text before tool parse" | tail -8 | sed 's/^/  /'
    exit 1
fi

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
#   15-17. Thinking x tools x stream grid, and stream == non-stream split
#   18. Three tool chats over a three-tool catalog parse to structured calls
#   19. Multi-line code round trip: a writeFile call and a multi-line tool result
#   20. /v1/messages stream with thinking + tools (Claude Code's shape): block lifecycle, no tag leak
#   21. Reasoning usage: billed reasoning tokens equal the delivered reasoning, on seven turn shapes
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
    calls, c, r, fin, cn, rn, done, usage, uchoices = {}, "", "", None, 0, 0, False, {}, None
    for o in sse(resp):
        if o is None:
            done = True
            break
        if o.get("usage"):
            usage, uchoices = o["usage"], len(o.get("choices") or [])
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
    out.update(finish=fin, content=c, reasoning=r, content_chunks=cn, reasoning_chunks=rn, done=done, usage=usage, usage_choices=uchoices, calls=[calls[k] for k in sorted(calls)])
elif not stream:
    d = json.load(resp)
    blocks = d.get("content", [])
    out.update(finish=d.get("stop_reason"), usage=d.get("usage") or {},
               text="".join(b.get("text", "") for b in blocks if b["type"] == "text"),
               thinking="".join(b.get("thinking", "") for b in blocks if b["type"] == "thinking"),
               calls=[{"name": b["name"], "args": json.dumps(b.get("input", {}))} for b in blocks if b["type"] == "tool_use"])
else:
    # The content-block lifecycle is checked as the stream goes: every delta and stop names an open block,
    # no index is reused while open, nothing is left open at message_stop.
    blocks, live, err, fin, stopped, saw_text, after_text = {}, {}, None, None, False, False, 0
    for ev in sse(resp):
        if ev is None:
            break
        t = ev.get("type")
        if t == "content_block_start":
            i, cb = ev["index"], ev["content_block"]
            if i in live:
                err = err or f"start index {i} while already open as {live[i]}"
            live[i] = cb["type"]
            blocks[i] = {"type": cb["type"], "name": cb.get("name"), "text": "", "json": ""}
            after_text += 1 if cb["type"] == "thinking" and saw_text else 0
            saw_text = saw_text or cb["type"] == "text"
        elif t == "content_block_delta":
            i, d = ev["index"], ev["delta"]
            if i not in live:
                err = err or f"delta for unopened index {i}"
            b = blocks.get(i)
            if b is not None:
                b["text"] += d.get("text") or d.get("thinking") or ""
                b["json"] += d.get("partial_json", "")
            if d.get("type") == "text_delta" and ("</think>" in d.get("text", "") or "<think>" in d.get("text", "")):
                err = err or f"think tag leaked in text_delta: {d.get('text')!r}"
        elif t == "content_block_stop":
            if ev["index"] not in live:
                err = err or f"stop for unopened index {ev['index']}"
            else:
                del live[ev["index"]]
        elif t == "message_delta":
            fin = (ev.get("delta") or {}).get("stop_reason") or fin
        elif t == "message_stop":
            stopped = True
            if live:
                err = err or f"blocks still open at message_stop: {sorted(live)}"
    err = err or (None if stopped else "no message_stop event")
    # A thought opened after visible text carries its own bytes: the split once ran over the whole buffer, so
    # the answer rode out as reasoning a second time.
    lead = ""
    for i in sorted(blocks):
        b = blocks[i]
        if b["type"] == "text" and not lead:
            lead = b["text"].strip()
        elif b["type"] == "thinking" and len(lead) >= 2 and b["text"].startswith(lead):
            err = err or f"thinking block {i} repeats the text delivered before it: {b['text'][:60]!r}"
    count = lambda ty: sum(1 for b in blocks.values() if b["type"] == ty)
    out.update(finish=fin, err=err, n_text=count("text"), n_thinking=count("thinking"), n_tool=count("tool_use"),
               thought_after_text=after_text,
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

# ntok <text>: the token count of the server's own tokenizer
ntok() {
    python3 -I -c 'import json, sys, urllib.request
req = urllib.request.Request(sys.argv[1] + "/tokenize", data=json.dumps({"content": sys.argv[2]}).encode(), headers={"Content-Type": "application/json"})
print(len(json.load(urllib.request.urlopen(req, timeout=60))["tokens"]))' "$SUSHI_URL" "$1"
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
    chk "12. stream thinking: content deltas follow the thought (or the cap cut it)" "r['content_chunks']>0 or r['finish']=='length'"
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

# ── 15. The thinking x tools x stream grid. The no-thinking half: a plain answer carries no reasoning, a
#      tool turn carries a call and no reasoning. GLM has no thinking-off mode, so it skips the half. ──
CMD_PROMPT=$(user 'Run the command: echo hello')
if [ "$GLM" = 1 ]; then
    echo -e "  ${YELLOW}SKIP${NC} 15. no-thinking permutations: GLM has no thinking-off mode"
else
    R=$(ask chat 0 "{\"model\":\"x\",\"messages\":[$(user 'What is 2+2? Answer in one word.')],\"max_tokens\":300,\"temperature\":0,\"enable_thinking\":false}")
    chk "15. no thinking, no tools, non-stream: content, no reasoning_content" "r['content'] and not r['reasoning']"
    R=$(ask chat 0 "{\"model\":\"x\",\"messages\":[$CMD_PROMPT],\"tools\":[$SHELL_TOOL],\"max_tokens\":300,\"temperature\":0,\"enable_thinking\":false}")
    chk "15. no thinking, tools, non-stream: finish tool_calls, call shell, no reasoning_content" "r['finish']=='tool_calls' and r['calls'] and r['calls'][0]['name']=='shell' and not r['reasoning']"
    R=$(ask chat 1 "{\"model\":\"x\",\"messages\":[$(user 'What is 2+2? One word.')],\"max_tokens\":300,\"temperature\":0,\"enable_thinking\":false}")
    chk "15. no thinking, no tools, stream: content deltas, no reasoning deltas, [DONE]" "r['content_chunks']>0 and r['reasoning_chunks']==0 and r['done']"
    R=$(ask chat 1 "{\"model\":\"x\",\"messages\":[$CMD_PROMPT],\"tools\":[$SHELL_TOOL],\"max_tokens\":300,\"temperature\":0,\"enable_thinking\":false}")
    chk "15. no thinking, tools, stream: tool_calls delta, finish tool_calls, no reasoning deltas" "r['calls'] and r['finish']=='tool_calls' and r['reasoning_chunks']==0"
    chk_clean "15. no thinking, tools, stream: no tag leak in content" content
fi

if [ "$THINKS" = 1 ]; then
    # ── 16. The thinking half. What a model does with its thought is its choice: the split holds either way ──
    R=$(ask chat 0 "{\"model\":\"x\",\"messages\":[$CMD_PROMPT],\"tools\":[$SHELL_TOOL],\"max_tokens\":2000,\"temperature\":0,$THINK}")
    chk "16. thinking, tools, non-stream: a call (or the thought was cut)" "r['calls'] or (r['finish']=='length' and r['reasoning'])"
    chk "16. thinking, tools, non-stream: with no thought, nothing leaked to content" "r['reasoning'] or not r['content']"
    chk_clean "16. thinking, tools, non-stream: no tag leak in content" content

    R=$(ask chat 1 "{\"model\":\"x\",\"messages\":[$(user 'What is 15 times 17? Think step by step then use shell to verify with: echo $((15*17))')],\"tools\":[$SHELL_TOOL],\"max_tokens\":2000,\"temperature\":0,$THINK}")
    chk "16. thinking, tools, stream: reasoning deltas, or the call survived" "r['reasoning'] or r['calls']"
    chk "16. thinking, tools, stream: a call, an answer, or a thought the cap cut (reasoning kept)" "r['calls'] or r['content'] or (r['finish']=='length' and r['reasoning'])"
    chk_clean "16. thinking, tools, stream: no tag leak in content" content
    chk "16. thinking, tools, stream: no raw think tag in reasoning" "not any(t in r['reasoning'] for t in ('<think','</think'))"

    R=$(ask chat 1 "{\"model\":\"x\",\"messages\":[$(user 'Say hello in French. Do not use any tools.')],\"tools\":[$SHELL_TOOL],\"max_tokens\":2000,\"temperature\":0,$THINK}")
    chk "16. thinking, tools declared, stream: the stream carries data" "r['content'] or r['reasoning'] or r['calls']"
    chk_clean "16. thinking, tools declared, stream: no tag leak in content" content

    # ── 17. The same request streamed and not splits the SAME at temperature 0, byte for byte ──
    SPLIT_BODY="\"model\":\"x\",\"messages\":[$(user 'What is 15 times 17?')],\"max_tokens\":3000,\"temperature\":0,$THINK"
    R=$(ask chat 0 "{$SPLIT_BODY}")
    NS=$(jget "$R" "json.dumps([r['reasoning'].strip(), r['content'].strip()])")
    R=$(ask chat 1 "{$SPLIT_BODY}")
    ST=$(jget "$R" "json.dumps([r['reasoning'].strip(), r['content'].strip()])")
    check "17. stream and non-stream split identically" "$([ -n "$NS" ] && [ "$NS" = "$ST" ] && echo 1 || echo 0)"
    chk "17. the answer is content, not reasoning (or the cap cut the thought)" "r['content'] or r['finish']=='length'"
fi

# ── 18. Three tool chats with a three-tool catalog: the parser's output is a structured call, never tool-call XML
#      in content, and a direct answer is not an empty reply. Thinking is off where the arch has an off mode;
#      GLM asks effort low instead. ──
NOTHINK='"enable_thinking":false'
[ "$GLM" = 1 ] && NOTHINK='"reasoning_effort":"low"'
TOOLS3='[{"type":"function","function":{"name":"shell","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},{"type":"function","function":{"name":"webSearch","description":"Search the web with DuckDuckGo","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},{"type":"function","function":{"name":"writeFile","description":"Write a small file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]'

tool_parse() { # tool_parse <desc> <system prompt> <user prompt>
    R=$(ask chat 0 "{\"model\":\"x\",\"messages\":[{\"role\":\"system\",\"content\":\"$2\"},$(user "$3")],\"tools\":$TOOLS3,\"max_tokens\":2000,\"temperature\":0,$NOTHINK}")
    chk "18. $1: no tool-call XML in content" "not any(t in r['content'] for t in ('<tool_call','</tool_call>','</tool_request>'))"
    chk "18. $1: a well-formed call (declared name, JSON-object args, finish tool_calls) or a non-empty answer" "(r['finish']=='tool_calls' and r['calls'][0]['name'] in ('shell','webSearch','writeFile') and isinstance(json.loads(r['calls'][0]['args']),dict)) if r['calls'] else (r['content'].strip() or (r['finish']=='length' and r['reasoning']))"
}

tool_parse "shell prompt" "You can call tools. Use the shell tool to run commands." "List the files in /tmp using the shell tool."
tool_parse "webSearch prompt" "Use the webSearch tool when the user asks to research something." "Research the top 5 open source inference apps."
tool_parse "writeFile prompt" "Use the writeFile tool to create files." "Create a file at /tmp/hello.txt containing the single word 'hello'."

# ── 19. Multi-line code round trip: a writeFile call carrying code with nested braces keeps its path and content,
#      and a tool result that is itself multi-line code comes back to a well-formed turn. Whether the model calls
#      the tool is its choice: a turn with no call reports a SKIP note instead of failing. ──
SYS_A='You MUST call the writeFile tool. Do not respond with text. Call writeFile with path=\"test.js\" and content that is a simple Express server with one GET route that returns JSON {\"status\":\"ok\"}.'
WF_TOOL='{"type":"function","function":{"name":"writeFile","description":"Write a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}'
R=$(ask chat 0 "{\"model\":\"x\",\"messages\":[{\"role\":\"system\",\"content\":\"$SYS_A\"},$(user 'Create a test.js file with an Express server')],\"tools\":[$WF_TOOL],\"max_tokens\":3000,\"temperature\":0,$NOTHINK}")
if [ "$(jget "$R" "1 if r['calls'] else 0")" = 1 ]; then
    chk "19. writeFile code: path and multi-line content with nested braces survive in the args" "json.loads(r['calls'][0]['args']).get('path') and len(json.loads(r['calls'][0]['args']).get('content',''))>=20 and '{' in json.loads(r['calls'][0]['args'])['content']"
else
    echo -e "  ${YELLOW}SKIP${NC} 19. writeFile code: the model made no call (finish=$(jget "$R" "r['finish']"))"
fi

RF_TOOL='{"type":"function","function":{"name":"readFile","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}'
RF_TAIL='{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"readFile","arguments":"{\"path\": \"server.js\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"1| const express = require(\u0027express\u0027);\n2| const app = express();\n3| const cruises = [\n4|   { id: 1, title: \"Med Dream\", price: 999 },\n5|   { id: 2, title: \"Caribbean\", price: 799 },\n6| ];\n7| app.get(\"/api/cruises\", (req, res) => res.json(cruises));\n8| app.listen(3456);"}'
R=$(ask chat 0 "{\"model\":\"x\",\"messages\":[{\"role\":\"system\",\"content\":\"You are helpful. Be brief.\"},$(user 'Read the server.js file'),$RF_TAIL],\"tools\":[$RF_TOOL],\"max_tokens\":2000,\"temperature\":0,$NOTHINK}")
chk "19. tool result of multi-line code: the round trip is well-formed (finish, no markup in content, JSON args)" "r['http']==200 and r['finish'] in ('stop','length','tool_calls') and not any(t in r['content'] for t in ('<tool_call','<|tool_call','<tool_response','<|tool_response','</think')) and all(isinstance(json.loads(c['args']),dict) for c in r['calls'])"

# ── 20. /v1/messages stream with thinking + tools, Claude Code's shape (it always sends tools). The block
#      lifecycle is protocol-valid, no think tag rides a text delta, thinking arrives as thinking blocks, and the
#      turn ends in a tool_use block or a text answer. What the model writes is its choice, so a cap-cut turn
#      waives the content bars. ──
CC_TOOLS='[{"name":"get_weather","description":"Get current weather for a city","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]'
CC_THINK='"thinking":{"type":"enabled","budget_tokens":8000}'
if [ "$THINKS" = 1 ]; then
    R=$(ask messages 1 "{\"model\":\"m\",\"max_tokens\":4000,\"temperature\":0,$CC_THINK,\"tools\":$CC_TOOLS,\"messages\":[$(user 'What is 17+25? Answer with just the number. Do not use any tools.')]}")
    chk "20. thinking+tools stream, plain answer: protocol-valid block lifecycle, no think-tag leak" "r['err'] is None"
    chk "20. thinking+tools stream, plain answer: thinking arrives as thinking block(s)" "r['n_thinking']>=1"
    chk "20. thinking+tools stream, plain answer: the visible answer arrives as text block(s)" "r['n_text']>=1 or r['finish']=='max_tokens'"

    R=$(ask messages 1 "{\"model\":\"m\",\"max_tokens\":4000,\"temperature\":0,$CC_THINK,\"tools\":$CC_TOOLS,\"messages\":[$(user 'What is the weather in Paris right now? Use the get_weather tool.')]}")
    chk "20. thinking+tools stream, tool-call turn: protocol-valid block lifecycle, no think-tag leak" "r['err'] is None"
    chk "20. thinking+tools stream, tool-call turn: tool_use block emitted" "r['n_tool']>=1 or r['finish']=='max_tokens'"
    chk "20. thinking+tools stream, tool-call turn: stop_reason is tool_use" "r['finish']=='tool_use' or r['finish']=='max_tokens'"
fi

# A thought that opens after visible text, thinking off: the thought opens while the text block is still open.
# GLM has no thinking-off mode, so its row is the 400. Elsewhere a run that wrote no thought after text is
# reported, not failed.
READ_TOOLS='[{"name":"Read","description":"Read a file","input_schema":{"type":"object","properties":{"file_path":{"type":"string"}},"required":["file_path"]}}]'
R=$(ask messages 1 "{\"model\":\"m\",\"max_tokens\":400,\"temperature\":0,\"thinking\":{\"type\":\"disabled\"},\"tools\":$READ_TOOLS,\"messages\":[$(user 'Reply with the word Done, then a new line containing exactly <think>plan</think> and then a new line with the answer to 2+2. Do not use tools.')]}")
if [ "$GLM" = 1 ]; then
    chk "20. thought after visible text, thinking disabled: refused with a 400 naming the efforts GLM serves" "r['http']==400 and 'low, high, max' in r.get('error','')"
else
    chk "20. thought after visible text, thinking disabled: protocol-valid block lifecycle, each block owns its index" "r['err'] is None"
    [ "$(jget "$R" "r['thought_after_text']")" -ge 1 ] 2>/dev/null || echo "    note: the model wrote no thought after text; this run did not exercise the shape"
fi

if [ "$THINKS" = 1 ]; then
    # A thought that quotes its own tags in code spans stays one thought: inside a code span a tag is text, so
    # the thought ends only at the real close and no thought opens after the answer's text.
    R=$(ask messages 1 "{\"model\":\"m\",\"max_tokens\":4000,\"temperature\":0,\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":2000},\"tools\":$CC_TOOLS,\"messages\":[$(user 'A parser reads Qwen output where reasoning sits between <think> and </think>. For the raw string `<think>plan A</think>Done`, what are the reasoning and the content? Reason it through, quoting the exact tags as you go. In your final answer never write the tags, call them OPEN and CLOSE. Do not use tools.')]}")
    chk "20. thought quoting its own tags: protocol-valid block lifecycle" "r['err'] is None"
    chk "20. thought quoting its own tags: one thought, never reopened after text" "r['n_thinking']==1 and r['thought_after_text']==0"
fi

# ── 21. Reasoning usage: the delivered reasoning is exactly what usage bills, on every surface a turn can take
#      (plain, forced tool call, JSON mode, a one-token reasoning budget), stream and not ──
ANSWER_TOOL='{"type":"function","function":{"name":"answer","description":"Return the integer answer","parameters":{"type":"object","properties":{"value":{"type":"integer"}},"required":["value"]}}}'
usage_case() { # usage_case <name> <0|1 stream> [extra request fields, each with a leading comma]
    local so=""
    [ "$2" = 1 ] && so=',"stream_options":{"include_usage":true}'
    R=$(ask chat "$2" "{\"model\":\"x\",\"messages\":[$(user 'Calculate 123 * 456. Return the integer result.')],\"reasoning_effort\":\"xhigh\",\"temperature\":0,\"max_tokens\":256$so${3:-}}")
    local expected billed tol=0
    # Billing tokenizes the server's reasoning text; the client retokenizes the joined deltas. The deferred JSON
    # grammar path can differ by one boundary token (billed 123, delivered 122 on qwen4_exp).
    [ "$1" = json-stream ] && tol=1
    expected=$(ntok "$(jget "$R" "r['reasoning']")")
    billed=$(jget "$R" "r['usage']['completion_tokens_details']['reasoning_tokens']")
    chk "21. $1: usage reasoning_tokens is a non-negative int equal to the tokens of the delivered reasoning (billed ${billed:-?}, delivered ${expected:-?})" "type(r['usage']['completion_tokens_details']['reasoning_tokens']) is int and r['usage']['completion_tokens_details']['reasoning_tokens'] in range($expected - $tol, $expected + $tol + 1)"
    chk "21. $1: usage totals add up and prompt_tokens_details carries cached_tokens" "r['usage']['total_tokens']==r['usage']['prompt_tokens']+r['usage']['completion_tokens'] and 'cached_tokens' in r['usage']['prompt_tokens_details']"
    [ "$2" != 1 ] || chk "21. $1: the usage event repeats no choice" "r['usage_choices']==0"
}
if [ "$THINKS" = 1 ]; then
    usage_case stream 1
    usage_case nonstream 0
    usage_case tool-stream 1 ",\"tools\":[$ANSWER_TOOL],\"tool_choice\":\"required\""
    usage_case tool-nonstream 0 ",\"tools\":[$ANSWER_TOOL],\"tool_choice\":\"required\""
    usage_case json-stream 1 ',"response_format":{"type":"json_object"}'
    usage_case budget-stream 1 ',"reasoning_budget_tokens":1'
    usage_case budget-nonstream 0 ',"reasoning_budget_tokens":1'
fi

echo
echo "Results: $PASS passed, $FAIL failed"
if [ "$FAIL" -gt 0 ]; then
    echo -e "  ${YELLOW}--- raw model output dumps (corpus harvesting) ---${NC}"
    sushi_log_since | grep -a "raw generated text before tool parse" | tail -8 | sed 's/^/  /'
    exit 1
fi

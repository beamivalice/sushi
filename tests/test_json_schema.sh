#!/bin/bash
# sushi-test: tier=live boot=api
# Integration test: verify response_format.json_schema is grammar-enforced.
#
# Without grammar enforcement, the model often emits prose around the JSON, or
# closes objects too early, especially under quantization. With enforcement, the
# token mask makes those outputs unreachable — every byte of the response must
# satisfy the schema. Cases A-F add the Responses and tool_choice:none spellings, answered under a prompt that
# asks for extra fields and code fences, so a clean reply is the mask's work.
#
# The last section is the #331 guard: completed schema JSON reaches CONTENT on every mask-building surface (chat,
# Responses, Messages), a completion cap spent in reasoning keeps empty content with `length`, and protocol-looking
# strings in schema data survive every surface with thinking on and off.
#
# Attaches to the shared api boot: tests/run.sh --tier live --model <m> --only test_json_schema.sh

. "$(dirname "$0")/lib/boot.sh"
sushi_attach
BASE=$SUSHI_URL
PASS=0
FAIL=0
TOTAL=0

echo "=== JSON Schema Constrained Generation Test ==="
echo "Server: $BASE ($SUSHI_ARCH)"
echo ""

# A model that can switch thinking off answers inside the small caps below. One that cannot (GLM) thinks
# first: it gets a large cap, and a reply that spent the whole cap in reasoning (content empty, finish
# `length`) is the checkpoint's choice, not a schema failure.
read -r OFF_OK CAPPED EFFORT_MED EFFORT_LOW < <(curl -sf "$BASE/v1/models" | python3 -I -c '
import sys, json
m = json.load(sys.stdin)["data"][0]
e = m.get("reasoning_efforts", [])
pick = lambda *w: next((x for x in w if x in e), "on")
print(int("off" in e), int(bool(m.get("reasoning_effort_budgets"))), pick("medium", "low"), pick("low"))')
if [ "$OFF_OK" = 1 ]; then
    THINK='"enable_thinking": false,'
    THINK_OBJ='{"enable_thinking":false}'
else
    THINK='"reasoning_effort": "low",'
    THINK_OBJ='{"reasoning_effort":"low"}'
fi
TOL_BASE=$((1 - OFF_OK))
cap() { if [ "$OFF_OK" = 1 ]; then echo "$1"; else echo 4096; fi; }

run_test() {
    local name="$1"
    local result="$2"
    local detail="$3"
    TOTAL=$((TOTAL + 1))
    if [ "$result" = "PASS" ]; then
        PASS=$((PASS + 1))
        echo "  PASS: $name"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $name — $detail"
    fi
}

# chat_parse <body> <stream>: prints {"finish": .., "content": ..} for a chat completion or its SSE stream.
chat_parse() {
    python3 -I -c '
import sys, json
raw, stream = sys.argv[1], sys.argv[2] == "true"
finish, buf = None, []
try:
    if stream:
        for line in raw.splitlines():
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            for c in json.loads(line[6:]).get("choices", []):
                finish = c.get("finish_reason") or finish
                buf.append(c.get("delta", {}).get("content") or "")
    else:
        c = json.loads(raw)["choices"][0]
        finish, buf = c.get("finish_reason"), [c["message"].get("content") or ""]
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit()
print(json.dumps({"finish": finish, "content": "".join(buf)}))' "$1" "$2"
}

# schema_check <kind> [tolerate]: stdin is chat_parse output; prints ok, ok:cap-spent-in-reasoning (accepted when
# tolerate is 1, default TOL_BASE) or fail:<why>.
schema_check() {
    python3 -I -c '
import sys, json
kind, tolerate = sys.argv[1], sys.argv[2] == "1"
p = json.load(sys.stdin)
if "error" in p:
    print("fail:unparsable_response:" + p["error"]); sys.exit()
text = p["content"]
if tolerate and p["finish"] == "length" and text == "":
    print("ok:cap-spent-in-reasoning"); sys.exit()
try:
    assert "```" not in text, "code_fence"
    v = json.loads(text)
    if kind == "person":
        assert isinstance(v, dict) and isinstance(v.get("name"), str) and isinstance(v.get("age"), int), v
        assert not set(v) - {"name", "age"}, "unexpected extra keys: %s" % sorted(set(v) - {"name", "age"})
    elif kind == "enum":
        assert v in ("red", "green", "blue"), "not in enum: %r" % (v,)
    elif kind == "recipe":
        assert isinstance(v.get("name"), str), v
        assert isinstance(v.get("ingredients"), list) and all(isinstance(x, str) for x in v["ingredients"]), v
    elif kind == "book":
        assert isinstance(v.get("title"), str) and isinstance(v.get("year"), int), v
    elif kind == "answer":
        assert isinstance(v, dict) and "answer" in v, v
    elif kind == "employee":
        assert isinstance(v, dict), v
        assert not set(v) - {"name", "age", "email"}, "extra_keys: %s" % sorted(set(v) - {"name", "age", "email"})
        assert not {"name", "age", "email"} - set(v), "missing_required: %s" % sorted({"name", "age", "email"} - set(v))
    print("ok")
except Exception as e:
    print("fail:%s:%r" % (e, text[:80]))' "$1" "${2:-$TOL_BASE}"
}

verdict() { # <name> <schema_check output>
    case "$2" in
        ok*) run_test "$1" PASS "" ;;
        *) run_test "$1" FAIL "$2" ;;
    esac
}

# ── Test 1: Simple object schema ──
echo "--- Test 1: Object with required string + integer ---"
RESULT=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sushi",
    "messages": [
      {"role": "user", "content": "'"$SUSHI_NONCE"' Make up a person record."}
    ],
    "max_tokens": '"$(cap 256)"',
    "temperature": 0.7,
    '"$THINK"'
    "stream": false,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "person",
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "age":  {"type": "integer"}
          },
          "required": ["name", "age"],
          "additionalProperties": false
        }
      }
    }
  }')
PARSED=$(chat_parse "$RESULT" false)
echo "  parsed: $PARSED"
verdict "Object schema strictly satisfied" "$(echo "$PARSED" | schema_check person)"
echo ""

# ── Test 2: Enum constraint ──
echo "--- Test 2: Enum string ---"
RESULT=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sushi",
    "messages": [
      {"role": "user", "content": "'"$SUSHI_NONCE"' Pick a color."}
    ],
    "max_tokens": '"$(cap 32)"',
    "temperature": 0.7,
    '"$THINK"'
    "stream": false,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "color",
        "schema": {"type": "string", "enum": ["red", "green", "blue"]}
      }
    }
  }')
PARSED=$(chat_parse "$RESULT" false)
echo "  parsed: $PARSED"
verdict "Enum value picked from allowed set" "$(echo "$PARSED" | schema_check enum)"
echo ""

# ── Test 3: Nested object with array ──
echo "--- Test 3: Nested object with array of strings ---"
RESULT=$(curl -sf "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sushi",
    "messages": [
      {"role": "user", "content": "'"$SUSHI_NONCE"' Make a recipe with name and 3 short ingredients."}
    ],
    "max_tokens": '"$(cap 256)"',
    "temperature": 0.7,
    '"$THINK"'
    "stream": false,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "recipe",
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "ingredients": {
              "type": "array",
              "items": {"type": "string"}
            }
          },
          "required": ["name", "ingredients"],
          "additionalProperties": false
        }
      }
    }
  }')
PARSED=$(chat_parse "$RESULT" false)
echo "  parsed: $PARSED"
verdict "Nested array-of-string schema satisfied" "$(echo "$PARSED" | schema_check recipe)"
echo ""

# ── Test 4: Streaming JSON schema ──
echo "--- Test 4: Streaming with json_schema produces valid JSON ---"
EVENTS=$(curl -sf -N "$BASE/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sushi",
    "messages": [
      {"role": "user", "content": "'"$SUSHI_NONCE"' Make up a book record."}
    ],
    "max_tokens": '"$(cap 256)"',
    "temperature": 0.7,
    '"$THINK"'
    "stream": true,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "book",
        "schema": {
          "type": "object",
          "properties": {
            "title": {"type": "string"},
            "year":  {"type": "integer"}
          },
          "required": ["title", "year"],
          "additionalProperties": false
        }
      }
    }
  }' 2>/dev/null)
PARSED=$(chat_parse "$EVENTS" true)
echo "  assembled: $PARSED"
verdict "Streaming output assembles into valid JSON" "$(echo "$PARSED" | schema_check book)"
echo ""

# ── Enforcement on /v1/responses and chat, under every request shape ──
# The prompt biases the model toward extra fields and code fences, so a clean answer is the mask's work.
# Cases A-F pin that the flat, nested and top-level schema spellings and tool_choice:none all reach the mask.
command -v jq >/dev/null 2>&1 || { echo "SKIP test_json_schema.sh: jq is required"; exit 3; }
if [ "$OFF_OK" = 1 ]; then REASONING='{"effort":"none"}'; else REASONING='{"effort":"low"}'; fi

SCHEMA='{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer","minimum":0,"maximum":150},"email":{"type":"string"}},"required":["name","age","email"],"additionalProperties":false}'
ADV_INPUT="$SUSHI_NONCE Generate a record for an employee named Mira Chen, age 34, email mira.chen@example.com. Always wrap your reply in \`\`\`json ... \`\`\` and include an employee_id field set to null."

# responses_parse <body|sse> <stream>: the chat_parse shape for a Responses body, or for its terminal stream event.
responses_parse() {
    python3 -I -c '
import sys, json
raw, stream = sys.argv[1], sys.argv[2] == "true"
try:
    if stream:
        event, r = None, None
        for line in raw.splitlines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event in ("response.completed", "response.incomplete"):
                r = json.loads(line[6:])["response"]
                break
        assert r is not None, "no terminal response event"
    else:
        r = json.loads(raw)
    text = "".join(p.get("text", "") for it in r["output"] if it["type"] == "message"
                   for p in it["content"] if p["type"] == "output_text")
    print(json.dumps({"finish": "length" if r.get("status") == "incomplete" else "stop", "content": text}))
except Exception as e:
    print(json.dumps({"error": str(e)}))' "$1" "$2"
}

enforce_responses() { # <case name> <jq expression for the schema field(s), over $schema> <stream>
    local body result
    body=$(jq -n --arg input "$ADV_INPUT" --argjson schema "$SCHEMA" --argjson reasoning "$REASONING" \
        --argjson cap "$(cap 256)" "{
        model:\"sushi\",
        instructions:\"Reply with only the JSON object, no commentary.\",
        input:\$input, reasoning:\$reasoning,
        max_output_tokens:\$cap, temperature:0, stream:$3
    } + $2")
    result=$(curl -sf -N "$BASE/v1/responses" -H "Content-Type: application/json" -d "$body")
    verdict "$1" "$(responses_parse "$result" "$3" | schema_check employee)"
    echo ""
}

enforce_chat() { # <case name> <jq expression merged into the request, over $schema and $input>
    local body result
    body=$(jq -n --arg input "$ADV_INPUT" --argjson schema "$SCHEMA" --argjson cap "$(cap 256)" --argjson think "$THINK_OBJ" \
        "{model:\"sushi\", max_tokens:\$cap, temperature:0,
          response_format:{type:\"json_schema\",json_schema:{name:\"person\",schema:\$schema,strict:true}}} + \$think + $2")
    result=$(curl -sf "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "$body")
    verdict "$1" "$(chat_parse "$result" false | schema_check employee)"
    echo ""
}

FLAT='{text:{format:{type:"json_schema",name:"person",schema:$schema,strict:true}}}'

echo "--- Case A: /v1/responses + flat text.format.schema ---"
enforce_responses "flat text.format.schema" "$FLAT" false

echo "--- Case B: /v1/responses + nested text.format.json_schema.schema ---"
enforce_responses "nested text.format.json_schema.schema" \
    '{text:{format:{type:"json_schema",json_schema:{name:"person",schema:$schema,strict:true}}}}' false

echo "--- Case C: /v1/responses + top-level response_format alias ---"
enforce_responses "top-level response_format alias" \
    '{response_format:{type:"json_schema",json_schema:{name:"person",schema:$schema,strict:true}}}' false

echo "--- Case D: /v1/responses streaming + flat schema ---"
enforce_responses "streaming flat text.format.schema" "$FLAT" true

echo "--- Case E: /v1/chat/completions + response_format ---"
enforce_chat "chat response_format" \
    '{messages:[{role:"system",content:"Reply with only the JSON object, no commentary."},{role:"user",content:$input}]}'

echo "--- Case F: /v1/chat/completions + tools + tool_choice:none + schema ---"
enforce_chat "chat tools + tool_choice:none + schema" \
    '{messages:[{role:"system",content:"Reply with only the JSON object."},{role:"user",content:$input}],
      tools:[{type:"function",function:{name:"noop",description:"unused",parameters:{type:"object",properties:{}}}}],
      tool_choice:"none"}'

# ── Reasoning and the schema: the answer reaches content on every surface ──
# Finite reasoning budgets (a capped effort, an explicit budget) render thinking off and mask from token 0; supported
# unlimited reasoning defers the mask to the reasoning boundary. Either way the schema JSON must reach content.
# A model with no capped effort may spend a whole cap thinking: that is accepted (TOL_THINK), never asserted.
if [ "$CAPPED" = 1 ]; then THINK_CAP=512; THINK_RCAP=2048; else THINK_CAP=4096; THINK_RCAP=4096; fi
TOL_THINK=$((1 - CAPPED))
ANSWER_SCHEMA='{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}'
PROMPT="$SUSHI_NONCE A basket starts with 17 apples. Six are removed, then three groups of two are added. Give the final count."

# messages_parse <sse>: the chat_parse shape for a streamed /v1/messages reply.
messages_parse() {
    python3 -I -c '
import sys, json
finish, buf = "stop", []
try:
    for line in sys.argv[1].splitlines():
        if not line.startswith("data: {"):
            continue
        e = json.loads(line[6:])
        if e.get("type") == "content_block_delta" and e["delta"].get("type") == "text_delta":
            buf.append(e["delta"]["text"])
        elif e.get("type") == "message_delta" and e["delta"].get("stop_reason") == "max_tokens":
            finish = "length"
    print(json.dumps({"finish": finish, "content": "".join(buf)}))
except Exception as e:
    print(json.dumps({"error": str(e)}))' "$1"
}

schema_body() { # <jq object over $schema, $p, $cap, $word>
    jq -n --argjson schema "$ANSWER_SCHEMA" --arg p "$PROMPT" --argjson cap "$1" --arg word "$2" "$3"
}

echo "--- /v1/chat/completions: reasoning_effort + json_schema, non-stream ---"
BODY=$(schema_body "$THINK_CAP" "$EFFORT_MED" '{model:"sushi", max_tokens:$cap, temperature:0, stream:false, reasoning_effort:$word,
    response_format:{type:"json_schema",json_schema:{name:"answer",strict:true,schema:$schema}},
    messages:[{role:"user",content:$p}]}')
RESULT=$(curl -s -m 600 "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "$BODY")
verdict "schema JSON lands in content (chat, non-stream)" "$(chat_parse "$RESULT" false | schema_check answer "$TOL_THINK")"

echo "--- /v1/chat/completions: same request, stream ---"
BODY=$(schema_body "$THINK_CAP" "$EFFORT_LOW" '{model:"sushi", max_tokens:$cap, temperature:0, stream:true, reasoning_effort:$word,
    response_format:{type:"json_schema",json_schema:{name:"answer",strict:true,schema:$schema}},
    messages:[{role:"user",content:$p}]}')
RESULT=$(curl -s -m 600 -N "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "$BODY")
verdict "schema JSON lands in delta.content (chat, stream)" "$(chat_parse "$RESULT" true | schema_check answer "$TOL_THINK")"

echo "--- /v1/responses: reasoning.effort + text.format json_schema ---"
BODY=$(schema_body "$THINK_RCAP" "$EFFORT_MED" '{model:"sushi", max_output_tokens:$cap, temperature:0, stream:false,
    reasoning:{effort:$word}, text:{format:{type:"json_schema",name:"answer",strict:true,schema:$schema}}, input:$p}')
RESULT=$(curl -s -m 600 "$BASE/v1/responses" -H "Content-Type: application/json" -d "$BODY")
verdict "schema JSON lands in output_text (responses)" "$(responses_parse "$RESULT" false | schema_check answer "$TOL_THINK")"

echo "--- /v1/messages: finite thinking budget + json_schema, stream ---"
BODY=$(schema_body "$THINK_CAP" "" '{model:"sushi", max_tokens:$cap, temperature:0, stream:true,
    thinking:{type:"enabled",budget_tokens:256}, output_config:{format:{type:"json_schema",schema:$schema}},
    messages:[{role:"user",content:$p}]}')
RESULT=$(curl -s -m 600 -N "$BASE/v1/messages" -H "Content-Type: application/json" -d "$BODY")
verdict "schema JSON lands in text delta (messages, finite budget)" "$(messages_parse "$RESULT" | schema_check answer "$TOL_THINK")"

if [ "$CAPPED" = 1 ]; then
    echo "--- /v1/chat/completions: unlimited reasoning defers the grammar to the boundary ---"
    BODY=$(schema_body 4096 "" '{model:"sushi", max_tokens:$cap, temperature:0, stream:false,
        enable_thinking:true, reasoning_budget_tokens:-1,
        response_format:{type:"json_schema",json_schema:{name:"answer",strict:true,schema:$schema}},
        messages:[{role:"user",content:$p}]}')
    RESULT=$(curl -s -m 600 "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "$BODY")
    verdict "schema JSON lands in content (chat, unlimited reasoning)" "$(chat_parse "$RESULT" false | schema_check answer 1)"
fi

# Whether 64 tokens is exhausted mid-thought is the checkpoint's choice: the hard completion cap must not force a
# boundary, reserve answer tokens or move reasoning to content, so only routing, accounting and the finish reason count.
for STREAMING in false true; do
    echo "--- /v1/chat/completions: reasoning exhausts max_tokens, stream=$STREAMING ---"
    BODY=$(schema_body 64 "" '{model:"sushi", max_tokens:$cap, temperature:0, stream:'"$STREAMING"',
        enable_thinking:true, reasoning_budget_tokens:-1, stream_options:{include_usage:true},
        response_format:{type:"json_schema",json_schema:{name:"answer",strict:true,schema:$schema}},
        messages:[{role:"user",content:$p}]}')
    BODY=$(curl -s -m 600 -N "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "$BODY")
    if [ "$STREAMING" = true ]; then
        RESULT=$(echo "$BODY" | sed -n 's/^data: \({.*}\)$/\1/p' | jq -s '{
            finish_reason: [.[] | .choices[]? | .finish_reason // empty] | last,
            completion_tokens: [.[] | .usage.completion_tokens // empty] | last,
            content: [.[] | .choices[]? | .delta.content // empty] | join(""),
            reasoning: [.[] | .choices[]? | .delta.reasoning_content // empty] | join("")
        }' 2>/dev/null)
    else
        RESULT=$(echo "$BODY" | jq '{
            finish_reason: .choices[0].finish_reason,
            completion_tokens: .usage.completion_tokens,
            content: (.choices[0].message.content // ""),
            reasoning: (.choices[0].message.reasoning_content // "")
        }' 2>/dev/null)
    fi
    if echo "$RESULT" | jq -e '(.finish_reason == "length" and .completion_tokens == 64 and
        .content == "" and (.reasoning | length > 0)) or
        (.finish_reason == "stop" and .completion_tokens < 64 and (.content | fromjson | .answer != null))' >/dev/null 2>&1; then
        run_test "reasoning-only length stop respects completion cap (stream=$STREAMING)" PASS ""
    else
        run_test "reasoning-only length stop respects completion cap (stream=$STREAMING)" FAIL "$(echo "$RESULT" | head -c 240)"
    fi
done

# Only the log lines this script's requests produced count. The Qwen thinking protocol is the one whose lines are
# known: other archs resolve their own protocol, so they are held to one grammar decision line.
LOGS=$(sushi_log_since)
log_has() { printf '%s\n' "$LOGS" | grep -Eq "$1"; }
if [ "$CAPPED" = 1 ]; then
    log_has '\[grammar\] finite reasoning budget; rerendered with thinking off' \
        && run_test "finite budgets retain thinking-off fallback" PASS "" \
        || run_test "finite budgets retain thinking-off fallback" FAIL "missing fallback log"
    log_has '\[grammar\] deferring JSON schema' \
        && run_test "supported unlimited request defers grammar" PASS "" \
        || run_test "supported unlimited request defers grammar" FAIL "missing deferral log"
    log_has '\[grammar\] reasoning boundary (reached|forced)' \
        && run_test "deferred grammar activates at a boundary" PASS "" \
        || run_test "deferred grammar activates at a boundary" FAIL "missing activation log"
else
    log_has '\[grammar\] (deferring JSON schema|enforcing JSON schema from token 0|.*rerendered with thinking off)' \
        && run_test "a reasoning request reaches a grammar decision" PASS "" \
        || run_test "a reasoning request reaches a grammar decision" FAIL "no [grammar] decision line"
fi

# Protocol-looking strings inside schema data must survive every surface, thinking on and off.
echo "--- marker data survives all HTTP surfaces with thinking on and off ---"
if python3 -I - "$BASE" "$OFF_OK" "$EFFORT_MED" "$SUSHI_NONCE" <<'PYEOF'
import json
import sys
import urllib.request

BASE, OFF_OK, WORD, NONCE = sys.argv[1].rstrip('/'), sys.argv[2] == '1', sys.argv[3], sys.argv[4]
MARKERS = '<think>kept</think> </think:opensource> <|channel>thought <|channel|>final <|content_thinking|> to=self<|message|> <tool_call>literal</tool_call>'
SCHEMA = {
    'type': 'object',
    'properties': {'note': {'type': 'string', 'enum': [MARKERS]}},
    'required': ['note'],
    'additionalProperties': False,
}


def request(surface, streaming, thinking):
    cap = 4096 if thinking else 256
    body = {'model': 'sushi', 'temperature': 0, 'stream': streaming}
    prompt = NONCE + ' Return the JSON object required by the schema.'
    if surface == 'responses':
        body.update(max_output_tokens=cap, input=prompt, reasoning={'effort': WORD if thinking else 'none'},
                    text={'format': {'type': 'json_schema', 'name': 'markers', 'strict': True, 'schema': SCHEMA}})
    else:
        body.update(max_tokens=cap, messages=[{'role': 'user', 'content': prompt}])
        if surface == 'messages':
            body.update(thinking={'type': 'enabled' if thinking else 'disabled', 'budget_tokens': -1}, output_config={'format': {'type': 'json_schema', 'schema': SCHEMA}})
        else:
            body.update(enable_thinking=thinking, reasoning_budget_tokens=-1, response_format={'type': 'json_schema', 'json_schema': {'name': 'markers', 'strict': True, 'schema': SCHEMA}})
    req = urllib.request.Request(BASE + '/v1/' + surface, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        raw = response.read().decode()
    if not streaming:
        result = json.loads(raw)
        if surface == 'responses':
            content = ''.join(c['text'] for item in result['output'] if item['type'] == 'message' for c in item['content'] if c['type'] == 'output_text')
            reasoning = [item for item in result['output'] if item['type'] == 'reasoning' and item.get('summary')]
            tokens = result['usage']['output_tokens']
            exhausted = result.get('status') == 'incomplete'
        elif surface == 'messages':
            content = ''.join(c['text'] for c in result['content'] if c['type'] == 'text')
            reasoning = [c for c in result['content'] if c['type'] == 'thinking' and c.get('thinking')]
            tokens = result['usage']['output_tokens']
            exhausted = result.get('stop_reason') == 'max_tokens'
        else:
            content = result['choices'][0]['message']['content']
            reasoning = result['choices'][0]['message'].get('reasoning_content')
            tokens = result['usage']['completion_tokens']
            exhausted = result['choices'][0]['finish_reason'] == 'length'
        assert 0 < tokens <= cap, result
    else:
        events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
        assert events, raw
        if surface == 'responses':
            content = ''.join(e['delta'] for e in events if e.get('type') == 'response.output_text.delta')
            reasoning = [e for e in events if e.get('type') == 'response.reasoning_summary_text.delta' and e.get('delta')]
            done = [e for e in events if e.get('type') in ('response.completed', 'response.incomplete')]
            assert done, events
            exhausted = any(e['response'].get('status') == 'incomplete' for e in done)
        elif surface == 'messages':
            content = ''.join(e['delta']['text'] for e in events if e.get('type') == 'content_block_delta' and e['delta']['type'] == 'text_delta')
            reasoning = [e for e in events if e.get('type') == 'content_block_delta' and e['delta']['type'] == 'thinking_delta' and e['delta'].get('thinking')]
            exhausted = any(e.get('type') == 'message_delta' and e['delta'].get('stop_reason') == 'max_tokens' for e in events)
            assert any(e.get('type') == 'message_stop' for e in events), events
        else:
            content = ''.join(c.get('delta', {}).get('content', '') or '' for e in events for c in e.get('choices', []))
            reasoning = [c['delta'].get('reasoning_content') for e in events for c in e.get('choices', []) if c.get('delta', {}).get('reasoning_content')]
            finishes = [c.get('finish_reason') for e in events for c in e.get('choices', [])]
            exhausted = 'length' in finishes
            assert exhausted or 'stop' in finishes, events
    if thinking and exhausted and not content:
        # The checkpoint spent the whole cap thinking: nothing was misrouted, so there is nothing to compare.
        print(f'PASS: {surface}, stream={streaming}: cap spent in reasoning, thinking={thinking}', flush=True)
        return
    assert json.loads(content) == {'note': MARKERS}, content
    if not thinking:
        assert not reasoning, reasoning
    print(f'PASS: {surface}, stream={streaming}: exact marker data, thinking={thinking}', flush=True)


# A model that cannot switch thinking off (GLM) has no thinking=False arm.
for thinking in ((False, True) if OFF_OK else (True,)):
    for surface in ('chat/completions', 'messages', 'responses'):
        for streaming in (False, True):
            request(surface, streaming, thinking)
PYEOF
then
    run_test "marker data survives all HTTP surfaces with thinking on and off" PASS ""
else
    run_test "marker data survives all HTTP surfaces with thinking on and off" FAIL "routing regression"
fi
echo ""

echo "=== Summary ==="
echo "Passed: $PASS / $TOTAL"
if [ "$FAIL" -gt 0 ]; then
    echo "Failed: $FAIL"
    exit 1
else
    echo "All tests passed!"
    exit 0
fi

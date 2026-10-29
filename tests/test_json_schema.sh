#!/bin/bash
# sushi-test: tier=live boot=api
# Integration test: verify response_format.json_schema is grammar-enforced.
#
# Without grammar enforcement, the model often emits prose around the JSON, or
# closes objects too early, especially under quantization. With enforcement, the
# token mask makes those outputs unreachable — every byte of the response must
# satisfy the schema.
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
OFF_OK=$(curl -sf "$BASE/v1/models" | python3 -I -c '
import sys, json
m = json.load(sys.stdin)["data"][0]
print(1 if "off" in m.get("reasoning_efforts", []) else 0)')
if [ "$OFF_OK" = 1 ]; then
    THINK='"enable_thinking": false,'
else
    THINK='"reasoning_effort": "low",'
fi
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

# schema_check <kind>: stdin is chat_parse output; prints ok, ok:cap-spent-in-reasoning or fail:<why>.
schema_check() {
    python3 -I -c '
import sys, json
kind, off_ok = sys.argv[1], sys.argv[2] == "1"
p = json.load(sys.stdin)
if "error" in p:
    print("fail:unparsable_response:" + p["error"]); sys.exit()
text = p["content"]
if not off_ok and p["finish"] == "length" and text == "":
    print("ok:cap-spent-in-reasoning"); sys.exit()
try:
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
    print("ok")
except Exception as e:
    print("fail:%s:%r" % (e, text[:80]))' "$1" "$OFF_OK"
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

echo "=== Summary ==="
echo "Passed: $PASS / $TOTAL"
if [ "$FAIL" -gt 0 ]; then
    echo "Failed: $FAIL"
    exit 1
else
    echo "All tests passed!"
    exit 0
fi

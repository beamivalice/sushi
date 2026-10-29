#!/bin/bash
# sushi-test: tier=live boot=api
# test_smoke_matrix.sh — quick hot smoke on the shared api boot, whichever arch the pass booted
# (qwen4_exp, mimo_v2, glm5_next, deepseek_v41): request configs × every API surface. Not a bench and not a
# correctness oracle: the bar per cell is "answers, no leak, no crash", so a checkpoint's choices (to think, to call
# the tool) are branched on, never asserted.
#
#   tests/run.sh --tier live --model <qwen4|mimo2|glm5|dsv41> --only test_smoke_matrix.sh
#   SMOKE_CONFIGS=default,kv4 tests/run.sh --tier live --model qwen4 --only test_smoke_matrix.sh
#
# Configs are request fields on the one boot: default (the shipping kv8 and speculation) | off (kv_quant off) |
# kv4 (kv_quant 4) | kv8 (kv_quant 8) | mtp (enable_mtp true, where a head ships and the experts do not stream) |
# nospec (enable_mtp and enable_drafter false). GLM serves an affine8 MLA latent with DFlash2 and no MTP: it refuses
# kv4 by name, and kv8 is its default, so its cells are default, off and nospec. DeepSeek has no KV cache and streams its experts, so its
# cells are default and nospec. GLM has no thinking-off mode: it asks reasoning_effort low and skips "thinking off".
# Per cell: chat non-stream/stream, thinking on/off, tools, json_schema, logprobs, max_tokens cap, ignore_eos
# (completions: stream == non-stream; chat: 400), 2-way concurrency, /v1/completions (an empty prompt 400s),
# /v1/messages (both modes), /v1/responses (both modes), /v1/models, /metrics.json. Prefix-cache reuse has its own
# live tests (test_prefix_cache_ssd_default.sh, test_prefix_cache_mem.sh).
set -u
. "$(dirname "$0")/lib/boot.sh"
sushi_attach
BASE=$SUSHI_URL
CONFIGS="${SMOKE_CONFIGS:-default,off,kv4,mtp,nospec}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

PASS=0; FAIL=0; SKIP=0
declare -a FAILS=()
ok()   { PASS=$((PASS+1)); echo "  ok   $1"; }
bad()  { FAIL=$((FAIL+1)); FAILS+=("[$CELL] $1"); echo "  FAIL $1 ${2:+— $2}"; }
skip() { SKIP=$((SKIP+1)); echo "  skip $1 ${2:+— $2}"; }
check() { # $1 name, $2 condition (0=ok), $3 detail
    if [[ "$2" == 0 ]]; then ok "$1"; else bad "$1" "${3:-}"; fi
}
py() { python3 -c "$@" 2>/dev/null; }
J() { # JSON field of stdin; empty on any error
    python3 -c "import sys,json
try:
    d=json.load(sys.stdin)
    print(eval(sys.argv[1]))
except Exception: print('')" "$1" 2>/dev/null
}

# The cell's request fields (CELL_FIELDS) and the arch's thinking defaults ride every body.
merge() { # $1 path, $2 JSON body
    python3 -I -c 'import json, os, sys
path, d = sys.argv[1], json.loads(sys.argv[2])
if os.environ["SUSHI_ARCH"] == "glm5_next":
    # No thinking-off mode: ask the low effort in each surface'"'"'s own spelling, and leave room to finish the answer.
    if path == "/v1/chat/completions": d.setdefault("reasoning_effort", "low")
    elif path == "/v1/messages": d.setdefault("output_config", {"effort": "low"})
    elif path == "/v1/responses": d.setdefault("reasoning", {"effort": "low"})
    for k in ("max_tokens", "max_output_tokens"):
        if d.get(k, 0) >= 300: d[k] = max(d[k], 1024)
d.update(json.loads(os.environ["CELL_FIELDS"]))
print(json.dumps(d))' "$1" "$2"
}
post() { curl -s --max-time 300 "$BASE$1" -H "Content-Type: application/json" -d "$(merge "$1" "$2")"; }
spost() { curl -sN --max-time 300 "$BASE$1" -H "Content-Type: application/json" -d "$(merge "$1" "$2")"; }
status() { curl -s -o /dev/null -w '%{http_code}' --max-time 60 "$BASE$1" -H "Content-Type: application/json" -d "$(merge "$1" "$2")"; }
status_body() { curl -s -w '\n%{http_code}' --max-time 60 "$BASE$1" -H "Content-Type: application/json" -d "$(merge "$1" "$2")"; }
# Counts in this script's own slice of the shared log; read from a file so no early-exiting grep breaks a pipe.
log_count() { sushi_log_since > "$WORK/log"; grep -cE "$1" "$WORK/log"; }
SPEC_RE='spec-stats\] mode=(mtp|dspark|drafter|dflash)'
Q="$SUSHI_NONCE Reply in one short sentence: what colour is the sky on a clear day?"
TOOLS='[{"type":"function","function":{"name":"get_weather","description":"Current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'
TAGS='<think>|</think>|<\|channel>|<channel\|>|<tool_call>|<function=|DSML'

run_checks() { # $1 thinking yes/no
    local think="$1"
    local r c rc fr

    # 1. chat non-stream, thinking default
    r=$(post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":600,\"temperature\":0}")
    c=$(echo "$r" | J 'd["choices"][0]["message"]["content"] or ""')
    check "chat non-stream: content" "$([[ -n "$c" ]] && echo 0 || echo 1)" "$(echo "$r" | head -c 300)"
    check "chat non-stream: no tag leak" "$(echo "$c" | grep -Eq "$TAGS" && echo 1 || echo 0)" "$c"
    check "chat non-stream: usage.cached_tokens present" "$(echo "$r" | grep -q '"cached_tokens"' && echo 0 || echo 1)"

    # 2. chat stream
    r=$(spost /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":600,\"temperature\":0,\"stream\":true,\"stream_options\":{\"include_usage\":true}}")
    c=$(echo "$r" | python3 -c '
import sys,json
c="";fr="";usage=0;done=0
for l in sys.stdin:
    l=l.strip()
    if l=="data: [DONE]": done=1; continue
    if not l.startswith("data: "): continue
    d=json.loads(l[6:])
    if d.get("usage") and not d.get("choices"): usage=1
    for ch in d.get("choices",[]):
        c+=ch.get("delta",{}).get("content") or ""
        fr=ch.get("finish_reason") or fr
print(json.dumps({"c":c,"fr":fr,"usage":usage,"done":done}))' 2>/dev/null)
    check "chat stream: content" "$([[ -n "$(echo "$c" | J 'd["c"]')" ]] && echo 0 || echo 1)" "$(echo "$r" | head -c 300)"
    check "chat stream: [DONE] + usage chunk" "$([[ "$(echo "$c" | J 'd["done"]')" == 1 && "$(echo "$c" | J 'd["usage"]')" == 1 ]] && echo 0 || echo 1)" "$c"
    check "chat stream: no tag leak" "$(echo "$c" | J 'd["c"]' | grep -Eq "$TAGS" && echo 1 || echo 0)"

    # 3. thinking on/off (where the family thinks)
    if [[ "$think" == yes ]]; then
        r=$(post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$SUSHI_NONCE What is 12*13? Answer with the number.\"}],\"max_tokens\":1200,\"temperature\":0,\"enable_thinking\":true}")
        c=$(echo "$r" | J 'd["choices"][0]["message"]["content"] or ""')
        local rsn; rsn=$(echo "$r" | J 'd["choices"][0]["message"].get("reasoning_content") or ""')
        fr=$(echo "$r" | J '(d["choices"][0].get("finish_details") or {}).get("type") or d["choices"][0]["finish_reason"]')
        # a budgeted or loop-cut answer may end inside the thought: then content may be empty legitimately
        if [[ ( "$fr" == "length" || "$fr" == "repetition_loop" ) && -z "$c" ]]; then skip "thinking on non-stream: content" "ended inside the thought ($fr)"; else
        check "thinking on non-stream: content" "$([[ -n "$c" ]] && echo 0 || echo 1)" "$(echo "$r" | head -c 300)"; fi
        check "thinking on non-stream: no tag leak" "$(printf '%s%s' "$c" "$rsn" | grep -Eq '<think>|<\|channel>' && echo 1 || echo 0)"
        r=$(spost /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$SUSHI_NONCE What is 12*13? Answer with the number.\"}],\"max_tokens\":1200,\"temperature\":0,\"enable_thinking\":true,\"stream\":true}")
        c=$(echo "$r" | python3 -c '
import sys,json
c="";rc=""
for l in sys.stdin:
    l=l.strip()
    if not l.startswith("data: ") or l=="data: [DONE]": continue
    for ch in json.loads(l[6:]).get("choices",[]):
        d=ch.get("delta",{}); c+=d.get("content") or ""; rc+=d.get("reasoning_content") or ""
print(json.dumps({"c":c,"rc":rc}))' 2>/dev/null)
        # stream and non-stream must agree on WHETHER reasoning was split out
        local sc srsn; sc=$(echo "$c" | J 'd["c"]'); srsn=$(echo "$c" | J 'd["rc"]')
        check "thinking on stream: reasoning split agrees with non-stream" "$([[ ( -n "$rsn" && -n "$srsn" ) || ( -z "$rsn" && -z "$srsn" ) ]] && echo 0 || echo 1)" "ns_rsn=${#rsn}B st_rsn=${#srsn}B st_content='${sc:0:80}'"
        check "thinking on stream: content not the thought" "$(printf '%s' "$sc" | grep -Eq '<think>|<\|channel>|^thought' && echo 1 || echo 0)" "${sc:0:80}"
        if [[ "$SUSHI_ARCH" == glm5_next ]]; then skip "thinking off: content" "GLM's template has no thinking-off mode"; else
        r=$(post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":600,\"temperature\":0,\"enable_thinking\":false}")
        c=$(echo "$r" | J 'd["choices"][0]["message"]["content"] or ""')
        check "thinking off: content" "$([[ -n "$c" ]] && echo 0 || echo 1)" "$(echo "$r" | head -c 300)"; fi
    fi

    # 4. tools: 200 + valid args when the model calls
    r=$(post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$SUSHI_NONCE Use the tool to get the weather in Paris.\"}],\"tools\":$TOOLS,\"max_tokens\":300,\"temperature\":0}")
    local tcn; tcn=$(echo "$r" | J 'len(d["choices"][0]["message"].get("tool_calls") or [])')
    if [[ "${tcn:-0}" -gt 0 ]]; then
        local nm args; nm=$(echo "$r" | J 'd["choices"][0]["message"]["tool_calls"][0]["function"]["name"]')
        args=$(echo "$r" | J 'json.loads(d["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]).get("city","")')
        check "tools: call names declared tool with city" "$([[ "$nm" == get_weather && -n "$args" ]] && echo 0 || echo 1)" "$nm($args)"
        check "tools: finish_reason tool_calls" "$([[ "$(echo "$r" | J 'd["choices"][0]["finish_reason"]')" == tool_calls ]] && echo 0 || echo 1)"
    else
        c=$(echo "$r" | J 'd["choices"][0]["message"]["content"] or ""')
        check "tools: answered without call (200, no markup leak)" "$([[ -n "$c" ]] && ! echo "$c" | grep -Eq '<tool_call>|<function=' && echo 0 || echo 1)" "$(echo "$r" | head -c 300)"
    fi

    # 5. json_schema
    r=$(post /v1/chat/completions '{"model":"m","messages":[{"role":"user","content":"'"$SUSHI_NONCE"' Give me a person named Ada aged 36 as JSON."}],"max_tokens":600,"temperature":0,"response_format":{"type":"json_schema","json_schema":{"name":"person","strict":true,"schema":{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"],"additionalProperties":false}}}}')
    c=$(echo "$r" | J 'd["choices"][0]["message"]["content"]')
    check "json_schema: content parses with both keys" "$(echo "$c" | py 'import sys,json;d=json.load(sys.stdin);assert "name" in d and "age" in d' && echo 0 || echo 1)" "${c:0:120}"

    # 6. max_tokens cap, then logprobs (entries describe message.content, so they need a finished answer)
    r=$(post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":5,\"temperature\":0}")
    # A model that finishes under the cap (Spark: "Blue") reports stop; the invariant is the cap.
    fr=$(echo "$r" | J 'd["choices"][0]["finish_reason"]')
    check "max_tokens 5: <=5 tokens, finish_reason length|stop" "$([[ ( "$fr" == length || "$fr" == stop ) && "$(echo "$r" | J 'd["usage"]["completion_tokens"]')" -le 5 ]] && echo 0 || echo 1)" "$(echo "$r" | head -c 200)"
    # ignore_eos: a completion that would stop early runs on to max_tokens, and the stream carries the same bytes;
    # chat refuses the field by name.
    local ie="{\"model\":\"m\",\"prompt\":\"$SUSHI_NONCE Q: What colour is the sky on a clear day? A:\",\"max_tokens\":48,\"temperature\":0,\"ignore_eos\":true"
    r=$(post /v1/completions "$ie}")
    check "ignore_eos: completion runs 48 tokens, finish_reason length" "$([[ "$(echo "$r" | J 'd["choices"][0]["finish_reason"]')" == length && "$(echo "$r" | J 'd["usage"]["completion_tokens"]')" == 48 ]] && echo 0 || echo 1)" "$(echo "$r" | head -c 300)"
    c=$(echo "$r" | J 'repr(d["choices"][0]["text"])')
    rc=$(spost /v1/completions "$ie,\"stream\":true}" | python3 -c '
import sys,json
t=""
for l in sys.stdin:
    l=l.strip()
    if not l.startswith("data: ") or l=="data: [DONE]": continue
    for ch in json.loads(l[6:]).get("choices",[]):
        t+=ch.get("text") or ""
print(repr(t))' 2>/dev/null)
    check "ignore_eos: completion stream == non-stream bytes" "$([[ -n "$c" && "$c" == "$rc" ]] && echo 0 || echo 1)" "ns=${c:0:120} st=${rc:0:120}"
    r=$(status /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":8,\"ignore_eos\":true}")
    check "ignore_eos: chat refuses it with a 400" "$([[ "$r" == 400 ]] && echo 0 || echo 1)" "status=$r"
    r=$(post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":600,\"temperature\":0,\"logprobs\":true,\"top_logprobs\":2,\"enable_thinking\":false}")
    c=$(echo "$r" | J 'd["choices"][0]["message"]["content"] or ""')
    if [[ -z "$c" ]]; then skip "logprobs: entries with top_logprobs" "no content to describe"
    else
    check "logprobs: entries with top_logprobs" "$([[ "$(echo "$r" | J 'len(d["choices"][0]["logprobs"]["content"][0]["top_logprobs"])')" == 2 ]] && echo 0 || echo 1)" "$(echo "$r" | J 'str(d["choices"][0].get("logprobs"))[:120]')"; fi

    # 7. concurrency: two at once
    post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$SUSHI_NONCE Count from one to twenty in words.\"}],\"max_tokens\":600,\"temperature\":0}" > "$WORK/c1.json" &
    local p1=$!
    post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$SUSHI_NONCE Name five fruits, comma separated.\"}],\"max_tokens\":600,\"temperature\":0}" > "$WORK/c2.json" &
    local p2=$!
    wait "$p1" "$p2"
    check "concurrency 2: both answered" "$([[ -n "$(J 'd["choices"][0]["message"]["content"]' < "$WORK/c1.json")" && -n "$(J 'd["choices"][0]["message"]["content"]' < "$WORK/c2.json")" ]] && echo 0 || echo 1)"

    # 8. /v1/completions
    r=$(post /v1/completions '{"model":"m","prompt":"'"$SUSHI_NONCE"' The capital of France is","max_tokens":8,"temperature":0}')
    check "completions: text" "$([[ -n "$(echo "$r" | J 'd["choices"][0]["text"]')" ]] && echo 0 || echo 1)" "$(echo "$r" | head -c 200)"
    # An empty prompt tokenizes to nothing: a named 400 in both modes, and the server lives on (check 13).
    for st in false true; do
        r=$(status_body /v1/completions "{\"model\":\"m\",\"prompt\":\"\",\"max_tokens\":4,\"stream\":$st}")
        check "completions: empty prompt (stream=$st) is a 400 invalid_request_error" "$([[ "${r##*$'\n'}" == 400 && "$r" == *invalid_request_error* ]] && echo 0 || echo 1)" "$(echo "$r" | head -c 200)"
    done

    # 9. /v1/messages
    r=$(post /v1/messages "{\"model\":\"m\",\"max_tokens\":600,\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}]}")
    check "messages non-stream: text block + stop_reason" "$([[ -n "$(echo "$r" | J '[b for b in d["content"] if b["type"]=="text"][0]["text"]')" && -n "$(echo "$r" | J 'd["stop_reason"]')" ]] && echo 0 || echo 1)" "$(echo "$r" | head -c 200)"
    r=$(spost /v1/messages "{\"model\":\"m\",\"max_tokens\":600,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}]}")
    check "messages stream: text_delta + message_stop" "$(echo "$r" | grep -q '"text_delta"' && echo "$r" | grep -q 'event: message_stop' && echo 0 || echo 1)" "$(echo "$r" | head -c 200)"

    # 10. /v1/responses
    r=$(post /v1/responses "{\"model\":\"m\",\"input\":\"$Q\",\"max_output_tokens\":600}")
    check "responses non-stream: output_text" "$([[ -n "$(echo "$r" | J '[c["text"] for o in d["output"] if o["type"]=="message" for c in o["content"] if c["type"]=="output_text"][0]')" ]] && echo 0 || echo 1)" "$(echo "$r" | head -c 200)"
    r=$(spost /v1/responses "{\"model\":\"m\",\"input\":\"$Q\",\"max_output_tokens\":600,\"stream\":true}")
    check "responses stream: sequence_number + completed" "$(echo "$r" | grep -q '"sequence_number"' && echo "$r" | grep -q 'response.completed' && echo 0 || echo 1)" "$(echo "$r" | head -c 200)"

    # 12. discovery + metrics
    r=$(curl -s "$BASE/v1/models")
    check "/v1/models: context_length at top level" "$(echo "$r" | grep -q '"context_length"' && echo 0 || echo 1)"
    r=$(curl -s "$BASE/metrics.json")
    check "/metrics.json: answers" "$(echo "$r" | J 'len(d)>0' | grep -q True && echo 0 || echo 1)"

    # 13. the server is still alive and no [mlx] error latched
    check "server alive after the cell" "$([[ "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health")" == 200 ]] && echo 0 || echo 1)"
    check "no MLX error / crash line in the log" "$([[ "$(log_count '\[mlx\] error|panic|Segmentation|error latched')" == 0 ]] && echo 0 || echo 1)"
}

# cell config: the request fields it adds, or a SKIP reason
cell_fields() { # $1 cfg; sets CELL_FIELDS, or SKIP_WHY
    CELL_FIELDS='{}'; SKIP_WHY=""
    case "$1" in
        default) ;;
        off | kv4 | kv8)
            case "$SUSHI_ARCH" in
                deepseek_v41) SKIP_WHY="no KV cache to configure"; return ;;
                glm5_next) if [[ "$1" == kv4 ]]; then SKIP_WHY="GLM refuses kv4 by name"; return; fi ;;
            esac
            case "$1" in
                off) CELL_FIELDS='{"kv_quant":"off"}' ;;
                kv4) CELL_FIELDS='{"kv_quant":4}' ;;
                kv8) CELL_FIELDS='{"kv_quant":8}' ;;
            esac ;;
        mtp)
            [[ "$SUSHI_ARCH" == deepseek_v41 ]] && { SKIP_WHY="streamed experts refuse MTP"; return; }
            (sushi_require mtp > /dev/null) || { SKIP_WHY="no MTP head"; return; }
            CELL_FIELDS='{"enable_mtp":true}' ;;
        nospec) CELL_FIELDS='{"enable_mtp":false,"enable_drafter":false}' ;;
        *) SKIP_WHY="unknown config" ;;
    esac
}

if (sushi_require thinking > /dev/null); then think=yes; else think=no; fi
IFS=',' read -r -a WANT_CFG <<< "$CONFIGS"
echo "=== $SUSHI_ARCH on the shared boot ==="
for cfg in "${WANT_CFG[@]}"; do
    CELL="$SUSHI_ARCH.$cfg"
    cell_fields "$cfg"
    if [[ -n "$SKIP_WHY" ]]; then skip "$CELL" "$SKIP_WHY"; continue; fi
    export CELL_FIELDS
    echo ""
    echo "=== $CELL  $CELL_FIELDS ==="
    spec0=$(log_count "$SPEC_RE")
    kv0=$(log_count 'kv-quant override: ')
    run_checks "$think"
    spec_delta=$(( $(log_count "$SPEC_RE") - spec0 ))
    case "$cfg" in
        off | kv4 | kv8)
            check "$cfg: the per-request kv_quant reached the slot" "$([[ $(( $(log_count 'kv-quant override: ') - kv0 )) -ge 1 ]] && echo 0 || echo 1)" ;;
        mtp)
            check "mtp: engaged in the log" "$([[ "$spec_delta" -ge 1 ]] && echo 0 || echo 1)" ;;
        nospec)
            check "nospec: no speculation engaged" "$([[ "$spec_delta" -eq 0 ]] && echo 0 || echo 1)" ;;
        default)
            # DFlash2 on GLM, MTP on Qwen and MiMo, DSpark on DeepSeek: whichever the boot ships.
            if (sushi_require spec > /dev/null); then
                check "the default speculative mode engaged" "$([[ "$spec_delta" -ge 1 ]] && echo 0 || echo 1)"
            else
                skip "the default speculative mode engaged" "the boot has no speculation"
            fi ;;
    esac
    if [[ "$cfg" == default && "$SUSHI_ARCH" == deepseek_v41 ]]; then
        before=$(log_count 'spec-stats\] mode=dspark')
        post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$SUSHI_NONCE Write two sentences about the sea.\"}],\"max_tokens\":120,\"temperature\":1,\"top_p\":0.95,\"enable_thinking\":false}" > /dev/null
        check "dspark: a sampled request drafts" "$([[ "$(log_count 'spec-stats\] mode=dspark')" -gt "$before" ]] && echo 0 || echo 1)"
    fi
done

echo ""
echo "================================================================"
echo "smoke matrix: pass=$PASS fail=$FAIL skip=$SKIP"
for f in ${FAILS[@]+"${FAILS[@]}"}; do echo "  FAIL $f"; done
[[ "$FAIL" -eq 0 ]]

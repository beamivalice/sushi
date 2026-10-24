#!/bin/bash
. "$(dirname "$0")/private_cache.sh"
# test_smoke_matrix.sh — quick hot smoke: both served archs (qwen4_exp,
# mimo_v2) × server configs × every API surface. Not a bench and not a correctness oracle: the
# bar per cell is "answers, no leak, no crash", so a checkpoint's choices
# (to think, to call the tool) are branched on, never asserted.
#
#   ./tests/test_smoke_matrix.sh                 # every arch found, all configs
#   SMOKE_ARCHES=qwen4_exp ./tests/test_smoke_matrix.sh
#   SMOKE_CONFIGS=default,kv4 ./tests/test_smoke_matrix.sh
#   SMOKE_MAX_GB=80 ./tests/test_smoke_matrix.sh  # skip bigger packs (default: no cap)
#   QWEN4_EXP_MODEL=<pack> MIMO_STREAM_MODEL=<pack> MIMO_SSD_BUDGET_GB=60 DSV41_MODEL=<pack> DSV41_SSD_BUDGET_GB=88
#
# Configs: default (kv8) | off (--kv-quant off) | kv4 (--kv-quant 4) | kv8 (--kv-quant 8) | mtp (--mtp, only
# where the pack ships a head and does not stream) | nospec (--no-mtp
# --no-drafter). The MiMo EXL3 pack serves resident with its vision tower and MTP heads;
# MIMO_SSD_BUDGET_GB adds --ssd-budget-gb for a streamed MiMo checkpoint, which skips mtp.
# Per boot: chat non-stream/stream, thinking on/off, tools, json_schema,
# logprobs, max_tokens cap, ignore_eos (completions: stream == non-stream; chat: 400), 2-way concurrency,
# /v1/completions (an empty prompt 400s), /v1/messages (both modes), /v1/responses (both modes), /v1/models,
# /metrics.json. Prefix-cache reuse has its own live tests (test_prefix_cache_ssd_default.sh, test_prefix_cache_mem.sh).
set -uo pipefail
cd "$(dirname "$0")/.."

BINARY="${BINARY:-./zig-out/bin/sushi}"
PORT="${PORT:-11431}"
BASE="http://127.0.0.1:$PORT"
MAX_GB="${SMOKE_MAX_GB:-0}"
OUT="${SMOKE_OUT:-$HOME/.sushi/runs/smoke-matrix-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT/home"

[[ -x "$BINARY" ]] || { echo "[fatal] $BINARY missing — zig build -Doptimize=ReleaseFast"; exit 1; }

# arch|thinking(yes/no)|candidate paths (first that exists wins)
ARCHES=(
    "qwen4_exp|yes|${QWEN4_EXP_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen3.8-Flash-Next-Sushi-3bpw}"
    "mimo_v2|yes|${MIMO_STREAM_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/MiMo-V2.6-Flash-Sushi-2.3bpw}"
    "deepseek_v41|yes|${DSV41_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/deepseek-ai/DeepSeek-V4.1-Flash-1017}"
)
CONFIGS="${SMOKE_CONFIGS:-default,off,kv4,mtp,nospec}"

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

SERVER_PID=""
boot() { # $1 model path, $2... extra flags
    local model="$1"; shift
    # Isolated HOME: ~/.sushi/model-settings.json decides every key a cell leaves
    # unflagged, so a real profile would silently change what the cell measures.
    HOME="$OUT/home" "$BINARY" --model "$model" --serve --host 127.0.0.1 --port "$PORT" --log-level info --metrics "$@" \
        > "$OUT/$CELL.server.log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 600); do
        curl -sf "$BASE/health" >/dev/null 2>&1 && curl -sf "$BASE/v1/models" 2>/dev/null | grep -q '"id"' && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "  server died at boot"; return 1; }
        sleep 1
    done
    echo "  server never became ready"; return 1
}
stop() {
    [[ -n "$SERVER_PID" ]] || return
    kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""
    # macOS hands an exited server's GPU memory back seconds later; the next boot's preflight reads free RAM.
    sleep "${SMOKE_SETTLE_S:-10}"
}
trap stop EXIT

post() { curl -s --max-time 300 "$BASE$1" -H "Content-Type: application/json" -d "$2"; }
Q='Reply in one short sentence: what colour is the sky on a clear day?'
TOOLS='[{"type":"function","function":{"name":"get_weather","description":"Current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'
TAGS='<think>|</think>|<\|channel>|<channel\|>|<tool_call>|<function=|DSML'

run_checks() { # $1 thinking yes/no, $2 has_spec yes/no
    local think="$1"
    local r c rc fr

    # 1. chat non-stream, thinking default
    r=$(post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":600,\"temperature\":0}")
    c=$(echo "$r" | J 'd["choices"][0]["message"]["content"] or ""')
    check "chat non-stream: content" "$([[ -n "$c" ]] && echo 0 || echo 1)" "$(echo "$r" | head -c 300)"
    check "chat non-stream: no tag leak" "$(echo "$c" | grep -Eq "$TAGS" && echo 1 || echo 0)" "$c"
    check "chat non-stream: usage.cached_tokens present" "$(echo "$r" | grep -q '"cached_tokens"' && echo 0 || echo 1)"

    # 2. chat stream
    r=$(curl -sN --max-time 300 "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
        -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":600,\"temperature\":0,\"stream\":true,\"stream_options\":{\"include_usage\":true}}")
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
        r=$(post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 12*13? Answer with the number.\"}],\"max_tokens\":1200,\"temperature\":0,\"enable_thinking\":true}")
        c=$(echo "$r" | J 'd["choices"][0]["message"]["content"] or ""')
        local rsn; rsn=$(echo "$r" | J 'd["choices"][0]["message"].get("reasoning_content") or ""')
        fr=$(echo "$r" | J '(d["choices"][0].get("finish_details") or {}).get("type") or d["choices"][0]["finish_reason"]')
        # a budgeted or loop-cut answer may end inside the thought: then content may be empty legitimately
        if [[ ( "$fr" == "length" || "$fr" == "repetition_loop" ) && -z "$c" ]]; then skip "thinking on non-stream: content" "ended inside the thought ($fr)"; else
        check "thinking on non-stream: content" "$([[ -n "$c" ]] && echo 0 || echo 1)" "$(echo "$r" | head -c 300)"; fi
        check "thinking on non-stream: no tag leak" "$(printf '%s%s' "$c" "$rsn" | grep -Eq '<think>|<\|channel>' && echo 1 || echo 0)"
        r=$(curl -sN --max-time 300 "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
            -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 12*13? Answer with the number.\"}],\"max_tokens\":1200,\"temperature\":0,\"enable_thinking\":true,\"stream\":true}")
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
        r=$(post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":600,\"temperature\":0,\"enable_thinking\":false}")
        c=$(echo "$r" | J 'd["choices"][0]["message"]["content"] or ""')
        check "thinking off: content" "$([[ -n "$c" ]] && echo 0 || echo 1)" "$(echo "$r" | head -c 300)"
    fi

    # 4. tools: 200 + valid args when the model calls
    r=$(post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Use the tool to get the weather in Paris.\"}],\"tools\":$TOOLS,\"max_tokens\":300,\"temperature\":0}")
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
    r=$(post /v1/chat/completions '{"model":"m","messages":[{"role":"user","content":"Give me a person named Ada aged 36 as JSON."}],"max_tokens":600,"temperature":0,"response_format":{"type":"json_schema","json_schema":{"name":"person","strict":true,"schema":{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"],"additionalProperties":false}}}}')
    c=$(echo "$r" | J 'd["choices"][0]["message"]["content"]')
    check "json_schema: content parses with both keys" "$(echo "$c" | py 'import sys,json;d=json.load(sys.stdin);assert "name" in d and "age" in d' && echo 0 || echo 1)" "${c:0:120}"

    # 6. max_tokens cap, then logprobs (entries describe message.content, so they need a finished answer)
    r=$(post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":5,\"temperature\":0}")
    # A model that finishes under the cap (Spark: "Blue") reports stop; the invariant is the cap.
    fr=$(echo "$r" | J 'd["choices"][0]["finish_reason"]')
    check "max_tokens 5: <=5 tokens, finish_reason length|stop" "$([[ ( "$fr" == length || "$fr" == stop ) && "$(echo "$r" | J 'd["usage"]["completion_tokens"]')" -le 5 ]] && echo 0 || echo 1)" "$(echo "$r" | head -c 200)"
    # ignore_eos: a completion that would stop early runs on to max_tokens, and the stream carries the same bytes;
    # chat refuses the field by name.
    local ie="{\"model\":\"m\",\"prompt\":\"Q: What colour is the sky on a clear day? A:\",\"max_tokens\":48,\"temperature\":0,\"ignore_eos\":true"
    r=$(post /v1/completions "$ie}")
    check "ignore_eos: completion runs 48 tokens, finish_reason length" "$([[ "$(echo "$r" | J 'd["choices"][0]["finish_reason"]')" == length && "$(echo "$r" | J 'd["usage"]["completion_tokens"]')" == 48 ]] && echo 0 || echo 1)" "$(echo "$r" | head -c 300)"
    c=$(echo "$r" | J 'repr(d["choices"][0]["text"])')
    rc=$(curl -sN --max-time 300 "$BASE/v1/completions" -H "Content-Type: application/json" -d "$ie,\"stream\":true}" | python3 -c '
import sys,json
t=""
for l in sys.stdin:
    l=l.strip()
    if not l.startswith("data: ") or l=="data: [DONE]": continue
    for ch in json.loads(l[6:]).get("choices",[]):
        t+=ch.get("text") or ""
print(repr(t))' 2>/dev/null)
    check "ignore_eos: completion stream == non-stream bytes" "$([[ -n "$c" && "$c" == "$rc" ]] && echo 0 || echo 1)" "ns=${c:0:120} st=${rc:0:120}"
    r=$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 "$BASE/v1/chat/completions" -H "Content-Type: application/json" \
        -d "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":8,\"ignore_eos\":true}")
    check "ignore_eos: chat refuses it with a 400" "$([[ "$r" == 400 ]] && echo 0 || echo 1)" "status=$r"
    r=$(post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}],\"max_tokens\":600,\"temperature\":0,\"logprobs\":true,\"top_logprobs\":2,\"enable_thinking\":false}")
    c=$(echo "$r" | J 'd["choices"][0]["message"]["content"] or ""')
    if [[ -z "$c" ]]; then skip "logprobs: entries with top_logprobs" "no content to describe"
    elif [[ "$model" == *.gguf ]]; then skip "logprobs: entries with top_logprobs" "engine-backed: logprobs not surfaced (known gap)"; else
    check "logprobs: entries with top_logprobs" "$([[ "$(echo "$r" | J 'len(d["choices"][0]["logprobs"]["content"][0]["top_logprobs"])')" == 2 ]] && echo 0 || echo 1)" "$(echo "$r" | J 'str(d["choices"][0].get("logprobs"))[:120]')"; fi

    # 7. concurrency: two at once
    post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from one to twenty in words.\"}],\"max_tokens\":600,\"temperature\":0}" > "$OUT/$CELL.c1.json" &
    local p1=$!
    post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Name five fruits, comma separated.\"}],\"max_tokens\":600,\"temperature\":0}" > "$OUT/$CELL.c2.json" &
    local p2=$!
    wait "$p1" "$p2"
    check "concurrency 2: both answered" "$([[ -n "$(J 'd["choices"][0]["message"]["content"]' < "$OUT/$CELL.c1.json")" && -n "$(J 'd["choices"][0]["message"]["content"]' < "$OUT/$CELL.c2.json")" ]] && echo 0 || echo 1)"

    # 8. /v1/completions
    r=$(post /v1/completions '{"model":"m","prompt":"The capital of France is","max_tokens":8,"temperature":0}')
    check "completions: text" "$([[ -n "$(echo "$r" | J 'd["choices"][0]["text"]')" ]] && echo 0 || echo 1)" "$(echo "$r" | head -c 200)"
    # An empty prompt tokenizes to nothing: a named 400 in both modes, and the server lives on (check 13).
    for st in false true; do
        r=$(curl -s -w '\n%{http_code}' --max-time 60 "$BASE/v1/completions" -H "Content-Type: application/json" -d "{\"model\":\"m\",\"prompt\":\"\",\"max_tokens\":4,\"stream\":$st}")
        check "completions: empty prompt (stream=$st) is a 400 invalid_request_error" "$([[ "${r##*$'\n'}" == 400 && "$r" == *invalid_request_error* ]] && echo 0 || echo 1)" "$(echo "$r" | head -c 200)"
    done

    # 9. /v1/messages
    r=$(post /v1/messages "{\"model\":\"m\",\"max_tokens\":600,\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}]}")
    check "messages non-stream: text block + stop_reason" "$([[ -n "$(echo "$r" | J '[b for b in d["content"] if b["type"]=="text"][0]["text"]')" && -n "$(echo "$r" | J 'd["stop_reason"]')" ]] && echo 0 || echo 1)" "$(echo "$r" | head -c 200)"
    r=$(curl -sN --max-time 300 "$BASE/v1/messages" -H "Content-Type: application/json" -d "{\"model\":\"m\",\"max_tokens\":600,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"$Q\"}]}")
    check "messages stream: text_delta + message_stop" "$(echo "$r" | grep -q '"text_delta"' && echo "$r" | grep -q 'event: message_stop' && echo 0 || echo 1)" "$(echo "$r" | head -c 200)"

    # 10. /v1/responses
    r=$(post /v1/responses "{\"model\":\"m\",\"input\":\"$Q\",\"max_output_tokens\":600}")
    check "responses non-stream: output_text" "$([[ -n "$(echo "$r" | J '[c["text"] for o in d["output"] if o["type"]=="message" for c in o["content"] if c["type"]=="output_text"][0]')" ]] && echo 0 || echo 1)" "$(echo "$r" | head -c 200)"
    r=$(curl -sN --max-time 300 "$BASE/v1/responses" -H "Content-Type: application/json" -d "{\"model\":\"m\",\"input\":\"$Q\",\"max_output_tokens\":600,\"stream\":true}")
    check "responses stream: sequence_number + completed" "$(echo "$r" | grep -q '"sequence_number"' && echo "$r" | grep -q 'response.completed' && echo 0 || echo 1)" "$(echo "$r" | head -c 200)"

    # 12. discovery + metrics
    r=$(curl -s "$BASE/v1/models")
    check "/v1/models: context_length at top level" "$(echo "$r" | grep -q '"context_length"' && echo 0 || echo 1)"
    r=$(curl -s "$BASE/metrics.json")
    check "/metrics.json: answers" "$(echo "$r" | J 'len(d)>0' | grep -q True && echo 0 || echo 1)"

    # 13. the server is still alive and no [mlx] error latched
    check "server alive after the cell" "$(kill -0 "$SERVER_PID" 2>/dev/null && echo 0 || echo 1)"
    check "no MLX error / crash line in the log" "$(grep -Eq '\[mlx\] error|panic|Segmentation|error latched' "$OUT/$CELL.server.log" && echo 1 || echo 0)"
}

has_mtp_head() { # an `mtp.` tensor in any safetensors HEADER (in-checkpoint or sidecar)
    python3 - "$1" <<'PY'
import glob, json, os, struct, sys
d = sys.argv[1]
files = glob.glob(os.path.join(d, "*.safetensors")) + glob.glob(os.path.join(d, "mtp*", "*.safetensors"))
for f in files:
    try:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            keys = json.loads(fh.read(n)).keys()
    except Exception:
        continue
    if any(".mtp." in k or k.startswith("mtp.") for k in keys):
        sys.exit(0)
sys.exit(1)
PY
}

IFS=',' read -r -a WANT_CFG <<< "$CONFIGS"
for entry in "${ARCHES[@]}"; do
    IFS='|' read -r arch think rest <<< "$entry"
    if [[ -n "${SMOKE_ARCHES:-}" ]] && ! [[ ",$SMOKE_ARCHES," == *",$arch,"* ]]; then continue; fi
    model=""
    IFS='|' read -r -a cands <<< "$rest"
    for p in "${cands[@]}"; do [[ -e "$p" ]] && { model="$p"; break; }; done
    if [[ -z "$model" ]]; then CELL="$arch"; skip "$arch" "no checkpoint on this box"; continue; fi
    gb=$(( $(du -sk "$model" 2>/dev/null | cut -f1) / 1024 / 1024 ))
    if [[ "$MAX_GB" -gt 0 && "$gb" -gt "$MAX_GB" ]]; then CELL="$arch"; skip "$arch" "${gb} GB > SMOKE_MAX_GB=$MAX_GB"; continue; fi

    arch_flags=()
    [[ "$arch" == mimo_v2 ]] && arch_flags=(${MIMO_SSD_BUDGET_GB:+--ssd-budget-gb "$MIMO_SSD_BUDGET_GB"})
    # Its routed experts always stream, so the budget is required; its KV is not a KV cache, so only default and nospec differ.
    [[ "$arch" == deepseek_v41 ]] && arch_flags=(--ssd-budget-gb "${DSV41_SSD_BUDGET_GB:-88}")

    for cfg in "${WANT_CFG[@]}"; do
        CELL="$arch.$cfg"
        flags=(${arch_flags[@]+"${arch_flags[@]}"})
        if [[ "$arch" == deepseek_v41 && "$cfg" != default && "$cfg" != nospec ]]; then skip "$CELL" "no KV cache to configure"; continue; fi
        case "$cfg" in
            default) ;;
            off)     flags+=(--kv-quant off) ;;
            kv4)     flags+=(--kv-quant 4) ;;
            kv8)     flags+=(--kv-quant 8) ;;
            mtp)     [[ "$arch" == mimo_v2 && -n "${MIMO_SSD_BUDGET_GB:-}" ]] && { skip "$CELL" "streamed experts refuse MTP"; continue; }
                     has_mtp_head "$model" || { skip "$CELL" "no MTP head"; continue; }; flags+=(--mtp) ;;
            nospec)  flags+=(--no-mtp --no-drafter) ;;
            *) skip "$CELL" "unknown config"; continue ;;
        esac
        echo ""
        echo "=== $CELL  ($(basename "$model"), ${gb} GB) ==="
        t0=$(date +%s)
        if ! boot "$model" ${flags[@]+"${flags[@]}"}; then bad "boot" "$(tail -3 "$OUT/$CELL.server.log" | tr '\n' ' ')"; stop; break; fi
        echo "  booted in $(( $(date +%s) - t0 ))s"
        run_checks "$think"
        if [[ "$cfg" == mtp ]]; then
            check "mtp: engaged in the log" "$(grep -Eq 'spec-stats\] mode=(mtp|dspark)' "$OUT/$CELL.server.log" && echo 0 || echo 1)"
        fi
        if [[ "$arch" == deepseek_v41 && "$cfg" == default ]]; then
            before=$(grep -c 'spec-stats\] mode=dspark' "$OUT/$CELL.server.log")
            post /v1/chat/completions "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"Write two sentences about the sea.\"}],\"max_tokens\":120,\"temperature\":1,\"top_p\":0.95,\"enable_thinking\":false}" > /dev/null
            check "dspark: a sampled request drafts" "$([[ "$(grep -c 'spec-stats\] mode=dspark' "$OUT/$CELL.server.log")" -gt "$before" ]] && echo 0 || echo 1)"
        fi
        if [[ "$cfg" == nospec ]]; then
            check "nospec: no speculation engaged" "$(grep -Eq 'spec-stats\] mode=(mtp|dspark|drafter|dflash)' "$OUT/$CELL.server.log" && echo 1 || echo 0)"
        fi
        stop
    done
done

echo ""
echo "================================================================"
echo "smoke matrix: pass=$PASS fail=$FAIL skip=$SKIP   logs: $OUT"
for f in ${FAILS[@]+"${FAILS[@]}"}; do echo "  FAIL $f"; done
[[ "$FAIL" -eq 0 ]]

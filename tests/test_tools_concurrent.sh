#!/bin/bash
# sushi-test: tier=live boot=api
# Guard: concurrent agent loops each get THEIR OWN tool calls.
#
# Batched decode and tool buffering are individually tested and have never been
# tested together. With tools present the server buffers tokens for detection
# per slot, and slots entering a batch mid-generation drain lazy pipeline state
# first — so a cross-slot leak here looks like one client receiving another
# client's arguments, which no single-stream test can see.
#
# N concurrent streams, each planting a DIFFERENT unique token in its prompt and
# asking for one tool call carrying it back. Assertions:
#   • every stream completes (no lost request, [DONE] seen)
#   • every stream's call carries ITS OWN token and no other stream's
#   • every stream's arguments are valid JSON, no markup in visible content
#   • the server is alive and functional afterwards
#
# Whether a checkpoint calls the tool is its choice: a stream without a call is checked only for leaks, and
# a round where no stream called it is a SKIP. Forcing the call would serialise its slot and defeat the test.
#
# Attaches to the shared api boot (4 slots): tests/run.sh --only test_tools_concurrent.sh --model <m>.
#   CONC_N=4  — number of concurrent agent loops

set -u

. "$(dirname "$0")/lib/boot.sh"
sushi_attach
sushi_require batching
N="${CONC_N:-4}"
BASE=$SUSHI_URL
RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
PASS=0; FAIL=0
ok()  { echo -e "  ${GREEN}PASS${NC} $1"; PASS=$((PASS+1)); }
bad() { echo -e "  ${RED}FAIL${NC} $1"; shift; for l in "$@"; do echo "        $l"; done; FAIL=$((FAIL+1)); }

OUTDIR=$(mktemp -d /tmp/tools_conc_out.XXXXXX)
trap 'rm -rf "$OUTDIR"' EXIT

# Distinct, non-overlapping tokens so a cross-slot leak is unambiguous.
TOKENS=(ZULU-1001 YANKEE-2002 XRAY-3003 WHISKEY-4004 VICTOR-5005 TANGO-6006 SIERRA-7007 ROMEO-8008)

echo "launching $N concurrent tool-calling streams"
# Collect the stream PIDs: a bare `wait` would also wait on any other background job of the runner.
STREAM_PIDS=""
for i in $(seq 0 $((N-1))); do
    TOK="${TOKENS[$i]}"
    python3 - "$BASE" "$TOK" "$OUTDIR/$i.json" "$SUSHI_NONCE" <<'PY' &
import json, sys, urllib.request
base, tok, out, nonce = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
body = {"model": "sushi",
        "messages": [{"role": "user", "content":
            "%s The session code is %s. Call write_file exactly once to save the session "
            "code to notes.txt, putting the code in the content argument." % (nonce, tok)}],
        "tools": [{"type": "function", "function": {
            "name": "write_file", "description": "Write a file",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"]}}}],
        "max_tokens": 4096, "reasoning_effort": "low", "temperature": 0, "stream": True}
chunks, saw_done, calls, content, reasoning = [], False, {}, [], []
try:
    r = urllib.request.urlopen(urllib.request.Request(
        base + "/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"}), timeout=900)
    for line in r:
        line = line.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            saw_done = True
            continue
        d = json.loads(payload)
        for ch in d.get("choices", []):
            delta = ch.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
            for tc in (delta.get("tool_calls") or []):
                idx = tc.get("index", 0)
                slot = calls.setdefault(idx, {"name": "", "args": ""})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]
except Exception as e:
    json.dump({"error": str(e)}, open(out, "w")); raise SystemExit
json.dump({"token": tok, "done": saw_done, "calls": list(calls.values()),
           "content": "".join(content), "reasoning": "".join(reasoning)}, open(out, "w"))
PY
    STREAM_PIDS="$STREAM_PIDS $!"
done
for pid in $STREAM_PIDS; do wait "$pid" 2>/dev/null; done

ALL_TOKENS="${TOKENS[*]:0:$N}"
CALLED=0
for i in $(seq 0 $((N-1))); do
    TOK="${TOKENS[$i]}"
    F="$OUTDIR/$i.json"
    [ -s "$F" ] || { bad "[stream $i $TOK] produced no result file"; continue; }
    V=$(python3 - "$F" "$TOK" "$ALL_TOKENS" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
mine, everyone = sys.argv[2], sys.argv[3].split()
if "error" in d:
    print("err|%s" % d["error"]); raise SystemExit
calls = d["calls"]
seen = d["content"] + d["reasoning"]
if not calls:
    foreign = [t for t in everyone if t != mine and t in seen]
    print("nocall|%d|%s|content=%r" % (int(d["done"]), ",".join(foreign), d["content"][:80])); raise SystemExit
try:
    args = json.loads(calls[0]["args"]); valid = 1
except Exception:
    args, valid = {}, 0
blob = json.dumps(args)
carried = 1 if mine in blob else 0
foreign = [t for t in everyone if t != mine and t in blob]
foreign += [t for t in everyone if t != mine and t in seen and t not in foreign]
leak = any(t in (d["content"] or "") for t in ("<tool_call", "<|tool_call", "<function=", "<｜DSML｜"))
print("ok|%d|%d|%d|%s|%d" % (int(d["done"]), valid, carried, ",".join(foreign), int(leak)))
PY
)
    case "$V" in
        err\|*)    bad "[stream $i $TOK] transport error" "${V#err|}"; continue ;;
        nocall\|*)
            IFS='|' read -r _ NC_DONE NC_FOREIGN NC_CONTENT <<< "$V"
            echo "  note: [stream $i $TOK] made no tool call ($NC_CONTENT)"
            [ "$NC_DONE" = "1" ] && ok "[stream $i $TOK] terminated with [DONE]" || bad "[stream $i $TOK] no [DONE]"
            [ -z "$NC_FOREIGN" ] && ok "[stream $i $TOK] carried no other stream's code" \
                                 || bad "[stream $i $TOK] CROSS-SLOT LEAK" "found other streams' codes: $NC_FOREIGN"
            continue ;;
    esac
    CALLED=$((CALLED+1))
    DONE=$(echo "$V" | cut -d'|' -f2); VALID=$(echo "$V" | cut -d'|' -f3)
    CARRIED=$(echo "$V" | cut -d'|' -f4); FOREIGN=$(echo "$V" | cut -d'|' -f5)
    LEAK=$(echo "$V" | cut -d'|' -f6)
    [ "$DONE" = "1" ]    && ok "[stream $i $TOK] terminated with [DONE]"     || bad "[stream $i $TOK] no [DONE]"
    [ "$VALID" = "1" ]   && ok "[stream $i $TOK] arguments valid JSON"       || bad "[stream $i $TOK] arguments not valid JSON"
    [ "$CARRIED" = "1" ] && ok "[stream $i $TOK] carried its OWN code"       || bad "[stream $i $TOK] lost its own code"
    [ -z "$FOREIGN" ]    && ok "[stream $i $TOK] carried no other stream's code" \
                         || bad "[stream $i $TOK] CROSS-SLOT LEAK" "found other streams' codes: $FOREIGN"
    [ "$LEAK" = "0" ]    && ok "[stream $i $TOK] no markup in visible content" || bad "[stream $i $TOK] markup leaked"
done

# Batched decode has to have actually happened, or this ran N serial requests.
if sushi_log_since | grep -qE "batch|concurrent"; then
    ok "server logged batched/concurrent decode activity"
else
    echo "  note: no batch line in the log — streams may have been serialized"
fi

R=$(curl -sf -m 60 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"sushi\",\"messages\":[{\"role\":\"user\",\"content\":\"$SUSHI_NONCE Say OK.\"}],\"max_tokens\":8,\"temperature\":0}" 2>/dev/null)
[ -n "$R" ] && ok "server still functional after the concurrent round" \
            || bad "server unresponsive after the concurrent round"

echo
echo "tools under concurrency: $PASS passed, $FAIL failed"
if [ "$FAIL" -eq 0 ] && [ "$CALLED" -eq 0 ]; then
    echo "SKIP $(basename "$0"): no stream called the tool, so no call could leak"
    exit 3
fi
[ "$FAIL" -eq 0 ]

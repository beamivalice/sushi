#!/bin/bash
# sushi-test: tier=live boot=own pack=each boots=2
# The SSD prefix tier across a restart, on the pass model: one persist boot and one restart boot (one label, one
# private cache root). SSD-only (`--prefix-cache-mem off`), so every warm request in either boot comes off the disk.
#
# Boot 1 persists, boot 2 restores:
#   [1] a long text prompt persists chunked entries (`[disk-cache] persisted`, meta.json + c*.safetensors on disk); a
#       multi-turn extension appends instead of rewriting; a warm re-issue restores from SSD.
#   [2] a text -> image -> text conversation: its append restores PAST the image (cached_tokens over PAST_BAR, a
#       bar only rows computed under that image can clear), a different image under the same placeholders restores
#       at most the text before it (issue #47). Cache invariants only: nothing asks the model to read the picture.
#   [3] after the restart the same text prompt restores from SSD with the warm reply's bytes (a restore at the same
#       point is exact; a hybrid's restore is not bit-identical to a cold prefill, so the baseline is the warm
#       reply); the restore shape line matches the arch (Qwen `ssm@`, MiMo `ring@`); the appended turn restores its
#       own prompt; the media conversation restores past its image.
#   [4] GLM (DFlash2 in the default boot): the assistant context restores beside the trunk (`dflash context
#       restored`) and the per-draft rate stays within 5 points of the persist boot's.
#   [5] qwen4 only: a restore wider than the fd limit (the restart boot runs under a lowered `ulimit -n`) succeeds,
#       and an unreadable chunk falls back to a cold prefill that answers.
set -uo pipefail
. "$(dirname "$0")/lib/boot.sh"

command -v jq > /dev/null || { echo "needs jq"; exit 1; }
KEY=$(sushi_model_key "${SUSHI_PASS_MODEL:-qwen4}") || exit 1
MODEL=$(sushi_pack "$KEY") || exit $?
ARCH=$(sushi_arch "$MODEL")
F1="$SUSHI_ROOT/tests/fixtures/street-name-signs.jpg"
F2="$SUSHI_ROOT/tests/fixtures/house.jpeg"
for f in "$F1" "$F2"; do [ -f "$f" ] || { echo "SKIP $(basename "$0"): fixture $f missing"; exit 3; }; done
PAST_BAR="${RESTART_PAST_BAR:-1000}"
SLACK=64
LABEL=restart
WORK=$(mktemp -d)
trap 'chmod -R u+rw "$WORK" 2> /dev/null; rm -rf "$WORK"' EXIT
pass=0; fail=0
check() { if [ "$2" = "$3" ]; then echo "  ok   $1"; pass=$((pass + 1)); else echo "  FAIL $1: got '$2' want '$3'"; fail=$((fail + 1)); fi; }
ge() { [ "$1" -ge "$2" ] 2> /dev/null && echo 1 || echo 0; }

# GLM's original template has no thinking-off mode, and MTP exists only on qwen4 and mimo2.
case "$ARCH" in glm5_next) THINK='{"reasoning_effort":"low"}' ;; *) THINK='{"enable_thinking":false}' ;; esac
case "$ARCH" in qwen4_exp | mimo_v2) THINK=$(jq -c '. + {enable_mtp:false}' <<< "$THINK") ;; esac
FD_ARM=0
[ "$ARCH" = qwen4_exp ] && FD_ARM=1

pyh() { # a request-body builder and a poster, keyed by its first argument
    python3 -I - "$@" <<'PY'
import base64, json, os, sys, urllib.request

kind = sys.argv[1]
think = json.loads(os.environ["THINK"])


def write(path, payload):
    payload.update(think)
    json.dump(payload, open(path, "w"))


if kind == "text":  # text <main|ext|long> <out> [reply file]
    variant, out = sys.argv[2], sys.argv[3]
    n = 1200 if variant == "long" else 80
    lines = ["You are reviewing a project log. Read it fully, then answer."]
    for i in range(1, n + 1):
        lines.append(f"Log entry {i}: subsystem {i % 17} reported state change {i * 31 % 997} with latency {i * 7 % 131} ms and checksum {i * i % 9973}.")
    lines.append("Question: which subsystem appears in log entry 42? Answer with one short sentence.")
    messages = [{"role": "user", "content": "\n".join(lines)}]
    if variant == "ext":
        messages += [{"role": "assistant", "content": open(sys.argv[4]).read()},
                     {"role": "user", "content": "Now answer the same question for log entry 43."}]
    write(out, {"model": "sushi", "messages": messages, "max_tokens": 24, "temperature": 0, "stream": False})
elif kind == "media":  # media <image> <turns> <out>
    path, turns, out = sys.argv[2], int(sys.argv[3]), sys.argv[4]
    url = "data:image/jpeg;base64," + base64.b64encode(open(path, "rb").read()).decode()
    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                     {"type": "text", "text": "Describe this picture in one sentence."}]},
        {"role": "assistant", "content": "A street scene."},
        {"role": "user", "content": "Notes: " + "lorem ipsum dolor sit amet " * 420},
        {"role": "assistant", "content": "Noted."},
    ]
    for i in range(turns):
        messages.append({"role": "user", "content": f"Turn {i}. Reply with ALPHA only."})
        if i + 1 < turns:
            messages.append({"role": "assistant", "content": "ALPHA"})
    write(out, {"model": "sushi", "messages": messages, "max_tokens": 16, "temperature": 0})
elif kind == "ask":  # ask <body> <reply out>: prints "<cached_n> <prompt_n>"; a refused body prints ERR ERR
    body, reply = sys.argv[2], sys.argv[3]
    req = urllib.request.Request(os.environ["SUSHI_URL"] + "/v1/chat/completions", data=open(body, "rb").read(),
                                 headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=1800))
    except Exception as e:
        print("the-server-refused-this-body:", e, file=sys.stderr)
        print("ERR ERR")
        sys.exit()
    m = d["choices"][0]["message"]
    open(reply, "w").write((m.get("reasoning_content") or "") + "\x01" + (m.get("content") or ""))
    t = d.get("timings") or {}
    cached = t.get("cached_n", d["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0))
    print(cached, d["usage"]["prompt_tokens"])
PY
}

ask() { # body -> "<cached> <prompt>"; the reply lands in $WORK/reply
    THINK="$THINK" pyh ask "$1" "$WORK/reply"
}
build() { THINK="$THINK" pyh "$@"; }

LOGOFF=0
boot() {
    LOGOFF=$(wc -c 2> /dev/null < "${SUSHI_RUN_DIR:-/nonexistent}/$LABEL.log" || echo 0)
    LOGOFF=${LOGOFF// /}
    # shellcheck disable=SC2046
    sushi_boot "$LABEL" "$MODEL" $(sushi_profile_flags api "$KEY") --prefix-cache-entries 4 --prefix-cache-mem off \
        --prefix-cache-disk 4GB --prefill-chunk 1024 $([ "$FD_ARM" = 1 ] && echo --ctx-size 65536) "$@" || exit $?
    CACHE="$SUSHI_RUN_DIR/$LABEL.kv-cache"
}
blog() { tail -c +$((LOGOFF + 1)) "$SUSHI_LOG"; }
disk_events() { blog | grep -cE '\[disk-cache\] (persisted|e[0-9]+ complete on disk)'; }
# The flush runs after the response: wait for the writer to go quiet before the entry is read back or the server stops.
settle_disk() {
    local prev=-1 now quiet=0 i
    for i in $(seq 1 90); do
        now=$(disk_events)
        if [ "$now" = "$prev" ] && [ "$now" -gt 0 ]; then quiet=$((quiet + 1)); else quiet=0; fi
        [ "$quiet" -ge 3 ] && return 0
        prev=$now
        sleep 1
    done
}
per_draft() { blog | grep -o 'per_draft_pct=[0-9.]*' | tail -1 | cut -d= -f2; }

echo "== boot 1: persist ($ARCH)"
boot
build text main "$WORK/main.json"
read -r C0 P0 <<< "$(ask "$WORK/main.json")"
cp "$WORK/reply" "$WORK/main.cold"
DF_COLD=$(per_draft)
check "cold text turn answered" "$([ "$C0" = ERR ] && echo 0 || echo 1)" "1"
check "cold: nothing cached" "$C0" "0"
settle_disk
check "[disk-cache] persisted fired" "$([ "$(blog | grep -c '\[disk-cache\] persisted')" -ge 1 ] && echo 1 || echo 0)" "1"
read -r CW PW <<< "$(ask "$WORK/main.json")"
cp "$WORK/reply" "$WORK/main.warm"
check "a warm re-issue restores the prompt (cached $CW of $PW)" "$(ge "$CW" $((P0 - SLACK)))" "1"
check "it came off the disk tier" "$([ "$(blog | grep -c '\[disk-cache\] restored')" -ge 1 ] && echo 1 || echo 0)" "1"
PERSISTS=$(blog | grep -c '\[disk-cache\] persisted')
build text ext "$WORK/ext.json" "$WORK/main.warm"
read -r CE PE <<< "$(ask "$WORK/ext.json")"
settle_disk
check "the extension turn reuses the first prompt (cached $CE of $PE)" "$(ge "$CE" $((P0 - SLACK)))" "1"
check "the extension turn persisted (an append, not a rewrite)" "$([ "$(blog | grep -c '\[disk-cache\] persisted')" -gt "$PERSISTS" ] && echo 1 || echo 0)" "1"

build media "$F1" 1 "$WORK/m1.json"
build media "$F1" 2 "$WORK/m2.json"
build media "$F2" 2 "$WORK/m3.json"
read -r R1 _ <<< "$(ask "$WORK/m1.json")"
check "media: cold turn answered, nothing cached" "$R1" "0"
read -r R2 _ <<< "$(ask "$WORK/m2.json")"
check "media: the same image with a turn appended restores PAST the image (cached $R2)" "$(ge "$R2" $((PAST_BAR + 1)))" "1"
read -r R3 _ <<< "$(ask "$WORK/m3.json")"
check "media: a different image invalidates its suffix (cached $R3)" "$([ "$R3" != ERR ] && [ "$R3" -le "$PAST_BAR" ] && echo 1 || echo 0)" "1"
settle_disk

TOP_FD=0
if [ "$FD_ARM" = 1 ]; then
    build text long "$WORK/long.json"
    for _ in 1 2 3 4 5 6; do
        ask "$WORK/long.json" > /dev/null
        settle_disk
        [ "$(find "$CACHE" -name 'c*.safetensors' 2> /dev/null | wc -l)" -ge 30 ] && break
    done
    TOP_FD=$(lsof -a -p "$SUSHI_PID" -d '0-65535' -F f 2> /dev/null | sed -n 's/^f//p' | sort -n | tail -1)
fi
sushi_stop 0

META=$(find "$CACHE" -name meta.json 2> /dev/null | wc -l | tr -d ' ')
CHUNKS=$(find "$CACHE" -name 'c*.safetensors' 2> /dev/null | wc -l | tr -d ' ')
check "on-disk layout: at least one meta.json and two chunks ($META meta, $CHUNKS chunks)" "$([ "$META" -ge 1 ] && [ "$CHUNKS" -ge 2 ] && echo 1 || echo 0)" "1"

echo "== boot 2: restart"
if [ "$FD_ARM" = 1 ]; then
    FD_LIMIT=$((TOP_FD + 16))
    echo "  persisted $CHUNKS chunks; serving top fd $TOP_FD; the restart runs under ulimit -n $FD_LIMIT"
    check "the fd arm persisted 30+ chunks, more than the fd headroom" "$([ "$CHUNKS" -ge 30 ] && echo 1 || echo 0)" "1"
    FD_SOFT=$(ulimit -S -n)
    ulimit -S -n "$FD_LIMIT"
    boot
    ulimit -S -n "$FD_SOFT"
else
    boot
fi
read -r CR PR <<< "$(ask "$WORK/main.json")"
cp "$WORK/reply" "$WORK/main.restart"
DF_HIT=$(per_draft)
check "[disk-cache] restored fired after the restart" "$([ "$(blog | grep -c '\[disk-cache\] restored .* tokens from SSD')" -ge 1 ] && echo 1 || echo 0)" "1"
check "the same prompt restores (cached $CR of $PR)" "$(ge "$CR" $((P0 - SLACK)))" "1"
check "the restored reply is the warm reply, byte for byte" "$(cmp -s "$WORK/main.warm" "$WORK/main.restart" && echo 1 || echo 0)" "1"
case "$ARCH" in
    qwen4_exp) check "the hybrid restore carries its SSM state" "$([ "$(blog | grep -cE '\[disk-cache\] restored .*\(ssm@[0-9]+\)')" -ge 1 ] && echo 1 || echo 0)" "1" ;;
    mimo_v2) check "the sliding ring restores from its ring file" "$([ "$(blog | grep -cE '\[disk-cache\] restored .*ring@')" -ge 1 ] && echo 1 || echo 0)" "1" ;;
esac
read -r CE2 _ <<< "$(ask "$WORK/ext.json")"
check "the appended turn restores its own prompt from SSD (cached $CE2 of $PE)" "$(ge "$CE2" $((PE - SLACK)))" "1"
read -r RR2 _ <<< "$(ask "$WORK/m2.json")"
check "media: the image prefix restores after the restart (cached $RR2)" "$(ge "$RR2" $((PAST_BAR + 1)))" "1"
if [ "$ARCH" = glm5_next ] && blog | grep -q 'DFlash drafter ready'; then
    check "the assistant context restored beside the trunk" "$([ "$(blog | grep -c 'dflash context restored')" -ge 1 ] && echo 1 || echo 0)" "1"
    check "the per-draft rate stays within 5 points across the restart (${DF_COLD:-none} -> ${DF_HIT:-none})" \
        "$(python3 -c "
import sys
try:
    print(1 if abs(float(sys.argv[1]) - float(sys.argv[2])) <= 5.0 else 0)
except ValueError:
    print(0)" "${DF_COLD:-x}" "${DF_HIT:-x}")" "1"
fi

if [ "$FD_ARM" = 1 ]; then
    read -r CL _ <<< "$(ask "$WORK/long.json")"
    check "a restore wider than the fd limit engages (cached $CL)" "$(ge "$CL" 30000)" "1"
    # The unreadable chunk: restore fails and a cold prefill still answers.
    chmod 000 "$(find "$CACHE" -name 'c000002.safetensors' | head -1)"
    read -r CU _ <<< "$(ask "$WORK/long.json")"
    check "an unreadable chunk: the restore fails and the request still answers" "$([ "$CU" != ERR ] && echo 1 || echo 0)" "1"
    check "  with the fallback named and no failed prefill" "$(blog | grep -q 'restore failed: .* falling back' && ! blog | grep -q 'prefill failed' && echo 1 || echo 0)" "1"
fi
sushi_stop 0
chmod -R u+rw "$CACHE" 2> /dev/null

echo "pass=$pass fail=$fail"
[ "$fail" = 0 ]

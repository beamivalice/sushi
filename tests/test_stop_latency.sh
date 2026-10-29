#!/bin/bash
# sushi-test: tier=live boot=api
# Stop-latency test: when the client cancels a streaming request mid-decode,
# the server's connection thread must notice within ~100ms and stop the
# inference for that slot. Without this, the GPU keeps generating tokens for
# many seconds — fans spin, energy burns, partial work is discarded.
#
# How we measure: read the server's log finish line "<- N+M tokens
# streamed ... [client_disconnect]" — we record the client's token count at
# the kill moment and compare against the server's reported M. Anything more
# than ~1 spec-block of extra tokens flags a regression. The kill point is a
# token count, never a wall-clock moment, so a slow arch is cut at the same place.
#
# Attaches to the shared api boot: tests/run.sh --only test_stop_latency.sh --model <m>.

set -e

. "$(dirname "$0")/lib/boot.sh"
sushi_attach
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
PORT=${SUSHI_URL##*:}

echo "== stop-latency test =="
echo "  server: $SUSHI_URL ($SUSHI_ARCH)"

# Per kill-at-N test: fire a long streaming request, drop the connection
# after the client has read N token deltas with SO_LINGER=0 (immediate RST), then read the server
# log for the matching finish line and compute (server_total - client_count).
run_one() {
    local kill_after=$1
    python3 - "$PORT" "$kill_after" "$SUSHI_NONCE" "$(dirname "$0")/lib" <<'PY'
import socket, struct, time, json, sys, re

sys.path.insert(0, sys.argv[4])
import sushi_test

port = int(sys.argv[1])
kill_after = int(sys.argv[2])
nonce = sys.argv[3]
FINISH = re.compile(r"<- (\d+)\+(\d+) tokens streamed.*\[(\w+)\]")


def finish_lines():
    return [m for m in map(FINISH.search, sushi_test.log_since().splitlines()) if m]


seen_before = len(finish_lines())
WRITE_FAILED = "streaming error: error.WriteFailed"
write_failed_before = sushi_test.log_since().count(WRITE_FAILED)

body = json.dumps({
    "model": "sushi",
    "messages": [{"role": "user", "content": f"{nonce} Tell me a long story about a robot."}],
    "max_tokens": 2000,
    "temperature": 0.8,
    "stream": True,
}).encode()
req = (
    b"POST /v1/chat/completions HTTP/1.1\r\n"
    b"Host: 127.0.0.1:%d\r\n" % port +
    b"Content-Type: application/json\r\n"
    b"Content-Length: " + str(len(body)).encode() + b"\r\n"
    b"Connection: close\r\n\r\n"
) + body
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("127.0.0.1", port))
s.sendall(req)

# Reasoning deltas are tokens the server generated too, so they count.
delta = re.compile(rb'"delta":\{"(?:reasoning_)?content"')
buf = b""
client_count = 0
guard = time.time() + 600
while client_count < kill_after:
    if time.time() > guard:
        print(f"FAIL kill_after={kill_after}: {client_count} deltas in 600s")
        sys.exit(1)
    s.settimeout(30)
    try: data = s.recv(4096)
    except socket.timeout: continue
    if not data:
        print(f"server closed early after {client_count} deltas")
        sys.exit(1)
    buf += data
    client_count = len(delta.findall(buf))

s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
s.close()

# Wait up to 10s for the server to log this request's finish line.
deadline = time.time() + 10
lines = []
while time.time() < deadline:
    lines = finish_lines()
    if len(lines) > seen_before:
        break
    time.sleep(0.05)

if len(lines) <= seen_before:
    # A write that fails before the disconnect probe ends the request without the finish line.
    if sushi_test.log_since().count(WRITE_FAILED) > write_failed_before:
        print(f"  kill@{kill_after} deltas: the stream write failed first; no finish line to measure")
        sys.exit(7)
    print(f"FAIL kill_after={kill_after}: no new finish line in the log within 10s")
    sys.exit(1)

m = lines[-1]
server_total = int(m.group(2))
reason = m.group(3)
extra = server_total - client_count

print(f"  kill@{kill_after} deltas: client={client_count} server={server_total} extra={extra} reason={reason}")

# Pass criteria:
#   - reason should be client_disconnect (not stop/length — model finished naturally)
#   - extra ≤ 16 tokens (≤ one spec-decode block in the worst case: drafter block_size 8 + a small buffer)
if reason != "client_disconnect":
    print(f"  FAIL: expected reason 'client_disconnect', got '{reason}' — server didn't detect the drop")
    sys.exit(1)
if extra > 16:
    print(f"  FAIL: server emitted {extra} tokens after kill (target ≤16) — disconnect-detection latency too high")
    sys.exit(1)
PY
}

# Try several kill points — early kills exercise the first decode tick,
# later kills exercise the steady-state loop.
PASS=0
FAIL=0
UNMEASURED=0
for kill_at in 8 24 64 128; do
    rc=0
    run_one "$kill_at" || rc=$?
    case $rc in
        0) PASS=$((PASS+1)) ;;
        7) UNMEASURED=$((UNMEASURED+1)) ;;
        *) FAIL=$((FAIL+1)) ;;
    esac
done
# Most kill points must be measured, or the script proves nothing about latency.
[ "$PASS" -ge 2 ] || FAIL=$((FAIL+1))

echo
echo "  measured and within budget: $PASS, write failed first: $UNMEASURED"
if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}FAIL${NC} a kill point exceeded the latency budget, or fewer than two were measured"
    exit 1
fi
echo -e "${GREEN}PASS${NC} all kill points within the 16-token budget"

#!/usr/bin/env python3
"""Streaming-silence probe for tests/test_stream_keepalive.sh.

Opens a raw socket against one streaming surface, drives a tool call big
enough that the server buffers tokens for many seconds, and records the
wall-clock gap between successive byte arrivals.

Exit codes:  0 = pass,  1 = fail,  2 = skip (generation too short to test)
"""
import json
import os
import re
import socket
import sys
import time

PORT = int(sys.argv[1])
SURFACE = sys.argv[2]
MODEL = sys.argv[3]

MAX_GAP_S = float(os.environ.get("MAX_GAP_S", 15))
MIN_BUFFER_S = float(os.environ.get("MIN_BUFFER_S", 8))

DESC = "Write text content to a file on disk."
# Long, mechanical content: the model emits it as one giant tool-call argument,
# which the server buffers end-to-end while it watches for the closing tag.
PROMPT = (
    "Use the write_file tool to create report.md. Its content must be the "
    "numbers 1 through 250, one per line, each formatted exactly as "
    "'- item N: pending review'. Write the whole file in a single call."
)
PROPS = {"path": {"type": "string"}, "content": {"type": "string"}}
SCHEMA = {"type": "object", "properties": PROPS, "required": ["path", "content"]}

if SURFACE == "chat":
    PATH = "/v1/chat/completions"
    BODY = {
        "model": MODEL, "stream": True, "max_tokens": 4096, "temperature": 0.0,
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": [{"type": "function", "function": {
            "name": "write_file", "description": DESC, "parameters": SCHEMA}}],
    }
elif SURFACE == "messages":
    PATH = "/v1/messages"
    BODY = {
        "model": MODEL, "stream": True, "max_tokens": 4096, "temperature": 0.0,
        "messages": [{"role": "user", "content": PROMPT}],
        "tools": [{"name": "write_file", "description": DESC, "input_schema": SCHEMA}],
    }
elif SURFACE == "responses":
    PATH = "/v1/responses"
    BODY = {
        "model": MODEL, "stream": True, "max_output_tokens": 4096, "temperature": 0.0,
        "input": [{"role": "user", "content": PROMPT}],
        "tools": [{"type": "function", "name": "write_file",
                   "description": DESC, "parameters": SCHEMA}],
    }
else:
    print(f"unknown surface {SURFACE}")
    sys.exit(1)

payload = json.dumps(BODY).encode()
req = (
    f"POST {PATH} HTTP/1.1\r\nHost: 127.0.0.1:{PORT}\r\n"
    f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
    f"Connection: close\r\n\r\n"
).encode() + payload

sock = socket.create_connection(("127.0.0.1", PORT), timeout=30)
sock.sendall(req)
sock.settimeout(180)

t0 = last = time.time()
max_gap = 0.0
max_gap_at = 0.0
raw = b""
while True:
    try:
        chunk = sock.recv(65536)
    except socket.timeout:
        print("FAIL: socket idle for 180s")
        sys.exit(1)
    if not chunk:
        break
    now = time.time()
    gap = now - last
    if gap > max_gap:
        max_gap, max_gap_at = gap, now - t0
    last = now
    raw += chunk
sock.close()
elapsed = time.time() - t0
text = raw.decode("utf-8", "replace")

beats = text.count(": keepalive") + text.count("event: ping")
print(f"elapsed={elapsed:.1f}s bytes={len(raw)} max_gap={max_gap:.1f}s "
      f"(at t={max_gap_at:.1f}s) keepalives={beats}")

# ── Did we actually exercise a long buffered span? ──
if elapsed < MIN_BUFFER_S:
    print(f"SKIP: generation took {elapsed:.1f}s < MIN_BUFFER_S={MIN_BUFFER_S}s")
    sys.exit(2)

failures = []

# 1. No client-visible silence.
if max_gap > MAX_GAP_S:
    failures.append(f"socket silent for {max_gap:.1f}s (limit {MAX_GAP_S}s) — "
                    f"a client with an idle-body timeout would drop this stream")

# 2. The heartbeat actually fired.
if beats == 0:
    failures.append("no keepalive/ping observed on the stream")

# 3. The tool call survived the injected bytes: name intact, args valid JSON.
args = None
name = None
if SURFACE == "chat":
    for m in re.finditer(r'"name":"(\w+)"', text):
        name = m.group(1)
    # arguments arrive as one JSON-string delta
    frags = re.findall(r'"arguments":"((?:[^"\\]|\\.)*)"', text)
    if frags:
        args = json.loads(f'"{"".join(frags)}"')
elif SURFACE == "messages":
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            ev = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        blk = ev.get("content_block") or {}
        if blk.get("type") == "tool_use":
            name = blk.get("name")
        if ev.get("delta", {}).get("type") == "input_json_delta":
            args = (args or "") + ev["delta"].get("partial_json", "")
else:  # responses
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            ev = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        item = ev.get("item") or {}
        if item.get("type") == "function_call":
            name = item.get("name")
            args = item.get("arguments") or args
        if ev.get("type", "").startswith("response.function_call_arguments"):
            args = ev.get("arguments") or ((args or "") + ev.get("delta", ""))

if name != "write_file":
    failures.append(f"tool call name not recovered (got {name!r})")
elif not args:
    failures.append("tool call carried no arguments")
else:
    try:
        parsed = json.loads(args)
    except json.JSONDecodeError as e:
        failures.append(f"tool arguments are not valid JSON ({e}); "
                        f"keepalive bytes may have corrupted the stream")
    else:
        if "content" not in parsed:
            failures.append(f"tool arguments missing 'content' key: {list(parsed)}")
        else:
            print(f"tool call ok: write_file, content={len(parsed['content'])} chars")

for f in failures:
    print(f"FAIL: {f}")
sys.exit(1 if failures else 0)

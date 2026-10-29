#!/usr/bin/env python3
# sushi-test: tier=live boot=api
"""Concurrent envelope-ID regression, attached to the shared api boot.

Four clients race per API shape, so the pass must batch. Passing runs are silent;
requests, replies and the observed ID sets are saved under --output (default: the run dir).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from urllib.request import Request, urlopen

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import sushi_test  # noqa: E402


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def request(base, endpoint, body, output, barrier):
    save(output.with_suffix(".request.json"), body)
    barrier.wait(timeout=900)
    req = Request(base + endpoint, data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=900) as response:
        raw = response.read().decode()
    output.with_suffix(".sse" if body["stream"] else ".response.json").write_text(raw)
    tool_calls = 0
    if body["stream"]:
        events = []
        done = False
        for line in raw.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                continue
            event = json.loads(data)
            assert "error" not in event, event
            events.append(event)
        if endpoint == "/v1/messages":
            ids = [event["message"]["id"] for event in events if event.get("type") == "message_start"]
            assert len(ids) == 1, events
            assert any(event.get("type") == "message_stop" for event in events), events
        else:
            assert done, raw
            ids = [event["id"] for event in events]
    else:
        reply = json.loads(raw)
        assert "error" not in reply, reply
        ids = [reply["id"]]
        if "tools" in body:
            calls = reply["choices"][0]["message"].get("tool_calls") or []
            tool_calls = len(calls)
    assert ids and all(isinstance(value, str) and value for value in ids), ids
    assert len(set(ids)) == 1, "one stream changed its envelope ID"
    return {"id": ids[0], "id_events": len(ids), "tool_calls": tool_calls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url")
    parser.add_argument("--model", default="default")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    nonce = sushi_test.attach()
    sushi_test.require("batching")
    base = args.url or sushi_test.url()
    output = args.output or Path(os.environ.get("SUSHI_RUN_DIR") or tempfile.gettempdir()) / f"completion-ids-{nonce}"
    output.mkdir(parents=True, exist_ok=False)
    # GLM's template refuses thinking off (400), so it thinks briefly instead.
    thinking = ({"reasoning_effort": "low", "reasoning_budget_tokens": 16}
                if os.environ.get("SUSHI_ARCH") == "glm5_next" else {"enable_thinking": False})
    common = dict({"model": args.model, "temperature": 0, "max_tokens": 32,
                   "enable_mtp": False, "enable_pld": False}, **thinking)
    chat = dict(common, messages=[{"role": "user", "content": f"{nonce} Explain sorting carefully with examples."}])
    completion = dict(common, prompt=f"{nonce} Sorting a list means")
    messages = dict(chat)
    tool = dict(chat, max_tokens=64, tools=[{"type": "function", "function": {
        "name": "emit", "description": "Emit the requested tag.", "parameters": {
            "type": "object", "properties": {"tag": {"type": "string", "enum": ["OWN_TOOL"]}},
            "required": ["tag"], "additionalProperties": False}}}],
        tool_choice={"type": "function", "function": {"name": "emit"}})
    tool["messages"] = [{"role": "user", "content": f"{nonce} Call emit once with tag OWN_TOOL."}]
    cases = (("chat", "/v1/chat/completions", chat),
             ("completion", "/v1/completions", completion),
             ("messages", "/v1/messages", messages),
             ("chat-tools", "/v1/chat/completions", tool))
    seen = set()
    results = []
    for name, endpoint, body in cases:
        for repeat in range(2):
            folder = output / f"{name}-{repeat + 1}"
            folder.mkdir()
            barrier = threading.Barrier(4)
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(request, base.rstrip("/"), endpoint,
                                       dict(body, stream=(i % 2 == 0)), folder / str(i), barrier)
                           for i in range(4)]
                replies = [future.result(timeout=900) for future in futures]
            ids = [reply["id"] for reply in replies]
            assert len(set(ids)) == len(ids), replies
            assert not seen.intersection(ids), "a later request reused an envelope ID"
            seen.update(ids)
            row = {"case": name, "repeat": repeat + 1, "replies": replies}
            save(folder / "evidence.json", row)
            results.append(row)
    save(output / "summary.json", {"requests": len(seen), "bursts": results,
         "tool_call_replies": sum(reply["tool_calls"] > 0 for row in results for reply in row["replies"]),
         "unique_across_requests": True, "stable_within_streams": True, "passing": True})


if __name__ == "__main__":
    main()

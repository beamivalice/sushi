#!/usr/bin/env python3
"""Byte-exact reference for the embedded DeepSeek-V4.1 chat template: the release's own
`encoding/encoding.py` renders the shapes the server sends, and the Zig test renders the same
cases through jinja.cpp.

  python3 -I tests/dump_dsv41_template_cases.py --encoding <DeepSeek-V4.1-Flash>/encoding \
      [--dump src/fixtures/dsv41_template_cases.json]

The server hands the template tool-call `arguments` as objects, omits `reasoning_content` when
absent, and passes `thinking_mode` and the effort word it mapped (low|high|max).
"""
import argparse
import importlib.util
import json
import sys

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}]
SIMPLE = [{"role": "user", "content": "Weather in Paris?"}]
WITH_SYSTEM = [{"role": "system", "content": "You are helpful."}] + SIMPLE
TOOL_ROUND = WITH_SYSTEM + [
    {"role": "assistant", "content": "", "reasoning_content": "Need the tool.",
     "tool_calls": [{"id": "tc_0", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}}]},
    {"role": "tool", "content": "Sunny, 22C", "tool_call_id": "tc_0"},
]
HISTORY = [
    {"role": "user", "content": "Why is the sky blue?"},
    {"role": "assistant", "content": "Rayleigh scattering.", "reasoning_content": "Shorter wavelengths scatter more."},
    {"role": "user", "content": "And sunsets?"},
]
PARALLEL = [
    {"role": "user", "content": "Weather in Paris and Rome?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
        {"id": "b", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]},
    {"role": "tool", "content": "Paris: sunny, 22C", "tool_call_id": "a"},
    {"role": "tool", "content": "Rome: cloudy, 19C", "tool_call_id": "b"},
]
LATE_SYSTEM = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello.", "reasoning_content": "Greet."},
    {"role": "system", "content": "Answer in French."},
    {"role": "user", "content": "How are you?"},
]
TWO_USERS = [{"role": "user", "content": "First."}, {"role": "user", "content": "Second."}]

# (label, messages, tools, thinking_mode, reasoning_effort)
CASES = [
    ("simple/chat", SIMPLE, None, "chat", "low"),
    ("simple/thinking low", SIMPLE, None, "thinking", "low"),
    ("simple/thinking high", SIMPLE, None, "thinking", "high"),
    ("simple/thinking max", SIMPLE, None, "thinking", "max"),
    ("system+tools/chat", WITH_SYSTEM, TOOLS, "chat", "low"),
    ("system+tools/thinking", WITH_SYSTEM, TOOLS, "thinking", "high"),
    ("tools, no system/thinking", SIMPLE, TOOLS, "thinking", "high"),
    ("tool round/chat", TOOL_ROUND, TOOLS, "chat", "low"),
    ("tool round/thinking", TOOL_ROUND, TOOLS, "thinking", "high"),
    ("history drops thinking/thinking", HISTORY, None, "thinking", "high"),
    ("history/chat", HISTORY, None, "chat", "low"),
    ("parallel calls/thinking", PARALLEL, TOOLS, "thinking", "high"),
    ("late system turn/thinking", LATE_SYSTEM, None, "thinking", "high"),
    ("consecutive users/chat", TWO_USERS, None, "chat", "low"),
]


def load_encoder(encoding_dir):
    spec = importlib.util.spec_from_file_location("dsv41_encoding", encoding_dir + "/encoding.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference(enc, messages, tools, mode, effort):
    msgs = json.loads(json.dumps(messages))
    for m in msgs:
        for tc in m.get("tool_calls", []):
            tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"], ensure_ascii=False)
    if tools:
        # the encoder takes tools on the leading system message, as the release's own inputs do
        if msgs and msgs[0]["role"] == "system":
            msgs[0]["tools"] = tools
        else:
            msgs.insert(0, {"role": "system", "content": "", "tools": tools})
    return enc.encode_messages(msgs, thinking_mode=mode, reasoning_effort=effort)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding", required=True)
    ap.add_argument("--dump")
    args = ap.parse_args()
    enc = load_encoder(args.encoding)
    out = []
    for label, messages, tools, mode, effort in CASES:
        out.append({"label": label, "messages": messages, "tools": tools, "thinking_mode": mode,
                    "reasoning_effort": effort, "expected": reference(enc, messages, tools, mode, effort)})
    if args.dump:
        with open(args.dump, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
            f.write("\n")
    else:
        for c in out:
            print(c["label"], repr(c["expected"][:120]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

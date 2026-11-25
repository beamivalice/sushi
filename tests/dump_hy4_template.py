#!/usr/bin/env python3
# sushi-test: tier=tool
"""Hy4-Flash's chat template rendered by transformers' apply_chat_template, the reference for hy4_chat.zig's cases.

    <venv with transformers at PR 50903 head c2ec8505>/bin/python -I tests/dump_hy4_template.py <Hy4-Flash checkout dir>

Writes src/hy4flash/fixtures/hy4_template_cases.json: each case's messages (tool-call arguments as the JSON text a
request carries), tools, thinking, effort word and preserve flag, and the text the reference renders.
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src/hy4flash/fixtures/hy4_template_cases.json"

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
CALL = {"id": "call_1", "name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}

CASES = [
    {"label": "system and user", "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]},
    {"label": "history thinking is dropped without tools", "messages": [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "add them"},
        {"role": "user", "content": "and 3+3?"}]},
    {"label": "preserved thinking keeps history thinking", "preserve": True, "messages": [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "add them"},
        {"role": "user", "content": "and 3+3?"}]},
    {"label": "no_think keeps history think tags empty", "thinking": False, "messages": [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "add them"},
        {"role": "user", "content": "and 3+3?"}]},
    {"label": "tools, a call and its result", "tools": [WEATHER], "messages": [
        {"role": "system", "content": "You help."},
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "reasoning_content": "use the tool", "tool_calls": [CALL]},
        {"role": "tool", "content": "{\"temp\": 18}", "tool_call_id": "call_1"}]},
    {"label": "max effort with tools", "effort": "max", "tools": [WEATHER], "messages": [
        {"role": "user", "content": "Weather in Paris?"}]},
]


def hf_messages(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        if "tool_calls" in m:
            m["tool_calls"] = [{"id": c["id"], "type": "function",
                                "function": {"name": c["name"], "arguments": json.loads(c["arguments"])}}
                               for c in m["tool_calls"]]
        out.append(m)
    return out


def main():
    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    cases = []
    for c in CASES:
        thinking = c.get("thinking", True)
        effort = "no_think" if not thinking else c.get("effort", "high")
        kw = {"reasoning_effort": effort}
        if c.get("preserve"):
            kw["preserved_thinking"] = True
        text = tok.apply_chat_template(hf_messages(c["messages"]), tools=c.get("tools"), tokenize=False,
                                       add_generation_prompt=True, **kw)
        cases.append({**c, "thinking": thinking, "sends": None if not thinking else c.get("effort"),
                      "preserve": bool(c.get("preserve")), "want": text})
    OUT.write_text(json.dumps({"generator": "tests/dump_hy4_template.py: transformers c2ec8505 apply_chat_template",
                               "cases": cases}, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(cases)} cases -> {OUT.relative_to(ROOT)}")


main()

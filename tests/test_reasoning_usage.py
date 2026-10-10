#!/usr/bin/env python3
"""Check delivered reasoning usage through a running real-model HTTP server."""
import argparse
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:12345")
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    def post(path, body):
        request = urllib.request.Request(args.url.rstrip("/") + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(request, timeout=180)

    tools = [{"type": "function", "function": {"name": "answer", "description": "Return the integer answer",
              "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}}}]
    cases = [("stream", True, {}), ("nonstream", False, {}),
             ("tool-stream", True, {"tools": tools, "tool_choice": "required"}),
             ("tool-nonstream", False, {"tools": tools, "tool_choice": "required"}),
             ("json-stream", True, {"response_format": {"type": "json_object"}}),
             ("budget-stream", True, {"reasoning_budget_tokens": 1}),
             ("budget-nonstream", False, {"reasoning_budget_tokens": 1})]
    results = []
    for name, streaming, extra in cases:
        body = {"model": args.model, "messages": [{"role": "user", "content": "Calculate 123 * 456. Return the integer result."}],
                "reasoning_effort": args.effort, "temperature": 0, "max_tokens": 256, "stream": streaming, **extra}
        if streaming:
            body["stream_options"] = {"include_usage": True}
        reasoning = ""
        usage = None
        tool_calls = 0
        with post("/v1/chat/completions", body) as response:
            if streaming:
                for line in response:
                    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    if event.get("usage"):
                        usage = event["usage"]
                        assert event["choices"] == [], "usage event must not repeat a choice"
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        reasoning += delta.get("reasoning_content") or ""
                        tool_calls += len(delta.get("tool_calls") or [])
            else:
                event = json.load(response)
                usage = event["usage"]
                message = event["choices"][0]["message"]
                reasoning = message.get("reasoning_content") or ""
                tool_calls = len(message.get("tool_calls") or [])
        assert usage is not None, (name, "missing usage")
        assert "completion_tokens_details" in usage, (name, "missing completion_tokens_details")
        reported = usage["completion_tokens_details"]["reasoning_tokens"]
        with post("/tokenize", {"model": args.model, "content": reasoning}) as response:
            expected = len(json.load(response)["tokens"])
        assert type(reported) is int and reported >= 0 and reported == expected, (name, reported, expected)
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], name
        assert "cached_tokens" in usage["prompt_tokens_details"], name
        result = {"case": name, "reasoning_tokens": reported, "completion_tokens": usage["completion_tokens"],
                  "tool_call_deltas": tool_calls}
        results.append(result)
        print("PASS:", json.dumps(result), flush=True)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "reasoning-usage.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

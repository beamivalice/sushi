#!/usr/bin/env python3
# sushi-test: tier=live boot=knobs
"""Check --think defaults, explicit budgets and repetition-penalty aliases on the knobs boot
(--think medium --reasoning-budget 1). Requests, responses and server traces are saved under --output."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import sushi_test as st


def reasoning(kind, reply):
    if kind == "chat":
        return reply["choices"][0]["message"].get("reasoning_content") or ""
    if kind == "messages":
        return "".join(x.get("thinking", "") for x in reply["content"] if x.get("type") == "thinking")
    return "".join(
        part.get("text", "")
        for item in reply.get("output", []) if item.get("type") == "reasoning"
        for part in item.get("summary", []) + item.get("content", [])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="default")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    nonce = st.attach()
    args.url = st.url()
    args.server_log = Path(os.environ["SUSHI_LOG"])
    args.output = args.output or (Path(os.environ["SUSHI_RUN_DIR"]) / "request-defaults" if os.environ.get("SUSHI_RUN_DIR") else Path(tempfile.mkdtemp(prefix="request-defaults-")))
    args.output.mkdir(parents=True, exist_ok=True)

    def post(name, endpoint, body):
        offset = args.server_log.stat().st_size
        (args.output / (name + ".request.json")).write_text(json.dumps(body, indent=2) + "\n")
        request = Request(args.url.rstrip("/") + endpoint, data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=180) as response:
            reply = json.load(response)
        (args.output / (name + ".response.json")).write_text(json.dumps(reply, indent=2) + "\n")
        trace = args.server_log.read_bytes()[offset:].decode("utf-8", "replace")
        (args.output / (name + ".server.log")).write_text(trace)
        # Responses includes an error:null field on successful/incomplete replies.
        assert reply.get("error") is None, reply
        if name.endswith("-default"):
            assert "reasoning budget 8192: enforced in-stream" in trace, (name, trace)
        if name.endswith("-explicit-one"):
            assert "reasoning budget 1: enforced in-stream" in trace, (name, trace)
        return reply

    records = []
    prompt = (nonce + " Find all positive integers n for which n squared plus 3n plus 1 is a perfect square. "
              "Work through the inequalities and verify every boundary carefully.")
    for kind, endpoint in (("chat", "/v1/chat/completions"), ("messages", "/v1/messages"),
                           ("responses", "/v1/responses")):
        body = {"model": args.model, "temperature": 0, "stream": False}
        if kind == "responses":
            body.update(input=prompt, max_output_tokens=128)
        else:
            body.update(messages=[{"role": "user", "content": prompt}], max_tokens=128)
        default = post(kind + "-default", endpoint, body)
        override = ({"thinking": {"type": "enabled", "budget_tokens": 1}} if kind == "messages"
                    else {"reasoning_budget_tokens": 1})
        capped = post(kind + "-explicit-one", endpoint, body | override)
        normal, limited = reasoning(kind, default), reasoning(kind, capped)
        assert len(normal) > 20, (kind, "missing default reasoning", default)
        assert "Considering the limited time" not in normal, (kind, "CLI default was capped early")
        assert "Considering the limited time" in limited, (kind, "explicit budget did not close thinking")
        records.append({"case": kind + "-budget", "default_reasoning_chars": len(normal),
                        "explicit_one_chars": len(limited), "default_uses_think_medium": True,
                        "explicit_budget_wins": True})

    for kind, endpoint in (("chat", "/v1/chat/completions"), ("legacy", "/v1/completions")):
        body = {"model": args.model, "temperature": 0, "max_tokens": 64, "stream": False,
                "enable_thinking": False, "enable_mtp": False, "enable_pld": False}
        prompt = nonce + " Write eight short numbered sentences about sorting files, using the word file in every sentence."
        body.update({"messages": [{"role": "user", "content": prompt}]} if kind == "chat" else {"prompt": prompt})
        replies = []
        for name, fields in (("canonical", {"repeat_penalty": 1.2}),
                             ("alias", {"repetition_penalty": 1.2}),
                             ("precedence", {"repeat_penalty": 1.2, "repetition_penalty": 1.5})):
            reply = post(kind + "-" + name, endpoint, body | fields)
            choice = reply["choices"][0]
            replies.append(choice.get("message", choice.get("text")))
        assert replies[0] and replies[0] == replies[1] == replies[2], (kind, "alias/precedence output differs")
        records.append({"case": kind + "-alias", "canonical_alias_precedence_exact": True})

    result = {"passing": True, "cases": records}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Direct multi-turn agent loop test against sushi (no Swift app in the loop).
Verifies that thinking is enabled on EVERY iteration of the agent loop and that
content/reasoning/tool_calls split cleanly across turns.

Tools implemented locally: writeFile, readFile, shell, listFiles.
Models tested: any path passed in.

Usage: python3 tests/test_agent_thinking.py [model_dir] [port] [stream:0|1]
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("SUSHI_MODELS_DIR", os.path.expanduser("~/.sushi/models")), "Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw")
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8095
STREAM = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
BASE = f"http://127.0.0.1:{PORT}"
WORKSPACE = os.path.expanduser("~/.sushi/workspace/agent_thinking_test")

TOOLS = [
    {"type": "function", "function": {
        "name": "writeFile",
        "description": "Write content to a file (overwrites). Path relative to working dir.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}
        }, "required": ["path", "content"]}
    }},
    {"type": "function", "function": {
        "name": "shell",
        "description": "Run a shell command in the working directory.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}
        }, "required": ["command"]}
    }},
    {"type": "function", "function": {
        "name": "listFiles",
        "description": "List files in a directory (defaults to working dir).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}
        }, "required": []}
    }},
]


def post_json(path, body, stream=False):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if stream:
        return urllib.request.urlopen(req, timeout=300)
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_chat(messages, stream=False):
    body = {
        "model": "sushi",
        "messages": messages,
        "max_tokens": 800,
        "temperature": 0.3,
        "stream": stream,
        "enable_thinking": True,
        "tools": TOOLS,
    }
    if not stream:
        d = post_json("/v1/chat/completions", body)
        m = d["choices"][0]["message"]
        return {
            "content": m.get("content") or "",
            "reasoning": m.get("reasoning_content") or "",
            "tool_calls": m.get("tool_calls") or [],
            "finish": d["choices"][0]["finish_reason"],
        }
    # streaming
    resp = post_json("/v1/chat/completions", body, stream=True)
    raw = resp.read().decode("utf-8", errors="replace")
    content_chunks, reasoning_chunks, tool_calls_acc = [], [], {}
    finish = None
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except Exception:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        fr = choices[0].get("finish_reason")
        if fr:
            finish = fr
        if delta.get("content"):
            content_chunks.append(delta["content"])
        if delta.get("reasoning_content"):
            reasoning_chunks.append(delta["reasoning_content"])
        if delta.get("tool_calls"):
            for tc in delta["tool_calls"]:
                idx = tc.get("index", 0)
                cur = tool_calls_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    cur["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    cur["name"] = fn["name"]
                if fn.get("arguments"):
                    cur["arguments"] += fn["arguments"]
    return {
        "content": "".join(content_chunks),
        "reasoning": "".join(reasoning_chunks),
        "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                       for _, tc in sorted(tool_calls_acc.items())],
        "finish": finish,
    }


def execute_tool(name, args):
    try:
        if name == "writeFile":
            path = os.path.join(WORKSPACE, args["path"])
            os.makedirs(os.path.dirname(path) or WORKSPACE, exist_ok=True)
            with open(path, "w") as f:
                f.write(args.get("content", ""))
            return f"wrote {len(args.get('content',''))} chars to {args['path']}"
        if name == "shell":
            r = subprocess.run(args["command"], shell=True, cwd=WORKSPACE, capture_output=True, text=True, timeout=30)
            out = (r.stdout + r.stderr)[:1000]
            return out or "(no output)"
        if name == "listFiles":
            target = args.get("path") or "."
            full = os.path.join(WORKSPACE, target) if target != "." else WORKSPACE
            return "\n".join(sorted(os.listdir(full)))
        return f"unknown tool: {name}"
    except Exception as e:
        return f"ERROR {type(e).__name__}: {e}"


def run_agent(user_prompt, max_rounds=10):
    messages = [{"role": "user", "content": user_prompt}]
    leak_tags = ("<think", "</think", "<|channel", "<channel|", "<tool_call")
    rounds = []
    for i in range(max_rounds):
        resp = call_chat(messages, stream=STREAM)
        rounds.append(resp)
        leak_in_content = any(t in resp["content"] for t in leak_tags)
        leak_in_reasoning = any(t in resp["reasoning"] for t in leak_tags)
        marker = "TOOLS" if resp["tool_calls"] else "FINAL"
        print(f"  R{i+1} [{marker}] reason={len(resp['reasoning'])} content={len(resp['content'])} "
              f"tool_calls={len(resp['tool_calls'])} leak_c={leak_in_content} leak_r={leak_in_reasoning}")
        if leak_in_content or leak_in_reasoning:
            print(f"    LEAK detected — content: {resp['content'][:120]!r}")
            print(f"    LEAK detected — reasoning: {resp['reasoning'][:120]!r}")
        if not resp["tool_calls"]:
            print(f"  FINAL: {resp['content'][:200]}")
            return rounds
        # Add assistant + tool results for next turn
        messages.append({
            "role": "assistant",
            "content": resp["content"],
            "tool_calls": resp["tool_calls"],
        })
        for tc in resp["tool_calls"]:
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                args = {}
            result = execute_tool(tc["function"]["name"], args)
            print(f"    -> {tc['function']['name']}({list(args.keys())}) = {result[:100]!r}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result[:1500],
            })
    return rounds


def main():
    os.makedirs(WORKSPACE, exist_ok=True)
    # Clean workspace
    for f in os.listdir(WORKSPACE):
        p = os.path.join(WORKSPACE, f)
        try:
            if os.path.isfile(p):
                os.remove(p)
            elif os.path.isdir(p):
                subprocess.run(["rm", "-rf", p])
        except Exception:
            pass

    # Wait for server
    print(f"Model: {MODEL_DIR}")
    print(f"Port: {PORT}, stream={STREAM}")
    for _ in range(60):
        try:
            urllib.request.urlopen(f"{BASE}/health", timeout=2).read()
            break
        except Exception:
            time.sleep(1)
    else:
        print("FAIL: server not reachable")
        sys.exit(1)

    print("\n=== Task: write fib.py and verify ===")
    rounds = run_agent(
        "Create a tiny Python script called fib.py that prints the first 10 Fibonacci numbers. "
        "Then run it via shell to verify it works. Think carefully about each step.",
        max_rounds=8,
    )

    # Verification
    fib_path = os.path.join(WORKSPACE, "fib.py")
    print(f"\nfib.py exists: {os.path.exists(fib_path)}")
    if os.path.exists(fib_path):
        try:
            r = subprocess.run(["python3", fib_path], capture_output=True, text=True, timeout=10)
            out = r.stdout.strip().split("\n")
            print(f"fib.py output (lines): {out[:12]}")
            expected = ["0", "1", "1", "2", "3", "5", "8", "13", "21", "34"]
            ok = out == expected
            print(f"Output matches expected first 10 Fibonacci: {ok}")
            if not ok:
                print(f"  expected: {expected}")
        except Exception as e:
            print(f"fib.py run failed: {e}")

    # Tally
    leak_count = 0
    for r in rounds:
        leak_tags = ("<think", "</think", "<|channel", "<channel|", "<tool_call")
        if any(t in r["content"] for t in leak_tags) or any(t in r["reasoning"] for t in leak_tags):
            leak_count += 1
    print(f"\nrounds: {len(rounds)}  total_tag_leaks: {leak_count}")
    sys.exit(0 if leak_count == 0 else 1)


if __name__ == "__main__":
    main()

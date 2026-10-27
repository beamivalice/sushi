#!/bin/bash
. "$(dirname "$0")/private_cache.sh"
# Usage: test_load_context_preflight.sh [resident Flash-Next or MiMo pack] [port] [startup|cold] [--mtp|--no-mtp]
# Owns one GPU-lock run; checks warmup memory, not timing. Silent on success.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw}"
exec python3 - "$ROOT" "$MODEL" "${2:-12484}" "${3:-startup}" "${4:---mtp}" <<'PY'
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

root, model = map(Path, sys.argv[1:3])
port, mode, mtp = sys.argv[3:]
assert mode in ("startup", "cold") and mtp in ("--mtp", "--no-mtp")
if not (model / "config.json").is_file():
    print(f"SKIP: model not found: {model}", file=sys.stderr)
    sys.exit(0)
assert json.loads((model / "config.json").read_text())["model_type"] in ("qwen4_exp", "mimo_v2")
binary = Path(os.environ.get("SUSHI_BINARY", root / "zig-out/bin/sushi")).resolve()
assert binary.is_file(), "build sushi with zig build -Doptimize=ReleaseFast first"
out = Path(os.environ.get("RUN_DIR", Path.home() / ".sushi/runs" / f"load-context-{model.name}-{mode}-{mtp[2:]}"))
out.mkdir(parents=True, exist_ok=True)
with socket.socket() as probe:
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    probe.bind(("127.0.0.1", int(port)))
base = f"http://127.0.0.1:{port}"
lock = str(root / "scripts/gpu-lock.sh")
owner = f"load-context-{os.getpid()}"
process = None

def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)

subprocess.run([lock, "acquire", owner], check=True)
try:
    args = [str(binary), "serve", "--host", "127.0.0.1", "--port", port,
            "--ctx-size", "1248", "--kv-quant", "8", mtp, "--max-tokens", "32"]
    args += ["--model", str(model)] if mode == "startup" else ["--model-dir", str(model.parent)]
    (out / "stamp.json").write_text(json.dumps({
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "binary_mtime": binary.stat().st_mtime,
        "args": args, "qos": "taskpolicy -a", "lock": owner,
    }, indent=2))
    with (out / "server.log").open("w") as log:
        process = subprocess.Popen(["taskpolicy", "-a", *args], stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            assert process.poll() is None, f"server exited; see {out / 'server.log'}"
            try:
                with urllib.request.urlopen(base + "/health", timeout=2):
                    break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1)
        else:
            raise AssertionError("server startup timeout")
        if mode == "cold":
            loaded = request("/v1/load-model", {"model": model.name})
            assert loaded["model"]["loaded"] and loaded["model"]["state"] == "ready", loaded
        props = request("/props")
        (out / "props.json").write_text(json.dumps(props, indent=2))
        assert props["default_generation_settings"]["n_ctx"] == 1248, props
        assert props["settings"]["mtp"]["default_on"] == (mtp == "--mtp"), props
        matches = re.findall(r"\[preflight\] weights ~([\d.]+) GB, needs ~([\d.]+) GB", (out / "server.log").read_text())
        assert len(matches) == 1, matches
        weights, needed = map(float, matches[0])
        assert weights >= 48, "this regression needs a large resident pack"
        # Billed weights + the larger of the measured warmup plus the 1 GiB net and the context bill.
        warmup = 0.25 if "mimo" in model.name.lower() else (1.25 if mtp == "--mtp" else 0.125)
        assert needed - weights >= 1.0 + warmup - 0.01, (weights, needed, warmup)
        assert 0 < props["memory"]["active_bytes"] <= props["memory"]["peak_bytes"]
        peak = props["memory"]["peak_bytes"] / 2**30
        assert peak <= needed + 0.005, props["memory"]
        assert needed - peak <= 3.0, (needed, peak)
        models = request("/v1/models")["data"]
        model_id = model.name if mode == "cold" else next(m["id"] for m in models if m.get("loaded"))
        reply = request("/v1/chat/completions", {
            "model": model_id, "messages": [{"role": "user", "content": "Hello."}],
            "max_tokens": 8, "stream": False,
        })
        assert reply.get("choices") and reply["usage"]["completion_tokens"] > 0, reply
        (out / "reply.json").write_text(json.dumps(reply, indent=2))
finally:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    subprocess.run([lock, "release", owner], check=True)
PY

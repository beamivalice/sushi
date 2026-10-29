#!/usr/bin/env python3
# sushi-test: tier=host
"""One-shot CLI regressions; set SUSHI_PROMPT_MODEL for the live model arms."""
import os
from pathlib import Path
import socket
import subprocess
import tempfile

binary = str(Path(os.environ.get("SUSHI_BIN", "zig-out/bin/sushi")).resolve())
# The SSD prefix cache is on by default: a live arm must not touch the real ~/.sushi/kv-cache.
os.environ.setdefault("SUSHI_PREFIX_CACHE_DIR", tempfile.mkdtemp(prefix="sushi-test-kv-"))


def run(*args, ok=False, timeout=30, gpu=False):
    lock = str(Path(__file__).resolve().parents[1] / "scripts/gpu-lock.sh")
    if gpu:
        subprocess.run([lock, "acquire", "prompt-flags-smoke"], check=True, capture_output=True)
    try:
        command = ["taskpolicy", "-a", binary] if gpu else [binary]
        result = subprocess.run([*command, *args], capture_output=True, timeout=timeout)
        output = os.environ.get("SUSHI_PROMPT_OUTPUT_DIR")
        if gpu and output:
            folder = Path(output)
            folder.mkdir(parents=True, exist_ok=True)
            n = len(list(folder.glob("*.stdout")))
            (folder / f"{n}.stdout").write_bytes(result.stdout)
            (folder / f"{n}.stderr").write_bytes(result.stderr)
    finally:
        if gpu:
            subprocess.run([lock, "release", "prompt-flags-smoke"], check=True, capture_output=True)
    assert (result.returncode == 0) == ok, (args, result.returncode, result.stderr.decode(errors="replace"))
    return result


for alias in ("-p", "--prompt"):
    assert b"expects a value" in run(alias).stderr
    assert b"requires --model" in run(alias, "hello").stderr
    for conflict in (("--serve",), ("--port", "12345"), ("--host", "0.0.0.0")):
        assert b"cannot be combined" in run("--model", "/missing-model", alias, "hello", *conflict).stderr
    assert b"cannot be combined" in run("run", "/missing-model", alias, "hello", "--tool", "on").stderr
    assert b"only valid with" in run("--model", "/missing-model", alias, "hello", "--tool", "on").stderr

# An unrelated service on the ordinary default port must never receive the prompt.
with socket.socket() as blocker, tempfile.TemporaryDirectory() as tmp:
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        blocker.bind(("127.0.0.1", 12345))
        blocker.listen()
        occupied_here = True
    except OSError:
        occupied_here = False  # A real service already supplies the occupied-port arm.
    result = run("--model", tmp, "-p", "exact prompt")
    assert b"already in use" not in result.stderr
    if occupied_here:
        blocker.settimeout(0.05)
        try:
            connection, _ = blocker.accept()
            connection.close()
            raise AssertionError("one-shot contacted the unrelated default listener")
        except TimeoutError:
            pass

model = os.environ.get("SUSHI_PROMPT_MODEL")
if model:
    base = ("--model", model, "--no-vision", "--ctx-size", "4096", "--max-tokens", "24", "--temp", "0", "--log-file", "off")
    prompt = "Reply with exactly: hello"
    plain = run(*base, "--prompt", prompt, "--think", "off", "--no-mtp", ok=True, timeout=300, gpu=True)
    streamed = run("run", model, *base[2:], "-p", prompt, "--think", "off", "--no-mtp", "--stream", ok=True, timeout=300, gpu=True)
    assert plain.stdout == streamed.stdout, (plain.stdout, streamed.stdout)
    assert b"\x1b" not in plain.stdout
    # The checkpoint chooses whether to reason; only request acceptance and clean exit are invariant.
    run(*base, "-p", prompt, "--think", "xhigh", "--stream", "--reasoning-budget", "4", "--top-p", "0.9", "--top-k", "20", ok=True, timeout=300, gpu=True)
    refused = run(*base, "-p", prompt, "--think", "high", timeout=300, gpu=True)
    assert b"not supported" in refused.stderr  # Qwen3.8 effort vocabulary.

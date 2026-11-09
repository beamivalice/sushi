#!/usr/bin/env python3
# sushi-test: tier=host
"""A POST /v1/load-model on a model whose last load failed runs a new attempt instead of answering from the failure."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def main():
    binary = Path(os.environ.get("BINARY", "zig-out/bin/sushi")).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        home = root / "home"
        home.mkdir()
        model = root / "models" / "org" / "broken"
        model.mkdir(parents=True)
        config = json.dumps({
            "model_type": "qwen4_exp", "hidden_size": 64, "num_hidden_layers": 1,
            "num_attention_heads": 1, "num_key_value_heads": 1,
            "intermediate_size": 128, "vocab_size": 32,
        })
        (model / "config.json").write_text(config)
        (model / "model.safetensors").write_bytes(b"bad!")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"

        def request(path, body=None):
            data = None if body is None else json.dumps(body).encode()
            req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as error:
                return error.code, json.load(error)

        def entry():
            return next(m for m in request("/v1/models")[1]["data"] if m["id"] == "org/broken")

        logfile = root / "server.log"
        with logfile.open("w+") as log:
            server = subprocess.Popen([str(binary), "--serve", "--port", str(port), "--model-dir", str(root / "models"),
                                       "--no-update-check", "--log-level", "debug"],
                                      stdout=log, stderr=log, env={**os.environ, "HOME": str(home)})
            try:
                for _ in range(100):
                    if server.poll() is not None:
                        raise AssertionError("server exited before readiness")
                    try:
                        if request("/health")[0] == 200:
                            break
                    except (OSError, ValueError):
                        time.sleep(0.1)
                else:
                    raise AssertionError("server did not become ready")
                assert request("/v1/load-model", {"model": "org/broken"})[0] >= 400
                first = entry()
                assert first["state"] == "error", first
                # The pack changes under the failure: a new request must meet the new condition, not the stored error.
                (model / "config.json").unlink()
                assert request("/v1/load-model", {"model": "org/broken"})[0] >= 400
                second = entry()
                assert second["state"] == "error", second
                assert second["error"] != first["error"], f"the second load answered from the stored failure {first['error']}"
                # An ordinary request keeps failing fast with the stored error, even once the pack is fixed...
                (model / "config.json").write_text(config)
                chat = {"model": "org/broken", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
                assert request("/v1/chat/completions", chat)[0] >= 400
                assert entry()["error"] == second["error"], entry()
                # ...until a saved setting that loading reads ends the failure, and the next request loads again.
                assert request("/v1/settings", {"model": "org/broken", "set": {"ctx_size": 4096}})[0] == 200
                assert entry()["state"] == "unloaded", entry()
                assert request("/v1/chat/completions", chat)[0] >= 400
                third = entry()
                assert third["state"] == "error" and third["error"] != second["error"], third
                # A setting the load never reads leaves the failure in place.
                assert request("/v1/settings", {"model": "org/broken", "set": {"max_tokens": 64}})[0] == 200
                assert entry()["state"] == "error", entry()
            except BaseException:
                log.flush()
                print(logfile.read_text(), file=sys.stderr)
                raise
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# sushi-test: tier=host
"""A failed model load becomes retryable through HTTP rescan, without a real model."""
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def main():
    binary = Path(os.environ.get("BINARY", "zig-out/bin/sushi")).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = root / "org" / "broken"
        model.mkdir(parents=True)
        (model / "config.json").write_text(json.dumps({
            "model_type": "qwen4_exp", "hidden_size": 64, "num_hidden_layers": 1,
            "num_attention_heads": 1, "num_key_value_heads": 1,
            "intermediate_size": 128, "vocab_size": 32,
        }))
        weights = model / "model.safetensors"
        weights.write_bytes(b"bad!")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"

        def request(path, body=None):
            data = None if body is None else json.dumps(body).encode()
            req = urllib.request.Request(base + path, data=data,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as error:
                return error.code, json.load(error)

        def entry():
            return next(m for m in request("/v1/models")[1]["data"] if m["id"] == "org/broken")

        with (root / "server.log").open("w+") as log:
            server = subprocess.Popen([str(binary), "--serve", "--port", str(port),
                                       "--model-dir", str(root), "--no-update-check"],
                                      stdout=log, stderr=log)
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
                assert entry()["state"] == "unloaded"
                assert request("/v1/load-model", {"model": "org/broken"})[0] >= 400
                assert entry()["state"] == "error"
                weights.write_bytes(b"still invalid, but changed")
                assert request("/v1/models/rescan", {}) == (200, {"added": 0})
                current = entry()
                assert current["state"] == "unloaded", current
                assert current["bytes_on_disk"] == weights.stat().st_size, current
            except BaseException:
                log.flush()
                log.seek(0)
                print(log.read(), file=__import__("sys").stderr)
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

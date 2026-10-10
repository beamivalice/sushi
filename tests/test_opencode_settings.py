#!/usr/bin/env python3
"""Test effort selection and persistent settings against a hermetic models API."""
import argparse
import http.server
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import threading

MODELS = [
    {"id": "qwen", "loaded": True, "capabilities": ["chat"], "context_length": 248000,
     "reasoning_efforts": ["off", "low", "medium", "xhigh"]},
    {"id": "glm", "loaded": False, "capabilities": ["chat"], "context_length": 1048576,
     "reasoning_efforts": ["low", "high", "max"]},
]


class Fixture(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"data": MODELS} if self.path == "/v1/models" else {"status": "ok"}).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", default="zig-out/bin/sushi")
    options = parser.parse_args()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with tempfile.TemporaryDirectory(prefix="opencode-settings-") as temp:
            root = Path(temp)
            env = dict(os.environ, XDG_CONFIG_HOME=str(root))
            command = [str(Path(options.bin).resolve()), "launch", "opencode", "--url",
                       f"http://127.0.0.1:{server.server_port}", "--model", "qwen", "--print"]

            def launch(*args):
                return subprocess.run(command + list(args), env=env, capture_output=True, text=True, timeout=30)

            def config(result):
                assert result.returncode == 0, result.stderr
                line = next(line for line in result.stdout.splitlines() if line.startswith("export OPENCODE_CONFIG_CONTENT="))
                return json.loads(shlex.split(line)[1].split("=", 1)[1])

            for effort in ["off", "low", "medium", "xhigh"]:
                models = config(launch("--think", effort))["provider"]["sushi"]["models"]
                assert models["qwen"]["options"] == {"reasoningEffort": effort}
                assert "options" not in models["glm"]
            assert "options" not in config(launch())["provider"]["sushi"]["models"]["qwen"]
            assert launch("--think", "high").returncode != 0
            assert not (root / "opencode").exists()
            directory = root / "opencode"
            directory.mkdir()
            path = directory / "opencode.jsonc"
            original = ('{ // retained in backup\n"plugin":["./plugins/custom.js",],'
                        '"provider":{"other":{"name":"keep"},"sushi":{"models":{'
                        '"qwen":{"options":{"reasoningEffort":"medium","temperature":0.1}},'
                        '"glm":{"limit":{"context":248000,"output":32768}}}}}}')
            path.write_text(original)
            result = launch("--persist", "--think", "xhigh")
            assert result.returncode == 0, result.stderr
            saved = json.loads(path.read_text())
            models = saved["provider"]["sushi"]["models"]
            assert saved["provider"]["other"] == {"name": "keep"}
            assert saved["plugin"] == ["./plugins/custom.js"]
            assert saved["model"] == "sushi/qwen"
            assert models["qwen"]["options"] == {"reasoningEffort": "xhigh", "temperature": 0.1}
            assert models["glm"]["limit"]["context"] == 248000
            assert models["glm"]["limit"]["output"] <= 32768
            backups = list(directory.glob("*.sushi-backup-*"))
            assert len(backups) == 1 and backups[0].read_text() == original
            assert path.stat().st_mode & 0o777 == 0o600
            before = path.read_bytes()
            assert launch("--persist").returncode == 0
            assert path.read_bytes() == before and len(list(directory.glob("*.sushi-backup-*"))) == 1
            assert launch("--persist", "--think", "high").returncode != 0
            assert path.read_bytes() == before
            path.write_text("{ broken")
            assert launch("--persist").returncode != 0 and path.read_text() == "{ broken"
            path.write_text(original)
            other = directory / "opencode.json"
            other.write_text("{}")
            assert launch("--persist").returncode != 0 and path.read_text() == original
            other.unlink()
            path.unlink()
            assert launch("--persist", "--think", "low").returncode == 0
            assert json.loads(other.read_text())["provider"]["sushi"]["models"]["qwen"]["options"]["reasoningEffort"] == "low"
        print("PASS: effort validation, opt-in save, merge, backup, idempotence, invalid and ambiguous configs")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()

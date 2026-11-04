#!/usr/bin/env python3
# sushi-test: tier=host
"""CPU-only `sushi launch dsh` contract against a stub /v1/models server.

python3 tests/test_dsh_launch.py --bin <sushi>
No model, GPU or network beyond a loopback stub; HOME is a temp dir. Checks the generated patch
layer, that the user's own dsh patch layer is left byte-identical, and the script that starts dsh.
"""
import argparse
import hashlib
import http.server
import json
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

MODELS = [
    {"id": "DeepSeek \"V4.1\"", "capabilities": ["chat", "vision"], "loaded": True,
     "meta": {"context_length": 1048576}, "reasoning_efforts": ["off", "low", "high", "max"],
     "default_reasoning_effort": "high"},
    {"id": "text-only", "capabilities": ["chat"], "context_length": 24576},
    {"id": "embedding-only", "capabilities": ["embeddings"]},
]
USER_LAYER = "- id: ui-settings-general\n  name: x\n  config:\n    welcomeNoticeVersion: 1\n"


class Stub(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        payload = {"data": MODELS} if self.path == "/v1/models" else {"status": "ok"}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    binary = os.path.abspath(ap.parse_args().bin)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp) / "home"
        layer = home / ".dsh/profiles/web/cordis.patch.yml"
        layer.parent.mkdir(parents=True)
        layer.write_text(USER_LAYER)
        before = digest(layer)
        env = {**os.environ, "HOME": str(home)}

        printed = subprocess.run([binary, "launch", "dsh", "--url", url, "--print", "--", "--no-open"],
                                 env=env, text=True, capture_output=True, timeout=60)
        assert printed.returncode == 0, printed.stderr
        assert '--profile web --patch "$HOME/.sushi/dsh/patch.yml" \'--no-open\'' in printed.stdout, printed.stdout
        assert digest(layer) == before, "the user's patch layer was modified"
        assert not any(p.name != "cordis.patch.yml" for p in layer.parent.iterdir() if p.is_file()), "stray file in the profile"

        patch = (home / ".sushi/dsh/patch.yml").read_text()
        for want in (f'baseURL: "{url}/v1"', 'Authorization: "Bearer sushi"', "contextWindow: 1048576",
                     "maxTokens: 256000", "input: [text, image]", "contextWindow: 24576", "maxTokens: 12288",
                     "input: [text]", "reasoning: high", '"max": "max"'):
            assert want in patch, (want, patch)
        assert "embedding-only" not in patch, patch

        if shutil.which("ruby"):
            parsed = json.loads(subprocess.run(
                ["ruby", "-ryaml", "-rjson", "-e", "puts JSON.generate(YAML.safe_load(File.read(ARGV[0])))", str(home / ".sushi/dsh/patch.yml")],
                text=True, capture_output=True, check=True).stdout)
            assert [e["id"] for e in parsed] == ["agent-default-model", "llm-pi-ai"], parsed
            assert parsed[0]["config"]["model"] == MODELS[0]["id"], parsed[0]
            ids = [m["id"] for m in parsed[1]["config"]["providers"]["sushi"]["models"]]
            assert ids == [MODELS[0]["id"], "text-only"], ids
            efforts = parsed[1]["config"]["providers"]["sushi"]["models"][0]["reasoningEfforts"]
            assert efforts == {"off": "off", "low": "low", "high": "high", "max": "max"}, efforts

        bindir = Path(tmp) / "bin"
        bindir.mkdir()
        log = Path(tmp) / "dsh.args"
        fake = bindir / "dsh"
        fake.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {log}\n')
        fake.chmod(0o755)
        ran = subprocess.run([binary, "launch", "dsh", "--url", url, "--", "--no-open"],
                             env={**env, "PATH": f"{bindir}:{os.environ['PATH']}"}, text=True,
                             capture_output=True, timeout=60)
        assert ran.returncode == 0, ran.stderr
        assert log.read_text().split("\n")[:-1] == ["--profile", "web", "--patch", str(home / ".sushi/dsh/patch.yml"), "--no-open"], log.read_text()
        assert digest(layer) == before


if __name__ == "__main__":
    main()

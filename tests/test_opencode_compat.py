#!/usr/bin/env python3
"""Exercise generated launch scripts against both OpenCode CLI flag contracts."""
import argparse
import http.server
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import threading


class Fixture(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        data = {"data": [{"id": "fixture", "capabilities": ["chat"], "loaded": True,
                          "context_length": 32768}]} if self.path == "/v1/models" else {"status": "ok"}
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", default="zig-out/bin/sushi")
    options = parser.parse_args()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with tempfile.TemporaryDirectory(prefix="opencode-compat-") as temp:
            stub = Path(temp) / "opencode"
            stub.write_text("#!/bin/sh\nif [ \"$1\" = --help ]; then\n"
                            "  [ \"$FIXTURE_STANDALONE\" = 1 ] && echo --standalone\n  exit 0\nfi\n"
                            "python3 -c 'import sys,json;print(json.dumps(sys.argv[1:]))' \"$@\"\n")
            stub.chmod(0o755)
            for extras in [[], ["run", "it's a prompt"]]:
                command = [str(Path(options.bin).resolve()), "launch", "opencode", "--url", base, "--print"]
                if extras:
                    command += ["--", *extras]
                result = subprocess.run(command, text=True, capture_output=True, timeout=30)
                assert result.returncode == 0, result.stderr
                for support in ["0", "1"]:
                    env = dict(os.environ, PATH=temp + os.pathsep + os.environ["PATH"], FIXTURE_STANDALONE=support)
                    run = subprocess.run(["/bin/zsh", "-c", result.stdout], env=env, text=True, capture_output=True, timeout=30)
                    assert run.returncode == 0, run.stderr
                    expected = (["run"] if extras else []) + (["--standalone"] if support == "1" else []) + extras[1:]
                    assert json.loads(run.stdout) == expected, (support, run.stdout, expected)
        print("PASS: OpenCode with/without standalone, default command and subcommand")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()

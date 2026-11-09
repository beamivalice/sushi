#!/usr/bin/env python3
# sushi-test: tier=host
"""Launch-flag refusals, subcommand help and the request framing of a model-less server.

No weights load and nothing listens on a port the owner uses: every server here is headless over an empty
--model-dir on a port above 18000, with a private HOME.
"""
import http.server
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time

binary = str(Path(os.environ.get("SUSHI_BIN", "zig-out/bin/sushi")).resolve())
root = Path(tempfile.mkdtemp(prefix="sushi-cli-flags-"))
home = root / "home"
(home / ".sushi").mkdir(parents=True)
(home / "models").mkdir()
env = dict(os.environ, HOME=str(home), SUSHI_PREFIX_CACHE_DIR=str(root / "kv"))
env.pop("SUSHI_API_KEY_TEST_UNSET", None)
env["SUSHI_API_KEY_TEST_SET"] = "k"
failures = []


def check(cond, what, detail=""):
    if not cond:
        failures.append(f"{what}: {detail}")


def run(*args, timeout=60, cwd=None):
    result = subprocess.run([binary, *args], env=env, capture_output=True, timeout=timeout, cwd=cwd or root)
    return result.returncode, result.stdout.decode(errors="replace"), result.stderr.decode(errors="replace")


def free_port(start):
    for port in range(start, start + 200):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port")


# A flag's value must be one it can take, and a refusal comes before anything starts.
refusals = [
    (("serve", "--port", "70000"), "--port: expected an integer in 1..65535"),
    (("serve", "--port", "0"), "--port: expected an integer in 1..65535"),
    (("serve", "--port", "-1"), "--port: expected an integer in 1..65535"),
    (("serve", "--temp", "-1"), "--temp: expected a number of at least 0"),
    (("serve", "--top-p", "2"), "--top-p: expected a number above 0 and at most 1"),
    (("serve", "--top-p", "0"), "--top-p: expected a number above 0 and at most 1"),
    (("serve", "--max-tokens", "0"), "--max-tokens: expected a whole number of at least 1"),
    (("serve", "--api-key", "--port", "18901"), "expects a value"),
    (("serve", "--model-dir", "--port", "18901"), "expects a value"),
    (("serve", "--log-file", "--port", "18901"), "expects a value"),
    (("serve", "--port", "18901", "--api-key-env", "SUSHI_API_KEY_TEST_UNSET"), "SUSHI_API_KEY_TEST_UNSET"),
    (("serve", "--mtp-typical", "x"), "--mtp-typical: expected a number above 0; got 'x'"),
    (("serve", "--mtp-tokenv3", "2"), "--mtp-tokenv3: expected a number from 0 to 1; got '2'"),
    (("serve", "--mtp-typical", "0.2", "--mtp-tokenv3", "0.5"), "conflict; pass one"),
    (("serve", "--mtp", "--no-mtp"), "--mtp and --no-mtp conflict; pass one"),
    (("serve", "--no-mtp", "--mtp"), "--mtp and --no-mtp conflict; pass one"),
    (("serve", "--api-key", ""), "--api-key: the key is empty"),
    (("serve", "--max-concurrent", "0"), "--max-concurrent: expected a whole number of at least 1"),
    (("serve", "--prefill-decode-share", "0.95"), "--prefill-decode-share: expected a number from 0 to 0.9"),
    (("serve", "--prefill-decode-share", "inf"), "--prefill-decode-share: expected a number from 0 to 0.9"),
    (("repack",), "a pack is required"),
]
for args, message in refusals:
    code, out, err = run(*args)
    check(code != 0 and message in err, f"refusal {args}", f"rc={code} {err[-300:]}")
    check(not any(name in err for name in ("InvalidAcceptanceValue", "ConflictingAcceptanceModes", "UnknownSubcommand")), f"no raw error name {args}", err[-300:])

# A set variable passes the key check; the bad --temp after it is what stops the run.
code, _, err = run("serve", "--api-key-env", "SUSHI_API_KEY_TEST_SET", "--temp", "-1")
check(code != 0 and "--temp" in err and "--api-key-env" not in err, "a set --api-key-env is accepted", err[-300:])
check(not (root / "--port 18901").exists(), "--log-file did not take --port as its path")



def registry_cap(*flags):
    """The `[registry] ... max_resident_mem=` field a model-less server logs at boot, then the server is stopped."""
    port = free_port(18971)
    log_path = root / f"registry-{port}.log"
    with open(log_path, "wb") as sink:
        proc = subprocess.Popen(
            [binary, "serve", "--model-dir", str(home / "models"), "--port", str(port), "--no-update-check", *flags],
            env=env, stdout=sink, stderr=sink,
        )
    try:
        deadline = time.time() + 30
        while time.time() < deadline and proc.poll() is None:
            text = log_path.read_text(errors="replace")
            if "max_resident_mem=" in text:
                return text.split("max_resident_mem=", 1)[1].split()[0]
            time.sleep(0.25)
        return log_path.read_text(errors="replace")[-300:]
    finally:
        proc.terminate()
        proc.wait(timeout=30)


# `auto` means auto wherever it stands: after a size it must not become the 0 that disables the cap.
for flags, want in [
    (("--max-resident-mem", "auto"), "GB"),
    (("--max-resident-mem", "1GB", "--max-resident-mem", "auto"), "GB"),
    (("--max-resident-mem", "1GB"), "1.0"),
    (("--max-resident-mem", "0"), "unlimited"),
]:
    got = registry_cap(*flags)
    if want == "GB":
        check(got.replace(".", "").isdigit() and got not in ("1.0", "0.0"), f"{flags} is the machine's auto cap", got)
    else:
        check(got.startswith(want), f"{flags} is {want}", got)

# A subcommand's own errors are sentences, and its --help is usage on stdout.
for args, message in [
    (("kld", "bogus"), "unknown subcommand"),
    (("convert", "bogus"), "unknown subcommand"),
    (("bench", "--bogus"), "--bogus: unrecognized argument"),
]:
    code, out, err = run(*args)
    text = out + err
    check(code != 0 and message in text and "UnknownSubcommand" not in text and "UnknownArgument" not in text, f"error text {args}", text[-300:])
for args, needle in [
    (("run", "--help"), "Usage: sushi"),
    (("pull", "--help"), "usage: sushi pull"),
    (("list", "-h"), "usage: sushi list"),
    (("launch", "-h"), "usage: sushi launch"),
    (("launch", "claude", "--help"), "usage: sushi launch"),
    (("repack", "--help"), "usage: sushi repack"),
    (("repack", "-h"), "usage: sushi repack"),
]:
    code, out, err = run(*args)
    check(code == 0 and needle in out, f"help {args}", f"rc={code} out={out[:80]!r} err={err[-200:]}")

# A model path that does not exist is named; the default models folder being absent says nothing.
port = free_port(18911)
code, out, err = run("serve", "--model", str(root / "no-such-model"), "--port", str(port), "--log-file", "off")
check(code != 0 and "no-such-model" in err and "FileNotFound" not in err, "missing --model is named", err[-400:])
check("--model-dir scan failed" not in err, "no scan warning for the absent default folder", err[-400:])

# kld compare reads its fixture before it loads the model: the model folder here is empty, so a load would fail first.
code, out, err = run("kld", "compare", "--model", str(root / "empty-model"), "--fixture", str(root / "no-fixture"))
check(code != 0 and "--fixture" in err + out and "no-fixture" in err + out and "BaselineUnreadable" not in err + out, "kld fixture checked first", err[-300:] + out[-300:])
check("Loading" not in err + out and "cannot read the model" not in err + out, "kld did not start a model load", err[-300:] + out[-300:])


# launch asks the server it is pointed at, never a fixed port: a fake listener answers, and the default port comes
# from the settings file.
class Fake(http.server.BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *_):
        pass

    def do_GET(self):
        Fake.seen.append(self.path)
        if self.path == "/health":
            body = b'{"status":"ok"}'
        elif self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "fake-model", "capabilities": ["chat"], "meta": {"context_length": 65536}}]}).encode()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


fake_port = free_port(18921)
fake = http.server.ThreadingHTTPServer(("127.0.0.1", fake_port), Fake)
threading.Thread(target=fake.serve_forever, daemon=True).start()
code, out, err = run("launch", "claude", "--print", "--port", str(fake_port))
check(code == 0 and f"http://127.0.0.1:{fake_port}" in out and "fake-model" in out, "launch --print on --port", f"rc={code} {err[-300:]}")
(home / ".sushi" / "model-settings.json").write_text(json.dumps({"server": {"port": fake_port}}))
Fake.seen.clear()
code, out, err = run("launch", "claude", "--print", "--model", "fake-model")
check(code == 0 and f"http://127.0.0.1:{fake_port}" in out and "/health" in Fake.seen, "launch with no port follows server.port of the settings", f"rc={code} {err[-300:]}")
fake.shutdown()
dead = free_port(18951)
code, out, err = run("launch", "claude", "--print", "--port", str(dead))
check(code == 1 and "--port" in err and "--url" in err and str(dead) in err, "a dead launch target names how to redirect it", err[-300:])
(home / ".sushi" / "model-settings.json").unlink()


# Request framing against a server with no model.
def read_response(sock):
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            return data
        data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    length = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":")[1])
    while len(body) < length:
        chunk = sock.recv(4096)
        if not chunk:
            break
        body += chunk
    return head + b"\r\n\r\n" + body


port = free_port(18931)
assert port not in (11250, 12345)
server = subprocess.Popen(
    [binary, "serve", "--model-dir", str(home / "models"), "--port", str(port), "--log-file", "off", "--no-update-check"],
    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
try:
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            break
        except OSError:
            time.sleep(0.25)
    body = b'{"repo":"other/Pack"}'

    def connect():
        s = socket.create_connection(("127.0.0.1", port), timeout=10)
        return s

    # Every answer says Connection: close, and a client that reuses the socket anyway sees a clean end, not a reset.
    with connect() as s:
        s.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: keep-alive\r\n\r\n")
        first = read_response(s)
        check(first.startswith(b"HTTP/1.1 200") and b"\r\nConnection: close\r\n" in first, "health answers Connection: close", first[:200])
        try:
            s.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
            second = s.recv(4096)
            check(second == b"", "a reused socket ends with EOF", second[:100])
        except ConnectionResetError:
            failures.append("a reused socket was reset")

    # A chunked body is decoded like its Content-Length twin.
    with connect() as s:
        wire = b"".join(b"%x\r\n%s\r\n" % (len(body[i:i + 5]), body[i:i + 5]) for i in range(0, len(body), 5)) + b"0\r\n\r\n"
        s.sendall(b"POST /v1/downloads HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n" + wire)
        reply = read_response(s)
        check(reply.startswith(b"HTTP/1.1 400") and b"beamster only" in reply, "chunked body decoded", reply[:300])
    with connect() as s:
        s.sendall(b"POST /v1/downloads HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: gzip\r\n\r\n")
        reply = read_response(s)
        check(reply.startswith(b"HTTP/1.1 501"), "an unsupported transfer coding is refused", reply[:100])

    # Expect: 100-continue is answered before the body is sent.
    with connect() as s:
        s.sendall(b"POST /v1/downloads HTTP/1.1\r\nHost: x\r\nExpect: 100-continue\r\nContent-Length: %d\r\n\r\n" % len(body))
        s.settimeout(3)
        interim = s.recv(4096)
        check(interim == b"HTTP/1.1 100 Continue\r\n\r\n", "100 Continue sent before the body", interim)
        s.sendall(body)
        reply = read_response(s)
        check(reply.startswith(b"HTTP/1.1 400") and b"beamster only" in reply, "body after 100 Continue is served", reply[:200])
    with connect() as s:
        s.sendall(b"POST /v1/downloads HTTP/1.1\r\nHost: x\r\nExpect: 200-ok\r\nContent-Length: %d\r\n\r\n" % len(body))
        reply = read_response(s)
        check(reply.startswith(b"HTTP/1.1 417"), "an unknown expectation is a 417", reply[:100])
finally:
    server.terminate()
    try:
        server.wait(timeout=20)
    except subprocess.TimeoutExpired:
        server.kill()

if failures:
    print("\n".join("FAIL: " + f for f in failures), file=sys.stderr)
    sys.exit(1)

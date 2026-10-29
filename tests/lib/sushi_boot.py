#!/usr/bin/env python3
# sushi-test: tier=lib
"""The Python twin of tests/lib/boot.sh for `boot=own` Python scripts: the GPU lock, a private HOME and cache root,
a free port, `taskpolicy -a`, a bounded wait for `Model ready`, a stop by PID and the <label>.END marker."""
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from shutil import which

EXIT_SKIP = 3
ROOT = Path(os.environ.get("SUSHI_ROOT", Path(__file__).resolve().parents[2]))
BIN = Path(os.environ.get("SUSHI_BIN", ROOT / "zig-out/bin/sushi"))
LOCK = ROOT / "scripts/gpu-lock.sh"


def skip(reason):
    print(f"SKIP {os.path.basename(sys.argv[0])}: {reason}")
    sys.exit(EXIT_SKIP)


def pack(key, required=True):
    """The pack dir for a model key, alias or path (boot.sh's table); SKIP when it is missing, or None if not required."""
    r = subprocess.run(["/bin/bash", "-c", f'. "{ROOT}/tests/lib/boot.sh"; sushi_pack "$1"', "_", key],
                       capture_output=True, text=True)
    if r.returncode == EXIT_SKIP:
        if not required:
            return None
        skip(r.stderr.strip().removeprefix("SKIP: "))
    r.check_returncode()
    return Path(r.stdout.strip())


def run_dir():
    d = os.environ.get("SUSHI_RUN_DIR") or str(Path.home() / ".sushi/runs" / f"tests-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}")
    os.environ["SUSHI_RUN_DIR"] = d
    Path(d).mkdir(parents=True, exist_ok=True)
    return Path(d)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def gpu_lock(label):
    owner = f"{os.environ.get('GPU_LOCK_OWNER', 'tests')}-{label}"
    subprocess.run([str(LOCK), "acquire", owner], check=True)
    try:
        yield owner
    finally:
        subprocess.run([str(LOCK), "release", owner], check=False)


def private_env(label):
    """An environment with a private HOME and prefix-cache root under the run dir, the way sushi_boot gives one."""
    home = run_dir() / f"{label}.home"
    cache = run_dir() / f"{label}.kv-cache"
    (home / ".sushi").mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    return {**os.environ, "HOME": str(home), "SUSHI_PREFIX_CACHE_DIR": str(cache)}


def request(base, route, body=None, timeout=180):
    """(status, json) of a GET or POST; an HTTP error is a status, not an exception."""
    import json
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + route, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


class Server:
    """One sushi server under the lock. `model` boots one pack, `model_dir` serves a directory headless; flags follow."""

    def __init__(self, label, model=None, model_dir=None, flags=(), env=None):
        self.label = label
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log = run_dir() / f"{label}.log"
        self.env = env if env is not None else private_env(label)
        argv = [str(BIN), "--serve"]
        argv += ["--model", str(model)] if model else []
        argv += ["--model-dir", str(model_dir)] if model_dir else []
        argv += ["--host", "127.0.0.1", "--port", str(self.port), "--log-file", "off"]
        argv += list(flags)
        self.argv = (["taskpolicy", "-a"] if which("taskpolicy") else []) + argv
        self.proc = None
        self._lock = None

    def start(self):
        if not BIN.is_file():
            raise AssertionError(f"{BIN} missing: zig build -Doptimize=ReleaseFast")
        self._lock = gpu_lock(self.label)
        self._lock.__enter__()
        self.offset = self.log.stat().st_size if self.log.exists() else 0
        with self.log.open("ab") as log:
            self.proc = subprocess.Popen(self.argv, stdout=log, stderr=subprocess.STDOUT, env=self.env)
        return self

    def log_text(self):
        with self.log.open("rb") as f:
            f.seek(self.offset)
            return f.read().decode("utf-8", "replace")

    def wait_ready(self, timeout=None):
        """Until /health answers and the log says `Model ready`; a headless server has no model, so /health alone."""
        timeout = timeout or float(os.environ.get("SUSHI_BOOT_TIMEOUT_S", "300"))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise AssertionError(f"{self.label}: server died during boot; log {self.log}")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2):
                    if "Model ready" in self.log_text() or "--model" not in self.argv:
                        return self
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
            time.sleep(0.25)
        raise AssertionError(f"{self.label}: not ready after {timeout:.0f}s; log {self.log}")

    def wait_exit(self, timeout):
        """The exit status of a server that is expected to refuse and leave."""
        return self.proc.wait(timeout=timeout)

    def stop(self, status=0, gone_ok=False):
        """TERM, then KILL after the wait; releases the lock and writes <label>.END. A server already gone is a crash
        unless the script expected it to leave (a refusal)."""
        if self.proc is None:
            return
        state = "clean"
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=float(os.environ.get("SUSHI_STOP_WAIT_S", "60")))
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
                state = "killed"
        elif gone_ok:
            state = "gone"
        elif self.proc.returncode != 0:
            state = "crash"
            print(f"[boot] CRASH: server '{self.label}' was already gone, exit status {self.proc.returncode}; log: {self.log}",
                  file=sys.stderr)
        (run_dir() / f"{self.label}.END").write_text(f"exit={status} server={state}\n")
        self.proc = None
        self._lock.__exit__(None, None, None)

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, *_):
        self.stop(1 if exc_type else 0)

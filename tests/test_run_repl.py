#!/usr/bin/env python3
# sushi-test: tier=live boot=own pack=default boots=1
"""Exercise `sushi run` on a real TTY: startup, one chat turn and /bye. `run` starts its own server, so the script
holds the GPU lock and a private HOME around it instead of calling Server."""
import os
import pathlib
import pty
import re
import select
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib"))
import sushi_boot as b  # noqa: E402


def main():
    model = b.pack("qwen4")
    out = b.run_dir() / "repl"
    out.mkdir(parents=True, exist_ok=True)
    port = b.free_port()
    command = [
        str(b.BIN), "run", str(model.resolve()),
        "--host", "127.0.0.1", "--port", str(port),
        "--kv-quant", "8", "--ctx-size", "4096", "--prefill-chunk", "512",
        "--max-tokens", "32", "--no-mtp", "--no-vision",
        "--prefix-cache-entries", "0", "--log-file", str(out / "server.log"),
    ]
    transcript = bytearray()
    sent = finished = False
    with b.gpu_lock("repl"):
        master, slave = pty.openpty()
        child = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave,
                                 env=b.private_env("repl"), start_new_session=True)
        os.close(slave)
        deadline = time.monotonic() + float(os.environ.get("SUSHI_BOOT_TIMEOUT_S", "600"))
        try:
            while time.monotonic() < deadline:
                readable, _, _ = select.select([master], [], [], 0.25)
                if readable:
                    try:
                        data = os.read(master, 65536)
                    except OSError:
                        break
                    if not data:
                        break
                    transcript.extend(data)
                    if not sent and b"chat is live" in transcript:
                        os.write(master, b"Reply with one short greeting.\n")
                        sent = True
                    if sent and not finished and re.search(rb"\[\d+ tokens, [\d.]+ tok/s\]", transcript):
                        os.write(master, b"/bye\n")
                        finished = True
                if child.poll() is not None:
                    break
        finally:
            if child.poll() is None and finished:
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            os.close(master)
            (out / "transcript.txt").write_bytes(transcript)
            (b.run_dir() / "repl.END").write_text(f"exit={child.returncode}\n")
    if not sent or not finished or child.returncode != 0:
        print(f"FAIL: chat_ready={sent}, reply_completed={finished}, exit={child.returncode}; diagnostics: {out}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

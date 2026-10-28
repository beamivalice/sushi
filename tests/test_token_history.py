#!/usr/bin/env python3
"""Publication/parity and optional B-before-A overlap against a caller-owned server and private writable store."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
from pathlib import Path
import re
import stat
import struct
import threading
import time
import urllib.request
from urllib.parse import urlsplit

SEPARATOR = 1 << 30


def post(url, path, body):
    request = urllib.request.Request(url.rstrip("/") + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=180)


def snapshot(store, identity):
    lock = store / "store.lock"
    payload = store / "tokens.bin"
    if not lock.exists() or not payload.exists():
        return []
    for path, mode in ((store, 0o700), (lock, 0o600), (payload, 0o600)):
        info = path.lstat()
        assert not stat.S_ISLNK(info.st_mode), f"symlink in private store: {path.name}"
        assert stat.S_IMODE(info.st_mode) == mode, f"non-private {path.name}"
        if path != store:
            assert info.st_nlink == 1, f"hard-linked {path.name}"
    with lock.open("rb") as held:
        fcntl.flock(held, fcntl.LOCK_SH)
        data = payload.read_bytes()
    assert data[:8] == b"MLXLOOK1" and data[8:40] == identity, "wrong tokenizer/store header"
    assert len(data) >= 40 and (len(data) - 40) % 4 == 0, "truncated store"
    documents, current = [], []
    for (token,) in struct.iter_unpack("<I", data[40:]):
        if token == SEPARATOR:
            assert current, "empty document"
            documents.append(current)
            current = []
        else:
            assert token < SEPARATOR, "invalid token"
            current.append(token)
    assert not current, "unterminated document"
    return documents


def overlap(args, identity):
    """Seed a real history lease, then publish B while A continues using that lease."""
    prompts = {
        "A": (
            "Continue this Python module, defining similar vector functions.\n\n"
            "def add(a, b):\n    return [x + y for x, y in zip(a, b)]\n\n"
            "def subtract(a, b):\n    return [x - y for x, y in zip(a, b)]\n\n"
            "def multiply(a, b):\n"
        ),
        "B": "List the integers from 1 through 80, separated by commas.\n1, 2, 3,",
    }
    tokenizer = json.loads((args.model_dir / "tokenizer.json").read_bytes())
    special_ids = {token["id"] for token in tokenizer.get("added_tokens", []) if token.get("special") is True}

    def api(path, body):
        with post(args.url, path, body) as response:
            return json.load(response)

    prompt_ids = {label: api("/tokenize", {"model": args.model, "content": prompt})["tokens"]
                  for label, prompt in prompts.items()}
    assert all(len(ids) + 192 <= 4096 for ids in prompt_ids.values()), "fixture exceeds the 4096-token context budget"
    log_start = args.server_log.stat().st_size

    def log_text():
        with args.server_log.open("rb") as log:
            assert log.seek(0, 2) >= log_start, "server log truncated during the test"
            log.seek(log_start)
            return log.read().decode(errors="replace")

    def history_rounds(text):
        return len(re.findall(r"\[mtp-round\][^\n]* source=history(?:\s|$)", text))

    def documents_when(count):
        deadline = time.monotonic() + 10
        while True:
            documents = snapshot(args.store, identity)
            if len(documents) >= count or time.monotonic() >= deadline:
                assert len(documents) == count, "missing or duplicate terminal publication"
                return documents
            time.sleep(0.02)

    emitted = threading.Event()

    def stream(name, label, limit, signal=False):
        body = {
            "model": args.model, "prompt": prompt_ids[label], "max_tokens": limit,
            "temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0, "repeat_penalty": 1,
            "enable_mtp": True, "enable_drafter": False, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True},
        }
        result = {"name": name, "label": label, "limit": limit, "started": time.monotonic(),
                  "events": [], "text": "", "done": False}
        chunks = 0
        try:
            with post(args.url, "/v1/completions", body) as response:
                for line in response:
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        result["done"] = True
                        break
                    event = json.loads(data)
                    assert not event.get("error"), event
                    now = time.monotonic()
                    result["events"].append({"at": now, "event": event})
                    if event.get("usage") is not None:
                        result["usage"] = event["usage"]
                    for choice in event.get("choices", []):
                        text = choice.get("text") or ""
                        if text:
                            result["text"] += text
                            result.setdefault("first_emitted", now)
                            result["last_emitted"] = now
                            chunks += 1
                            if signal and chunks >= 8:
                                emitted.set()
                        if choice.get("finish_reason") is not None:
                            result["finish"] = choice["finish_reason"]
            assert result["done"] and "usage" in result, f"{name}: incomplete SSE response"
            assert result.get("finish") == "length", f"UNEXERCISED: {name} did not reach its length cap"
            assert result["usage"]["completion_tokens"] == limit, f"{name}: wrong emitted-token count"
            assert result["text"], f"UNEXERCISED: {name} emitted no visible text"
            return result
        except Exception as error:
            result["error"] = str(error)
            raise
        finally:
            result["finished"] = time.monotonic()
            (args.output / f"history-{name}.json").write_text(json.dumps(result, indent=2) + "\n")

    seed = stream("seed", "A", 96)
    documents_when(1)
    a_log_start = len(log_text())
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(stream, "A", "A", 192, True)
            deadline = time.monotonic() + 180
            while not emitted.is_set() or not history_rounds(log_text()[a_log_start:]):
                if future_a.done():
                    future_a.result()
                    raise AssertionError("UNEXERCISED: A ended before emitting eight chunks and borrowing history; use --log-level debug and enabled MTP lookup")
                assert time.monotonic() < deadline, "UNEXERCISED: A did not emit and acquire history before the overlap deadline"
                if emitted.is_set():
                    time.sleep(0.02)
                else:
                    emitted.wait(0.02)
            assert not future_a.done(), "UNEXERCISED: A already ended before B could start"
            future_b = pool.submit(stream, "B", "B", 24)
            b = future_b.result()
            assert not future_a.done(), "UNEXERCISED: B did not finish before A"
            intermediate = documents_when(2)
            observed_b = time.monotonic()
            assert not future_a.done(), "UNEXERCISED: B's durable publication was not observed while A was active"
            a = future_a.result()

        assert a["first_emitted"] < b["started"] < b["finished"] < a["last_emitted"] <= a["finished"], "UNEXERCISED: A did not emit both before B started and after B finished"
        assert observed_b < a["last_emitted"], "UNEXERCISED: A emitted no output after B's durable publication"
        request_overlap = min(a["finished"], b["finished"]) - max(a["started"], b["started"])
        emission_overlap = min(a["last_emitted"], b["last_emitted"]) - max(a["first_emitted"], b["first_emitted"])
        assert request_overlap > 0 and emission_overlap > 0, "UNEXERCISED: request/emission intervals did not overlap"
        documents = documents_when(3)
        assert documents[:2] == intermediate, "A's append lost or rewrote the seed/B documents"
        checked = []
        for document, record in zip(documents, (seed, b, a)):
            prefix = prompt_ids[record["label"]]
            assert document[:len(prefix)] == prefix, f"{record['name']}: wrong prompt or publication order"
            output = document[len(prefix):]
            assert len(output) == record["limit"], f"{record['name']}: unseen speculative tail or missing emitted tokens"
            # ignore_eos completions count all emitted IDs but hide special:true tokens on the wire.
            visible = [token for token in output if token not in special_ids]
            assert api("/detokenize", {"model": args.model, "tokens": visible})["content"] == record["text"], f"{record['name']}: durable output differs from the stream"
            checked.append({"name": record["name"], "prompt_tokens": len(prefix), "output_tokens": len(output), "hidden_special_tokens": len(output) - len(visible)})
        log = log_text()
        native = re.findall(r"\[spec-stats\] mode=mtp attempts=\d+", log)
        assert len(native) == 3, "UNEXERCISED: native MTP mode did not complete all three requests, or another client used this server"
        publication = f"[history] published prompt_tokens={len(prompt_ids['B'])} output_tokens=24"
        assert log.count(publication) == 1, "B's successful publication was not logged exactly once"
        after_b = log.split(publication, 1)[1]
        assert not any(message in log for message in ("StoreBusy", "[history] proposal tier refused:", "[history] publication failed:", "[history] publication refused:")), "history tier refused or failed; inspect history-server.log"
        summary = {
            "documents": checked, "finish_order": ["B", "A"],
            "request_overlap_seconds": request_overlap, "emission_overlap_seconds": emission_overlap,
            "b_publication_observed": observed_b, "a_history_rounds_after_b": history_rounds(after_b),
            "post_publication_history_verification_exercised": bool(history_rounds(after_b)),
            "requests": [{key: record[key] for key in ("name", "started", "first_emitted", "last_emitted", "finished", "limit", "finish")} for record in (seed, b, a)],
            "scope": "HTTP overlap, exact publication and lease teardown; post-publication verified lookup is covered only when selected, not semantic answers, universal serial equivalence or throughput",
        }
        (args.output / "history-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("PASS: overlapping native-MTP streams retain exact seed/B/A documents; post-publication history selection is reported separately")
    finally:
        (args.output / "history-server.log").write_text(log_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="sushi")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True, help="Isolated initially empty lookup_store, explicitly configured writable by the caller")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, help="Raw prompt already rendered with the checkpoint's chat template; default is Qwen with thinking off")
    parser.add_argument("--overlap", action="store_true", help="Cold seed96 then overlapping A192/B24 streams; B must publish before A and A must retain usable history")
    parser.add_argument("--server-log", type=Path, help="Caller-owned dedicated server log, required with --overlap; start server with --log-level debug")
    args = parser.parse_args()
    url = urlsplit(args.url)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost") or not url.port or url.username or url.password or url.path not in ("", "/") or url.query or url.fragment:
        parser.error("use an owned plain-HTTP loopback server with an explicit port inside a coordinated GPU reservation")
    if args.overlap and (args.server_log is None or args.prompt_file is not None):
        parser.error("--overlap requires --server-log and uses its own bounded raw continuation prompts, not --prompt-file")
    identity = hashlib.sha256((args.model_dir / "tokenizer.json").read_bytes()).digest()
    assert not snapshot(args.store, identity), "refusing a populated history store"
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.overlap:
        overlap(args, identity)
        return
    prompt = args.prompt_file.read_text() if args.prompt_file else (
        "<|im_start|>user\nWrite a Python function named sum_squares that returns the sum of squared integers in a list."
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    with post(args.url, "/tokenize", {"model": args.model, "content": prompt}) as response:
        prompt_ids = json.load(response)["tokens"]
    records = []
    for name, speculative, streaming in (("serial", False, False), ("mtp", True, False), ("mtp-stream", True, True)):
        body = {"model": args.model, "prompt": prompt_ids, "max_tokens": 64, "temperature": 0, "enable_mtp": speculative, "enable_drafter": False, "stream": streaming}
        if streaming:
            body["stream_options"] = {"include_usage": True}
        with post(args.url, "/v1/completions", body) as response:
            if streaming:
                pieces, usage, finish, ended = [], None, None, False
                for line in response:
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        ended = True
                        break
                    event = json.loads(data)
                    if event.get("usage") is not None:
                        usage = event["usage"]
                    for choice in event.get("choices", []):
                        pieces.append(choice.get("text", ""))
                        if choice.get("finish_reason") is not None:
                            finish = choice["finish_reason"]
                assert ended and usage is not None and finish is not None, "incomplete SSE response"
                text = "".join(pieces)
            else:
                data = json.load(response)
                text, usage = data["choices"][0]["text"], data["usage"]
                finish = data["choices"][0]["finish_reason"]
        records.append({"name": name, "text": text, "count": usage["completion_tokens"], "finish": finish})
    assert records[1]["text"] == records[2]["text"], "greedy MTP stream/non-stream publication differs"
    expected = [record for record in records if record["count"] > 0]
    deadline = time.monotonic() + 10
    while True:
        documents = snapshot(args.store, identity)
        if len(documents) >= len(expected) or time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    assert len(documents) == len(expected), "missing or duplicate terminal publication"
    for document, record in zip(documents, expected):
        assert document[:len(prompt_ids)] == prompt_ids, "history prompt differs from submitted IDs"
        output = document[len(prompt_ids):]
        assert len(output) == record["count"], "unpublished speculative tail or missing emitted token"
        with post(args.url, "/detokenize", {"model": args.model, "tokens": output}) as response:
            assert json.load(response)["content"] == record["text"], "durable output differs from engine response"
    summary = {"documents": len(documents), "requests": [{key: value for key, value in record.items() if key != "text"} for record in records], "serial_matches_mtp": records[0]["text"] == records[1]["text"], "scope": "publication/stream parity, not universal variable-width serial equivalence, throughput or proof of draft-source engagement"}
    (args.output / "history-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PASS: exact private history documents and greedy MTP stream/non-stream publication" if expected else "SKIP: no emitted tokens to exercise persistence")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# sushi-test: tier=lib
"""A stand-in for huggingface.co, serving the repo folders under ROOT (tests/test_pull_integrity.sh).

    hf_stub.py ROOT PORT

ROOT/<org>/<repo>/... is a repo's files, re-read on every request so a test edits them between pulls.
.safetensors files are listed as LFS (sha256 under lfs.oid), the rest with their git blob sha1.
ROOT/.corrupt lists paths whose bytes are served altered while the listing still declares the real content id.
ROOT/.requests gets one line per file request, ROOT/.ranges one "path start" line per ranged one.
ROOT/.private lists repo ids answered 401 everywhere and left out of the author listing (/api/models?author=).
ROOT/.rate, when present, caps file downloads at that many bytes per second.
"""
import hashlib
import http.server
import json
import os
import sys
import time

ROOT = os.path.abspath(sys.argv[1])


def blob_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def listing(repo_dir):
    out = []
    for base, _, names in os.walk(repo_dir):
        for name in sorted(names):
            full = os.path.join(base, name)
            rel = os.path.relpath(full, repo_dir)
            data = open(full, "rb").read()
            entry = {"type": "file", "path": rel, "size": len(data), "oid": blob_sha1(data)}
            if name.endswith(".safetensors"):
                entry["lfs"] = {"oid": hashlib.sha256(data).hexdigest(), "size": len(data)}
            out.append(entry)
    return out


def marked(name, item):
    path = os.path.join(ROOT, name)
    return os.path.isfile(path) and item in open(path).read().split()


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_bytes(self, code, body, ctype="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        parts = path.strip("/").split("/")
        if parts == ["api", "models"]:
            author = dict(kv.partition("=")[::2] for kv in query.split("&")).get("author", "")
            org_dir = os.path.join(ROOT, author)
            ids = sorted("%s/%s" % (author, n) for n in os.listdir(org_dir)) if author and os.path.isdir(org_dir) else []
            rows = [{"id": i, "private": False} for i in ids if not marked(".private", i)]
            return self.send_bytes(200, json.dumps(rows).encode(), "application/json")
        repo_id = "/".join(parts[2:4] if parts[:2] == ["api", "models"] else parts[:2])
        if marked(".private", repo_id):
            return self.send_bytes(401, b"{}")
        if parts[:2] == ["api", "models"] and len(parts) >= 4 and parts[-2:] == ["tree", "main"]:
            repo_dir = os.path.join(ROOT, *parts[2:-2])
            if not os.path.isdir(repo_dir):
                return self.send_bytes(404, b"{}")
            return self.send_bytes(200, json.dumps(listing(repo_dir)).encode(), "application/json")
        if parts[:2] == ["api", "models"] and len(parts) == 4:
            if not os.path.isdir(os.path.join(ROOT, *parts[2:])):
                return self.send_bytes(404, b"{}")
            siblings = [{"rfilename": f["path"], "size": f["size"]} for f in listing(os.path.join(ROOT, *parts[2:]))]
            return self.send_bytes(200, json.dumps({"sha": "stubcommit0001", "siblings": siblings}).encode(), "application/json")
        if "resolve" in parts:
            i = parts.index("resolve")
            rel = "/".join(parts[i + 2:])
            full = os.path.join(ROOT, *parts[:i], rel)
            if not os.path.isfile(full):
                return self.send_bytes(404, b"")
            with open(os.path.join(ROOT, ".requests"), "a") as f:
                f.write(rel + "\n")
            data = open(full, "rb").read()
            corrupt = os.path.join(ROOT, ".corrupt")
            if os.path.isfile(corrupt) and rel in open(corrupt).read().split():
                data = bytes([data[0] ^ 0xFF]) + data[1:]
            start = 0
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                start = int(rng[6:].split("-")[0])
                with open(os.path.join(ROOT, ".ranges"), "a") as f:
                    f.write("%s %d\n" % (rel, start))
            if start >= len(data) > 0:
                return self.send_bytes(416, b"")
            body = data[start:]
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body)))
            if start:
                self.send_header("Content-Range", "bytes %d-%d/%d" % (start, len(data) - 1, len(data)))
            self.end_headers()
            rate_file = os.path.join(ROOT, ".rate")
            rate = int(open(rate_file).read()) if os.path.isfile(rate_file) else 0
            step = max(1, rate // 10) if rate else len(body) or 1
            for i in range(0, len(body), step):
                self.wfile.write(body[i:i + step])
                if rate:
                    self.wfile.flush()
                    time.sleep(0.1)
            return
        self.send_bytes(404, b"")


http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[2])), Handler).serve_forever()

#!/usr/bin/env python3
"""A stand-in for huggingface.co, serving the repo folders under ROOT (tests/test_pull_integrity.sh).

    hf_stub.py ROOT PORT

ROOT/<org>/<repo>/... is a repo's files, re-read on every request so a test edits them between pulls.
.safetensors files are listed as LFS (sha256 under lfs.oid), the rest with their git blob sha1.
ROOT/.corrupt lists paths whose bytes are served altered while the listing still declares the real content id.
ROOT/.requests gets one line per file request.
"""
import hashlib
import http.server
import json
import os
import sys

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
        path = self.path.split("?")[0]
        parts = path.strip("/").split("/")
        if parts[:2] == ["api", "models"] and len(parts) >= 4 and parts[-2:] == ["tree", "main"]:
            repo_dir = os.path.join(ROOT, *parts[2:-2])
            if not os.path.isdir(repo_dir):
                return self.send_bytes(404, b"{}")
            return self.send_bytes(200, json.dumps(listing(repo_dir)).encode(), "application/json")
        if parts[:2] == ["api", "models"] and len(parts) == 4:
            if not os.path.isdir(os.path.join(ROOT, *parts[2:])):
                return self.send_bytes(404, b"{}")
            return self.send_bytes(200, json.dumps({"sha": "stubcommit0001"}).encode(), "application/json")
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
            if start >= len(data) > 0:
                return self.send_bytes(416, b"")
            body = data[start:]
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body)))
            if start:
                self.send_header("Content-Range", "bytes %d-%d/%d" % (start, len(data) - 1, len(data)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_bytes(404, b"")


http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[2])), Handler).serve_forever()

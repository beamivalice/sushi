#!/usr/bin/env python3
import http.server
import json
import os
import subprocess
import sys
import threading


class Handler(http.server.BaseHTTPRequestHandler):
    metrics_requests = 0

    def log_message(self, *args):
        pass

    def do_GET(self):
        status = 200
        if self.path == "/health":
            body = {"model_state": "ready"}
        elif self.path == "/v1/models":
            body = {"data": [{"id": "fixture", "loaded": True}]}
        elif self.path == "/props":
            body = {"memory": {"active_bytes": 1024}}
        elif self.path == "/metrics.json":
            Handler.metrics_requests += 1
            step = Handler.metrics_requests
            if step == 3:
                status, body = 500, {"error": "temporary failure"}
            elif step == 6:
                status, body = 503, {"error": "metrics not enabled — start with --metrics"}
            else:
                tokens = {1: 100, 2: 160, 4: 2000, 5: 2060}[step]
                body = {"gauges": {"generation_tokens_live": tokens, "prefill_tokens_live": tokens}}
        else:
            status, body = 404, {"error": "not found"}
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        environment = os.environ.copy()
        environment["SUSHI_METRICS_FIXTURE_URL"] = f"http://127.0.0.1:{server.server_port}"
        subprocess.run([sys.argv[1]], check=True, timeout=30, env=environment)
    finally:
        server.shutdown()

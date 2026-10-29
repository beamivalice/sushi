#!/usr/bin/env python3
import http.server
import json
import subprocess
import sys
import threading


class Handler(http.server.BaseHTTPRequestHandler):
    paused = False
    active = 1
    model = "current"
    selected = False

    def log_message(self, *args):
        pass

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != '/control/status':
            return self.reply(404, {'error': 'Unsupported'})
        if self.headers.get('Authorization') != 'Bearer control-fixture':
            return self.reply(401, {'error': 'Unauthorized'})
        self.reply(200, {'desktop': {'paused': Handler.paused, 'models': [{'id': 'current', 'path': '/models/current'}, {'id': 'other', 'path': '/models/other'}, {'id': 'broken', 'path': '/models/broken'}]},
                         'model': Handler.model, 'idle_controller': {'state': 'ready', 'active_requests': Handler.active}})

    def do_POST(self):
        if self.path != '/control/action':
            return self.reply(404, {'error': 'Unsupported'})
        if self.headers.get('Authorization') != 'Bearer control-fixture':
            return self.reply(401, {'error': 'Unauthorized'})
        data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        action = data['action']
        if action in ('select', 'load', 'unload'):
            if Handler.active:
                return self.reply(409, {'error': 'Requests active'})
            if action == 'select':
                body_path = data['path']
                if body_path not in ('/models/other', '/models/broken'):
                    return self.reply(400, {'error': 'Unexpected selection'})
                Handler.model = body_path.rsplit('/', 1)[-1]
                Handler.selected = True
            if action == 'load' and Handler.model == 'broken':
                return self.reply(500, {'error': 'Load failed'})
        elif action in ('pause', 'resume'):
            Handler.paused = action == 'pause'
            if action == 'resume':
                Handler.active = 0
        else:
            return self.reply(400, {'error': 'Unknown action'})
        self.reply(200, {'ok': True})



with http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        subprocess.run([sys.argv[1], f'http://127.0.0.1:{server.server_port}'], check=True, timeout=30)
    finally:
        server.shutdown()

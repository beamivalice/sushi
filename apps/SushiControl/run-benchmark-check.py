import http.server, json, subprocess, sys, threading, time
requests = []
class Handler(http.server.BaseHTTPRequestHandler):
 def log_message(self, *args): pass
 def do_GET(self):
  if self.path in ['/oversize-declared', '/oversize-streamed']:
   data = b'{"padding":"'+b'x'*(8*1024*1024)+b'"}'
   self.send_response(200)
   if self.path == '/oversize-declared': self.send_header('Content-Length',str(len(data)))
   self.end_headers()
   try: self.wfile.write(data)
   except (BrokenPipeError, ConnectionResetError): pass
   return
  if self.path == '/echo-secret':
   data=json.dumps({'error':{'message':'fixture-key '+ 'x'*10000}}).encode()
   self.send_response(401); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
   return
  if self.path == '/redirect':
   self.send_response(302); self.send_header('Location', '/unexpected'); self.end_headers()
  else:
   requests.append('unexpected redirect'); self.send_response(500); self.end_headers()
 def do_POST(self):
  try:
   body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
   assert self.path == '/v1/chat/completions'
   assert self.headers.get('Authorization') == 'Bearer fixture-key'
   assert body['temperature'] == 0 and body['max_tokens'] == 32 and body['stream'] is False
   assert body['chat_template_kwargs']['enable_thinking'] is False
   requests.append(body['model'])
   if body['model'] == 'slow': time.sleep(2)
   data = json.dumps({'choices':[{'finish_reason':'length'}], 'usage':{'prompt_tokens':100,'completion_tokens':20,'prompt_tokens_details':{'cached_tokens':80}},'timings':{'prompt_n':100,'cached_n':80,'prompt_ms':10,'predicted_n':20,'predicted_ms':200}}).encode()
   self.send_response(200); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
  except (BrokenPipeError, ConnectionResetError): pass
  except Exception as error:
   requests.append('error: '+str(error)); self.send_response(500); self.end_headers()
server = http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
try:
 subprocess.run([sys.argv[1],f'http://127.0.0.1:{server.server_port}'],check=True,timeout=40)
 assert requests.count('fixture') == 2 and not any(x.startswith('error') or x == 'unexpected redirect' for x in requests), requests
finally:
 server.shutdown(); server.server_close(); thread.join(timeout=2)

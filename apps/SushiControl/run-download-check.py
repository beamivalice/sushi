import http.server, json, os, pathlib, subprocess, sys, tempfile, threading, time
payloads={'config.json':b'{"model_type":"qwen4_exp"}','model.safetensors':bytes([42])*(2*1024*1024)}
class Handler(http.server.BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_GET(self):
  if self.path.startswith('/api/models/'):
   body=json.dumps([{'type':'file','path':name,'size':len(data)} for name,data in payloads.items()]).encode()
   self.send_response(200); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body); return
  name=self.path.rsplit('/',1)[-1]
  if name not in payloads: self.send_error(404); return
  data=payloads[name]; start=0
  if 'Range' in self.headers: start=int(self.headers['Range'].split('=')[1].split('-')[0])
  self.send_response(206 if start else 200); self.send_header('Content-Length',str(len(data)-start)); self.send_header('Accept-Ranges','bytes')
  if start: self.send_header('Content-Range',f'bytes {start}-{len(data)-1}/{len(data)}')
  self.end_headers()
  try:
   for index in range(start,len(data),32768):
    self.wfile.write(data[index:index+32768]); self.wfile.flush()
    if name.endswith('.safetensors'): time.sleep(.05)
  except (BrokenPipeError,ConnectionResetError): pass
server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
threading.Thread(target=server.serve_forever,daemon=True).start()
try:
 with tempfile.TemporaryDirectory(prefix='sushi-download-check-') as directory:
  home=pathlib.Path(directory); (home/'bin').mkdir()
  shim=home/'bin/curl'
  shim.write_text('#!'+sys.executable+'\nimport os,sys\nargs=sys.argv[1:]\nfor index,arg in enumerate(args):\n if arg.startswith("https://huggingface.co/"):\n  args[index]="http://127.0.0.1:'+str(server.server_port)+'/"+arg[len("https://huggingface.co/"):]\nos.execv("/usr/bin/curl",["curl"]+args)\n')
  shim.chmod(0o755)
  environment=dict(os.environ,HOME=directory,PATH=str(home/'bin')+':'+os.environ['PATH'])
  environment.pop('HF_TOKEN',None)
  environment.pop('HUGGINGFACE_HUB_TOKEN',None)
  subprocess.run([sys.argv[2],str(pathlib.Path(sys.argv[1]).resolve())],env=environment,check=True,timeout=60)
finally:
 server.shutdown(); server.server_close()

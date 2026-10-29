import json, os, pathlib, secrets, socket, subprocess, sys, tempfile, time, urllib.request
root = pathlib.Path(__file__).resolve().parent
binary = pathlib.Path(sys.argv[1]).resolve()
results = []
with tempfile.TemporaryDirectory(prefix='sushi-app-check-') as directory:
 home = pathlib.Path(directory); (home/'models').mkdir()
 for metrics in (False, True):
  with socket.socket() as sock:
   sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
  key = secrets.token_hex(16)
  args = [str(binary),'serve','--host','127.0.0.1','--port',str(port),'--model-dir',str(home/'models'),'--api-key',key,'--api-key-strict','--ctx-size','16384','--max-resident-mem','17179869184','--idle-evict-secs','900','--kv-quant','8']
  if metrics: args.append('--metrics')
  with open(root/f'stock-{metrics}.log','w') as log:
   process = subprocess.Popen(args,env=dict(os.environ, HOME=directory),stdout=log,stderr=log)
   try:
    endpoint = f'http://127.0.0.1:{port}'; deadline = time.monotonic()+30
    while time.monotonic()<deadline:
     if process.poll() is not None: raise RuntimeError('Server exited; inspect log')
     try:
      request=urllib.request.Request(endpoint+'/health',headers={'Authorization':'Bearer '+key})
      with urllib.request.urlopen(request,timeout=1): break
     except Exception: time.sleep(.2)
    else: raise TimeoutError('Server startup')
    result=subprocess.run([str(root/'macos/build/integration-tests'),endpoint,key,'metrics' if metrics else 'no-metrics'],capture_output=True,text=True,timeout=40)
    results.append({'metrics':metrics,'exit':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    if result.returncode: raise RuntimeError(result.stderr)
   finally:
    process.terminate()
    try: process.wait(timeout=15)
    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
(root/'stock-check.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))

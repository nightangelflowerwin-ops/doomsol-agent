from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
import json,time,base64,threading
ROOT=Path(__file__).resolve().parents[1]; FR=ROOT/"replays"/"frames";FR.mkdir(parents=True,exist_ok=True)
lock=threading.Lock();commands=[];status={"attached":False};latest={"data":None,"frame_id":0,"timestamp":None,"request_id":None};eval_results={}
class H(BaseHTTPRequestHandler):
 def sendj(self,c=200,b=None):
  x=json.dumps(b or {}).encode();self.send_response(c);self.send_header("Content-Type","application/json");self.send_header("Access-Control-Allow-Origin","*");self.end_headers();self.wfile.write(x)
 def do_OPTIONS(self):
  self.send_response(204);self.send_header("Access-Control-Allow-Origin","*");self.send_header("Access-Control-Allow-Methods","GET,POST,OPTIONS");self.send_header("Access-Control-Allow-Headers","Content-Type");self.end_headers()
 def body(self):
  return json.loads(self.rfile.read(int(self.headers.get("Content-Length","0"))) or b"{}")
 def do_GET(self):
  if self.path=="/commands":
   with lock:o=commands[:];commands.clear()
   self.sendj(b={"commands":o})
  elif self.path=="/status":
   with lock:self.sendj(b=status.copy())
  elif self.path=="/latest-frame":
   with lock:self.sendj(b=latest.copy())
  elif self.path=="/eval-result":
   with lock:
    if eval_results:self.sendj(b=eval_results.pop(next(iter(eval_results))))
    else:self.sendj(b={})
  else:self.sendj(404,{"error":"not found"})
 def do_POST(self):
  if self.path=="/status":
   with lock:status.update(self.body())
   self.sendj(b={"ok":True})
  elif self.path=="/command":
   with lock:commands.append(self.body())
   self.sendj(b={"ok":True})
  elif self.path=="/eval-result":
   b=self.body();rid=b.get("id")
   if rid:
    with lock:eval_results[rid]=b
   self.sendj(b={"ok":True})
  elif self.path=="/frame":
   b=self.body();data=b.get("data")
   if data:
    ts=int(b.get('timestamp',time.time()*1000))
    with lock:
     latest.update(data=data,frame_id=latest["frame_id"]+1,timestamp=ts,request_id=b.get("requestId"))
     frame_id=latest["frame_id"]
    (FR/f"{ts}_{frame_id}.jpg").write_bytes(base64.b64decode(data))
   self.sendj(b={"ok":True,"frame_id":latest["frame_id"]})
  else:self.sendj(404,{"error":"not found"})
 def log_message(self,*a):pass
print("DoomSol local bridge: http://127.0.0.1:8765");ThreadingHTTPServer(("127.0.0.1",8765),H).serve_forever()

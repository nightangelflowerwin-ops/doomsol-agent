import argparse, base64, json, time, uuid, io
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError
from PIL import Image
from closed_loop import run as run_closed_loop
BASE='http://127.0.0.1:8765'; ROOT=Path(__file__).resolve().parents[1]; REPLAYS=ROOT/'replays'; REPLAYS.mkdir(exist_ok=True)
def http(path,method='GET',body=None,timeout=3):
    data=None if body is None else json.dumps(body).encode(); req=Request(BASE+path,data=data,method=method,headers={'Content-Type':'application/json'} if data else {})
    with urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode() or '{}')
def command(c): return http('/command','POST',c)
def capture(): return command({'type':'capture'})
def key(k,down,code=None,vk=None): return command({'type':'key','key':k,'down':down,'code':code or k,'vk':vk})
def mouse(dx=0,dy=0): return command({'type':'mouse','event':'mouseMoved','x':0,'y':0,'deltaX':dx,'deltaY':dy,'buttons':0})
def click(x,y): return command({'type':'click','x':x,'y':y})
def latest():
    d=http('/latest-frame').get('data'); return base64.b64decode(d) if d else None
def save(run,b,label):
    if b:(run/'frames').mkdir(exist_ok=True); (run/'frames'/f'{time.time_ns()}_{label}.jpg').write_bytes(b)
def ocr(b):
    try:
        import pytesseract
        im=Image.open(io.BytesIO(b)); w,h=im.size; crop=im.crop((0,0,w,min(h,280))).resize((w*2,min(h,280)*2)); return pytesseract.image_to_string(crop,config='--psm 6').replace('\n',' | ')
    except Exception as e:return 'OCR_ERROR:'+str(e)
def eval_js(expr,wait=5):
    rid=str(uuid.uuid4()); command({'type':'eval','id':rid,'expression':expr}); end=time.time()+wait
    while time.time()<end:
        r=http('/eval-result')
        if r.get('id')==rid:return r
        time.sleep(.05)
    return {'error':'timeout'}
def newrun(mode):
    rid=time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]; run=REPLAYS/rid; (run/'frames').mkdir(parents=True); m={'run_id':rid,'mode':mode,'started':time.time(),'actions':[]}; return run,m
def rec(m,a):m['actions'].append({'t':time.time()-m['started'],**a})
def scout(seconds):
    run,m=newrun('scout')
    for i in range(int(seconds*5)):
        capture(); time.sleep(.2); b=latest(); save(run,b,'scout')
        if i%5==0 and b:print(f'[{i/5:4.1f}s] {ocr(b)[:220]}')
    m['finished']=time.time();m['frames']=len(list((run/'frames').glob('*.jpg')));(run/'meta.json').write_text(json.dumps(m,indent=2));print('Scout saved:',run)
def baseline(seconds):
    run,m=newrun('baseline_sweep'); click(750,500);rec(m,{'action':'click_lock','x':750,'y':500});time.sleep(.5);key('w',True);rec(m,{'action':'keydown','key':'w'});end=time.time()+seconds;i=0
    try:
        while time.time()<end:
            dx=55 if i%2==0 else -55;mouse(dx,0);rec(m,{'action':'mouse','dx':dx});key('Control',True,'ControlLeft',17);time.sleep(.07);key('Control',False,'ControlLeft',17);rec(m,{'action':'fire_tap'})
            if i%10==0:key('space',True,'Space',32);time.sleep(.04);key('space',False,'Space',32);rec(m,{'action':'use'})
            if i%5==0:capture();time.sleep(.04);b=latest();save(run,b,'live');print(f'[{time.time():.0f}] {ocr(b)[:220]}' if b else '')
            i+=1;time.sleep(.10)
    finally:key('w',False);key('Control',False,'ControlLeft',17)
    b=latest();m.update(finished=time.time(),duration=time.time()-m['started'],frames=len(list((run/'frames').glob('*.jpg'))),final_ocr=ocr(b or b''));(run/'meta.json').write_text(json.dumps(m,indent=2));print('Baseline saved:',run)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['status','discover','scout','baseline','closed-loop'],default='status');ap.add_argument('--seconds',type=float,default=20);ap.add_argument('--live',action='store_true',help='Send bounded aim/fire input; closed-loop is dry-run by default.');a=ap.parse_args()
    try:
        print('Bridge:',http('/status'))
        if a.mode=='discover':
            expr='''(()=>({url:location.href,title:document.title,canvases:[...document.querySelectorAll("canvas")].map((c,i)=>({i,width:c.width,height:c.height,clientWidth:c.clientWidth,clientHeight:c.clientHeight})),bodyText:document.body.innerText.slice(0,6000),storage:Object.keys(localStorage),globals:Object.keys(window).filter(k=>/doom|game|player|engine|level|canvas/i.test(k)).slice(0,250)}))()''';print(json.dumps(eval_js(expr),indent=2,ensure_ascii=False))
        elif a.mode=='scout':scout(a.seconds)
        elif a.mode=='baseline':baseline(a.seconds)
        elif a.mode=='closed-loop':run_closed_loop(ROOT,http,command,key,mouse,a.seconds,a.live)
    except URLError as e:print('Bridge unavailable:',e)
if __name__=='__main__':main()

import base64, json, time, uuid
from pathlib import Path
from perception import TargetTracker, compare_frames

def fresh_frame(http,command,previous_id=-1,timeout=3):
    request_id=str(uuid.uuid4());started=time.time();command({'type':'capture','requestId':request_id})
    while time.time()-started<timeout:
        frame=http('/latest-frame')
        if frame.get('data') and frame.get('frame_id',0)>previous_id and frame.get('request_id')==request_id:
            frame['bytes']=base64.b64decode(frame.pop('data'));frame['latency_ms']=round((time.time()-started)*1000,1);return frame
        time.sleep(.025)
    raise TimeoutError('No correlated fresh frame returned')

def choose_action(perception,aim_threshold=55):
    target,aim=perception.get('target'),perception.get('aim_error')
    if not target or target['confidence']<.30:return {'type':'none','reason':'no_stable_confident_target'}
    if aim['distance']>aim_threshold:return {'type':'aim','dx':max(-60,min(60,round(aim['dx']*.18))),'dy':max(-35,min(35,round(aim['dy']*.18))),'target_id':target['id']}
    return {'type':'fire','target_id':target['id']}

def run(root,http,command,key,mouse,seconds,live=False):
    replay=root/'replays'/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]);(replay/'frames').mkdir(parents=True)
    meta={'schema_version':2,'mode':'closed_loop','dry_run':not live,'started_at':time.time(),'events':[],
          'limitations':['Visual candidates are not confirmed enemies.','Fire confirmation requires observable post-fire visual evidence.']}
    def record(kind,**fields):meta['events'].append({'sequence':len(meta['events'])+1,'t':round(time.time()-meta['started_at'],4),'kind':kind,**fields})
    def save(frame,label):
        path=replay/'frames'/f"{frame['frame_id']:06d}_{label}.jpg";path.write_bytes(frame['bytes']);return str(path.relative_to(replay))
    tracker=TargetTracker();last=-1;end=time.time()+seconds
    try:
        while time.time()<end:
            before=fresh_frame(http,command,last);last=before['frame_id'];perception=tracker.observe(before['bytes'])
            record('observation',frame_id=last,frame_timestamp=before.get('timestamp'),capture_latency_ms=before['latency_ms'],frame=save(before,'before'),perception=perception)
            action=choose_action(perception);record('decision',action=action)
            if not live or action['type']=='none':record('action',action=action,sent=False,result='dry_run' if not live else 'skipped');time.sleep(.08);continue
            sent=time.time()
            if action['type']=='aim':mouse(action['dx'],action['dy'])
            else:key('Control',True,'ControlLeft',17);time.sleep(.06);key('Control',False,'ControlLeft',17)
            record('action',action=action,sent=True)
            after=fresh_frame(http,command,last);last=after['frame_id'];delta=compare_frames(before['bytes'],after['bytes'])
            result=('confirmed' if delta['changed'] else 'failed') if action['type']=='aim' else ('confirmed' if (delta['center_change'] or 0)>=8 else 'inconclusive')
            record('validation',action=action,frame_id=last,frame=save(after,'after'),validation={'result':result,'expected':'visual state change' if action['type']=='aim' else 'central post-fire visual evidence','observed':delta,'action_to_frame_ms':round((time.time()-sent)*1000,1),'fresh_frame':after['frame_id']>before['frame_id']})
    finally:
        key('Control',False,'ControlLeft',17)
        meta['finished_at']=time.time();meta['duration']=round(meta['finished_at']-meta['started_at'],3)
        (replay/'meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('Closed-loop replay saved:',replay)

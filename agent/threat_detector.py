"""Small local enemy detector trained from Doom sprite crops and replay backgrounds."""
from io import BytesIO
from math import hypot
from pathlib import Path

import numpy as np
from PIL import Image


def _torch():
    import torch
    from torch import nn
    return torch, nn


class ThreatNet:
    """Factory kept import-safe for installations that only run rule-based mode."""
    @staticmethod
    def build():
        torch, nn = _torch()
        return nn.Sequential(
            nn.Conv2d(3, 12, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(12, 20, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(20, 28, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(28, 1),
        )


def image_tensor(images, size=64):
    torch, _ = _torch()
    rows=[]
    for image in images:
        rgb=image.convert('RGB').resize((size,size),Image.Resampling.BILINEAR)
        rows.append(np.asarray(rgb,dtype=np.float32).transpose(2,0,1)/255.0)
    return torch.from_numpy(np.stack(rows))


class LearnedThreatDetector:
    """Batched multiscale window detector returning a target compatible with hybrid_loop."""
    def __init__(self, checkpoint=None, threshold=None):
        torch, _ = _torch()
        path=Path(checkpoint or Path(__file__).resolve().parents[1]/'models'/'doom_threat_detector_v1'/'threat_detector.pt')
        self.available=path.exists();self.path=path;self.model=None
        self.threshold=float(threshold or .80);self.labels=['enemy']
        self._tracks=[];self._next_track_id=1;self._frame_index=0
        if not self.available:return
        payload=torch.load(path,map_location='cpu',weights_only=False)
        self.model=ThreatNet.build();self.model.load_state_dict(payload['state_dict']);self.model.eval()
        # Live promotion prioritizes precision: validation-F1 calibration may
        # never lower the deployed threshold below the reviewed safe floor.
        self.threshold=max(.80,float(threshold or payload.get('threshold',self.threshold)));self.labels=payload.get('labels',self.labels)

    def reset(self):
        self._tracks=[];self._next_track_id=1;self._frame_index=0

    def _track(self,box,confidence):
        """Associate detections by center distance instead of scan-grid identity."""
        self._frame_index+=1
        cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
        max_distance=max(box[2]-box[0],box[3]-box[1])*1.8
        live=[t for t in self._tracks if self._frame_index-t['last_seen']<=2]
        match=min(live,key=lambda t:hypot(t['center'][0]-cx,t['center'][1]-cy),default=None)
        if match is None or hypot(match['center'][0]-cx,match['center'][1]-cy)>max_distance:
            match={'id':f'threat-track-{self._next_track_id}','age':0,'center':(cx,cy),'last_seen':self._frame_index}
            self._next_track_id+=1;live.append(match)
        match['age']+=1;match['center']=(cx,cy);match['box']=box
        match['confidence']=confidence;match['last_seen']=self._frame_index
        self._tracks=live
        return match

    @staticmethod
    def viewport(w,h):return (round(w*.165),round(h*.105),round(w*.835),round(h*.665))
    @staticmethod
    def crosshair(w,h):return (w*.50,h*.419)

    def observe(self,raw):
        if not self.available:return {'available':False,'target':None,'reason':'checkpoint_missing'}
        torch, _ = _torch();image=Image.open(BytesIO(raw)).convert('RGB');w,h=image.size;view=self.viewport(w,h)
        vw,vh=view[2]-view[0],view[3]-view[1];windows=[];boxes=[]
        # Classic Doom sprites are tall. Scan three bounded scales with 50% overlap.
        for frac in (.18,.30,.46):
            wh=max(48,round(vh*frac));ww=max(32,round(wh*.68));sx=max(20,ww//2);sy=max(24,wh//2)
            for y in range(view[1],max(view[1]+1,view[3]-wh+1),sy):
                for x in range(view[0],max(view[0]+1,view[2]-ww+1),sx):
                    box=(x,y,min(view[2],x+ww),min(view[3],y+wh))
                    # Exclude the weapon/HUD band, a dominant source of high
                    # confidence corpse and pistol false positives.
                    if (box[2]-box[0]>=24 and box[3]-box[1]>=36
                            and (box[1]+box[3])/2 <= h*.575):
                        boxes.append(box);windows.append(image.crop(box))
        if not windows:return {'available':True,'target':None,'reason':'no_windows'}
        with torch.inference_mode():scores=torch.sigmoid(self.model(image_tensor(windows))).flatten().numpy()
        order=np.argsort(scores)[::-1];best_i=int(order[0]);best=float(scores[best_i]);best_box=boxes[best_i]
        if best<self.threshold:
            self._frame_index+=1;self._tracks=[t for t in self._tracks if self._frame_index-t['last_seen']<=2]
            return {'available':True,'target':None,'best_score':round(best,3),'threshold':round(self.threshold,3)}
        # Weighted local consensus suppresses isolated wall-texture false positives.
        cx=(best_box[0]+best_box[2])/2;cy=(best_box[1]+best_box[3])/2
        neighbors=[float(scores[i]) for i in order[:8] if hypot((boxes[i][0]+boxes[i][2])/2-cx,(boxes[i][1]+boxes[i][3])/2-cy)<max(best_box[2]-best_box[0],best_box[3]-best_box[1])*.65]
        consensus=sum(s>=self.threshold*.88 for s in neighbors)
        if consensus<2 and best<min(.96,self.threshold+.12):
            return {'available':True,'target':None,'best_score':round(best,3),'threshold':round(self.threshold,3),'consensus':consensus}
        track=self._track(best_box,best);confirmed=track['age']>=3
        cross=self.crosshair(w,h);aim={'dx':round(cx-cross[0],2),'dy':round(cy-cross[1],2),'distance':round(hypot(cx-cross[0],cy-cross[1]),2)}
        candidate={'id':track['id'],'class':'enemy','box':list(best_box),'center':[round(cx,1),round(cy,1)],'confidence':round(best,3),'track_age':track['age']}
        return {'available':True,'target':candidate if confirmed else None,'raw_candidate':candidate,
                'aim_error':aim if confirmed else None,'best_score':round(best,3),'threshold':round(self.threshold,3),
                'consensus':consensus,'track_age':track['age'],'confirmed':confirmed}

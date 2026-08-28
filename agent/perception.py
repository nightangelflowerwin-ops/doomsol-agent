from dataclasses import dataclass, asdict
from io import BytesIO
from math import hypot
from PIL import Image, ImageChops, ImageStat

@dataclass
class Candidate:
    id: str
    box: tuple
    center: tuple
    confidence: float
    velocity: tuple = (0.0, 0.0)
    age: int = 1

    def json(self):
        value = asdict(self)
        value["box"], value["center"], value["velocity"] = map(list, (self.box, self.center, self.velocity))
        return value

class TargetTracker:
    """Conservative visual candidates; this is not an enemy classifier."""
    def __init__(self, grid=12, threshold=20, max_distance=120):
        self.grid, self.threshold, self.max_distance = grid, threshold, max_distance
        self.previous_image, self.tracks, self.next_id = None, [], 1

    def observe(self, raw):
        image = Image.open(BytesIO(raw)).convert("RGB"); w, h = image.size
        if self.previous_image is None or self.previous_image.size != image.size:
            self.previous_image = image
            return {"resolution":[w,h],"candidates":[],"target":None,"aim_error":None,
                    "movement":{"state":"unknown","global_change":0.0,"center_change":0.0}}
        diff = ImageChops.difference(self.previous_image, image).convert("L")
        global_change = ImageStat.Stat(diff).mean[0]
        center_change = ImageStat.Stat(diff.crop((int(w*.25),int(h*.2),int(w*.75),int(h*.8)))).mean[0]
        small = diff.resize((self.grid,self.grid), Image.Resampling.BOX)
        cells=[]
        for i, score in enumerate(small.getdata()):
            x,y=i%self.grid,i//self.grid
            if score >= self.threshold and 1 <= y < self.grid-2:
                box=(round(x*w/self.grid),round(y*h/self.grid),round((x+1)*w/self.grid),round((y+1)*h/self.grid))
                cells.append((box,score))
        candidates=[]; unused=list(self.tracks)
        for box,score in sorted(cells,key=lambda item:item[1],reverse=True)[:8]:
            cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
            prior=min(unused,key=lambda t:hypot(t.center[0]-cx,t.center[1]-cy),default=None)
            if prior and hypot(prior.center[0]-cx,prior.center[1]-cy)<=self.max_distance:
                unused.remove(prior); c=Candidate(prior.id,box,(cx,cy),round(min(1.0,score/80),3),(cx-prior.center[0],cy-prior.center[1]),prior.age+1)
            else:
                c=Candidate(f"target-{self.next_id:04d}",box,(cx,cy),round(min(1.0,score/80),3)); self.next_id+=1
            candidates.append(c)
        self.tracks, self.previous_image = candidates, image
        stable=[c for c in candidates if c.age>=2]
        target=min(stable,key=lambda c:hypot(c.center[0]-w/2,c.center[1]-h/2),default=None)
        aim=None if target is None else {"dx":round(target.center[0]-w/2,2),"dy":round(target.center[1]-h/2,2),
                                        "distance":round(hypot(target.center[0]-w/2,target.center[1]-h/2),2)}
        state="stationary" if global_change<2 else ("turning" if center_change>8 else "moving")
        return {"resolution":[w,h],"candidates":[c.json() for c in candidates],"target":target.json() if target else None,
                "aim_error":aim,"movement":{"state":state,"global_change":round(global_change,3),"center_change":round(center_change,3)}}

def compare_frames(before, after):
    a,b=Image.open(BytesIO(before)).convert("RGB"),Image.open(BytesIO(after)).convert("RGB")
    if a.size!=b.size:return {"mean_change":None,"center_change":None,"changed":False}
    d=ImageChops.difference(a,b).convert("L"); w,h=a.size
    mean=ImageStat.Stat(d).mean[0]; center=ImageStat.Stat(d.crop((int(w*.35),int(h*.3),int(w*.65),int(h*.7)))).mean[0]
    return {"mean_change":round(mean,3),"center_change":round(center,3),"changed":mean>=2.0}

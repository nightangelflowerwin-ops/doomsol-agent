"""Train a compact enemy/non-enemy crop detector entirely from local assets."""
import argparse,copy,json,random,time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image,ImageEnhance,ImageOps

from threat_detector import ThreatNet,image_tensor


def augment_enemy(image,rng):
    scale=rng.uniform(.55,1.35);w=max(18,round(image.width*scale));h=max(28,round(image.height*scale))
    image=image.resize((w,h),Image.Resampling.BILINEAR)
    if rng.random()<.5:image=ImageOps.mirror(image)
    image=ImageEnhance.Brightness(image).enhance(rng.uniform(.55,1.35))
    image=ImageEnhance.Contrast(image).enhance(rng.uniform(.75,1.35))
    # Preserve sprite context while varying framing/occlusion.
    pad_x=round(w*rng.uniform(.02,.35));pad_y=round(h*rng.uniform(.02,.22))
    canvas=Image.new('RGB',(w+2*pad_x,h+2*pad_y),(rng.randrange(8,35),rng.randrange(5,28),rng.randrange(4,25)))
    if image.mode=='RGBA':canvas.paste(image,(pad_x,pad_y),image.getchannel('A'))
    else:canvas.paste(image.convert('RGB'),(pad_x,pad_y))
    if rng.random()<.25:
        cut=round(canvas.height*rng.uniform(.05,.22));canvas.paste((15,12,10),(0,canvas.height-cut,canvas.width,canvas.height))
    return canvas


def negative_crops(paths,count,rng):
    out=[];attempts=0
    while len(out)<count and attempts<count*12:
        attempts+=1;path=rng.choice(paths)
        try:image=Image.open(path).convert('RGB')
        except Exception:continue
        w,h=image.size;view=(round(w*.165),round(h*.105),round(w*.835),round(h*.665));vw,vh=view[2]-view[0],view[3]-view[1]
        ch=max(40,round(vh*rng.uniform(.16,.48)));cw=max(30,round(ch*rng.uniform(.55,.90)))
        if cw>=vw or ch>=vh:continue
        x=rng.randrange(view[0],view[2]-cw);y=rng.randrange(view[1],view[3]-ch)
        out.append(image.crop((x,y,x+cw,y+ch)))
    return out


def replay_false_positive_crops(root,runs,rng):
    """Mine the exact learned-threat boxes from human-reviewed zero-kill gates."""
    out=[]
    for run in runs:
        meta_path=root/'replays'/run/'meta.json'
        if not meta_path.exists():continue
        meta=json.loads(meta_path.read_text(encoding='utf-8'));latest_frame=None
        for event in meta.get('events',[]):
            if event.get('kind')=='observation':latest_frame=event.get('frame')
            action=event.get('action') or {}
            threat=action.get('learned_threat') or event.get('learned_threat') or {}
            target=threat.get('target')
            if event.get('kind')!='action' or action.get('detector')!='learned_threat' or not target or not latest_frame:continue
            frame_path=root/'replays'/run/Path(latest_frame)
            if not frame_path.exists():continue
            image=Image.open(frame_path).convert('RGB');box=[int(v) for v in target['box']]
            for _ in range(8):
                pad_x=rng.randint(-8,12);pad_y=rng.randint(-6,10)
                x1=max(0,box[0]-pad_x);y1=max(0,box[1]-pad_y);x2=min(image.width,box[2]+pad_x);y2=min(image.height,box[3]+pad_y)
                if x2-x1>=24 and y2-y1>=36:out.append(image.crop((x1,y1,x2,y2)))
    return out


def metrics(logits,y,threshold):
    p=(logits.sigmoid()>=threshold).float();tp=float(((p==1)&(y==1)).sum());fp=float(((p==1)&(y==0)).sum());fn=float(((p==0)&(y==1)).sum());tn=float(((p==0)&(y==0)).sum())
    return {'precision':round(tp/max(1,tp+fp),4),'recall':round(tp/max(1,tp+fn),4),'f1':round(2*tp/max(1,2*tp+fp+fn),4),'accuracy':round((tp+tn)/max(1,len(y)),4),'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--epochs',type=int,default=14);ap.add_argument('--seed',type=int,default=1701);ap.add_argument('--positive-samples',type=int,default=900);ap.add_argument('--negative-samples',type=int,default=1400);ap.add_argument('--hard-negative-run',action='append',default=[]);a=ap.parse_args()
    import torch
    rng=random.Random(a.seed);torch.manual_seed(a.seed)
    doomsol_paths=sorted((a.root/'agent'/'assets'/'doom_enemy_templates').glob('*.png'))
    wad_paths=sorted((a.root/'agent'/'assets'/'wad_enemy_sprites').glob('*/*.png'))
    template_paths=doomsol_paths+wad_paths
    frame_paths=sorted((a.root/'replays').glob('**/*.jpg'))
    if not template_paths:raise SystemExit('No enemy templates found.')
    if len(frame_paths)<20:raise SystemExit('Need at least 20 real replay frames for hard negatives.')
    templates=[Image.open(p).convert('RGB') for p in template_paths]
    positives=[augment_enemy(rng.choice(templates),rng) for _ in range(a.positive_samples)]
    mined_negatives=replay_false_positive_crops(a.root,a.hard_negative_run,rng)
    negatives=negative_crops(frame_paths,a.negative_samples,rng)+mined_negatives
    samples=positives+negatives;labels=[1.0]*len(positives)+[0.0]*len(negatives);order=list(range(len(samples)));rng.shuffle(order)
    split=round(len(order)*.82);train_i,val_i=order[:split],order[split:]
    x=image_tensor(samples);y=torch.tensor(labels,dtype=torch.float32).unsqueeze(1)
    model=ThreatNet.build();opt=torch.optim.AdamW(model.parameters(),lr=1.8e-3,weight_decay=1e-4);loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([len(negatives)/len(positives)]))
    history=[];best_loss=float('inf');best_epoch=0;best_state=None
    for epoch in range(a.epochs):
        model.train();batch_order=train_i[:];rng.shuffle(batch_order);total=0.0
        for start in range(0,len(batch_order),64):
            idx=batch_order[start:start+64];logits=model(x[idx]);loss=loss_fn(logits,y[idx]);opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())*len(idx)
        model.eval()
        with torch.inference_mode():val_logits=model(x[val_i]);val_loss=float(loss_fn(val_logits,y[val_i]))
        history.append({'epoch':epoch+1,'train_loss':round(total/len(train_i),5),'validation_loss':round(val_loss,5)})
        if val_loss<best_loss:
            best_loss=val_loss;best_epoch=epoch+1;best_state=copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch+1}/{a.epochs}: train={history[-1]['train_loss']} val={history[-1]['validation_loss']}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():val_logits=model(x[val_i])
    choices=np.arange(.55,.951,.01);ranked=[(metrics(val_logits,y[val_i],float(t)),float(t)) for t in choices]
    precision_candidates=[z for z in ranked if z[0]['precision']>=.90]
    best_metric,threshold=max(precision_candidates or ranked,key=lambda z:(z[0]['recall'],z[0]['precision'],z[1]))
    threshold=max(.80,threshold)
    out=a.root/'models'/'doom_threat_detector_v1';out.mkdir(parents=True,exist_ok=True)
    torch.save({'schema_version':1,'state_dict':model.state_dict(),'threshold':threshold,'labels':['enemy'],'input_size':64,'seed':a.seed},out/'threat_detector.pt')
    report={'schema_version':2,'model':'doom_threat_detector_v1','trained_at':time.time(),'seed':a.seed,'templates':[str(p.relative_to(a.root/'agent'/'assets')) for p in template_paths],
            'source_counts':{'doomsol_enemy_crops':len(doomsol_paths),'local_wad_living_enemy_sprites':len(wad_paths)},
            'positive_samples':len(positives),'negative_samples':len(negatives),'mined_hard_negatives':len(mined_negatives),'hard_negative_runs':a.hard_negative_run,'train_samples':len(train_i),'validation_samples':len(val_i),
            'best_epoch':best_epoch,'best_validation_loss':round(best_loss,5),'checkpoint_policy':'lowest validation loss',
            'threshold':round(threshold,3),'validation':best_metric,'history':history,
            'limitations':['Local WAD art is pretraining data and may differ from the future DoomSol renderer','Replay hard negatives may contain unlabeled distant enemies','Promotion requires bounded DoomSol precision/recall review']}
    (out/'training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()

"""Inspect held-out false positives/misses at a fixed deployment threshold."""
import json
import cv2
import numpy as np
from .common import DATA, MODELS, REPORTS, setup, save_json
from .inference import Detector, annotate

def box_iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]);x2,y2=min(a[2],b[2]),min(a[3],b[3])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/max(union,1e-9)

def match(predictions,truth,threshold=.5):
    used=set();tp=0
    for p in sorted(predictions,key=lambda d:d['confidence'],reverse=True):
        candidates=[(box_iou(p['box'],t),i) for i,t in enumerate(truth) if i not in used]
        score,i=max(candidates,default=(0,-1))
        if score>=threshold:used.add(i);tp+=1
    return tp,len(predictions)-tp,len(truth)-tp

def run():
    setup()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    detector=Detector(MODELS/'best.onnx')
    records=[r for r in json.loads((DATA/'manifest.json').read_text()) if r['split']=='test']
    rows=[];pictures={}
    for r in records:
        im=cv2.imread(str(DATA/'images'/'test'/r['file_name']));h,w=im.shape[:2]
        labels=(DATA/'labels'/'test'/r['file_name'].replace('.jpg','.txt')).read_text().splitlines()
        truth=[]
        for line in labels:
            _,cx,cy,bw,bh=map(float,line.split());truth.append([(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h])
        pred,ms=detector.predict(im,.25)
        tp,fp,fn=match(pred,truth)
        rows.append({'file_name':r['file_name'],'video':r['group'],'tp':tp,'fp':fp,'fn':fn,'latency_ms':ms,'predictions':pred,'ground_truth':truth})
        pictures[r['file_name']]=im
    totals={k:sum(r[k] for r in rows) for k in ['tp','fp','fn']}
    totals['precision']=totals['tp']/max(totals['tp']+totals['fp'],1)
    totals['recall']=totals['tp']/max(totals['tp']+totals['fn'],1)
    save_json(REPORTS/'failure_cases.json',{'confidence':.25,'iou':.5,'backend':'custom ONNX demo pipeline','totals':totals,'images':rows})
    order=sorted(rows,key=lambda r:r['fp']+r['fn'],reverse=True)[:8]
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for ax,r in zip(axes.flat,order):
        im=annotate(pictures[r['file_name']],r['predictions'])
        for b in r['ground_truth']:
            x1,y1,x2,y2=map(round,b);cv2.rectangle(im,(x1,y1),(x2,y2),(255,170,50),1)
        ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.axis('off');ax.set_title(f"{r['file_name']}\nTP {r['tp']} / FP {r['fp']} / FN {r['fn']}",fontsize=8)
    fig.suptitle('Largest error counts: green predictions, blue ground truth | confidence 0.25, IoU 0.5')
    fig.tight_layout();fig.savefig(REPORTS/'failure_contact_sheet.jpg',dpi=140);plt.close(fig)
    samples=REPORTS/'samples';samples.mkdir(exist_ok=True)
    for r in sorted(rows,key=lambda r:r['tp'],reverse=True)[:3]:
        cv2.imwrite(str(samples/r['file_name']),annotate(pictures[r['file_name']],r['predictions']))

if __name__=='__main__':run()

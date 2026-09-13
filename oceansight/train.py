"""Equal-budget compact detectors; select using validation, reserve test."""
import argparse
import time
import shutil
from .common import ROOT, DATA, REPORTS, MODELS, setup, save_json

def train(epochs=10, device='mps', size=320):
    setup()
    from ultralytics import YOLO
    import torch
    torch.set_num_threads(4)
    comparisons=[]
    for name in ['yolo11n','yolov8n']:
        start=time.perf_counter()
        model=YOLO(f'{name}.pt')
        model.train(data=str(DATA/'dataset.yaml'), epochs=epochs, imgsz=size, batch=8, device=device,
                    workers=0, seed=42, deterministic=True, project=str(ROOT/'runs'), name=name,
                    exist_ok=False, patience=epochs, cache=False, amp=False, plots=True,
                    close_mosaic=2, verbose=False)
        best=ROOT/'runs'/name/'weights'/'best.pt'
        # Use trainer save_dir because reruns receive a fresh suffixed directory.
        best=model.trainer.save_dir/'weights'/'best.pt'
        dest=MODELS/f'{name}.pt'; shutil.copy2(best,dest)
        result=YOLO(dest).val(data=str(DATA/'dataset.yaml'), split='val', imgsz=size, batch=1, device='cpu', workers=0,
                              project=str(ROOT/'runs'),name=f'{name}_val',plots=True,verbose=False)
        comparisons.append({'model':name,'epochs':epochs,'imgsz':size,'train_device':device,
                            'elapsed_train_and_val_s':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.model.parameters()),
                            'size_mb':dest.stat().st_size/1e6,'validation':{k:float(v) for k,v in result.results_dict.items()}})
        save_json(REPORTS/'comparison.json',comparisons)
    winner=max(comparisons,key=lambda x:x['validation']['metrics/mAP50-95(B)'])
    shutil.copy2(MODELS/f"{winner['model']}.pt",MODELS/'best.pt')
    save_json(REPORTS/'selection.json',{'selected':winner['model'],'criterion':'highest validation mAP50-95; test never used for selection','imgsz':size})

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=10);p.add_argument('--device',default='mps');p.add_argument('--imgsz',type=int,default=320)
    a=p.parse_args(); train(a.epochs,a.device,a.imgsz)

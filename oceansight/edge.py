"""FP32 export, training-only static INT8 calibration, held-out evaluation/benchmark."""
import json
import time
import platform
import hashlib
from pathlib import Path
import cv2
import numpy as np
from .common import ROOT, DATA, MODELS, REPORTS, setup, save_json
from .inference import Detector, preprocess

def run():
    setup()
    import torch
    import onnx
    from ultralytics import YOLO
    from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantFormat, QuantType
    torch.set_num_threads(4)
    selection=json.loads((REPORTS/'selection.json').read_text());size=selection['imgsz']
    YOLO(MODELS/'best.pt').export(format='onnx',imgsz=size,opset=17,simplify=False,dynamic=False,batch=1,device='cpu')
    fp=MODELS/'best.onnx'; q=MODELS/'best_int8.onnx'
    records=json.loads((DATA/'manifest.json').read_text())
    calibration=[r for r in records if r['split']=='train'][:64]
    class Reader(CalibrationDataReader):
        def __init__(self):
            self.items=iter(calibration)
        def get_next(self):
            r=next(self.items,None)
            if r is None:return None
            im=cv2.imread(str(DATA/'images'/'train'/r['file_name']))
            return {'images':preprocess(im,size)[0]}
    quant_status={'status':'not_run'}
    try:
        quantize_static(str(fp),str(q),Reader(),quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8,weight_type=QuantType.QInt8,per_channel=True)
        onnx.checker.check_model(str(q))
        Detector(q)
        quant_status={'status':'completed','calibration_images':len(calibration),'calibration_split':'train','format':'QDQ U8 activations/S8 per-channel weights'}
    except Exception as exc:
        quant_status={'status':'failed','error':str(exc)}
    save_json(REPORTS/'quantization.json',quant_status)
    candidates=[('pytorch_fp32',MODELS/'best.pt'),('onnx_fp32',fp)]
    if quant_status['status']=='completed':candidates.append(('onnx_int8',q))
    rows=[]
    for label,path in candidates:
        metrics=YOLO(path,task='detect').val(data=str(DATA/'dataset.yaml'),split='test',imgsz=size,batch=1,device='cpu',workers=0,
                         project=str(ROOT/'runs'),name=f'test_{label}',plots=True,verbose=False)
        rows.append({'backend':label,'size_mb':path.stat().st_size/1e6,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'test':{k:float(v) for k,v in metrics.results_dict.items()}})
        save_json(REPORTS/'test_metrics.json',rows)
    # Same square tensors, batch=1, four CPU threads; forward-only comparison.
    test_records=[r for r in records if r['split']=='test'][:30]
    tensors=[preprocess(cv2.imread(str(DATA/'images'/'test'/r['file_name'])),size)[0] for r in test_records]
    pt=YOLO(MODELS/'best.pt').model.float().eval(); pt.fuse()
    benchmark=[]
    outputs={}
    for label,path in candidates:
        if label=='pytorch_fp32':
            @torch.inference_mode()
            def forward(x):
                y=pt(torch.from_numpy(x));return y[0].numpy() if isinstance(y,tuple) else y.numpy()
        else:
            detector=Detector(path)
            def forward(x):return detector.session.run(None,{detector.name:x})[0]
        for _ in range(5):forward(tensors[0])
        timings=[]
        for _ in range(3):
            for x in tensors:
                start=time.perf_counter();forward(x);timings.append((time.perf_counter()-start)*1000)
        outputs[label]=forward(tensors[0])
        benchmark.append({'backend':label,'samples':len(timings),'median_ms':float(np.median(timings)),
                          'p95_ms':float(np.percentile(timings,95)),'mean_ms':float(np.mean(timings))})
    parity={k:{'max_absolute_error':float(np.max(np.abs(v-outputs['pytorch_fp32']))),'mean_absolute_error':float(np.mean(np.abs(v-outputs['pytorch_fp32'])))} for k,v in outputs.items() if k!='pytorch_fp32'}
    save_json(REPORTS/'benchmark.json',{'hardware':platform.platform(),'processor':platform.machine(),'threads':4,'batch':1,'imgsz':size,
              'scope':'warm forward pass only; excludes image decode, letterbox and NMS; not a Jetson or Raspberry Pi benchmark',
              'warmup':5,'unique_images':len(tensors),'repeat':3,'results':benchmark,'raw_output_parity_first_test_image':parity})
    save_json(REPORTS/'deployment.json',{'model':'best.onnx','reason':'FP32 ONNX is the demo default; INT8 is an evaluated experimental alternative, not automatically selected using test results.'})
    print(benchmark)

if __name__=='__main__':run()

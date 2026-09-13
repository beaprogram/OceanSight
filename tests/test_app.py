from pathlib import Path
import pytest
from oceansight.common import MODELS,ROOT

def test_app_load_and_threshold_change():
    if not (MODELS/'best.onnx').exists():pytest.skip('Export model before integration test')
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
    assert not app.exception
    assert len(app.metric)==3
    app.slider[0].set_value(.9).run(timeout=30)
    assert not app.exception

def test_custom_onnx_matches_reference_on_real_images():
    if not (MODELS/'best.onnx').exists():pytest.skip('Export model before integration test')
    import json
    import cv2
    from ultralytics import YOLO
    from oceansight.common import DATA
    from oceansight.inference import Detector
    reference=YOLO(MODELS/'best.onnx',task='detect');custom=Detector(MODELS/'best.onnx')
    rows=[r for r in json.loads((DATA/'manifest.json').read_text()) if r['split']=='test'][:10]
    for r in rows:
        im=cv2.imread(str(DATA/'images'/'test'/r['file_name']))
        expected=reference.predict(im,imgsz=320,rect=False,conf=.25,iou=.45,device='cpu',verbose=False)[0].boxes
        actual,_=custom.predict(im,.25)
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected.xyxy.tolist()):assert a['box']==pytest.approx(b,abs=1.0)

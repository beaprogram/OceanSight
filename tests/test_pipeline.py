import numpy as np
import pytest
from oceansight.data import group_id,yolo_box
from oceansight.inference import preprocess,postprocess
from oceansight.failures import match

def test_group_frames_stay_together():
    assert group_id('vid_000021_frame000001.jpg')==group_id('vid_000021_frame000099.jpg')
    with pytest.raises(ValueError):group_id('../escape.jpg')

def test_box_clipping_and_invalid_labels():
    assert yolo_box([-10,0,100,50],100,50)==[.5,.5,1,1]
    with pytest.raises(ValueError):yolo_box([10,1,2,3],100,100)
    with pytest.raises(ValueError):yolo_box([float('nan'),1,2,3],100,100)

def test_letterbox_round_trip_and_nms():
    im=np.zeros((100,200,3),np.uint8);tensor,transform=preprocess(im,320)
    assert tensor.shape==(1,3,320,320)
    # Original box [20,10,120,60] -> letterboxed xywh [112,136,160,80].
    output=np.array([[[112,112],[136,136],[160,160],[80,80],[.9,.8]]],dtype=np.float32)
    result=postprocess(output,transform,im.shape)
    assert len(result)==1
    assert result[0]['box']==pytest.approx([20,10,120,60])
    assert postprocess(output,transform,im.shape,.95)==[]

def test_detection_matching_does_not_reuse_ground_truth():
    pred=[{'box':[0,0,10,10],'confidence':.9},{'box':[0,0,10,10],'confidence':.8}]
    assert match(pred,[[0,0,10,10]])==(1,1,0)
    assert match([],[[0,0,10,10]])==(0,0,1)

def test_real_manifest_disjointness():
    import json
    from oceansight.common import DATA
    if not (DATA/'manifest.json').exists():pytest.skip('Run dataset preparation first')
    rows=json.loads((DATA/'manifest.json').read_text())
    for field in ['file_name','group','sha256']:
        sets={s:{r[field] for r in rows if r['split']==s} for s in ['train','val','test']}
        assert not sets['train']&sets['val']
        assert not sets['train']&sets['test']
        assert not sets['val']&sets['test']

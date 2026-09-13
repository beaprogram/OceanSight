"""Run: streamlit run app.py --server.address 127.0.0.1"""
import json
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from oceansight.common import MODELS, REPORTS, DATA
from oceansight.inference import Detector, annotate
from oceansight.video import process_video

st.set_page_config(page_title='OceanSight Edge',page_icon='🌊',layout='wide')
st.markdown('<style>.stApp{background:#071c27}h1,h2,h3{color:#b7f4e0}div[data-testid="stMetric"]{background:#123440;padding:18px;border-radius:12px}</style>',unsafe_allow_html=True)
st.caption('OCEANSIGHT EDGE  /  APPLIED MARINE VISION')
st.title('Find debris. Understand the evidence.')
st.write('A local marine-debris detection prototype with video-disjoint evaluation and a transparent CPU deployment benchmark.')
st.caption('Research MVP • one debris class • human review required • no cloud upload')
demo,evidence,quality,about=st.tabs(['Inspect footage','Measured results','Data quality','How it works'])

@st.cache_resource
def load_detector(path,mtime):return Detector(path)

def read_report(name):
    p=REPORTS/name
    return json.loads(p.read_text()) if p.exists() else None

with demo:
    path=MODELS/'best.onnx'
    if not path.exists():st.info('Model preparation is not complete. Run the training and edge export scripts first.')
    else:
        detector=load_detector(str(path),path.stat().st_mtime)
        confidence=st.slider('Confidence threshold',.05,.95,.25,.05,help='Lower thresholds find more candidates but usually produce more false alarms. Confidence is not a calibrated probability.')
        upload=st.file_uploader('Upload an underwater image or short video',type=['jpg','jpeg','png','mp4','mov'])
        manifest=read_report('manifest.json') or []
        names=[r['file_name'] for r in manifest if r['split']=='test']
        sample=st.selectbox('Or inspect a held-out sample',names) if names else None
        if upload and upload.size>50*1024*1024:st.error('Use a file under 50 MB.')
        elif upload and upload.name.lower().endswith(('.mp4','.mov')):
            st.info('Processes at most 150 frames. Counts are detections per frame, not unique tracked objects.')
            if st.button('Analyze video',type='primary'):
                with tempfile.TemporaryDirectory() as temp:
                    source=Path(temp)/'clip.mp4';source.write_bytes(upload.getvalue())
                    output=Path(temp)/'annotated.mp4';preview=st.empty();progress=st.progress(0)
                    def update(i,frame):
                        if i%5==0:preview.image(frame,channels='BGR')
                        progress.progress((i+1)/150)
                    try:
                        rows=process_video(source,output,detector,confidence,callback=update)
                        st.line_chart(pd.DataFrame(rows).set_index('frame')['detections'])
                        st.download_button('Download per-frame results',pd.DataFrame(rows).to_csv(index=False),'oceansight-video.csv','text/csv')
                        st.download_button('Download annotated clip',output.read_bytes(),'oceansight-video.mp4','video/mp4')
                    except (ValueError,RuntimeError) as exc:st.error(str(exc))
                    finally:progress.empty()
        else:
            raw=upload.getvalue() if upload else ((DATA/'images'/'test'/sample).read_bytes() if sample and (DATA/'images'/'test'/sample).exists() else None)
            if raw:
                im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                if im is None:st.error('The image could not be decoded.')
                elif max(im.shape[:2])>6000:st.error('Please resize the image below 6000 pixels per side.')
                else:
                    detections,ms=detector.predict(im,confidence)
                    a,b,c=st.columns(3);a.metric('Debris candidates',len(detections));b.metric('Pipeline latency',f'{ms:.1f} ms');c.metric('Runtime','ONNX · CPU')
                    st.image(annotate(im,detections),channels='BGR',caption='Candidate detections need human review.')
                    st.download_button('Download detections',json.dumps(detections,indent=2),'detections.json','application/json')
                    ok,encoded=cv2.imencode('.jpg',annotate(im,detections))
                    if ok:st.download_button('Download annotated image',encoded.tobytes(),'oceansight.jpg','image/jpeg')
with evidence:
    st.subheader('Results generated on this Mac')
    comparison=read_report('comparison.json')
    if comparison:
        st.caption('Model selection uses validation mAP50–95. These are equal-budget compact YOLO models, not an exhaustive architecture comparison.')
        st.dataframe([{'Model':r['model'],'Epochs':r['epochs'],'Validation mAP50':round(r['validation']['metrics/mAP50(B)'],4),'Validation mAP50–95':round(r['validation']['metrics/mAP50-95(B)'],4)} for r in comparison],hide_index=True)
    else:st.info('Training comparison has not completed.')
    metrics=read_report('test_metrics.json');benchmark=read_report('benchmark.json')
    if metrics:
        st.subheader('Held-out test evaluation')
        st.dataframe([{'Runtime':r['backend'],'mAP50':round(r['test']['metrics/mAP50(B)'],4),'mAP50–95':round(r['test']['metrics/mAP50-95(B)'],4),'File MB':round(r['size_mb'],2)} for r in metrics],hide_index=True)
        if any(r['backend']=='onnx_int8' and r['test']['metrics/mAP50(B)']==0 for r in metrics):
            st.warning('The initial INT8 experiment produced zero test AP. It is retained as a failed optimization experiment; the demo uses FP32 ONNX.')
    if benchmark:
        st.subheader('Warm CPU forward-pass latency')
        st.dataframe([{'Runtime':r['backend'],'Median ms':round(r['median_ms'],2),'P95 ms':round(r['p95_ms'],2),'Measurements':r['samples']} for r in benchmark['results']],hide_index=True)
        st.caption(benchmark['scope'])
    if (REPORTS/'failure_contact_sheet.jpg').exists():st.image(str(REPORTS/'failure_contact_sheet.jpg'),caption='Failure review: green predictions, blue ground truth.')
with quality:
    st.subheader('Which training images deserve another look?')
    st.write('We cluster color and image-quality features and rank unusual images with Isolation Forest. An outlier is a review candidate, not proof of a bad label.')
    for name in ['quality_clusters.png','outlier_contact_sheet.jpg']:
        if (REPORTS/name).exists():st.image(str(REPORTS/name))
    if (REPORTS/'quality.csv').exists():st.dataframe(pd.read_csv(REPORTS/'quality.csv').head(30),hide_index=True)
with about:
    st.markdown('''**Pipeline:** pinned TrashCan mirror → source-video split → annotation checks → two detector experiments → held-out evaluation → ONNX + INT8 experiment → local review app.

**Scope:** all `trash_*` annotations become one marine-debris class. No material classification, object tracking, depth estimation, or robot control.

**Limits:** small third-party dataset subset, possible location overlap across videos, short training, and no physical edge-device test. Performance on a Mac does not establish performance on a Jetson or Raspberry Pi.

**Data:** [TrashCan / University of Minnesota](https://irvlab.cs.umn.edu/resources/trashcan), based on JAMSTEC imagery. This app is an independent portfolio prototype, not a DeepSense product.''')

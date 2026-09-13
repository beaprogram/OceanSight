# Resume-ready project wording

**OceanSight Edge — Marine Debris Detection & Edge Inference**
*Python, PyTorch, Ultralytics, OpenCV, scikit-learn, ONNX Runtime, FastAPI, Streamlit, Docker*

- Built an end-to-end underwater debris detection pipeline; compared YOLO11n and YOLOv8n using video-disjoint data and achieved 0.498 mAP50 on a 108-image reference holdout.
- Exported the selected detector to ONNX, measuring 16.98 ms median CPU forward latency versus 39.50 ms for PyTorch on Apple M4; delivered a Dockerized FastAPI image API and a Streamlit image/video demo.
- Analyzed 1,680 training images with ResNet18 embeddings, PCA, KMeans and Isolation Forest; added video-grouped error analysis and diagnosed confidence collapse in an INT8 export.

## How to use these bullets

Use two or three bullets depending on space. Add your real GitHub URL only after publishing; no repository URL is invented here. The measured latency excludes image decoding, preprocessing and NMS. Do not call mAP classification accuracy, claim Jetson/Pi performance, or describe the reused reference test set as a new untouched benchmark.

Only list the project after you can run it and explain its data split, evaluation, quantization failure and remaining limitations. The build used AI assistance; do not imply unaided authorship.

## Optional optimization bullet

- Reduced the ONNX model file by 72.4% with selective INT8 quantization, passing a validation gate of no more than 0.02 absolute mAP50–95 loss; measured quality and latency separately.

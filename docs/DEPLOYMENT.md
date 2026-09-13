# Deployment and verification

## Local demo

Use the SSD environment with `./start-demo.command`, or create a fresh Python 3.13 environment and install `requirements-demo.txt` to run inference without the training stack. Supply `models/best.onnx`, and optionally the validated `models/best_int8.onnx`. Start `streamlit run app.py --server.address 127.0.0.1` from the repository root.

The app can process your uploads without downloading training data. The reference-sample selector requires prepared dataset images; run the README data command if you want those examples. Results, graphs and experiment records are included as numerical evidence. Image contact sheets are generated locally from the source dataset.

## API contract

- `GET /health`: process health and artifact availability.
- `GET /ready`: validates that the ONNX model can load and returns its SHA-256 identifier.
- `POST /predict?confidence=0.25`: multipart field `file` containing a JPEG/PNG. Returns `detections`, `width`, `height`, `pipeline_ms`, `model_sha256`, and `confidence_threshold`.
- `GET /docs`: interactive OpenAPI documentation.

Each detection contains an `xyxy` box in original image pixel coordinates, a score and the label `marine_debris`. The API handles at most two simultaneous prediction calls per process. Invalid images/thresholds return 422; unavailable/busy inference returns 503. The upload read is bounded and image dimensions are checked before OpenCV decoding. There is no authentication, queue persistence or remote storage: keep the service bound to localhost unless you deliberately add deployment controls.

## Container

The Dockerfile uses a pinned Python base-image digest and a locked Linux inference environment. PyTorch, Ultralytics, datasets and model weights are not baked into the image. The server runs as an unprivileged user. A readiness healthcheck verifies the supplied model. The README uses `docker cp` so an external SSD need not be mounted into a VM.

Run `python -m oceansight.verify_service --url http://127.0.0.1:8502` against the verification container (or substitute your API port). The local container verification checks readiness, invalid input and a real prediction. Its numerical output should agree with the local ONNX pipeline within tolerance. Container latency is not compared with native CPU timing because VM resources and concurrent training can differ.

## Automated checks

`python -m pytest -q` exercises:

- normalized boxes, clipping and invalid annotations;
- source-video/exact-hash separation and frozen holdouts;
- letterbox reversal, NMS and one-to-one ground-truth matching;
- custom ONNX agreement with the Ultralytics reference on real images;
- app loading and confidence changes;
- bounded, decodable annotated video output;
- API health/readiness, malformed input, input limits and a real model;
- reproducible video-cluster bootstrap intervals.

Artifact-dependent checks explicitly skip in a source-only checkout. The GitHub Actions workflow runs source checks when the repository is published; remote Actions execution has not been claimed. Local source tests and lint are recorded under `reports/verification.txt`.

## Reproducibility and maintenance

The report manifest records source revision, SHA-256 hashes, labels and splits. Training uses a fixed seed and records raw results/configurations; bit-for-bit GPU determinism across software or hardware is not guaranteed. Model hashes identify exact deployed artifacts. JSON report writes replace files atomically. New CLI outputs never overwrite existing directories.

The development history preserves the first experiment. Later experiments should receive new report/model directories and a new genuinely unseen final dataset, rather than repeatedly tuning against the reference test set.

## Clean source-only check

After building `oceansight-edge:local`, run:

```sh
docker build -f Dockerfile.ci -t oceansight-edge:checks .
docker run --rm oceansight-edge:checks
```

This verifies the same source-only path used by CI without mounting the dataset or model. Remote GitHub Actions has not run unless its own status is recorded separately.

## Streamlit Community Cloud

Select repository `beaprogram/OceanSight`, branch `main`, entrypoint `cloud/app.py`, and Python 3.13. The entrypoint's `cloud/requirements.txt` takes priority over the training requirements at the repository root. It downloads and SHA-256 verifies the v1.0.0 ONNX artifacts and three reference examples from the pinned mirror. Example imagery is credited to JAMSTEC/TrashCan. No secrets or GPU are required.

The public profile caps video processing at 60 frames and 20 MB; images are limited to 10 MB, 20 megapixels and 6000 pixels per side. Uploads are processed on the hosted server and are not intentionally persisted by the application. Results stay in session memory. Recorded CPU timings are from the original Apple M4 experiment, not cloud benchmarks.

## Public deployment verified — 2026-09-13

Live URL: https://oceansight-edge.streamlit.app/

Deployed from commit `d6ff426126aa30a6bfeea1fd7a1b62d5a7410255`, branch `main`, entrypoint `cloud/app.py`. The actual host build used Python 3.14.7 and installed `cloud/requirements.txt`; the platform replaced PyArrow 25 with 24. Both FP32 and INT8 produced a detection on the first reference image. Measured-results and data-quality pages rendered successfully. A fresh anonymous HTTP session returned 200. These are deployment smoke checks, not a new accuracy or latency benchmark. Upload/video edge cases remain covered by local checks; they were not re-run on the public host.

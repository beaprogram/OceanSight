# OceanSight Edge data and model card

**Use:** local marine-debris candidate detection for human review. This is an independent portfolio implementation with a reproducible training/evaluation pipeline, not a robot-control system.

**Lineage:** JAMSTEC J-EDI → TrashCan researchers → public third-party Hugging Face CSV/image mirror → project subset. The mirror revision is pinned, and source annotations/dimensions/hashes are retained. Original segmentation masks are not used. Mirror conversion has not been checked against the original blocked archive.

**Task:** all source `trash_*` labels become one binary `marine_debris` detection class. Other source objects are background. The model does not classify plastic, metal or other materials.

**Splits:** 1,680 training images across 213 videos, 108 validation images across 44 videos, and 108 reference-test images across 45 videos. There is no cross-split video ID or exact-hash overlap. Validation/test membership is frozen from v1; test results were previously inspected, so this is a reused reference holdout. Videos may share locations/expeditions. The annotation-derived inventory may omit completely unlabeled negative frames.

**Models:** pretrained YOLO11n and YOLOv8n, fine-tuned under matching 30-epoch, 416-pixel, batch-eight, seed-42 settings. Validation mAP50–95 selects the model/checkpoint. It is a single-seed comparison of related detectors, not evidence of universal architecture superiority.

**Review:** interpretable quality features and frozen 512-D ResNet18 embeddings. Scaling/PCA/KMeans/Isolation Forest are fitted only on training data. The review contact sheet takes at most one image per video. Outliers are not assumed to be annotation mistakes and are not automatically removed.

**Optimization:** fixed-shape, batch-one ONNX export. The corrected INT8 recipe quantizes Conv operators after shape preprocessing and excludes DFL/output computations. Calibration uses 128 distinct training videos. A validation quality gate requires no more than 0.02 absolute mAP50–95 loss and nonzero AP. See the generated gate result; conversion success alone is insufficient. FP32 remains the default.

**Evaluation:** matched square preprocessing for all final backends; AP50 and AP50–95; fixed-threshold custom-pipeline counts; video-cluster bootstrap intervals for precision/recall; rotating-order CPU forward benchmarks. No physical edge-device/power measurements, multi-seed uncertainty, calibrated probabilities or independent new test are claimed.

**Serving:** Streamlit image/video review; bounded FastAPI image inference; Docker inference container without the training framework. Counts in video are per-frame detections, not unique objects. The local API is not authenticated and is meant for localhost use.

**Limits:** domain shift, small/occluded/low-contrast objects, annotation uncertainty and source correlation. No material classes, depth, navigation or autonomous removal. Review actual failure images before assigning a cause.

**Rights:** source AGPL-3.0; dataset images, annotations and pretrained weights have separate rights. Public accessibility is not permission to relicense or redistribute. Dataset imagery is not committed to Git.

**Evidence:** `docs/RESULTS.md`, `reports/`, `runs/`, `logs/` and the Git history. Resume wording is generated from the measured reports in `docs/RESUME.md`.

# OceanSight Edge — measured results

Generated from the completed local run. AP values are proportions, not classification accuracy.

## Dataset and protocol

| Split | Images | Videos | Debris boxes | Negative images |
|---|---:|---:|---:|---:|
| train | 1680 | 213 | 1467 | 421 |
| val | 108 | 44 | 94 | 30 |
| test | 108 | 45 | 94 | 24 |

Validation/test files preserved from v1; test was previously inspected and is a reused reference holdout, not a fresh final benchmark.

This is a custom video-disjoint subset, not the official TrashCan benchmark.

## Model comparison on validation

| Detector | Epochs | Input size | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| yolo11n | 30 | 416 | 0.4463 | 0.2255 |
| yolov8n | 30 | 416 | 0.4907 | 0.2312 |

Selected **yolov8n** using validation mAP50–95. One seed; two related YOLO architectures.

## Reference-test evaluation

| Runtime | mAP50 | mAP50–95 | File MB |
|---|---:|---:|---:|
| pytorch_fp32 | 0.4984 | 0.2742 | 6.21 |
| onnx_fp32 | 0.4984 | 0.2742 | 12.14 |
| onnx_int8 | 0.4849 | 0.2677 | 3.35 |

Every backend uses identical square preprocessing, batch one and the same test images. These scores do not select a model or quantization recipe.

## Architecture compute comparison

| Detector | Parameters | Checkpoint MB | CPU median ms | CPU P95 ms |
|---|---:|---:|---:|---:|
| yolo11n | 2,590,035 | 5.43 | 27.18 | 47.27 |
| yolov8n | 3,011,043 | 6.21 | 26.10 | 46.22 |

Native PyTorch FP32 fused forward only; identical validation images, four CPU threads, batch one; rotating detector order.

Accuracy-first model selection uses validation AP; these timings document the additional compute/size tradeoff.

## CPU timing

| Runtime | Median ms | P95 ms | Timed calls |
|---|---:|---:|---:|
| pytorch_fp32 | 39.50 | 58.64 | 90 |
| onnx_fp32 | 16.98 | 28.87 | 90 |
| onnx_int8 | 21.78 | 35.00 | 90 |

Warm network forward only; excludes decode, letterbox and NMS. Mac CPU, not a physical robot/Jetson/Pi.
Timing was measured in an interactive workstation session after training, without controlling power or thermal state. Architecture and runtime tables were measured separately; compare values within each table, not across sessions.
Apple M4 / 16 GB; 4 CPU threads; 416×416; 5 warmups; 30 validation images × 3 repeats; rotating backend order.

PyTorch checkpoints and FP32 ONNX use different serialization/storage conventions. Compare INT8 with FP32 ONNX when calculating compression. There are no physical edge-device or power measurements.

## Quantization diagnosis and gate

The archived first experiment quantized a combined coordinate/confidence output at scale 2.0351. All confidence scores became zero on ten checked validation images. The revised recipe uses shape preprocessing and Conv-only QDQ quantization, preserving box/confidence output calculations in floating point.

Validation gate **PASSED**: INT8 validation mAP50-95 loss <= 0.02 absolute, and AP > 0.

- onnx_fp32: validation mAP50–95 0.2312
- onnx_int8: validation mAP50–95 0.2142

FP32 is the accuracy-first demo default. Passing the gate allows the compact option; it does not imply faster inference.

## Fixed-threshold errors and uncertainty

At confidence 0.25 and IoU 0.5: **49 TP, 36 FP, 45 misses**.

Precision 0.576, 95% video-bootstrap interval [0.443, 0.727]. Recall 0.521, interval [0.393, 0.653].

Resample whole source videos with replacement; pool TP/FP/FN for each replicate. Conditional on this fixed model and reference holdout. Not training-seed uncertainty, AP intervals or evidence of location independence.

The custom demo uses fixed confidence and NMS thresholds, so these counts differ in definition from AP and validator precision/recall at its selected operating point.

## Unsupervised review

Extracted frozen 512-D ResNet18 features for 1680 training images. PCA(32) retains 63.0% of standardized feature variance; KMeans and Isolation Forest produce clusters and a video-diverse review queue.

The separate interpretable audit uses color, brightness, contrast and sharpness. Neither analysis assigns verified semantic class names or automatically deletes images.

## Comparison with the initial experiment

The expanded model improves validation AP over the initial run, but reference-test mAP50–95 is 0.2742 versus 0.2871 in v1. More data/training did not improve every score. Data volume, epochs and input size changed together, so this is not an isolated ablation or proof of a statistically significant difference.

## Scope of evidence

Training, local API/UI checks, numerical parity, failure analysis and container verification are recorded in the repository. Remote GitHub Actions execution, full original archive verification, additional detector families, multiple training seeds, physical edge hardware, calibrated confidence and unique-object tracking are not claimed.

## Additional fresh audit

After model/export selection, the FP32 deployment was evaluated on 87 images from 9 previously unused videos: mAP50 **0.9208**, mAP50–95 **0.6523**.

Small, availability-defined remaining-video sample. Not an official benchmark or a location-disjoint population estimate. Evaluate the already selected FP32 deployment only; do not tune on this audit.
The large gap from reference-test AP shows how strongly results depend on these small video samples. Do not present the higher audit score alone as general ocean performance.

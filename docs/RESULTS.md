# Measured results

Generated from local report files. Metrics are proportions, not percentages. Missing steps are explicitly marked.

## Dataset

| Split | Images | Videos | Debris boxes | Negative images |
|---|---:|---:|---:|---:|
| train | 503 | 199 | 415 | 143 |
| val | 108 | 44 | 94 | 30 |
| test | 108 | 45 | 94 | 24 |

Custom video-disjoint subset; not the official TrashCan benchmark.

## Validation model comparison

| Model | Epochs | mAP50 | mAP50–95 | Checkpoint MB |
|---|---:|---:|---:|---:|
| yolo11n | 10 | 0.3835 | 0.1855 | 5.42 |
| yolov8n | 10 | 0.3573 | 0.1981 | 6.20 |

Selected: **yolov8n**, using validation mAP50–95.

## Held-out test

| Backend | mAP50 | mAP50–95 | File MB |
|---|---:|---:|---:|
| pytorch_fp32 | 0.5086 | 0.2871 | 6.20 |
| onnx_fp32 | 0.5086 | 0.2871 | 12.11 |
| onnx_int8 | 0.0000 | 0.0000 | 3.40 |

## CPU forward benchmark

| Backend | Median ms | P95 ms | Repeated samples |
|---|---:|---:|---:|
| pytorch_fp32 | 8.84 | 11.11 | 90 |
| onnx_fp32 | 4.16 | 4.48 | 90 |
| onnx_int8 | 3.94 | 4.67 | 90 |

warm forward pass only; excludes image decode, letterbox and NMS; not a Jetson or Raspberry Pi benchmark
Batch 1, 4 CPU threads, 320×320; 5 warmup calls, 30 images × 3 repetitions. Sequential backend timing on an M4 MacBook Air; no confidence intervals or power measurements.

Checkpoint and ONNX file sizes use different serialization/precision conventions. Compare INT8 size with FP32 ONNX for the quantization compression ratio; the PyTorch checkpoint is not an FP32 tensor-storage baseline.

## Quantization

Status: completed

**Conversion succeeded but detection quality failed: INT8 test AP is zero. This model is not suitable for the demo.** The unselective quantization recipe remains as an explicitly failed experiment. Investigate output-range distortion and selective operator quantization on training/validation data in a future experiment; no further parameters were tuned against this test set.

## Demo-pipeline error analysis

At confidence 0.25 and IoU 0.5: **46 true positives, 33 false positives, 48 missed boxes**. Precision 0.582; recall 0.489.

These fixed-threshold counts come from the custom ONNX demo preprocessing/NMS. They differ in definition from AP and from the validator’s best-F1 operating-point precision/recall. See the locally generated failure contact sheet for the highest-error examples.

## Interpretation and next work

This is a short, single-seed subset experiment. It demonstrates the full workflow but does not establish reliable open-water performance. The comparison is between two related YOLO architectures. FP32 ONNX remains the app default; quantization is an experimental result, not an assumed improvement.

**Not run:** full original dataset verification/training, RT-DETR/Faster R-CNN, multi-seed uncertainty, semantic embedding analysis, physical edge-device profiling, calibrated confidence, tracking, material classification and production deployment.

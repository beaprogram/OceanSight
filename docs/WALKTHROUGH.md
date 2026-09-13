# Explain OceanSight Edge in an interview

## 1. Start with the operational problem

“Underwater footage takes time to review. I built a local tool that highlights candidate marine debris and shows the evidence behind its limitations.” The deliverable is a review workflow, not just a notebook, and not an autonomous cleanup robot.

## 2. Explain the data decision

Adjacent video frames look almost identical. A random image split can make the model appear to generalize when it has seen almost the same scene. We assign whole source videos to one split, then sample spread-out frames. This is why `group_id()` and the split-integrity test matter more than a decorative accuracy badge.

The binary class is a scope decision: learn where trash is before classifying its material. Animal/plant/ROV-only images provide useful negative examples. The mirror's annotation tables may omit fully unannotated images, so its background distribution is not the same as continuous real footage.

## 3. Explain training and model selection

Transfer learning starts with features learned from a general image dataset. Thirty epochs adapt those features to this small underwater sample. YOLO11n and YOLOv8n get the same input size, batch size, seed, split and epoch count; their wall-clock compute is not identical.

Validation chooses the winning checkpoint and detector. Test data is only used after selection. mAP50–95 checks whether confident boxes line up with ground truth at several localization thresholds. A high score on 108 test images is still narrow evidence, and one seed is not a significance test.

## 4. Explain unsupervised analysis

We do not supply labels to KMeans or Isolation Forest. One pipeline sees brightness, contrast, sharpness, color and saturation. A second uses frozen 512-dimensional ResNet18 features; PCA reduces those standardized features to 32 dimensions before clustering and outlier ranking. KMeans groups similar appearances; Isolation Forest ranks uncommon combinations. PCA compresses the standardized feature space for a plot. An unusual image could be valuable data, not junk. We therefore produce a review queue and keep all images.

## 5. Explain the edge tradeoff

ONNX represents the network for a deployment runtime. Static INT8 quantization uses representative training images to estimate activation ranges and stores many computations at lower precision. It can shrink a file yet run slower when CPU kernels or conversion overhead dominate. We measure that rather than assuming success.

Read `benchmark.json`: batch size, warmup, CPU thread count, number of images and included operations define what latency means. The app's pipeline time includes preprocessing and NMS, while the benchmark measures network forward time. Neither is a hardware claim about a Raspberry Pi.

## 6. Demonstrate a failure before proposing a fix

Open the failure contact sheet. Green is a prediction, blue is ground truth. At confidence .25 and IoU .5, unmatched predictions are false positives; unmatched labels are misses. A poorly localized detection can count as both. Inspect the actual image before claiming blur, occlusion, or label errors caused it.

Possible next experiments: higher input resolution for small objects, more video groups, longer training, manual label review, or a genuinely different detector family. Keep the test set out of that tuning loop by creating a new final holdout when extending this work.

## A concise demo sequence

1. Open a held-out sample; inspect boxes and adjust confidence to show the recall/false-alarm tradeoff.
2. Show validation comparison and final test results. Quote only the numbers in the generated report.
3. Show an outlier and explain why it needs human review.
4. Show an actual failure, then the ONNX/INT8 speed and quality tradeoff.
5. Close with the next experiment and the deployment constraints you have not tested.

Do not claim material classification, verified semantic cluster labels, Jetson performance, real-time video tracking, or broad robustness. Do not claim you personally wrote every line: this build was developed with AI assistance; demonstrate understanding by explaining and modifying it.

## The diagnosed INT8 failure

The first export quantized coordinates and confidence scores together at a step of about 2.04. Probabilities from 0 to 1 collapsed to zero. The corrected recipe compresses convolution layers and preserves the final output math in floating point. Show `quantization_diagnosis.json` and the validation gate: a smaller file is only useful if detections survive.

## Prove it can run elsewhere

Run a real image through the Docker API. Explain why its requirements do not include PyTorch: the exported ONNX graph is enough for inference. Distinguish the container readiness check, prediction response and model fingerprint from the separate native CPU speed experiment.

## Reproduce the debugging probe on the original Mac

`python -m oceansight.quant_diagnose` inspects the archived v1 output range. `python -m oceansight.quant_probe` validates the selective fix on those same archived weights with training-only calibration and validation evaluation. These commands require `models/baseline_v1/`, which is preserved locally.

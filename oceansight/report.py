"""Build results and resume wording strictly from the completed release evidence."""

import json
from .common import ROOT, REPORTS


def run():
    def read(name):
        return json.loads((REPORTS / name).read_text())

    data = read("dataset.json")
    comparison = read("comparison.json")
    selection = read("selection.json")
    metrics = read("test_metrics.json")
    benchmark = read("benchmark.json")
    gate = read("quantization_gate.json")
    failures = read("failure_cases.json")
    uncertainty = read("uncertainty.json")
    embeddings = read("embeddings.json")
    lines = [
        "# OceanSight Edge — measured results",
        "",
        "Generated from the completed local run. AP values are proportions, not classification accuracy.",
        "",
        "## Dataset and protocol",
        "",
        "| Split | Images | Videos | Debris boxes | Negative images |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, r in data["counts"].items():
        lines.append(
            f"| {split} | {r['images']} | {r['videos']} | {r['boxes']} | {r['negative_images']} |"
        )
    lines += [
        "",
        data["evaluation_note"],
        "",
        "This is a custom video-disjoint subset, not the official TrashCan benchmark.",
        "",
        "## Model comparison on validation",
        "",
        "| Detector | Epochs | Input size | mAP50 | mAP50–95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in comparison:
        v = r["validation"]
        lines.append(
            f"| {r['model']} | {r['epochs']} | {r['imgsz']} | {v['metrics/mAP50(B)']:.4f} | {v['metrics/mAP50-95(B)']:.4f} |"
        )
    lines += [
        "",
        f"Selected **{selection['selected']}** using validation mAP50–95. One seed; two related YOLO architectures.",
        "",
        "## Reference-test evaluation",
        "",
        "| Runtime | mAP50 | mAP50–95 | File MB |",
        "|---|---:|---:|---:|",
    ]
    for r in metrics:
        v = r["test"]
        lines.append(
            f"| {r['backend']} | {v['metrics/mAP50(B)']:.4f} | {v['metrics/mAP50-95(B)']:.4f} | {r['size_mb']:.2f} |"
        )
    lines += [
        "",
        "Every backend uses identical square preprocessing, batch one and the same test images. These scores do not select a model or quantization recipe.",
        "",
        "## CPU timing",
        "",
        "| Runtime | Median ms | P95 ms | Timed calls |",
        "|---|---:|---:|---:|",
    ]
    for r in benchmark["results"]:
        lines.append(
            f"| {r['backend']} | {r['median_ms']:.2f} | {r['p95_ms']:.2f} | {r['samples']} |"
        )
    lines += [
        "",
        benchmark["scope"],
        f"Apple M4 / 16 GB; {benchmark['threads']} CPU threads; {benchmark['imgsz']}×{benchmark['imgsz']}; 5 warmups; 30 validation images × 3 repeats; rotating backend order.",
        "",
        "PyTorch checkpoints and FP32 ONNX use different serialization/storage conventions. Compare INT8 with FP32 ONNX when calculating compression. There are no physical edge-device or power measurements.",
        "",
        "## Quantization diagnosis and gate",
        "",
        "The archived first experiment quantized a combined coordinate/confidence output at scale 2.0351. All confidence scores became zero on ten checked validation images. The revised recipe uses shape preprocessing and Conv-only QDQ quantization, preserving box/confidence output calculations in floating point.",
        "",
        f"Validation gate **{'PASSED' if gate['passed'] else 'FAILED'}**: {gate['criterion']}.",
        "",
    ]
    for r in gate["validation"]:
        lines.append(
            f"- {r['backend']}: validation mAP50–95 {r['metrics']['metrics/mAP50-95(B)']:.4f}"
        )
    lines += [
        "",
        "FP32 is the accuracy-first demo default. Passing the gate allows the compact option; it does not imply faster inference.",
        "",
        "## Fixed-threshold errors and uncertainty",
        "",
    ]
    t = failures["totals"]
    lo_p, hi_p = uncertainty["precision_95_percentile_interval"]
    lo_r, hi_r = uncertainty["recall_95_percentile_interval"]
    lines += [
        f"At confidence {failures['confidence']} and IoU {failures['iou']}: **{t['tp']} TP, {t['fp']} FP, {t['fn']} misses**.",
        "",
        f"Precision {t['precision']:.3f}, 95% video-bootstrap interval [{lo_p:.3f}, {hi_p:.3f}]. Recall {t['recall']:.3f}, interval [{lo_r:.3f}, {hi_r:.3f}].",
        "",
        uncertainty["method"] + " " + uncertainty["limitations"],
        "",
        "The custom demo uses fixed confidence and NMS thresholds, so these counts differ in definition from AP and validator precision/recall at its selected operating point.",
        "",
        "## Unsupervised review",
        "",
        f"Extracted frozen 512-D ResNet18 features for {embeddings['images']} training images. PCA(32) retains {embeddings['explained_variance']:.1%} of standardized feature variance; KMeans and Isolation Forest produce clusters and a video-diverse review queue.",
        "",
        "The separate interpretable audit uses color, brightness, contrast and sharpness. Neither analysis assigns verified semantic class names or automatically deletes images.",
        "",
        "## Scope of evidence",
        "",
        "Training, local API/UI checks, numerical parity, failure analysis and container verification are recorded in the repository. Remote GitHub Actions execution, full original archive verification, additional detector families, multiple training seeds, physical edge hardware, calibrated confidence and unique-object tracking are not claimed.",
        "",
    ]
    (ROOT / "docs" / "RESULTS.md").write_text("\n".join(lines))
    models = {r["backend"]: r for r in metrics}
    times = {r["backend"]: r for r in benchmark["results"]}
    fp = models["onnx_fp32"]
    pt_time = times["pytorch_fp32"]["median_ms"]
    fp_time = times["onnx_fp32"]["median_ms"]
    resume = [
        "# Resume-ready project wording",
        "",
        "**OceanSight Edge — Marine Debris Detection & Edge Inference**",
        "*Python, PyTorch, Ultralytics, OpenCV, scikit-learn, ONNX Runtime, FastAPI, Streamlit, Docker*",
        "",
        f"- Built an end-to-end underwater debris detection pipeline; compared YOLO11n and YOLOv8n using video-disjoint data and achieved {fp['test']['metrics/mAP50(B)']:.3f} mAP50 on a 108-image reference holdout.",
        f"- Exported the selected detector to ONNX, measuring {fp_time:.2f} ms median CPU forward latency versus {pt_time:.2f} ms for PyTorch on Apple M4; served image/video inference through FastAPI, Streamlit and a Docker container.",
        f"- Analyzed {embeddings['images']:,} training images with ResNet18 embeddings, PCA, KMeans and Isolation Forest; added video-grouped error analysis and diagnosed confidence collapse in an INT8 export.",
        "",
        "## How to use these bullets",
        "",
        "Use two or three bullets depending on space. Add your real GitHub URL only after publishing; no repository URL is invented here. The measured latency excludes image decoding, preprocessing and NMS. Do not call mAP classification accuracy, claim Jetson/Pi performance, or describe the reused reference test set as a new untouched benchmark.",
        "",
        "Only list the project after you can run it and explain its data split, evaluation, quantization failure and remaining limitations. The build used AI assistance; do not imply unaided authorship.",
        "",
    ]
    if gate["passed"] and "onnx_int8" in models:
        q = models["onnx_int8"]
        reduction = 1 - q["size_mb"] / fp["size_mb"]
        resume += [
            "## Optional optimization bullet",
            "",
            f"- Reduced the ONNX model file by {reduction:.1%} with selective INT8 quantization, passing a validation gate of no more than 0.02 absolute mAP50–95 loss; measured quality and latency separately.",
            "",
        ]
    (ROOT / "docs" / "RESUME.md").write_text("\n".join(resume))


if __name__ == "__main__":
    run()

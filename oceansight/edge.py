"""Selective INT8 export with a validation quality gate and matched CPU evaluation."""

import hashlib
import os
import shutil
import json
import platform
import time

import cv2
import numpy as np

from .common import ROOT, DATA, MODELS, REPORTS, setup, save_json
from .inference import Detector, preprocess


def metric_dict(result):
    return {k: float(v) for k, v in result.results_dict.items()}


def run():
    setup()
    import torch
    import onnx
    from ultralytics import YOLO
    from onnxruntime.quantization import (
        CalibrationDataReader,
        quantize_static,
        QuantFormat,
        QuantType,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    torch.set_num_threads(4)
    selection = json.loads((REPORTS / "selection.json").read_text())
    size = selection["imgsz"]
    export_dir = MODELS / "pending"
    export_dir.mkdir(exist_ok=True)
    shutil.copy2(MODELS / "best.pt", export_dir / "best.pt")
    YOLO(export_dir / "best.pt").export(
        format="onnx", imgsz=size, opset=17, simplify=False, dynamic=False, batch=1, device="cpu"
    )
    fp = export_dir / "best.onnx"
    q = export_dir / "best_int8.onnx"
    prepared = export_dir / "best_preprocessed.onnx"
    records = json.loads((DATA / "manifest.json").read_text())
    seen = set()
    calibration = []
    for r in records:
        if r["split"] == "train" and r["group"] not in seen:
            calibration.append(r)
            seen.add(r["group"])
            if len(calibration) == 128:
                break

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.items = iter(calibration)

        def get_next(self):
            r = next(self.items, None)
            if r is None:
                return None
            image = cv2.imread(str(DATA / "images" / "train" / r["file_name"]))
            return {"images": preprocess(image, size)[0]}

    quant_status = {"status": "not_run"}
    try:
        quant_pre_process(str(fp), str(prepared), skip_symbolic_shape=True)
        graph = onnx.load(str(prepared))
        excluded = [n.name for n in graph.graph.node if "/dfl/" in n.name]
        quantize_static(
            str(prepared),
            str(q),
            Reader(),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            op_types_to_quantize=["Conv"],
            nodes_to_exclude=excluded,
        )
        onnx.checker.check_model(str(q))
        Detector(q)
        quant_status = {
            "status": "completed",
            "calibration_images": len(calibration),
            "calibration_videos": len(seen),
            "calibration_split": "train",
            "recipe": "Shape-inferred graph; Conv-only QDQ; U8 activations/S8 per-channel weights; DFL excluded",
            "excluded_nodes": excluded,
            "calibration_files": [r["file_name"] for r in calibration],
        }
    except Exception as exc:
        quant_status = {"status": "failed", "error": str(exc)}
    save_json(REPORTS / "quantization.json", quant_status)
    candidates = [("pytorch_fp32", MODELS / "best.pt"), ("onnx_fp32", fp)]
    if quant_status["status"] == "completed":
        candidates.append(("onnx_int8", q))

    # Gate quantization on validation before looking at test results.
    validation = []
    for label, path in candidates[1:]:
        result = YOLO(path, task="detect").val(
            data=str(DATA / "dataset.yaml"),
            split="val",
            imgsz=size,
            batch=1,
            device="cpu",
            workers=0,
            rect=False,
            project=str(ROOT / "runs"),
            name=f"v2_gate_{label}",
            plots=False,
            verbose=False,
        )
        validation.append({"backend": label, "metrics": metric_dict(result)})
    passed = False
    if len(validation) == 2:
        fp_ap = validation[0]["metrics"]["metrics/mAP50-95(B)"]
        int_ap = validation[1]["metrics"]["metrics/mAP50-95(B)"]
        passed = int_ap > 0 and fp_ap - int_ap <= 0.02
    gate = {
        "criterion": "INT8 validation mAP50-95 loss <= 0.02 absolute, and AP > 0",
        "passed": passed,
        "validation": validation,
    }
    save_json(REPORTS / "quantization_gate.json", gate)

    rows = []
    for label, path in candidates:
        metrics = YOLO(path, task="detect").val(
            data=str(DATA / "dataset.yaml"),
            split="test",
            imgsz=size,
            batch=1,
            device="cpu",
            workers=0,
            rect=False,
            project=str(ROOT / "runs"),
            name=f"v2_test_{label}",
            plots=True,
            verbose=False,
        )
        rows.append(
            {
                "backend": label,
                "size_mb": path.stat().st_size / 1e6,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "test": metric_dict(metrics),
                "evaluation_note": "Same reference holdout previously inspected in v1; no model or quantization selection uses these test scores.",
            }
        )
        save_json(REPORTS / "test_metrics.json", rows)

    # Round-robin timings reduce backend-order and thermal drift confounding.
    sample_records = [r for r in records if r["split"] == "val"][:30]
    tensors = [
        preprocess(cv2.imread(str(DATA / "images" / "val" / r["file_name"])), size)[0]
        for r in sample_records
    ]
    pt = YOLO(MODELS / "best.pt").model.float().eval()
    pt.fuse()

    @torch.inference_mode()
    def pt_forward(x):
        result = pt(torch.from_numpy(x))
        return result[0].numpy() if isinstance(result, tuple) else result.numpy()

    forwards = {"pytorch_fp32": pt_forward}
    for label, path in candidates[1:]:
        detector = Detector(path)
        forwards[label] = lambda x, d=detector: d.session.run(None, {d.name: x})[0]
    for forward in forwards.values():
        for _ in range(5):
            forward(tensors[0])
    timings = {label: [] for label in forwards}
    labels = list(forwards)
    for repeat in range(3):
        for index, x in enumerate(tensors):
            shift = (repeat + index) % len(labels)
            for label in labels[shift:] + labels[:shift]:
                start = time.perf_counter()
                forwards[label](x)
                timings[label].append((time.perf_counter() - start) * 1000)
    benchmark = [
        {
            "backend": label,
            "samples": len(values),
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "mean_ms": float(np.mean(values)),
        }
        for label, values in timings.items()
    ]
    parity = {}
    for label in labels[1:]:
        differences = [np.abs(forwards[label](x) - pt_forward(x)) for x in tensors[:10]]
        parity[label] = {
            "max_absolute_error": float(max(d.max() for d in differences)),
            "mean_absolute_error": float(np.mean([d.mean() for d in differences])),
        }
    save_json(
        REPORTS / "benchmark.json",
        {
            "hardware": platform.platform(),
            "processor": platform.machine(),
            "chip": "Apple M4",
            "memory_gb": 16,
            "threads": 4,
            "batch": 1,
            "imgsz": size,
            "scope": "Warm network forward only; excludes decode, letterbox and NMS. Mac CPU, not a physical robot/Jetson/Pi.",
            "warmup": 5,
            "unique_images": len(tensors),
            "repeat": 3,
            "timing_order": "rotating round-robin backends per image",
            "image_split": "validation",
            "results": benchmark,
            "raw_output_parity_10_validation_images": parity,
        },
    )
    # Only promote complete, evaluated artifacts into the live demo paths.
    for artifact in [fp] + ([q] if quant_status["status"] == "completed" else []):
        staging = MODELS / (artifact.name + ".next")
        shutil.copy2(artifact, staging)
        os.replace(staging, MODELS / artifact.name)
    save_json(
        REPORTS / "deployment.json",
        {
            "model": "best.onnx",
            "default_confidence": 0.25,
            "int8_quality_gate_passed": passed,
            "reason": "FP32 ONNX remains the accuracy-first default; INT8 is available only as a validated compression experiment when its validation gate passes.",
        },
    )
    print(benchmark)


if __name__ == "__main__":
    run()

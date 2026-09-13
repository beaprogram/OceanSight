"""Measure both trained detector architectures on identical validation tensors."""

import json
import time
import cv2
import numpy as np
from .common import ROOT, DATA, REPORTS, save_json, setup
from .inference import preprocess


def run():
    setup()
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(4)
    comparison = json.loads((REPORTS / "comparison.json").read_text())
    if len(comparison) != 2:
        raise ValueError("Complete both model experiments first")
    size = comparison[0]["imgsz"]
    records = [r for r in json.loads((DATA / "manifest.json").read_text()) if r["split"] == "val"][
        :30
    ]
    tensors = [
        torch.from_numpy(
            preprocess(cv2.imread(str(DATA / "images" / "val" / r["file_name"])), size)[0]
        )
        for r in records
    ]
    networks = []
    for r in comparison:
        model = YOLO(ROOT / r["checkpoint"]).model.float().eval()
        model.fuse()
        networks.append(model)
    times = [[] for _ in networks]
    with torch.inference_mode():
        for model in networks:
            for _ in range(5):
                model(tensors[0])
        for repeat in range(3):
            for i, tensor in enumerate(tensors):
                order = [0, 1] if (i + repeat) % 2 == 0 else [1, 0]
                for index in order:
                    start = time.perf_counter()
                    networks[index](tensor)
                    times[index].append((time.perf_counter() - start) * 1000)
    rows = [
        {
            "model": r["model"],
            "validation_map50_95": r["validation"]["metrics/mAP50-95(B)"],
            "parameters": r["parameters"],
            "checkpoint_mb": r["size_mb"],
            "median_ms": float(np.median(times[i])),
            "p95_ms": float(np.percentile(times[i], 95)),
            "calls": len(times[i]),
        }
        for i, r in enumerate(comparison)
    ]
    save_json(
        REPORTS / "architecture_benchmark.json",
        {
            "scope": "Native PyTorch FP32 fused forward only; identical validation images, four CPU threads, batch one; rotating detector order.",
            "imgsz": size,
            "warmup": 5,
            "images": len(tensors),
            "repeats": 3,
            "selection_note": "Accuracy-first model selection uses validation AP; these timings document the additional compute/size tradeoff.",
            "results": rows,
        },
    )


if __name__ == "__main__":
    run()

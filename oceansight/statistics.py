"""Video-cluster bootstrap uncertainty for fixed-threshold detection counts."""

import json
from collections import defaultdict
import numpy as np
from .common import REPORTS, save_json


def bootstrap_counts(rows, repetitions=2000, seed=42):
    groups = defaultdict(lambda: np.zeros(3, dtype=int))
    for r in rows:
        groups[r["video"]] += np.array([r["tp"], r["fp"], r["fn"]])
    if not groups:
        raise ValueError("No evaluated videos")
    values = np.array(list(groups.values()))
    rng = np.random.default_rng(seed)
    summed = values[rng.integers(0, len(values), size=(repetitions, len(values)))].sum(axis=1)
    precision = summed[:, 0] / np.maximum(summed[:, 0] + summed[:, 1], 1)
    recall = summed[:, 0] / np.maximum(summed[:, 0] + summed[:, 2], 1)
    return {
        "video_groups": len(values),
        "repetitions": repetitions,
        "seed": seed,
        "precision_95_percentile_interval": np.quantile(precision, [0.025, 0.975]).tolist(),
        "recall_95_percentile_interval": np.quantile(recall, [0.025, 0.975]).tolist(),
        "method": "Resample whole source videos with replacement; pool TP/FP/FN for each replicate.",
        "limitations": "Conditional on this fixed model and reference holdout. Not training-seed uncertainty, AP intervals or evidence of location independence.",
    }


def run():
    failures = json.loads((REPORTS / "failure_cases.json").read_text())
    save_json(REPORTS / "uncertainty.json", bootstrap_counts(failures["images"]))


if __name__ == "__main__":
    run()

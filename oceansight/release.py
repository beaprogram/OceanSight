"""Package committed source plus fingerprint-verified inference models; no dataset images."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from .common import ROOT, REPORTS, MODELS


def build(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new release filename; existing files are not overwritten")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise RuntimeError("Commit the completed source and reports before packaging")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    metrics = {r["backend"]: r for r in json.loads((REPORTS / "test_metrics.json").read_text())}
    selection = json.loads((REPORTS / "selection.json").read_text())
    benchmark = json.loads((REPORTS / "benchmark.json").read_text())
    comparison = json.loads((REPORTS / "comparison.json").read_text())
    if len(comparison) != 2 or any(r["imgsz"] != selection["imgsz"] for r in comparison):
        raise ValueError("Both model experiments and selection must be complete and consistent")
    if selection["imgsz"] != benchmark["imgsz"]:
        raise ValueError("Evaluation and selected model have mismatched input sizes")
    deployment = json.loads((REPORTS / "deployment.json").read_text())
    models = [("onnx_fp32", "best.onnx")]
    if deployment.get("int8_quality_gate_passed"):
        models.append(("onnx_int8", "best_int8.onnx"))
    fingerprints = {}
    for backend, name in models:
        digest = hashlib.sha256((MODELS / name).read_bytes()).hexdigest()
        if digest != metrics[backend]["sha256"]:
            raise ValueError(f"{name} differs from evaluated artifact")
        fingerprints[name] = {"sha256": digest, "bytes": (MODELS / name).stat().st_size}
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name in files:
            if name and (ROOT / name).is_file():
                archive.write(ROOT / name, "OceanSight-Edge/" + name)
        for _, name in models:
            archive.write(MODELS / name, "OceanSight-Edge/models/" + name)
        archive.writestr(
            "OceanSight-Edge/ARTIFACTS.json",
            json.dumps(
                {
                    "source_commit": commit,
                    "models": fingerprints,
                    "dataset_images_included": False,
                    "purpose": "Local portable inference handoff; dataset and pretrained-model rights remain separate.",
                },
                indent=2,
            ),
        )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "archive": str(output),
                "bytes": output.stat().st_size,
                "source_commit": commit,
                "models": fingerprints,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    build(p.parse_args().output)

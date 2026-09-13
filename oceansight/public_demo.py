"""Fetch only fingerprint-verified release models and a few reference examples."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
import requests
from .common import MODELS, REPORTS, DATA


def fetch_verified(url, destination, sha256, max_bytes=20_000_000):
    destination = Path(destination)
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == sha256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with requests.get(url, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as file:
                temporary = Path(file.name)
                digest = hashlib.sha256()
                total = 0
                for chunk in response.iter_content(65536):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("Download exceeds expected size")
                    digest.update(chunk)
                    file.write(chunk)
        if digest.hexdigest() != sha256:
            raise ValueError("Downloaded artifact fingerprint does not match evaluated version")
        os.replace(temporary, destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare():
    metrics = json.loads((REPORTS / "test_metrics.json").read_text())
    for backend, filename in [("onnx_fp32", "best.onnx"), ("onnx_int8", "best_int8.onnx")]:
        record = next(r for r in metrics if r["backend"] == backend)
        fetch_verified(
            f"https://github.com/beaprogram/OceanSight/releases/download/v1.0.0/{filename}",
            MODELS / filename,
            record["sha256"],
        )
    examples = {
        "vid_000080_frame0000077.jpg",
        "vid_000205_frame0000041.jpg",
        "vid_000320_frame0000056.jpg",
    }
    manifest = json.loads((REPORTS / "manifest.json").read_text())
    for r in manifest:
        if r["file_name"] in examples and r["split"] == "test":
            base = "https://huggingface.co/datasets/anyaeross/trashcan1/resolve/52a49e9cb66002def6777e99d47e04301988f211"
            fetch_verified(
                f"{base}/{r['source_split']}/{r['file_name']}",
                DATA / "images" / "test" / r["file_name"],
                r["sha256"],
                5_000_000,
            )

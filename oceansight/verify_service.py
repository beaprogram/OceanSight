"""Verify a running service against the local model on a real reference image."""

import argparse
import hashlib
import json
import cv2
import numpy as np
import requests
from .common import DATA, MODELS, REPORTS, save_json
from .inference import Detector


def run(url):
    model = MODELS / "best.onnx"
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    records = json.loads((DATA / "manifest.json").read_text())
    record = next(r for r in records if r["split"] == "test" and r["debris_boxes"])
    source = DATA / "images" / "test" / record["file_name"]
    ready = requests.get(url + "/ready", timeout=30)
    ready.raise_for_status()
    assert ready.json()["model_sha256"] == digest
    with source.open("rb") as file:
        response = requests.post(
            url + "/predict", files={"file": (source.name, file, "image/jpeg")}, timeout=30
        )
    response.raise_for_status()
    actual = response.json()
    expected, _ = Detector(model).predict(cv2.imread(str(source)))
    assert actual["model_sha256"] == digest
    assert len(actual["detections"]) == len(expected)
    for a, b in zip(actual["detections"], expected):
        np.testing.assert_allclose(a["box"], b["box"], atol=0.1, rtol=1e-4)
        np.testing.assert_allclose(a["confidence"], b["confidence"], atol=1e-4)
    invalid = requests.post(
        url + "/predict", files={"file": ("bad.jpg", b"not an image", "image/jpeg")}, timeout=30
    )
    assert invalid.status_code == 422
    bad_threshold = requests.post(
        url + "/predict?confidence=2", files={"file": ("bad.jpg", b"bad", "image/jpeg")}, timeout=30
    )
    assert bad_threshold.status_code == 422
    save_json(
        REPORTS / "container_verification.json",
        {
            "url": url,
            "model_sha256": digest,
            "reference_image": source.name,
            "readiness": ready.json(),
            "prediction": actual,
            "native_detections": expected,
            "parity_passed": True,
            "invalid_image_status": invalid.status_code,
            "invalid_threshold_status": bad_threshold.status_code,
            "scope": "Real-image service/native parity and input rejection; container latency is not an edge hardware benchmark.",
        },
    )
    print("Service readiness, real-image parity and invalid-input checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8502")
    run(parser.parse_args().url.rstrip("/"))

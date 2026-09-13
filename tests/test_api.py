import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from oceansight.api import app, decode_image
from oceansight.common import MODELS


def test_api_validation_and_missing_model(monkeypatch):
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert (
        client.post(
            "/predict", files={"file": ("bad.jpg", b"not an image", "image/jpeg")}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/predict?confidence=2", files={"file": ("bad.jpg", b"x", "image/jpeg")}
        ).status_code
        == 422
    )
    monkeypatch.setenv("OCEANSIGHT_MODEL", "/nonexistent/oceansight.onnx")
    assert client.get("/ready").status_code == 503


def test_image_size_guard():
    with pytest.raises(ValueError):
        decode_image(b"")
    with pytest.raises(ValueError):
        decode_image(b"x" * (10 * 1024 * 1024 + 1))


def test_api_real_model():
    if not (MODELS / "best.onnx").exists():
        pytest.skip("Requires exported local model")
    client = TestClient(app)
    assert client.get("/ready").status_code == 200
    _, encoded = cv2.imencode(".png", np.zeros((100, 200, 3), np.uint8))
    response = client.post("/predict", files={"file": ("test.png", encoded.tobytes(), "image/png")})
    assert response.status_code == 200
    result = response.json()
    assert result["width"] == 200 and result["height"] == 100
    assert len(result["model_sha256"]) == 64
    assert result["pipeline_ms"] > 0

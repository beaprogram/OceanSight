"""Local prediction service: uvicorn oceansight.api:app --host 127.0.0.1."""

import hashlib
import io
import os
from functools import lru_cache
from pathlib import Path
from threading import BoundedSemaphore

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

from .common import MODELS
from .inference import Detector

app = FastAPI(
    title="OceanSight Edge",
    version="1.0.0",
    description="Local marine-debris candidate detection; one class; human review required.",
)
MAX_BYTES = 10 * 1024 * 1024
gate = BoundedSemaphore(2)


class Detection(BaseModel):
    box: tuple[float, float, float, float]
    confidence: float
    label: str


class Prediction(BaseModel):
    detections: list[Detection]
    width: int
    height: int
    pipeline_ms: float
    model_sha256: str
    confidence_threshold: float


def model_path():
    return Path(os.environ.get("OCEANSIGHT_MODEL", str(MODELS / "best.onnx")))


@lru_cache(maxsize=1)
def load_model(path, mtime):
    return Detector(path), hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode_image(raw):
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("Supply an image between 1 byte and 10 MB")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.width * im.height > 20_000_000 or max(im.size) > 6000:
                raise ValueError("Image exceeds 20 megapixels or 6000 pixels per side")
            if im.format not in ("JPEG", "PNG"):
                raise ValueError("Only JPEG and PNG images are supported")
            im.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid JPEG or PNG image") from exc
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the image")
    return image


@app.get("/health")
def health():
    return {"status": "ok", "model_available": model_path().is_file()}


@app.get("/ready")
def ready():
    path = model_path()
    if not path.is_file():
        raise HTTPException(503, "Model artifact is not installed")
    try:
        _, digest = load_model(str(path), path.stat().st_mtime_ns)
    except Exception as exc:
        raise HTTPException(503, "Model artifact could not be loaded") from exc
    return {"ready": True, "model_sha256": digest, "class_names": ["marine_debris"]}


@app.post("/predict", response_model=Prediction)
def predict(file: UploadFile = File(...), confidence: float = Query(0.25, ge=0.05, le=0.95)):
    if not gate.acquire(blocking=False):
        raise HTTPException(503, "Inference is busy; retry shortly")
    try:
        raw = file.file.read(MAX_BYTES + 1)
        try:
            image = decode_image(raw)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        path = model_path()
        if not path.is_file():
            raise HTTPException(503, "Model artifact is not installed")
        detector, digest = load_model(str(path), path.stat().st_mtime_ns)
        detections, ms = detector.predict(image, confidence)
        return Prediction(
            detections=detections,
            width=image.shape[1],
            height=image.shape[0],
            pipeline_ms=ms,
            model_sha256=digest,
            confidence_threshold=confidence,
        )
    finally:
        gate.release()
        file.file.close()

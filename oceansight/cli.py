"""Batch image or bounded video inference with machine-readable outputs."""

import argparse
import json
from pathlib import Path
import cv2
from .common import MODELS
from .inference import Detector, annotate
from .api import decode_image
from .video import process_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODELS / "best.onnx")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=150)
    args = parser.parse_args()
    if not 0.05 <= args.confidence <= 0.95:
        parser.error("confidence must be between .05 and .95")
    if not 1 <= args.max_frames <= 3000:
        parser.error("max-frames must be between 1 and 3000")
    if args.output.exists():
        parser.error("output directory already exists; choose a new directory")
    detector = Detector(args.model)
    args.output.mkdir(parents=True)
    if args.source.suffix.lower() in (".mp4", ".mov"):
        rows = process_video(
            args.source, args.output / "annotated.mp4", detector, args.confidence, args.max_frames
        )
        result = {
            "frames": rows,
            "count_definition": "detections per frame, not unique tracked objects",
        }
    else:
        image = decode_image(args.source.read_bytes())
        detections, ms = detector.predict(image, args.confidence)
        if not cv2.imwrite(str(args.output / "annotated.jpg"), annotate(image, detections)):
            raise RuntimeError("Could not save annotated image")
        result = {
            "detections": detections,
            "pipeline_ms": ms,
            "width": image.shape[1],
            "height": image.shape[0],
        }
    result["model"] = str(args.model)
    result["confidence_threshold"] = args.confidence
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(args.output / "results.json")


if __name__ == "__main__":
    main()

"""Bounded local video processing; counts describe frames, not unique objects."""

import math
import cv2
from .inference import annotate


def process_video(source, output, detector, confidence=0.25, max_frames=150, callback=None):
    cap = cv2.VideoCapture(str(source))
    writer = None
    rows = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and 0 < fps <= 120 else 25
        for i in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            # Bound video memory and output size for a local MVP.
            h, w = frame.shape[:2]
            if max(h, w) > 1280:
                scale = 1280 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2))
            frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError("Could not open the annotated-video encoder")
            detections, ms = detector.predict(frame, confidence)
            annotated = annotate(frame, detections)
            writer.write(annotated)
            rows.append({"frame": i, "detections": len(detections), "pipeline_ms": ms})
            if callback:
                callback(i, annotated)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if not rows:
        raise ValueError("The video could not be decoded")
    return rows

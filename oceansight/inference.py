"""CPU-only ONNX inference with explicit letterboxing and NMS."""

import time
import cv2
import numpy as np


def preprocess(image, size=320):
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (nw, nh))
    left, top = (size - nw) // 2, (size - nh) // 2
    padded = cv2.copyMakeBorder(
        resized,
        top,
        size - nh - top,
        left,
        size - nw - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    tensor = (
        np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
    )
    return tensor, (scale, left, top)


def postprocess(output, transform, shape, confidence=0.25, iou=0.45):
    predictions = output[0].T
    scores = predictions[:, 4]
    keep = scores >= confidence
    xywh = predictions[keep, :4]
    scores = scores[keep]
    if not len(scores):
        return []
    xywh[:, :2] -= xywh[:, 2:] / 2
    indexes = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), confidence, iou)
    scale, left, top = transform
    result = []
    for i in np.asarray(indexes).reshape(-1):
        x, y, w, h = xywh[i]
        box = [
            float(np.clip((x - left) / scale, 0, shape[1])),
            float(np.clip((y - top) / scale, 0, shape[0])),
            float(np.clip((x + w - left) / scale, 0, shape[1])),
            float(np.clip((y + h - top) / scale, 0, shape[0])),
        ]
        if box[2] > box[0] and box[3] > box[1]:
            result.append({"box": box, "confidence": float(scores[i]), "label": "marine_debris"})
    return result


class Detector:
    def __init__(self, path, threads=4):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.name = self.session.get_inputs()[0].name
        self.size = self.session.get_inputs()[0].shape[-1]

    def predict(self, image, confidence=0.25):
        start = time.perf_counter()
        tensor, transform = preprocess(image, self.size)
        output = self.session.run(None, {self.name: tensor})[0]
        detections = postprocess(output, transform, image.shape, confidence)
        return detections, (time.perf_counter() - start) * 1000


def annotate(image, detections):
    out = image.copy()
    for d in detections:
        x1, y1, x2, y2 = map(round, d["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (45, 220, 150), 2)
        cv2.putText(
            out,
            f"debris {d['confidence']:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (45, 220, 150),
            1,
            cv2.LINE_AA,
        )
    return out

"""Diagnose the archived v1 quantization failure using validation images only."""

import json
import cv2
import numpy as np
import onnx
from onnx import numpy_helper
from .common import DATA, MODELS, REPORTS, save_json
from .inference import Detector, preprocess


def run():
    records = [r for r in json.loads((DATA / "manifest.json").read_text()) if r["split"] == "val"][
        :10
    ]
    results = {}
    for filename in ["best.onnx", "best_int8.onnx"]:
        path = MODELS / "baseline_v1" / filename
        d = Detector(path)
        rows = []
        for r in records:
            image = cv2.imread(str(DATA / "images" / "val" / r["file_name"]))
            output = d.session.run(None, {d.name: preprocess(image, d.size)[0]})[0]
            scores = output[0, 4]
            rows.append(
                {
                    "file_name": r["file_name"],
                    "max_confidence": float(scores.max()),
                    "nonzero_scores": int(np.count_nonzero(scores)),
                }
            )
        graph = onnx.load(str(path))
        producer = next(n for n in graph.graph.node if graph.graph.output[0].name in n.output)
        initializers = {i.name: numpy_helper.to_array(i) for i in graph.graph.initializer}
        info = {"output_operator": producer.op_type, "images": rows}
        if producer.op_type == "DequantizeLinear":
            info["output_quantization_scale"] = np.asarray(initializers[producer.input[1]]).tolist()
        results[filename] = info
    save_json(
        REPORTS / "quantization_diagnosis.json",
        {
            "split": "validation",
            "source": "archived v1 exports",
            "results": results,
            "hypothesis": "A shared output quantization range across pixel coordinates and probabilities may round small confidence scores to zero. Inspect the measured output scale and confidence values; Conv-only quantization avoids quantizing this output concatenation.",
        },
    )


if __name__ == "__main__":
    run()

"""Validate the selective-quantization fix on archived v1 weights, using validation only."""

import json
import cv2
from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantFormat, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
import onnx
from oceansight.common import MODELS, DATA, REPORTS, save_json, setup
from oceansight.inference import preprocess


def run():
    setup()
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)
    records = [
        r for r in json.loads((DATA / "manifest.json").read_text()) if r["split"] == "train"
    ][:32]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.items = iter(records)

        def get_next(self):
            r = next(self.items, None)
            if r is None:
                return None
            return {
                "images": preprocess(
                    cv2.imread(str(DATA / "images" / "train" / r["file_name"])), 320
                )[0]
            }

    source = MODELS / "baseline_v1" / "best.onnx"
    prepared = MODELS / "probe_prepared.onnx"
    output = MODELS / "probe_fixed.onnx"
    quant_pre_process(str(source), str(prepared), skip_symbolic_shape=True)
    graph = onnx.load(str(prepared))
    excluded = [n.name for n in graph.graph.node if "/dfl/" in n.name]
    quantize_static(
        str(prepared),
        str(output),
        Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["Conv"],
        nodes_to_exclude=excluded,
    )
    results = []
    for label, path in [("fp32", source), ("conv_int8", output)]:
        metrics = YOLO(path, task="detect").val(
            data=str(DATA / "dataset.yaml"),
            split="val",
            imgsz=320,
            rect=False,
            batch=1,
            device="cpu",
            workers=0,
            plots=False,
            verbose=False,
            project=str(MODELS / "probe_runs"),
            name=label,
        )
        results.append(
            {
                "backend": label,
                "validation": {k: float(v) for k, v in metrics.results_dict.items()},
                "size_mb": path.stat().st_size / 1e6,
            }
        )
    save_json(
        REPORTS / "quantization_probe.json",
        {
            "purpose": "Verify proposed fix on archived v1 weights before final v2 export. Validation only.",
            "calibration_images": 32,
            "calibration_split": "train",
            "recipe": "preprocess + Conv-only QDQ + DFL excluded",
            "results": results,
        },
    )


if __name__ == "__main__":
    run()

"""Reserve previously unused video groups, then evaluate the frozen deployment once."""

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image
from .common import ROOT, DATA, REPORTS, MODELS, setup, save_json
from .data import BASE, group_id, fetch, yolo_box


def prepare():
    setup()
    target = REPORTS / "final_audit_manifest.json"
    reserved = target.exists()
    used = []
    for name in [DATA / "manifest.json", REPORTS / "baseline_v1" / "manifest.json"]:
        used.extend(json.loads(name.read_text()))
    used_groups = {r["group"] for r in used}
    used_hashes = {r["sha256"] for r in used}
    records = {}
    for split in ["train", "val", "test"]:
        with (DATA / "raw" / f"{split}.csv").open() as source:
            for row in csv.DictReader(source):
                name = row["file_name"]
                gid = group_id(name)
                if gid in used_groups:
                    continue
                rec = records.setdefault(
                    name,
                    {"file_name": name, "source_split": split, "group": gid, "annotations": []},
                )
                a = {k: v for k, v in row.items() if k != "file_name"}
                if a not in rec["annotations"]:
                    rec["annotations"].append(a)
    if reserved:
        protocol = json.loads((REPORTS / "final_audit_protocol.json").read_text())
        if hashlib.sha256(target.read_bytes()).hexdigest() != protocol["manifest_sha256"]:
            raise ValueError("Frozen audit manifest changed")
        records = {r["file_name"]: r for r in json.loads(target.read_text())}
        if {r["group"] for r in records.values()} & used_groups:
            raise ValueError("Frozen audit now overlaps an experiment split")
    if not records:
        raise ValueError("No unused source videos remain")

    def download(rec):
        path = DATA / "images" / "final_audit" / rec["file_name"]
        raw = fetch(f"{BASE}/{rec['source_split']}/{rec['file_name']}", path)
        digest = hashlib.sha256(raw).hexdigest()
        if reserved and digest != rec["sha256"]:
            raise ValueError("Downloaded audit image does not match its frozen fingerprint")
        if digest in used_hashes:
            raise ValueError("Final audit contains an exact previously used image")
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
        lines = []
        for a in rec["annotations"]:
            if a["label"].startswith("trash_"):
                box = yolo_box([a[k] for k in ["x_min", "y_min", "x_max", "y_max"]], w, h)
                lines.append("0 " + " ".join(f"{v:.8f}" for v in box))
        label = DATA / "labels" / "final_audit" / Path(rec["file_name"]).with_suffix(".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("\n".join(lines) + "\n" if lines else "")
        return {**rec, "sha256": digest, "width": w, "height": h, "debris_boxes": len(lines)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        manifest = list(pool.map(download, [records[k] for k in sorted(records)]))
    if not reserved:
        save_json(target, manifest)
    (DATA / "final_audit.txt").write_text(
        "\n".join(str(DATA / "images" / "final_audit" / r["file_name"]) for r in manifest) + "\n"
    )
    (DATA / "final_audit.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(DATA),
                "train": "train.txt",
                "val": "val.txt",
                "test": "final_audit.txt",
                "names": {0: "marine_debris"},
            }
        )
    )
    if not reserved:
        save_json(
            REPORTS / "final_audit_protocol.json",
            {
                "reserved_at_utc": datetime.now(timezone.utc).isoformat(),
                "policy": "All annotation-listed frames from every source video absent from BOTH v1 and v2 train/validation/reference-test manifests. Membership frozen before model evaluation.",
                "images": len(manifest),
                "videos": len({r["group"] for r in manifest}),
                "boxes": sum(r["debris_boxes"] for r in manifest),
                "manifest_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "limitations": "Small, availability-defined remaining-video sample. Not an official benchmark or a location-disjoint population estimate. Evaluate the already selected FP32 deployment only; do not tune on this audit.",
            },
        )


def evaluate(recheck=False):
    setup()
    if (REPORTS / "final_audit_results.json").exists() and not recheck:
        raise FileExistsError("Final audit already evaluated; do not silently reuse it for tuning")
    protocol = json.loads((REPORTS / "final_audit_protocol.json").read_text())
    digest = hashlib.sha256((REPORTS / "final_audit_manifest.json").read_bytes()).hexdigest()
    if digest != protocol["manifest_sha256"]:
        raise ValueError("Frozen final-audit membership changed")
    from ultralytics import YOLO
    import torch

    torch.set_num_threads(4)
    size = json.loads((REPORTS / "selection.json").read_text())["imgsz"]
    result = YOLO(MODELS / "best.onnx", task="detect").val(
        data=str(DATA / "final_audit.yaml"),
        split="test",
        imgsz=size,
        rect=False,
        batch=1,
        device="cpu",
        workers=0,
        project=str(ROOT / "runs"),
        name="final_audit_fp32",
        plots=True,
        verbose=False,
    )
    save_json(
        REPORTS / ("final_audit_recheck.json" if recheck else "final_audit_results.json"),
        {
            "backend": "onnx_fp32",
            "recheck": recheck,
            "images": protocol["images"],
            "videos": protocol["videos"],
            "model_sha256": hashlib.sha256((MODELS / "best.onnx").read_bytes()).hexdigest(),
            "imgsz": size,
            "metrics": {k: float(v) for k, v in result.results_dict.items()},
            "protocol": protocol,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "evaluate"])
    p.add_argument(
        "--recheck",
        action="store_true",
        help="Write a separate repeat-evaluation report without overwriting the original fresh audit",
    )
    a = p.parse_args()
    prepare() if a.action == "prepare" else evaluate(a.recheck)

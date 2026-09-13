"""Pinned mirror -> video-disjoint subset -> validated YOLO labels + manifest."""

import argparse
import csv
import hashlib
import io
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
from PIL import Image
import yaml
from .common import DATA, REPORTS, setup, save_json

REVISION = "52a49e9cb66002def6777e99d47e04301988f211"
BASE = f"https://huggingface.co/datasets/anyaeross/trashcan1/resolve/{REVISION}"


def group_id(filename):
    if "_frame" not in filename or Path(filename).name != filename:
        raise ValueError(f"Unexpected image name: {filename}")
    return filename.split("_frame")[0]


def yolo_box(box, width, height):
    x1, y1, x2, y2 = map(float, box)
    import math

    if not all(math.isfinite(x) for x in (x1, y1, x2, y2)):
        raise ValueError("Non-finite annotation")
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Degenerate annotation")
    return [
        (x1 + x2) / (2 * width),
        (y1 + y2) / (2 * height),
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]


def fetch(url, path):
    if path.exists():
        return path.read_bytes()
    last = None
    for _ in range(3):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
            return r.content
        except requests.RequestException as exc:
            last = exc
    raise last


def prepare(limit=720, seed=42, frames_per_video=6, preserve_holdouts=None):
    setup()
    if limit < 20 or frames_per_video < 1:
        raise ValueError("Use limit >= 20 and frames_per_video >= 1")
    records = {}
    for source in ["train", "val", "test"]:
        raw = fetch(f"{BASE}/{source}/metadata.csv", DATA / "raw" / f"{source}.csv")
        for row in csv.DictReader(io.StringIO(raw.decode())):
            name = row["file_name"]
            group_id(name)
            rec = records.setdefault(
                name, {"file_name": name, "source_split": source, "annotations": []}
            )
            annotation = {k: v for k, v in row.items() if k != "file_name"}
            if annotation not in rec["annotations"]:
                rec["annotations"].append(annotation)
    groups = defaultdict(list)
    for name in sorted(records):
        groups[group_id(name)].append(name)
    ids = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    split_ids = {
        "train": ids[: int(n * 0.7)],
        "val": ids[int(n * 0.7) : int(n * 0.85)],
        "test": ids[int(n * 0.85) :],
    }
    chosen = []
    for split, gids in split_ids.items():
        candidates = []
        for gid in gids:
            names = groups[gid]
            # Spread samples across each video rather than taking adjacent frames.
            count = min(frames_per_video, len(names))
            indexes = [round(i * (len(names) - 1) / max(count - 1, 1)) for i in range(count)]
            candidates.extend(names[i] for i in indexes)
        rng.shuffle(candidates)
        cap = int(limit * (0.7 if split == "train" else 0.15))
        for name in candidates[:cap]:
            chosen.append({**records[name], "split": split, "group": group_id(name)})

    if preserve_holdouts:
        import json

        locked = json.loads(Path(preserve_holdouts).read_text())
        held = [r for r in locked if r["split"] in ("val", "test")]
        if {r["group"] for r in chosen if r["split"] == "train"} & {r["group"] for r in held}:
            raise ValueError("Expanded training would overlap locked holdouts")
        chosen = [r for r in chosen if r["split"] == "train"] + held

    def download(rec):
        name, split = rec["file_name"], rec["split"]
        path = DATA / "images" / split / name
        raw = fetch(f"{BASE}/{rec['source_split']}/{name}", path)
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
        with Image.open(io.BytesIO(raw)) as im:
            w, h = im.size
        labels = []
        for a in rec["annotations"]:
            if a["label"].startswith("trash_"):
                b = yolo_box([a[k] for k in ["x_min", "y_min", "x_max", "y_max"]], w, h)
                labels.append("0 " + " ".join(f"{x:.8f}" for x in b))
        labelpath = DATA / "labels" / split / (Path(name).stem + ".txt")
        labelpath.parent.mkdir(parents=True, exist_ok=True)
        labelpath.write_text("\n".join(labels) + ("\n" if labels else ""))
        return {
            **rec,
            "width": w,
            "height": h,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "debris_boxes": len(labels),
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        manifest = list(pool.map(download, chosen))
    # Exact duplicate images across splits invalidate the experiment.
    seen = {}
    for rec in manifest:
        old = seen.setdefault(rec["sha256"], rec["split"])
        if old != rec["split"]:
            raise ValueError("Cross-split duplicate detected; revise split before training")
    save_json(DATA / "manifest.json", manifest)
    for split in split_ids:
        (DATA / f"{split}.txt").write_text(
            "\n".join(
                str(DATA / "images" / split / r["file_name"])
                for r in manifest
                if r["split"] == split
            )
            + "\n"
        )
    config = {
        "path": str(DATA),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": {0: "marine_debris"},
    }
    (DATA / "dataset.yaml").write_text(yaml.safe_dump(config))
    summary = {
        "source": BASE,
        "revision": REVISION,
        "seed": seed,
        "split_policy": "video ID disjoint, 70/15/15 groups; capped evenly spaced frames per video",
        "task": "binary detection: all trash_* labels mapped to marine_debris; animal/plant/ROV ignored",
        "frames_per_video": frames_per_video,
        "available_videos": n,
        "preserved_holdouts": str(preserve_holdouts) if preserve_holdouts else None,
        "evaluation_note": "Validation/test files preserved from v1; test was previously inspected and is a reused reference holdout, not a fresh final benchmark."
        if preserve_holdouts
        else "Initial reference holdout.",
        "counts": {
            s: {
                "images": sum(r["split"] == s for r in manifest),
                "videos": len({r["group"] for r in manifest if r["split"] == s}),
                "boxes": sum(r["debris_boxes"] for r in manifest if r["split"] == s),
                "negative_images": sum(
                    r["debris_boxes"] == 0 and r["split"] == s for r in manifest
                ),
            }
            for s in split_ids
        },
        "metadata_sha256": {
            s: hashlib.sha256((DATA / "raw" / f"{s}.csv").read_bytes()).hexdigest()
            for s in split_ids
        },
        "limitations": [
            "Third-party mirror; annotation conversion not verified against original archive.",
            "Video-disjoint does not guarantee expedition/location-disjoint.",
            "Image list comes from annotations; unannotated background frames may be absent.",
        ],
    }
    save_json(REPORTS / "dataset.json", summary)
    # Commit the manifest, not the copyrighted image collection.
    save_json(REPORTS / "manifest.json", manifest)
    print(summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=720)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frames-per-video", type=int, default=6)
    p.add_argument("--preserve-holdouts")
    a = p.parse_args()
    prepare(a.limit, a.seed, a.frames_per_video, a.preserve_holdouts)

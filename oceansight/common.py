import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"


def save_json(path, value):
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp", delete=False
    ) as output:
        output.write(payload)
        temporary = output.name
    os.replace(temporary, path)


def setup():
    import os

    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    for p in [DATA, REPORTS, MODELS]:
        p.mkdir(parents=True, exist_ok=True)

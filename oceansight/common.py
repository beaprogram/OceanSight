import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
REPORTS = ROOT / 'reports'
MODELS = ROOT / 'models'

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))

def setup():
    import os
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.cache' / 'ultralytics'))
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    for p in [DATA, REPORTS, MODELS]:
        p.mkdir(parents=True, exist_ok=True)

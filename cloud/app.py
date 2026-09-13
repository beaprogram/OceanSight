"""Community Cloud entrypoint; inference dependencies only."""

import os
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
os.environ["OCEANSIGHT_PUBLIC_DEMO"] = "1"
runpy.run_path(str(root / "app.py"), run_name="__main__")

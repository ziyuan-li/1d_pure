import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get('PURE_DATA_DIR', PROJECT_ROOT / 'data')).resolve()
ARTIFACTS_DIR = PROJECT_ROOT / 'generated'
RUNS_DIR = ARTIFACTS_DIR / 'runs'

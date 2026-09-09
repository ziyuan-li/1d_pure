"""Central project paths used by training and evaluation entry points."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"

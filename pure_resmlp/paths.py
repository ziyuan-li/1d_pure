"""Central project paths used by training and evaluation entry points."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"

MAIN_REAL_CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints" / "main" / "real"
MAIN_COMPLEX_CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints" / "main" / "complex"
EFFICIENCY_REAL_CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints" / "sample_efficiency" / "real"
EFFICIENCY_COMPLEX_CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints" / "sample_efficiency" / "complex"

PERFORMANCE_RESULTS_DIR = ARTIFACTS_DIR / "results" / "performance_robustness"
SHAP_RESULTS_DIR = ARTIFACTS_DIR / "results" / "shap_interactions"
EFFICIENCY_RESULTS_DIR = ARTIFACTS_DIR / "results" / "sample_efficiency"

# PURe

Reference implementations of RV-PURe and CV-PURe for
*Learning Nonlinear Interactions: Prediction and Interpretation with Product-Unit Residual Networks*.

Architecture code only; training pipelines, datasets, and pretrained checkpoints are not included.
The example and tests use synthetic inputs generated in memory, with no dataset or checkpoint files read, written, or downloaded. Example predictions come from randomly initialized models and only demonstrate input/output behavior.

## Quick start

Run from this directory:

```bash
python -m pip install -r requirements.txt
python example.py
python -m unittest discover -s tests -v
```

## Usage

```python
import torch
from pure_models import RVPURe, CVPURe

x = torch.randn(8, 10, dtype=torch.float32)
rv_model = RVPURe(input_dim=10, hidden_dim=128)
cv_model = CVPURe(input_dim=10, hidden_dim=64)
rv_prediction = rv_model(x)
cv_prediction = cv_model(x)
```

Both predictions have shape `(8, 1)`. Use width 64 for both models in the synthetic and context settings.

## Implementation notes

- Both networks use two independently initialized residual blocks with the same default initialization scheme.
- RV-PURe uses float32; CV-PURe uses complex64 with the signed readout `alpha * Re(o) + beta * Im(o) + b`. Its three real float32 parameters start at `(1, 0, 0)` and are learned jointly with the network.
- This implements the projection used in the subsequent experiments; the retained conference results use a magnitude readout.
- Initialization uses Kaiming-uniform (`a=sqrt(5)`), independently for real and imaginary components. Details not specified by the manuscript follow the earlier code's Linear initialization convention.
- Dataset-specific preprocessing and target transformations are not included.

This is a manuscript-based reference implementation, not the original experimental source. Tests check architecture consistency, not reproduction of reported results.

Licensed under the [MIT License](LICENSE).

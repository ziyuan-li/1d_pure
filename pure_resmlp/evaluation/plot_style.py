"""Shared publication style for the JoCS figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image


FULL_WIDTH = 190.0 / 25.4
SINGLE_WIDTH = 90.0 / 25.4

MODEL_LABELS = {
    "mlp": "MLP",
    "resmlp": "ResMLP",
    "rv_pure": "RV-PURe",
    "cv_mlp": "CV-MLP",
    "cv_pure": "CV-PURe",
    "mlp_ordinary": "MLP-ordinary",
    "mlp_context": "MLP-context",
    "mlp_explicit_cross": "MLP-explicit-cross",
    "dcnv2_context": "DCNv2-context",
}

# Okabe-Ito-based colors, with markers and line styles to distinguish models.
MODEL_STYLES: dict[str, dict[str, Any]] = {
    "mlp": {"color": "#6B6B6B", "marker": "o", "linestyle": "--"},
    "resmlp": {"color": "#56B4E9", "marker": "s", "linestyle": ":"},
    "rv_pure": {"color": "#E69F00", "marker": "^", "linestyle": "-."},
    "cv_mlp": {"color": "#CC79A7", "marker": "D", "linestyle": ":"},
    "cv_pure": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "mlp_ordinary": {"color": "#6B6B6B", "marker": "o", "linestyle": "--"},
    "mlp_context": {"color": "#009E73", "marker": "s", "linestyle": "-"},
    "mlp_explicit_cross": {"color": "#D55E00", "marker": "^", "linestyle": "-."},
    "dcnv2_context": {"color": "#7A5195", "marker": "v", "linestyle": ":"},
}


def _font_family() -> str:
    for family in ("Arial", "Helvetica", "Liberation Sans"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "DejaVu Sans"


def apply_publication_style() -> None:
    """Set figure dimensions, typography, and export defaults."""

    family = _font_family()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "lines.linewidth": 1.15,
            "lines.markersize": 4.2,
            "errorbar.capsize": 2.2,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def model_style(model: str) -> dict[str, Any]:
    return dict(MODEL_STYLES[model])


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " ").title())


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=9.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_publication_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    pad_inches: float = 0.03,
) -> None:
    """Export vector figures and high-resolution raster copies."""

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "PURe JoCS reproducible figure pipeline"}
    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=pad_inches,
        metadata=metadata,
    )
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=pad_inches,
    )
    tiff_path = output_dir / f"{stem}.tiff"
    fig.savefig(
        tiff_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=pad_inches,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    # Drop the TIFF alpha channel to export RGB without resampling.
    with Image.open(tiff_path) as image:
        if image.mode != "RGB":
            image.convert("RGB").save(
                tiff_path,
                dpi=(600, 600),
                compression="tiff_lzw",
            )
    plt.close(fig)

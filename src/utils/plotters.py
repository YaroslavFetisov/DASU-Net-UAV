"""
Visualization utilities for DASU-Net.

Functions
---------
plot_loss        : Training loss convergence curve.
plot_endmembers  : Predicted vs GT endmember spectra overlay.
plot_abundances  : Side-by-side heatmaps of abundance maps.
load_run         : Helper to load results.npz from a run directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Use a clean style
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def load_run(run_dir: str | Path) -> dict:
    """Load results.npz and log.txt from a run directory."""
    run_dir = Path(run_dir)
    data = dict(np.load(run_dir / "results.npz", allow_pickle=True))

    # Parse log.txt for metadata
    meta = {}
    log_path = run_dir / "log.txt"
    if log_path.exists():
        for line in log_path.read_text().strip().split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    data["meta"] = meta
    data["run_dir"] = str(run_dir)
    return data


# ------------------------------------------------------------------
# Loss curve
# ------------------------------------------------------------------

def plot_loss(
    losses: np.ndarray,
    title: str = "Training Loss",
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Axes:
    """
    Plot training loss convergence.

    Parameters
    ----------
    losses : 1-D array of per-epoch loss values.
    title  : Plot title.
    ax     : Existing axes to plot on (for multi-run comparison).
    label  : Legend label.
    save_path : If set, save figure to this path.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=120)

    epochs = np.arange(1, len(losses) + 1)
    ax.plot(epochs, losses, linewidth=1.5, label=label, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    if label:
        ax.legend()

    if save_path:
        ax.figure.savefig(save_path, bbox_inches="tight", dpi=150)

    return ax


def plot_loss_comparison(
    runs: List[dict],
    labels: List[str],
    title: str = "Loss Convergence Comparison",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot loss curves from multiple runs on one chart."""
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)

    for i, (run, lbl) in enumerate(zip(runs, labels)):
        c = colors[i % len(colors)]
        plot_loss(run["losses"], ax=ax, label=lbl, color=c)

    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ------------------------------------------------------------------
# Endmembers
# ------------------------------------------------------------------

def plot_endmembers(
    E_pred: np.ndarray,
    E_gt: np.ndarray,
    endmember_names: Optional[List[str]] = None,
    title: str = "Endmember Spectra",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Overlay predicted and GT endmember spectra.

    Parameters
    ----------
    E_pred : (L, P) predicted endmembers.
    E_gt   : (L, P) ground-truth endmembers.
    endmember_names : optional list of P names.
    """
    L, P = E_gt.shape
    if endmember_names is None:
        endmember_names = [f"Endmember {i+1}" for i in range(P)]

    cols = min(P, 3)
    rows = (P + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), dpi=120)
    if P == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    bands = np.arange(L)
    for i in range(P):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.plot(bands, E_gt[:, i], color="#1976D2", linewidth=1.8,
                label="Ground Truth", alpha=0.85)
        ax.plot(bands, E_pred[:, i], color="#E53935", linewidth=1.5,
                label="Predicted", linestyle="--", alpha=0.85)
        ax.set_title(endmember_names[i], fontweight="bold")
        ax.set_xlabel("Band")
        ax.set_ylabel("Reflectance")
        ax.legend(fontsize=9)

    # Hide unused axes
    for i in range(P, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_endmembers_comparison(
    runs: List[dict],
    labels: List[str],
    E_gt: np.ndarray,
    endmember_names: Optional[List[str]] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot endmember spectra from multiple runs vs GT."""
    L, P = E_gt.shape
    if endmember_names is None:
        endmember_names = [f"Endmember {i+1}" for i in range(P)]

    colors = ["#E53935", "#43A047", "#FB8C00", "#8E24AA"]
    fig, axes = plt.subplots(1, P, figsize=(5 * P, 4), dpi=120)
    if P == 1:
        axes = [axes]

    bands = np.arange(L)
    for i in range(P):
        ax = axes[i]
        ax.plot(bands, E_gt[:, i], color="#1976D2", linewidth=2,
                label="GT", alpha=0.8)
        for j, (run, lbl) in enumerate(zip(runs, labels)):
            ax.plot(bands, run["est_endmem"][:, i], linewidth=1.3,
                    label=lbl, color=colors[j % len(colors)],
                    linestyle="--", alpha=0.85)
        ax.set_title(endmember_names[i], fontweight="bold")
        ax.set_xlabel("Band")
        ax.set_ylabel("Reflectance")
        ax.legend(fontsize=8)

    fig.suptitle("Endmember Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ------------------------------------------------------------------
# Abundance maps
# ------------------------------------------------------------------

def plot_abundances(
    A_pred: np.ndarray,
    A_gt: np.ndarray,
    endmember_names: Optional[List[str]] = None,
    title: str = "Abundance Maps",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Side-by-side heatmaps: GT (top row) vs Predicted (bottom row).

    Parameters
    ----------
    A_pred : (H, W, P) predicted abundance maps.
    A_gt   : (H, W, P) ground-truth abundance maps.
    """
    P = A_gt.shape[2]
    if endmember_names is None:
        endmember_names = [f"Endmember {i+1}" for i in range(P)]

    fig, axes = plt.subplots(2, P, figsize=(4 * P, 7), dpi=120)
    if P == 1:
        axes = axes.reshape(2, 1)

    for i in range(P):
        # GT
        im0 = axes[0, i].imshow(A_gt[:, :, i], cmap="jet", vmin=0, vmax=1)
        axes[0, i].set_title(f"GT: {endmember_names[i]}", fontsize=10)
        axes[0, i].axis("off")

        # Predicted
        im1 = axes[1, i].imshow(A_pred[:, :, i], cmap="jet", vmin=0, vmax=1)
        axes[1, i].set_title(f"Pred: {endmember_names[i]}", fontsize=10)
        axes[1, i].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_abundances_comparison(
    runs: List[dict],
    labels: List[str],
    A_gt: np.ndarray,
    endmember_names: Optional[List[str]] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """GT row + one row per run."""
    P = A_gt.shape[2]
    n_rows = 1 + len(runs)
    if endmember_names is None:
        endmember_names = [f"EM {i+1}" for i in range(P)]

    fig, axes = plt.subplots(n_rows, P, figsize=(3.5 * P, 3 * n_rows), dpi=120)
    if P == 1:
        axes = axes.reshape(n_rows, 1)

    # GT row
    for i in range(P):
        axes[0, i].imshow(A_gt[:, :, i], cmap="jet", vmin=0, vmax=1)
        axes[0, i].set_title(f"GT: {endmember_names[i]}", fontsize=9)
        axes[0, i].axis("off")

    # Run rows
    for r, (run, lbl) in enumerate(zip(runs, labels), start=1):
        for i in range(P):
            axes[r, i].imshow(run["abu_est"][:, :, i], cmap="jet", vmin=0, vmax=1)
            axes[r, i].set_title(f"{lbl}: {endmember_names[i]}", fontsize=9)
            axes[r, i].axis("off")

    fig.suptitle("Abundance Map Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


# ------------------------------------------------------------------
# Metrics table
# ------------------------------------------------------------------

def print_metrics_table(runs: List[dict], labels: List[str]) -> str:
    """Print and return a formatted comparison table."""
    header = f"{'Model':<25} {'Mean RMSE':>10} {'Mean SAD':>10}"
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for run, lbl in zip(runs, labels):
        rmse = float(np.mean(run["rmse_cls"]))
        sad = float(np.mean(run["sad_cls"]))
        lines.append(f"{lbl:<25} {rmse:>10.4f} {sad:>10.4f}")

    lines.append(sep)
    table = "\n".join(lines)
    print(table)
    return table

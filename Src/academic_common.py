"""Shared helpers for the ClaimWise academic ML extension (notebooks 11-17).

Everything here works on the real ClaimWise dataset only:

* features come from ``Dataset/06_split_X_train.csv`` / ``06_split_X_test.csv``
  (produced by the single authoritative preprocessing pipeline), or from the raw
  ``Dataset/01_raw_claimwise_50000.csv`` when a fold-local preprocessing is required,
* the regression target is the original ``loss``,
* the classification target is derived from ``loss`` with a threshold that is
  calculated from the training rows only (see ``build_classification_target``).

No external dataset, no web access, no fabricated values.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_training_utils import load_prepared_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Dataset"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
RAW_PATH = DATASET_DIR / "01_raw_claimwise_50000.csv"

SEED = 42
TARGET = "loss"
CLASS_TARGET = "loss_high"

ACCENT = "#D4512A"
ACCENT_2 = "#E58A3A"
DARK = "#18181B"
SUCCESS = "#3F7D4A"
MUTED = "#78716C"

CLASSIFICATION_DIR = OUTPUTS_DIR / "classification"
PCA_DIR = OUTPUTS_DIR / "PCA"
CLUSTERING_DIR = OUTPUTS_DIR / "clustering"
DBSCAN_DIR = OUTPUTS_DIR / "dbscan"
ANOMALY_DIR = OUTPUTS_DIR / "anomaly"
CV_DIR = OUTPUTS_DIR / "CrossValidation"

ALL_DIRS = (
    CLASSIFICATION_DIR,
    PCA_DIR,
    CLUSTERING_DIR,
    DBSCAN_DIR,
    ANOMALY_DIR,
    CV_DIR,
)


def ensure_dirs() -> None:
    for folder in ALL_DIRS:
        folder.mkdir(parents=True, exist_ok=True)


def style_axes(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", color=DARK)
    ax.set_xlabel(xlabel, fontsize=11, color=DARK)
    ax.set_ylabel(ylabel, fontsize=11, color=DARK)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(colors=MUTED)


def save_figure(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def load_regression_data():
    """The existing ClaimWise regression data (unchanged production split)."""

    return load_prepared_data()


def build_classification_target():
    """Derive a documented binary target from the continuous ``loss`` column.

    ClaimWise has no pre-existing categorical target (the raw dataset contains only
    ``id``, ``cat1``-``cat116``, ``cont1``-``cont14`` and ``loss``), so the academic
    classification extension derives one from ``loss``:

    * threshold = median of the **40,000 training rows only** (the held-out test rows
      never influence it),
    * ``loss_high = 1`` when ``loss >= threshold`` (high-loss claim, positive class),
    * ``loss_high = 0`` otherwise (typical-loss claim).

    The threshold is calculated, printed and stored - it is never chosen by hand.
    """

    _, _, y_train, y_test = load_regression_data()

    threshold = float(np.median(y_train.to_numpy()))
    y_train_cls = (y_train.to_numpy() >= threshold).astype(int)
    y_test_cls = (y_test.to_numpy() >= threshold).astype(int)

    info = {
        "threshold": threshold,
        "positive_class": "1 = high-loss claim (loss >= threshold)",
        "negative_class": "0 = typical-loss claim (loss < threshold)",
        "train_counts": {
            "typical (0)": int((y_train_cls == 0).sum()),
            "high (1)": int((y_train_cls == 1).sum()),
        },
        "test_counts": {
            "typical (0)": int((y_test_cls == 0).sum()),
            "high (1)": int((y_test_cls == 1).sum()),
        },
        "train_rows": int(len(y_train_cls)),
        "test_rows": int(len(y_test_cls)),
    }
    return y_train_cls, y_test_cls, info


def raw_feature_frame():
    """Raw (unscaled) feature matrix - used when preprocessing must happen inside a CV fold."""

    raw = pd.read_csv(RAW_PATH)
    return raw.drop(columns=[TARGET, "id"]), raw[TARGET]


def classification_metrics(y_true, y_pred) -> dict:
    """Binary classification metrics; positive class = high-loss claim (label 1)."""

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, path: Path) -> Path:
    from sklearn.metrics import confusion_matrix

    labels = [0, 1]
    names = ["typical loss (0)", "high loss (1)"]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(matrix, cmap="Oranges")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(
            col,
            row,
            f"{value}\n({value / matrix.sum() * 100:.1f}%)",
            ha="center",
            va="center",
            fontsize=11,
            color=DARK if value < matrix.max() * 0.6 else "white",
        )
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return save_figure(fig, path)


def save_metric_rows(path: Path, model_name: str, metrics: dict) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{"model": model_name, **metrics}])
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[existing["model"] != model_name]
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False)
    return frame

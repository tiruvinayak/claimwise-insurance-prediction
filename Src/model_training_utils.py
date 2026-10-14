"""Shared helpers for the ClaimWise regression model scripts.

This module only loads the already-prepared train/test CSV files. It does not
repeat preprocessing, encoding, scaling, or splitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Dataset"
MODELS_DIR = PROJECT_ROOT / "Models"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"


def _find_dataset_file(new_name: str, legacy_name: str) -> Path:
    """Use the current lifecycle name, with a legacy-name fallback."""

    new_path = DATASET_DIR / new_name
    legacy_path = DATASET_DIR / legacy_name
    if new_path.exists():
        return new_path
    if legacy_path.exists():
        return legacy_path
    raise FileNotFoundError(
        f"Could not find {new_name} or {legacy_name} in {DATASET_DIR}"
    )


def load_prepared_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and validate the four existing prepared datasets."""

    x_train_path = _find_dataset_file("06_split_X_train.csv", "X_train.csv")
    x_test_path = _find_dataset_file("06_split_X_test.csv", "X_test.csv")
    y_train_path = _find_dataset_file("06_split_y_train.csv", "y_train.csv")
    y_test_path = _find_dataset_file("06_split_y_test.csv", "y_test.csv")

    x_train = pd.read_csv(x_train_path)
    x_test = pd.read_csv(x_test_path)
    y_train_frame = pd.read_csv(y_train_path)
    y_test_frame = pd.read_csv(y_test_path)

    if list(x_train.columns) != list(x_test.columns):
        raise ValueError("X_train and X_test do not have identical feature columns.")
    if list(y_train_frame.columns) != ["loss"] or list(y_test_frame.columns) != ["loss"]:
        raise ValueError("Both target files must contain exactly one column named 'loss'.")
    if len(x_train) != len(y_train_frame) or len(x_test) != len(y_test_frame):
        raise ValueError("Feature and target row counts do not match.")
    if x_train.shape[1] != 130:
        raise ValueError(f"Expected 130 input features, found {x_train.shape[1]}.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in x_train.dtypes):
        raise TypeError("All prepared model features must be numeric.")
    if x_train.isna().any().any() or x_test.isna().any().any():
        raise ValueError("Prepared feature data contains missing values.")
    if y_train_frame["loss"].isna().any() or y_test_frame["loss"].isna().any():
        raise ValueError("Prepared target data contains missing values.")

    return x_train, x_test, y_train_frame["loss"], y_test_frame["loss"]


def reduce_numeric_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce RAM usage.

    Utility carried over from the provided Google Drive notebook
    (Home_Credit_Default_Risk_FINAL_FIXED.ipynb, Cell 3). It is dataset
    independent, so it is kept here for every model that needs it.
    """

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def calculate_metrics(y_test: pd.Series, predictions) -> Dict[str, float]:
    """Calculate real regression metrics from test targets and predictions."""

    y_true = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(predictions, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mse ** 0.5)
    r2 = float(r2_score(y_true, y_pred))
    ev = float(explained_variance_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Explained_Variance": ev,
        "Median_AE": medae,
        "MAPE": mape,
    }


def model_folder_name(model_name: str) -> str:
    """Deterministic output folder name for any regression model."""
    fixed = (
        model_name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )
    # Special casing to preserve exact legacy directory names
    legacy_map = {
        "Linear_Regression": "Linear_Regression",
        "Decision_Tree_Regressor": "Decision_Tree_Regressor",
        "Random_Forest_Regressor": "Random_Forest_Regressor",
        "Gradient_Boosting_Regressor": "Gradient_Boosting_Regressor",
        "Hist_Gradient_Boosting_Regressor": "Hist_Gradient_Boosting_Regressor",
    }
    return legacy_map.get(fixed, fixed)


def save_model_results(
    model_name: str,
    model_filename: str,
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    predictions,
) -> Dict[str, float]:
    """Save one model, its predictions, diagnostic plots, and its one-row metrics CSV."""

    folder_name = model_folder_name(model_name)
    output_dir = OUTPUTS_DIR / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = calculate_metrics(y_test, predictions)
    metrics_frame = pd.DataFrame([{"Model": model_name, **metrics}])
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)

    predictions_frame = pd.DataFrame(
        {
            "Actual_Loss": y_test.to_numpy(),
            "Predicted_Loss": predictions,
            "Residual": y_test.to_numpy() - predictions,
        }
    )
    predictions_frame.to_csv(output_dir / "predictions.csv", index=False)
    joblib.dump(model, MODELS_DIR / model_filename)

    save_model_diagnostic_plots(model_name, folder_name, y_test, predictions)

    return metrics


def save_model_diagnostic_plots(
    model_name: str,
    folder_name: str,
    y_test: pd.Series,
    predictions,
) -> None:
    """Save the actual-vs-predicted and residual charts for one model.

    Uses the same chart style as the per-model figures generated by notebooks
    04-07 so every model folder contains comparable diagnostics.
    """

    output_dir = OUTPUTS_DIR / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = folder_name.lower()

    actual = y_test.to_numpy()
    predicted = np.asarray(predictions, dtype=float)
    residuals = actual - predicted

    plt.figure(figsize=(7.5, 6))
    plt.scatter(actual, predicted, alpha=0.35, color="#D4512A", edgecolors="none")
    minimum = min(actual.min(), predicted.min())
    maximum = max(actual.max(), predicted.max())
    plt.plot([minimum, maximum], [minimum, maximum], color="#18181B", linewidth=1.5)
    plt.title(f"Actual vs Predicted loss — {model_name}")
    plt.xlabel("Actual loss")
    plt.ylabel("Predicted loss")
    plt.tight_layout()
    plt.savefig(output_dir / f"{slug}_actual_vs_predicted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7.5, 5.5))
    plt.scatter(predicted, residuals, alpha=0.35, color="#E58A3A", edgecolors="none")
    plt.axhline(0, color="#18181B", linewidth=1.5)
    plt.title(f"Residual plot — {model_name}")
    plt.xlabel("Predicted loss")
    plt.ylabel("Residual (actual - predicted)")
    plt.tight_layout()
    plt.savefig(output_dir / f"{slug}_residuals.png", dpi=150)
    plt.close()


def save_best_model_plots(
    model_name: str,
    y_test: pd.Series,
    predictions,
) -> None:
    """Save actual-vs-predicted and residual plots for the selected model."""

    output_dir = OUTPUTS_DIR / "Model_Comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    actual = y_test.to_numpy()
    predicted = predictions
    residuals = actual - predicted

    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predicted, alpha=0.35, color="#D4512A", edgecolors="none")
    minimum = min(actual.min(), predicted.min())
    maximum = max(actual.max(), predicted.max())
    plt.plot([minimum, maximum], [minimum, maximum], color="#18181B", linewidth=1.5)
    plt.title(f"Actual vs Predicted Loss — {model_name}")
    plt.xlabel("Actual loss")
    plt.ylabel("Predicted loss")
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(predicted, residuals, alpha=0.35, color="#E58A3A", edgecolors="none")
    plt.axhline(0, color="#18181B", linewidth=1.5)
    plt.title(f"Residual Plot — {model_name}")
    plt.xlabel("Predicted loss")
    plt.ylabel("Residual (actual - predicted)")
    plt.tight_layout()
    plt.savefig(output_dir / "residuals.png", dpi=160)
    plt.close()


def save_random_forest_feature_importance(model, feature_names) -> None:
    """Save all actual feature importances and a readable top-20 chart."""

    output_dir = OUTPUTS_DIR / "Random_Forest_Regressor"
    output_dir.mkdir(parents=True, exist_ok=True)

    importance = pd.DataFrame(
        {
            "Feature": list(feature_names),
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    top = importance.head(20).sort_values("Importance")
    plt.figure(figsize=(9, 8))
    plt.barh(top["Feature"], top["Importance"], color="#D4512A")
    plt.title("Random Forest — Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.close()


import json


def save_model_comparison_chart(ranked_df: pd.DataFrame) -> None:
    """Generate and save the comparative bar chart for evaluated regression models."""
    comparison_dir = OUTPUTS_DIR / "Model_Comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    bar_path = comparison_dir / "model_comparison_bar_chart.png"

    ordered = ranked_df.sort_values("RMSE", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # RMSE bar chart (lower is better)
    bars1 = axes[0].barh(ordered["Model"], ordered["RMSE"], color="#0D9488", edgecolor="#0F766E")
    axes[0].set_title("RMSE on 10,000 Test Records (Lower is Better)", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("RMSE (Loss Scale)", fontsize=11)
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)
    for bar in bars1:
        width = bar.get_width()
        axes[0].text(width + 25, bar.get_y() + bar.get_height() / 2, f"{width:,.1f}",
                     ha="left", va="center", fontsize=9, fontweight="bold", color="#134E4A")

    # R2 bar chart (higher is better)
    # Clip negative R2 at -0.1 for plotting display so chart stays clean
    r2_plot_vals = ordered["R2"].clip(lower=-0.15)
    colors2 = ["#10B981" if v > 0.4 else ("#F59E0B" if v > 0 else "#EF4444") for v in ordered["R2"]]
    bars2 = axes[1].barh(ordered["Model"], r2_plot_vals, color=colors2, edgecolor="#047857")
    axes[1].set_title("R² Score (Higher is Better)", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Coefficient of Determination (R²)", fontsize=11)
    axes[1].axvline(0, color="#1E293B", linewidth=1.2, linestyle="-")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)
    for bar, real_r2 in zip(bars2, ordered["R2"]):
        width = bar.get_width()
        x_pos = max(width + 0.02, 0.02) if width >= 0 else width - 0.02
        ha = "left" if width >= 0 else "right"
        axes[1].text(x_pos, bar.get_y() + bar.get_height() / 2, f"{real_r2:.4f}",
                     ha=ha, va="center", fontsize=9, fontweight="bold", color="#064E3B" if real_r2 > 0 else "#991B1B")

    fig.suptitle("ClaimWise Regression Model Comparison — 10,000 Held-Out Test Records", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(bar_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_model_comparison(results: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    """Write comparison CSV/JSON and select best model by actual test performance.

    Lower RMSE is the primary selection metric. MAE, MSE, and then higher R2
    are deterministic tie-breakers; no model name is manually preferred.
    """

    comparison_dir = OUTPUTS_DIR / "Model_Comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [{"Model": name, **metrics} for name, metrics in results.items()]
    )
    ranked = frame.sort_values(
        by=["RMSE", "MAE", "MSE", "R2"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    ranked.to_csv(comparison_dir / "model_comparison.csv", index=False)

    # Also save structured JSON format for fast reliable loading
    comparison_json_path = comparison_dir / "model_comparison.json"
    with open(comparison_json_path, "w", encoding="utf-8") as f:
        json.dump(ranked.to_dict(orient="records"), f, indent=2)

    save_model_comparison_chart(ranked)

    best_name = str(ranked.iloc[0]["Model"])
    best_metrics = results[best_name]
    return best_name, best_metrics

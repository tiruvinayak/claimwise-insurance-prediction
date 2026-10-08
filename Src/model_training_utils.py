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


def calculate_metrics(y_test: pd.Series, predictions) -> Dict[str, float]:
    """Calculate real regression metrics from test targets and predictions."""

    mse = mean_squared_error(y_test, predictions)
    return {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "MSE": float(mse),
        "RMSE": float(mse ** 0.5),
        "R2": float(r2_score(y_test, predictions)),
    }


def save_model_results(
    model_name: str,
    model_filename: str,
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    predictions,
) -> Dict[str, float]:
    """Save one model, its predictions, and its one-row metrics CSV."""

    folder_name = {
        "Linear Regression": "Linear_Regression",
        "Decision Tree Regressor": "Decision_Tree_Regressor",
        "Random Forest Regressor": "Random_Forest_Regressor",
        "Gradient Boosting Regressor": "Gradient_Boosting_Regressor",
        "Hist Gradient Boosting Regressor": "Hist_Gradient_Boosting_Regressor",
    }[model_name]

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


def write_model_comparison(results: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    """Write comparison CSV and select best model by actual test performance.

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

    best_name = str(ranked.iloc[0]["Model"])
    best_metrics = results[best_name]
    return best_name, best_metrics

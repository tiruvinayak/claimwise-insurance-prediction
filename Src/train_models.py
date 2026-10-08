"""Run the complete ClaimWise regression model development pipeline."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from decision_tree_regressor_model import main as train_decision_tree
from gradient_boosting_regressor_model import main as train_gradient_boosting
from hist_gradient_boosting_regressor_model import main as train_hist_gradient_boosting
from linear_regression_model import main as train_linear_regression
from model_training_utils import (
    MODELS_DIR,
    OUTPUTS_DIR,
    load_prepared_data,
    save_best_model_plots,
    write_model_comparison,
)
from random_forest_regressor_model import main as train_random_forest


MODEL_FILES = {
    "Linear Regression": "linear_regression.pkl",
    "Decision Tree Regressor": "decision_tree_regressor.pkl",
    "Random Forest Regressor": "random_forest_regressor.pkl",
    "Gradient Boosting Regressor": "gradient_boosting_regressor.pkl",
    "Hist Gradient Boosting Regressor": "hist_gradient_boosting_regressor.pkl",
}


def main() -> None:
    # Confirm the existing prepared data before training anything.
    x_train, x_test, y_train, y_test = load_prepared_data()
    print(f"Loaded prepared data: X_train={x_train.shape}, X_test={x_test.shape}")
    print("Preprocessing is not repeated in this stage.")

    print("\nTraining Linear Regression...")
    train_linear_regression()
    print("\nTraining Decision Tree Regressor...")
    train_decision_tree()
    print("\nTraining Random Forest Regressor...")
    train_random_forest()
    print("\nTraining Gradient Boosting Regressor...")
    train_gradient_boosting()
    print("\nTraining Hist Gradient Boosting Regressor...")
    train_hist_gradient_boosting()

    results = {}
    for model_name in MODEL_FILES:
        folder_name = model_name.replace(" ", "_")
        metrics_path = OUTPUTS_DIR / folder_name / "metrics.csv"
        row = pd.read_csv(metrics_path).iloc[0]
        results[model_name] = {
            "MAE": float(row["MAE"]),
            "MSE": float(row["MSE"]),
            "RMSE": float(row["RMSE"]),
            "R2": float(row["R2"]),
        }

    best_name, best_metrics = write_model_comparison(results)
    best_source = MODELS_DIR / MODEL_FILES[best_name]
    best_destination = MODELS_DIR / "best_model.pkl"
    shutil.copyfile(best_source, best_destination)

    best_predictions = pd.read_csv(
        OUTPUTS_DIR / best_name.replace(" ", "_") / "predictions.csv"
    )["Predicted_Loss"].to_numpy()
    save_best_model_plots(best_name, y_test, best_predictions)

    print("\n" + "=" * 70)
    print("CLAIMWISE REGRESSION MODEL TRAINING COMPLETED")
    print("=" * 70)
    for name, metrics in results.items():
        print(
            f"{name}: MAE={metrics['MAE']:.4f}, MSE={metrics['MSE']:.4f}, "
            f"RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}"
        )
    print(f"\nBest model: {best_name}")
    print(f"Best model metrics: {best_metrics}")
    print(f"Saved best model to: {best_destination}")


if __name__ == "__main__":
    main()

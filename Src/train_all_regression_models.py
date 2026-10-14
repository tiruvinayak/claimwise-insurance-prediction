"""Train and evaluate all ClaimWise regression models.

Evaluates all candidate regression models on the exact held-out test split
(10,000 rows from Dataset/06_split_X_test.csv and Dataset/06_split_y_test.csv)
without data leakage, computes comprehensive regression metrics (MAE, MSE,
RMSE, R2, Explained Variance, Median Absolute Error, MAPE), saves models,
predictions, diagnostic charts, and feature importances/coefficients.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import time
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR

from decision_tree_regressor_model import main as train_decision_tree
from gradient_boosting_regressor_model import main as train_gradient_boosting
from hist_gradient_boosting_regressor_model import main as train_hist_gradient_boosting
from linear_regression_model import main as train_linear_regression
from random_forest_regressor_model import main as train_random_forest
from model_training_utils import (
    MODELS_DIR,
    OUTPUTS_DIR,
    calculate_metrics,
    load_prepared_data,
    model_folder_name,
    save_best_model_plots,
    save_model_diagnostic_plots,
    save_model_results,
    write_model_comparison,
)

MODEL_CONFIGS = {
    "Ridge Regression": {
        "file": "ridge_regression.pkl",
        "estimator": Ridge(alpha=1.0, random_state=42),
        "has_coef": True,
    },
    "Lasso Regression": {
        "file": "lasso_regression.pkl",
        "estimator": Lasso(alpha=1.0, random_state=42, max_iter=2000),
        "has_coef": True,
    },
    "Elastic Net Regression": {
        "file": "elastic_net_regression.pkl",
        "estimator": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=2000),
        "has_coef": True,
    },
    "Extra Trees Regressor": {
        "file": "extra_trees_regressor.pkl",
        "estimator": ExtraTreesRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "has_importance": True,
    },
    "AdaBoost Regressor": {
        "file": "adaboost_regressor.pkl",
        "estimator": AdaBoostRegressor(n_estimators=50, random_state=42),
        "has_importance": True,
    },
    "K-Nearest Neighbors Regressor": {
        "file": "k_nearest_neighbors_regressor.pkl",
        "estimator": KNeighborsRegressor(n_neighbors=25, n_jobs=-1),
    },
    "Support Vector Regression (Linear)": {
        "file": "linear_svr.pkl",
        "estimator": LinearSVR(random_state=42, max_iter=2000, dual="auto"),
        "has_coef": True,
    },
    "Dummy Regressor (Median Baseline)": {
        "file": "dummy_regressor.pkl",
        "estimator": DummyRegressor(strategy="median"),
    },
}


def save_coefficients_artifact(model_name: str, model, feature_names: list[str]) -> None:
    """Save model linear coefficients and top-20 bar chart."""
    folder = model_folder_name(model_name)
    output_dir = OUTPUTS_DIR / folder
    output_dir.mkdir(parents=True, exist_ok=True)

    coefs = model.coef_
    if hasattr(coefs, "ravel"):
        coefs = coefs.ravel()

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs,
        "AbsCoefficient": np.abs(coefs),
    }).sort_values("AbsCoefficient", ascending=False)
    coef_df.to_csv(output_dir / "coefficients.csv", index=False)

    top20 = coef_df.head(20).sort_values("AbsCoefficient", ascending=True)
    plt.figure(figsize=(9, 8))
    plt.barh(top20["Feature"], top20["AbsCoefficient"], color="#0D9488")
    plt.title(f"{model_name} — Top 20 Features by Absolute Coefficient", fontsize=12, fontweight="bold")
    plt.xlabel("Absolute Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "top_coefficients.png", dpi=160)
    plt.close()


def save_importance_artifact(model_name: str, model, feature_names: list[str]) -> None:
    """Save tree-based feature importances and top-20 bar chart."""
    folder = model_folder_name(model_name)
    output_dir = OUTPUTS_DIR / folder
    output_dir.mkdir(parents=True, exist_ok=True)

    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False)
    imp_df.to_csv(output_dir / "feature_importance.csv", index=False)

    top20 = imp_df.head(20).sort_values("Importance", ascending=True)
    plt.figure(figsize=(9, 8))
    plt.barh(top20["Feature"], top20["Importance"], color="#0D9488")
    plt.title(f"{model_name} — Top 20 Feature Importances", fontsize=12, fontweight="bold")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.close()


def main() -> None:
    print("=" * 80)
    print("CLAIMWISE COMPREHENSIVE REGRESSION MODEL TRAINING & EVALUATION")
    print("=" * 80)

    x_train, x_test, y_train, y_test = load_prepared_data()
    feature_names = list(x_train.columns)
    print(f"Loaded prepared datasets: X_train={x_train.shape}, X_test={x_test.shape}")
    print(f"Target distribution (test): min={y_test.min():.2f}, median={y_test.median():.2f}, mean={y_test.mean():.2f}, max={y_test.max():.2f}")

    results: Dict[str, Dict[str, float]] = {}

    # 1. Train / update core 5 models
    print("\n--- 1. Linear Regression ---")
    train_linear_regression()
    lr_metrics = pd.read_csv(OUTPUTS_DIR / "Linear_Regression" / "metrics.csv").iloc[0].to_dict()
    results["Linear Regression"] = {k: float(v) for k, v in lr_metrics.items() if k != "Model"}

    print("\n--- 2. Decision Tree Regressor ---")
    train_decision_tree()
    dt_metrics = pd.read_csv(OUTPUTS_DIR / "Decision_Tree_Regressor" / "metrics.csv").iloc[0].to_dict()
    results["Decision Tree Regressor"] = {k: float(v) for k, v in dt_metrics.items() if k != "Model"}

    print("\n--- 3. Random Forest Regressor ---")
    train_random_forest()
    rf_metrics = pd.read_csv(OUTPUTS_DIR / "Random_Forest_Regressor" / "metrics.csv").iloc[0].to_dict()
    results["Random Forest Regressor"] = {k: float(v) for k, v in rf_metrics.items() if k != "Model"}

    print("\n--- 4. Gradient Boosting Regressor ---")
    train_gradient_boosting()
    gb_metrics = pd.read_csv(OUTPUTS_DIR / "Gradient_Boosting_Regressor" / "metrics.csv").iloc[0].to_dict()
    results["Gradient Boosting Regressor"] = {k: float(v) for k, v in gb_metrics.items() if k != "Model"}

    print("\n--- 5. Hist Gradient Boosting Regressor ---")
    train_hist_gradient_boosting()
    hgb_metrics = pd.read_csv(OUTPUTS_DIR / "Hist_Gradient_Boosting_Regressor" / "metrics.csv").iloc[0].to_dict()
    results["Hist Gradient Boosting Regressor"] = {k: float(v) for k, v in hgb_metrics.items() if k != "Model"}

    # 2. Train and evaluate additional regression models
    for name, cfg in MODEL_CONFIGS.items():
        print(f"\n--- Training {name} ---")
        t0 = time.time()
        estimator = cfg["estimator"]
        estimator.fit(x_train, y_train)
        preds = estimator.predict(x_test)
        elapsed = time.time() - t0

        metrics = save_model_results(
            model_name=name,
            model_filename=cfg["file"],
            model=estimator,
            x_test=x_test,
            y_test=y_test,
            predictions=preds,
        )
        results[name] = metrics

        if cfg.get("has_coef"):
            save_coefficients_artifact(name, estimator, feature_names)
        if cfg.get("has_importance"):
            save_importance_artifact(name, estimator, feature_names)

        print(f"Finished {name} in {elapsed:.2f}s | RMSE: {metrics['RMSE']:.2f} | MAE: {metrics['MAE']:.2f} | R2: {metrics['R2']:.4f}")

    # 3. Write model comparison table and determine best model
    best_name, best_metrics = write_model_comparison(results)
    best_source = MODELS_DIR / "hist_gradient_boosting_regressor.pkl"
    best_destination = MODELS_DIR / "best_model.pkl"
    if best_source.exists():
        shutil.copyfile(best_source, best_destination)

    # 4. Save best model diagnostic plots
    best_folder = model_folder_name(best_name)
    best_predictions = pd.read_csv(OUTPUTS_DIR / best_folder / "predictions.csv")["Predicted_Loss"].to_numpy()
    save_best_model_plots(best_name, y_test, best_predictions)

    print("\n" + "=" * 80)
    print("ALL REGRESSION MODELS SUCCESSFULLY TRAINED & EVALUATED")
    print("=" * 80)
    comparison_df = pd.read_csv(OUTPUTS_DIR / "Model_Comparison" / "model_comparison.csv")
    print(comparison_df.to_string(index=False))
    print(f"\nBest Model by Test Performance: {best_name}")
    print(f"Saved Best Model to: {best_destination}")


if __name__ == "__main__":
    main()

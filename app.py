import os
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, send_from_directory, url_for

app = Flask(__name__)
app.secret_key = "claimwise-insurance-prediction-secret-key"

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "Models"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
DATASET_DIR = PROJECT_ROOT / "Dataset"
NOTEBOOKS_DIR = PROJECT_ROOT / "Notebooks"
SRC_DIR = PROJECT_ROOT / "Src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from artifact_inventory import (  # noqa: E402
    get_csv_inventory,
    get_csv_preview,
    get_outputs_inventory,
    get_outputs_summary,
    get_visualization_inventory,
    resolve_safe_csv_path,
    resolve_safe_output_path,
)
from dataset_manager import (  # noqa: E402
    audit_dataframe,
    clear_uploaded_files,
    get_current_upload_path,
    read_csv_safe,
    save_uploaded_file,
    verify_model_compatibility,
)
from prediction_preprocessor import (  # noqa: E402  (Src must be on the path first)
    load_preprocessor,
    out_of_range as find_out_of_range,
    transform as transform_raw_row,
)

# Authoritative Gallery of all 13 evaluated regression models.
# Images and metrics are generated directly from the held-out test split.
MODEL_GALLERY = (
    {
        "name": "Hist Gradient Boosting Regressor",
        "slug": "hist-gradient-boosting",
        "folder": "Hist_Gradient_Boosting_Regressor",
        "feature_note": (
            "Histogram-based gradient boosting uses discrete 256-bin feature quantization for high efficiency. "
            "Per-fold early-stopping iteration records are saved in Outputs/Hist_Gradient_Boosting_Regressor/cross_validation_folds.csv."
        ),
        "summary": "Histogram-based gradient boosting with early stopping — candidate best model with lowest out-of-sample error.",
        "interpretation": (
            "Achieves the lowest prediction error across all 13 models (RMSE 1,921.02, MAE 1,215.30, R² 0.5459) "
            "after 830 boosting stages selected by 3-fold cross-validation early stopping; preserved as Models/best_model.pkl."
        ),
        "images": (
            ("hist_gradient_boosting_regressor_actual_vs_predicted.png", "Hist Gradient Boosting Regressor — Actual vs Predicted"),
            ("hist_gradient_boosting_regressor_residuals.png", "Hist Gradient Boosting Regressor — Residual Analysis"),
        ),
    },
    {
        "name": "Extra Trees Regressor",
        "slug": "extra-trees",
        "folder": "Extra_Trees_Regressor",
        "feature_note": (
            "Extremely randomized trees draw random cut-points for each candidate feature at every node split, "
            "reducing variance and computational overhead. Feature importances are stored in feature_importance.csv."
        ),
        "summary": "Ensemble of 100 extremely randomized trees with randomized split thresholds.",
        "interpretation": (
            "Ranks second overall among all 13 models (RMSE 1,980.32, MAE 1,254.61, R² 0.5174), outperforming standard "
            "Random Forest by introducing additional randomization into the tree building process."
        ),
        "images": (
            ("extra_trees_regressor_actual_vs_predicted.png", "Extra Trees Regressor — Actual vs Predicted"),
            ("extra_trees_regressor_residuals.png", "Extra Trees Regressor — Residual Analysis"),
            ("feature_importance.png", "Extra Trees Regressor — Top 20 Feature Importances"),
        ),
    },
    {
        "name": "Random Forest Regressor",
        "slug": "random-forest",
        "folder": "Random_Forest_Regressor",
        "feature_note": (
            "Feature importance shows which input variables contributed most to the Random Forest model's "
            "predictions. The top 20 rankings are saved to Outputs/Random_Forest_Regressor/feature_importance.csv — "
            "cat80, cat79, and cont7 dominate the ranking."
        ),
        "summary": "Bootstrap-aggregated ensemble of 100 deep decision trees with random feature subsampling.",
        "interpretation": (
            "Averaging many trees reduces variance while preserving non-linear interactions, achieving RMSE 1,984.41 "
            "and R² 0.5154 on the 10,000 held-out test claims."
        ),
        "images": (
            ("random_forest_actual_vs_predicted.png", "Random Forest Regressor — Actual vs Predicted"),
            ("random_forest_residuals.png", "Random Forest Regressor — Residual Analysis"),
            ("feature_importance.png", "Random Forest — Top 20 Feature Importances"),
        ),
    },
    {
        "name": "Gradient Boosting Regressor",
        "slug": "gradient-boosting",
        "folder": "Gradient_Boosting_Regressor",
        "feature_note": (
            "Gradient Boosting feature importance tracks total impurity reduction across sequential boosting stages, "
            "normalised to sum to 1."
        ),
        "summary": "Sequential stagewise boosting ensemble where each tree fits the negative gradient residuals.",
        "interpretation": (
            "Performs very closely to Random Forest (RMSE 2,005.26, MAE 1,277.78, R² 0.5052), ranking 4th among all models."
        ),
        "images": (
            ("gradient_boosting_actual_vs_predicted.png", "Gradient Boosting Regressor — Actual vs Predicted"),
            ("gradient_boosting_residuals.png", "Gradient Boosting Regressor — Residual Analysis"),
            ("gradient_boosting_feature_importance.png", "Gradient Boosting Regressor — Top 20 Feature Importances"),
        ),
    },
    {
        "name": "Ridge Regression",
        "slug": "ridge-regression",
        "folder": "Ridge_Regression",
        "feature_note": (
            "L2 Tikhonov regularization penalizes the L2-norm of coefficient weights, shrinking collinear features "
            "and preventing numerical instability."
        ),
        "summary": "Linear regression with L2 weight regularization (alpha=1.0).",
        "interpretation": (
            "Achieves RMSE 2,050.45 and R² 0.4826, slightly outperforming ordinary least squares by stabilizing the 130 beta coefficients."
        ),
        "images": (
            ("ridge_regression_actual_vs_predicted.png", "Ridge Regression — Actual vs Predicted"),
            ("ridge_regression_residuals.png", "Ridge Regression — Residual Analysis"),
            ("top_coefficients.png", "Ridge Regression — Top 20 Features by Absolute Coefficient"),
        ),
    },
    {
        "name": "Linear Regression",
        "slug": "linear-regression",
        "folder": "Linear_Regression",
        "feature_note": (
            "The coefficient chart ranks features by the absolute size of their fitted linear coefficient — "
            "the larger the absolute coefficient, the more that feature moves the predicted loss."
        ),
        "summary": "Ordinary least-squares baseline that predicts loss as an unregularized linear combination.",
        "interpretation": (
            "Explains about 48.3% of the variance on test rows (RMSE 2,050.52, R² 0.4826). The linear constraint caps performance "
            "because insurance claim loss has strong non-linear interactions."
        ),
        "images": (
            ("linear_regression_actual_vs_predicted.png", "Linear Regression — Actual vs Predicted"),
            ("linear_regression_residuals.png", "Linear Regression — Residual Analysis"),
            ("linear_regression_top_coefficients.png", "Linear Regression — Top 20 Features by Absolute Coefficient"),
        ),
    },
    {
        "name": "Lasso Regression",
        "slug": "lasso-regression",
        "folder": "Lasso_Regression",
        "feature_note": (
            "L1 regularization drives coefficients of non-informative features to exact zero, performing embedded feature selection."
        ),
        "summary": "Linear regression with L1 norm penalty (alpha=1.0).",
        "interpretation": (
            "Achieves RMSE 2,051.88 and R² 0.4819 while enforcing sparsity across the 130 input variables."
        ),
        "images": (
            ("lasso_regression_actual_vs_predicted.png", "Lasso Regression — Actual vs Predicted"),
            ("lasso_regression_residuals.png", "Lasso Regression — Residual Analysis"),
            ("top_coefficients.png", "Lasso Regression — Top 20 Features by Absolute Coefficient"),
        ),
    },
    {
        "name": "Elastic Net Regression",
        "slug": "elastic-net",
        "folder": "Elastic_Net_Regression",
        "feature_note": (
            "Combines L1 and L2 penalties (l1_ratio=0.5) to balance sparsity and grouping of correlated continuous variables."
        ),
        "summary": "Regularized linear regression combining L1 (Lasso) and L2 (Ridge) penalties.",
        "interpretation": (
            "Yields RMSE 2,162.80 and R² 0.4244, trading peak R² for improved numerical stability on correlated inputs."
        ),
        "images": (
            ("elastic_net_regression_actual_vs_predicted.png", "Elastic Net Regression — Actual vs Predicted"),
            ("elastic_net_regression_residuals.png", "Elastic Net Regression — Residual Analysis"),
            ("top_coefficients.png", "Elastic Net Regression — Top 20 Features by Absolute Coefficient"),
        ),
    },
    {
        "name": "Support Vector Regression (Linear)",
        "slug": "support-vector-regression",
        "folder": "Support_Vector_Regression_Linear",
        "feature_note": (
            "Linear SVR fits a margin-based hyperplane with an epsilon-insensitive loss tube, making it robust against small noise."
        ),
        "summary": "Linear support vector machine optimizing epsilon-insensitive loss.",
        "interpretation": (
            "Achieves low median absolute error (785.42) with overall test RMSE 2,174.74 and R² 0.4180."
        ),
        "images": (
            ("support_vector_regression_linear_actual_vs_predicted.png", "Linear SVR — Actual vs Predicted"),
            ("support_vector_regression_linear_residuals.png", "Linear SVR — Residual Analysis"),
            ("top_coefficients.png", "Linear SVR — Top 20 Features by Absolute Coefficient"),
        ),
    },
    {
        "name": "Decision Tree Regressor",
        "slug": "decision-tree",
        "folder": "Decision_Tree_Regressor",
        "feature_note": (
            "Decision-tree feature importance is the total reduction in split impurity across the tree, normalised to sum to 1."
        ),
        "summary": "A single CART decision tree recursively partitioning the 130-feature space.",
        "interpretation": (
            "Predicts a piece-wise constant value with high variance, resulting in RMSE 2,638.94 and R² 0.1431."
        ),
        "images": (
            ("decision_tree_actual_vs_predicted.png", "Decision Tree Regressor — Actual vs Predicted"),
            ("decision_tree_residuals.png", "Decision Tree Regressor — Residual Analysis"),
            ("decision_tree_feature_importance.png", "Decision Tree Regressor — Top 20 Feature Importances"),
        ),
    },
    {
        "name": "K-Nearest Neighbors Regressor",
        "slug": "k-nearest-neighbors",
        "folder": "K_Nearest_Neighbors_Regressor",
        "feature_note": (
            "Non-parametric local regression predicting the arithmetic mean loss of the k=25 nearest neighbors in 130-dimensional space."
        ),
        "summary": "Instance-based regression using Euclidean distance across 25 nearest neighbors.",
        "interpretation": (
            "Suffers from the curse of dimensionality in 130D space (RMSE 2,673.83, R² 0.1203), confirming that tree ensembles handle high dimensionality better."
        ),
        "images": (
            ("k_nearest_neighbors_regressor_actual_vs_predicted.png", "KNN Regressor — Actual vs Predicted"),
            ("k_nearest_neighbors_regressor_residuals.png", "KNN Regressor — Residual Analysis"),
        ),
    },
    {
        "name": "Dummy Regressor (Median Baseline)",
        "slug": "dummy-regressor",
        "folder": "Dummy_Regressor_Median_Baseline",
        "feature_note": (
            "Zero-feature constant predictor: always predicts the training set median claim loss (~2,115.47) without reading any features."
        ),
        "summary": "Non-learning baseline model predicting the training set median.",
        "interpretation": (
            "Establishes the fundamental sanity check: RMSE 2,986.61, MAE 1,788.56, R² -0.0976. Confirms that all machine learning models add substantial predictive value."
        ),
        "images": (
            ("dummy_regressor_median_baseline_actual_vs_predicted.png", "Dummy Regressor — Actual vs Predicted"),
            ("dummy_regressor_median_baseline_residuals.png", "Dummy Regressor — Residual Analysis"),
        ),
    },
    {
        "name": "AdaBoost Regressor",
        "slug": "adaboost",
        "folder": "AdaBoost_Regressor",
        "feature_note": (
            "AdaBoost iteratively adjusts instance weights based on prediction errors on the previous estimator."
        ),
        "summary": "Adaptive boosting ensemble using decision stump base estimators.",
        "interpretation": (
            "Right-skewed heavy-tailed claims loss destabilizes standard AdaBoost exponential reweighting (RMSE 4,627.19, R² -1.63), demonstrating why gradient boosting with squared loss is necessary."
        ),
        "images": (
            ("adaboost_regressor_actual_vs_predicted.png", "AdaBoost Regressor — Actual vs Predicted"),
            ("adaboost_regressor_residuals.png", "AdaBoost Regressor — Residual Analysis"),
            ("feature_importance.png", "AdaBoost Regressor — Top 20 Feature Importances"),
        ),
    },
)

MODEL_METADATA = {
    "Hist Gradient Boosting Regressor": {
        "model_type": "Histogram-based Gradient Boosted Trees with Early Stopping",
        "family": "Boosting Ensemble",
        "algorithm": "Gradient boosting that discretizes continuous numerical features into integer-valued bins (256 bins), accelerating split evaluations. Fitted with 3-fold cross-validation early stopping and L2 regularization.",
        "cv_summary": "3-Fold Early-Stopping CV (median 830 iters) OOF MAE: 1,214.84, RMSE: 1,956.68, R²: 0.5484; 5-Fold CV: MAE 1,203.37 ± 10.99, RMSE 1,937.66 ± 34.40, R² 0.5532 ± 0.0056.",
    },
    "Extra Trees Regressor": {
        "model_type": "Extremely Randomized Trees Ensemble",
        "family": "Bagging Ensemble",
        "algorithm": "Ensemble of 100 randomized decision trees drawing random thresholds for each candidate split, reducing variance on continuous and encoded inputs.",
        "cv_summary": "Evaluated on 10,000 held-out test records: MAE 1,254.61, RMSE 1,980.32, R² 0.5174.",
    },
    "Random Forest Regressor": {
        "model_type": "Bootstrap Aggregated (Bagging) Ensemble of Decision Trees",
        "family": "Bagging Ensemble",
        "algorithm": "Ensemble of 100 deep decision trees fitted on bootstrap samples with random feature subsampling at each node (max_depth=20, min_samples_leaf=2). Predictions are the arithmetic mean of all individual tree estimates.",
        "cv_summary": "5-Fold CV: MAE 1,260.39 ± 5.03, RMSE 1,978.81 ± 28.28, R² 0.5339 ± 0.0116.",
    },
    "Gradient Boosting Regressor": {
        "model_type": "Sequential Stagewise Gradient Boosted Decision Trees",
        "family": "Boosting Ensemble",
        "algorithm": "150 sequential boosting iterations adding shallow regression trees (max_depth=3) to fit negative gradients (residuals) of the squared error loss function with shrinkage learning rate 0.05.",
        "cv_summary": "5-Fold CV: MAE 1,269.72 ± 13.74, RMSE 1,978.59 ± 41.74, R² 0.5339 ± 0.0179.",
    },
    "Ridge Regression": {
        "model_type": "L2-Regularized Linear Regression (Tikhonov)",
        "family": "Linear Model",
        "algorithm": "Ridge regression imposing an L2 penalty on beta coefficients to avoid multicollinearity and coefficient inflation in 130-dimensional space.",
        "cv_summary": "Evaluated on 10,000 held-out test records: MAE 1,333.49, RMSE 2,050.45, R² 0.4826.",
    },
    "Linear Regression": {
        "model_type": "Ordinary Least Squares (OLS) Linear Regression",
        "family": "Linear Model",
        "algorithm": "Closed-form analytical linear regression baseline estimating 130 beta coefficients to minimize the sum of squared residuals between observed and predicted losses.",
        "cv_summary": "5-Fold CV: MAE 1,327.86 ± 15.98, RMSE 2,093.48 ± 51.00, R² 0.4785 ± 0.0108.",
    },
    "Lasso Regression": {
        "model_type": "L1-Regularized Linear Regression (Lasso)",
        "family": "Linear Model",
        "algorithm": "L1-regularized regression promoting sparse coefficients, driving non-influential features to exact zero.",
        "cv_summary": "Evaluated on 10,000 held-out test records: MAE 1,333.58, RMSE 2,051.88, R² 0.4819.",
    },
    "Elastic Net Regression": {
        "model_type": "L1 + L2 Regularized Linear Regression",
        "family": "Linear Model",
        "algorithm": "Linearly combined penalty of L1 and L2 regularization (l1_ratio=0.5) balancing variable selection and correlated coefficient grouping.",
        "cv_summary": "Evaluated on 10,000 held-out test records: MAE 1,394.39, RMSE 2,162.80, R² 0.4244.",
    },
    "Support Vector Regression (Linear)": {
        "model_type": "Linear Support Vector Regression (Linear SVR)",
        "family": "Margin-Based",
        "algorithm": "Support vector regression using epsilon-insensitive loss tube and dual optimization, robust against small residual fluctuations.",
        "cv_summary": "Evaluated on 10,000 held-out test records: MAE 1,289.04, RMSE 2,174.74, R² 0.4180.",
    },
    "Decision Tree Regressor": {
        "model_type": "Single Recursive Partitioning Decision Tree",
        "family": "Decision Tree",
        "algorithm": "A single classification and regression tree (CART) recursively splitting the 130-dimensional feature space to maximize reduction in squared error (max_depth=20, min_samples_leaf=2).",
        "cv_summary": "5-Fold CV: MAE 1,613.58 ± 26.99, RMSE 2,618.39 ± 77.33, R² 0.1836 ± 0.0444.",
    },
    "K-Nearest Neighbors Regressor": {
        "model_type": "K-Nearest Neighbors Regression (k=25)",
        "family": "Instance-Based",
        "algorithm": "Non-parametric instance regression computing the uniform average target loss of the 25 nearest test points in Euclidean space.",
        "cv_summary": "Evaluated on 10,000 held-out test records: MAE 1,720.33, RMSE 2,673.83, R² 0.1203.",
    },
    "Dummy Regressor (Median Baseline)": {
        "model_type": "Constant Median Statistical Baseline",
        "family": "Baseline",
        "algorithm": "Baseline predictor that always returns the training set median claim loss (~2,115.47), ignoring all input features.",
        "cv_summary": "Evaluated on 10,000 held-out test records: MAE 1,788.56, RMSE 2,986.61, R² -0.0976.",
    },
    "AdaBoost Regressor": {
        "model_type": "Adaptive Boosting Regression Ensemble",
        "family": "Boosting Ensemble",
        "algorithm": "Sequential boosting ensemble adjusting training sample weights after each stage based on individual instance prediction errors.",
        "cv_summary": "Evaluated on 10,000 held-out test records: MAE 4,278.58, RMSE 4,627.19, R² -1.6346.",
    },
}

COMPARISON_CHART = (
    "Model_Comparison/model_comparison_bar_chart.png",
    "ClaimWise model comparison on the 10,000 test rows — RMSE (lower is better) and R² (higher is better).",
)

NOTEBOOK_PURPOSE = {
    "01": ("Data Inspection", "Audits the raw dataset: shape, data types, missing values, duplicates and the loss target."),
    "02": ("Data Preprocessing", "Runs the single authoritative pipeline and verifies every stage plus the 10 leakage checks."),
    "03": ("Exploratory Data Analysis", "Distribution, correlation and feature analysis on the preprocessed dataset."),
    "04": ("Linear Regression", "Trains, evaluates and saves the linear baseline."),
    "05": ("Decision Tree", "Trains, evaluates and saves the decision tree regressor."),
    "06": ("Random Forest", "Trains, evaluates and saves the random forest, including feature importances."),
    "07": ("Gradient Boosting", "Trains, evaluates and saves the gradient boosting regressor."),
    "08": ("Hist Gradient Boosting", "Trains, evaluates and saves the histogram gradient boosting model adapted from the provided code, including 3-fold cross-validation with early stopping."),
    "09": ("Model Comparison", "Trains all five models and ranks them by MAE, MSE, RMSE and R²."),
    "10": ("Final Prediction", "End-to-end prediction of claim loss with the saved best model."),
    "11": ("PCA Feature Analysis", "Principal component analysis of the 130 features: explained variance, component loadings and a PCA-space vs original-space modelling experiment."),
    "12": ("Logistic Regression Classification", "Derived high-loss target, logistic regression classifier, confusion matrix, precision/recall/F1 and ROC curve."),
    "13": ("Decision Tree Classification", "Decision tree classifier with confusion matrix, top features and a metric comparison against logistic regression."),
    "14": ("Clustering (K-Means)", "Unsupervised k-means clustering on the features only (loss excluded), with elbow and silhouette selection plus cluster profiling."),
    "15": ("DBSCAN", "Density-based clustering with k-distance eps selection, noise reporting and a PCA-20 variant."),
    "16": ("Anomaly Detection", "Isolation Forest anomaly detection on the features only — flagged records are unusual, never claimed to be fraud."),
    "17": ("5-Fold Cross-Validation", "Leakage-free 5-fold cross-validation for all five regressors and both classifiers, with the preprocessing fitted inside every fold."),
}

_LEAKAGE_CACHE = None


def _find_dataset_file(new_name, legacy_name):
    """Find a prepared file without changing the existing preprocessing."""
    new_path = DATASET_DIR / new_name
    legacy_path = DATASET_DIR / legacy_name
    if new_path.exists():
        return new_path
    if legacy_path.exists():
        return legacy_path
    return None


def load_model_results():
    """Read the real model-comparison CSV generated by model evaluation."""
    comparison_path = OUTPUTS_DIR / "Model_Comparison" / "model_comparison.csv"
    if not comparison_path.exists():
        return [], None

    comparison = pd.read_csv(comparison_path)
    rows = comparison.to_dict(orient="records")
    best_name = str(rows[0]["Model"]) if rows else None
    return rows, best_name


def load_prediction_data():
    """Load the actual prepared test features and targets for safe demo prediction."""
    x_path = _find_dataset_file("06_split_X_test.csv", "X_test.csv")
    y_path = _find_dataset_file("06_split_y_test.csv", "y_test.csv")
    if x_path is None or y_path is None:
        return None, None
    return pd.read_csv(x_path), pd.read_csv(y_path)["loss"]


def model_is_ready():
    return (MODELS_DIR / "best_model.pkl").exists() and bool(load_model_results()[0])


def _read_stage_rows():
    """Read the stage summary written by Src/claimwise_preprocessing_pipeline.py."""
    path = OUTPUTS_DIR / "Preprocessing" / "claimwise_preprocessing_stage_summary.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return {str(row["stage"]): row for row in frame.to_dict(orient="records")}


def load_stage_rows():
    """Feature-engineering stages only (target-only rows are omitted on purpose)."""
    rows = _read_stage_rows()
    return [row for key, row in rows.items() if not key.startswith(("6c", "6d"))]


def load_dataset_stats():
    """Dataset facts read from the real CSV files and the stage summary."""
    stages = _read_stage_rows()
    raw_stage = stages.get("1. Raw", {})
    final_stage = stages.get("5. Final preprocessed", {})
    train_stage = stages.get("6a. X_train", {})
    test_stage = stages.get("6b. X_test", {})

    def _int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    stats = {
        "raw_rows": _int(raw_stage.get("rows")),
        "raw_cols": _int(raw_stage.get("columns")),
        "final_rows": _int(final_stage.get("rows")),
        "final_cols": _int(final_stage.get("columns")),
        "feature_count": _int(train_stage.get("columns")),
        "train_rows": _int(train_stage.get("rows")),
        "test_rows": _int(test_stage.get("rows")),
        "missing": _int(raw_stage.get("missing_values")),
        "duplicates": _int(raw_stage.get("duplicate_rows")),
        "target_column": None,
        "removed_column": None,
        "categorical_count": None,
        "continuous_count": None,
    }

    raw_path = DATASET_DIR / "01_raw_claimwise_50000.csv"
    final_path = DATASET_DIR / "05_preprocessed_final.csv"
    if raw_path.exists() and final_path.exists():
        raw_columns = list(pd.read_csv(raw_path, nrows=0).columns)
        final_columns = set(pd.read_csv(final_path, nrows=0).columns)
        stats["target_column"] = "loss" if "loss" in final_columns else None
        stats["categorical_count"] = sum(1 for col in raw_columns if str(col).startswith("cat"))
        stats["continuous_count"] = sum(1 for col in raw_columns if str(col).startswith("cont"))
        if "id" in raw_columns and "id" not in final_columns:
            stats["removed_column"] = "id"

    return stats


def load_leakage_stats():
    """Scaled-feature means for train and test, verified by notebook 02."""
    global _LEAKAGE_CACHE
    if _LEAKAGE_CACHE is not None:
        return _LEAKAGE_CACHE

    train_path = _find_dataset_file("06_split_X_train.csv", "X_train.csv")
    test_path = _find_dataset_file("06_split_X_test.csv", "X_test.csv")
    if train_path is None or test_path is None:
        return {"available": False}

    x_train = pd.read_csv(train_path)
    x_test = pd.read_csv(test_path)
    continuous = [col for col in x_train.columns if str(col).startswith("cont")]

    _LEAKAGE_CACHE = {
        "available": True,
        "continuous_count": len(continuous),
        "train_max_abs_mean": float(x_train[continuous].mean().abs().max()),
        "test_max_abs_mean": float(x_test[continuous].mean().abs().max()),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }
    return _LEAKAGE_CACHE


def load_model_gallery():
    """Join the static model description with the real metrics CSV and image files."""

    comparison_rows, best_name = load_model_results()
    comparison_by_name = {str(row["Model"]): row for row in comparison_rows}

    gallery = []
    for model in MODEL_GALLERY:
        folder = model["folder"]
        metrics = comparison_by_name.get(model["name"])
        if metrics is None:
            metrics_path = OUTPUTS_DIR / folder / "metrics.csv"
            if metrics_path.exists():
                rows = pd.read_csv(metrics_path).to_dict(orient="records")
                metrics = rows[0] if rows else None

        images = []
        for filename, caption in model["images"]:
            if (OUTPUTS_DIR / folder / filename).exists():
                images.append(
                    {
                        "file": filename,
                        "src": f"{folder}/{filename}",
                        "alt": caption,
                        "caption": caption,
                        "featured": model["slug"] == "random-forest" and filename == "feature_importance.png",
                    }
                )

        meta = MODEL_METADATA.get(model["name"], {})
        rank = None
        for idx, row in enumerate(comparison_rows):
            if str(row.get("Model")) == model["name"]:
                rank = idx + 1
                break

        gallery.append(
            {
                "name": model["name"],
                "slug": model["slug"],
                "summary": model["summary"],
                "interpretation": model["interpretation"],
                "feature_note": model.get("feature_note"),
                "model_type": meta.get("model_type", "Regression Algorithm"),
                "algorithm": meta.get("algorithm", model["summary"]),
                "cv_summary": meta.get("cv_summary", "Evaluated on 10,000 test records."),
                "status": "Trained & Evaluated",
                "rank": rank,
                "metrics": metrics,
                "images": images,
                "is_best": bool(best_name) and model["name"] == best_name,
            }
        )
    gallery.sort(key=lambda m: (m["rank"] is None, m["rank"]))
    return gallery, best_name


def load_model_kpis():
    """Summary KPI metrics calculated from actual model evaluation inventory."""
    rows, best_name = load_model_results()
    if not rows:
        return {
            "total_models": 0,
            "best_model": "None",
            "lowest_mae": 0.0,
            "lowest_rmse": 0.0,
            "highest_r2": 0.0,
            "chart_count": 0,
        }
    maes = [float(r["MAE"]) for r in rows if "MAE" in r and pd.notna(r["MAE"])]
    rmses = [float(r["RMSE"]) for r in rows if "RMSE" in r and pd.notna(r["RMSE"])]
    r2s = [float(r["R2"]) for r in rows if "R2" in r and pd.notna(r["R2"])]
    vis_inv = get_visualization_inventory()
    model_charts = [v for v in vis_inv if v["page"] == "models"]
    return {
        "total_models": len(rows),
        "best_model": best_name or (str(rows[0]["Model"]) if rows else "N/A"),
        "lowest_mae": min(maes) if maes else 0.0,
        "lowest_rmse": min(rmses) if rmses else 0.0,
        "highest_r2": max(r2s) if r2s else 0.0,
        "chart_count": len(model_charts),
    }


def load_comparison_chart():
    """Return the comparison chart only if the pipeline actually produced it."""

    filename, caption = COMPARISON_CHART
    if not (OUTPUTS_DIR / filename).exists():
        return None
    return {"src": filename, "alt": caption, "caption": caption}


def load_notebooks():
    """List the real .ipynb files in Notebooks/ as an ordered workflow."""

    notebooks = []
    if not NOTEBOOKS_DIR.exists():
        return notebooks

    for path in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        prefix = path.name[:2]
        title, purpose = NOTEBOOK_PURPOSE.get(
            prefix, (path.stem.replace("_", " "), "Supporting project notebook.")
        )
        notebooks.append(
            {
                "filename": path.name,
                "number": prefix if prefix.isdigit() else "",
                "title": title,
                "purpose": purpose,
            }
        )
    return notebooks


def load_visualization_modules():
    """Build the verified 5-module visualization catalog.

    Each visualization entry provides:
    1. Module number and module name
    2. Visualization title
    3. Graph image path, alt text, and caption
    4. Short explanation of the concept
    5. What the graph shows
    6. Why the visualization is useful
    7. Dataset or model used to generate it
    8. Actual metric or data summary
    """
    modules = [
        {
            "number": "Module 1",
            "section_tag": "Section A: Dataset Overview",
            "name": "Dataset Understanding",
            "academic_title": "Module 1: Dataset Exploration and Initial Audit",
            "slug": "module-1",
            "summary": "Initial exploration and verification of the raw ClaimWise insurance claims dataset: dimensions, column types, target distribution, and feature partition.",
            "visualizations": [
                {
                    "id": "m1-v1",
                    "module_num": "Module 1",
                    "module_name": "Dataset Understanding",
                    "section_label": "Section A — Dataset Overview",
                    "title": "Distribution of Continuous Claim Loss (Raw Dataset)",
                    "src": "Data_Inspection/claimwise_raw_target_distribution.png",
                    "alt": "Histogram and Density of Claim Loss in Raw Dataset",
                    "caption": "Raw Claim Loss Probability Density Distribution (N = 50,000 records).",
                    "concept": "Univariate target exploration to examine density profile, right-skewness, and severity boundaries.",
                    "what_it_shows": "Strongly right-skewed target distribution with concentration of frequent modest claims and an extended tail of high-severity losses up to 85,923.56.",
                    "why_useful": "Demonstrates why regression error metrics (MAE and RMSE) are required rather than accuracy, and establishes the baseline for loss estimation.",
                    "dataset_or_model": "Dataset/01_raw_claimwise_50000.csv (Raw ClaimWise Dataset)",
                    "metric_summary": "Min: 10.00 | Median: 2,115.57 | Mean: 3,035.01 | Max: 85,923.56 across 50,000 records.",
                },
                {
                    "id": "m1-v2",
                    "module_num": "Module 1",
                    "module_name": "Dataset Understanding",
                    "section_label": "Section A — Dataset Overview",
                    "title": "Raw Feature Schema & Variable Types Composition",
                    "src": "Data_Inspection/claimwise_feature_types_composition.png",
                    "alt": "ClaimWise Feature Schema Composition across 132 columns",
                    "caption": "Structural Feature Breakdown: 116 Categorical, 14 Continuous, 1 Target, 1 ID.",
                    "concept": "Feature partitioning by measurement scale to define preprocessing requirements.",
                    "what_it_shows": "73 binary categorical columns (cat1–cat73), 43 multi-level categorical columns (cat74–cat116), 14 continuous numeric features (cont1–cont14), 1 target (loss), and 1 ID column.",
                    "why_useful": "Identifies that categorical encoding is needed for 116 features and standardization is required for 14 continuous features, while 'id' must be dropped to prevent false correlations.",
                    "dataset_or_model": "Dataset/01_raw_claimwise_50000.csv",
                    "metric_summary": "132 total columns -> 130 predictive features (116 categorical, 14 continuous).",
                },
            ],
        },
        {
            "number": "Module 2",
            "section_tag": "Sections E & F: Preprocessing & Data Integrity",
            "name": "Data Cleaning and Preprocessing",
            "academic_title": "Module 2: Data Cleaning, Imputation, Encoding, and Scaling",
            "slug": "module-2",
            "summary": "Step-by-step data cleansing, imputation, LabelEncoding, StandardScaler normalization, 80:20 train/test splitting, and empirical data leakage verification.",
            "visualizations": [
                {
                    "id": "m2-v1",
                    "module_num": "Module 2",
                    "module_name": "Data Cleaning and Preprocessing",
                    "section_label": "Section F — Preprocessing Visualizations",
                    "title": "Preprocessing Pipeline Stages & Data Leakage Verification",
                    "src": "Preprocessing/claimwise_preprocessing_pipeline.png",
                    "alt": "Preprocessing Pipeline Stages Dimensions and Leakage Check",
                    "caption": "Feature dimensionality progression across 6 pipeline stages and empirical scaler leakage check.",
                    "concept": "Sequential data transformation from raw CSV to model-ready train and test matrices while strictly isolating test rows from feature transformers.",
                    "what_it_shows": "Dimensional progression (132 -> 131 -> 130 features) and maximum absolute mean of scaled features: 0.000000 on training rows vs 0.023239 on test rows.",
                    "why_useful": "Empirically proves that StandardScaler was fitted exclusively on the 40,000 training rows and applied to the 10,000 test rows without leakage.",
                    "dataset_or_model": "Src/claimwise_preprocessing_pipeline.py across Dataset/ files",
                    "metric_summary": "Training max abs mean: 0.000000 (fitted on train); Test max abs mean: 0.023239 (transformed only).",
                },
                {
                    "id": "m2-v2",
                    "module_num": "Module 2",
                    "module_name": "Data Cleaning and Preprocessing",
                    "section_label": "Section E — Missing-Value Analysis",
                    "title": "Missing Value Integrity & Completeness Heatmap",
                    "src": "EDA_Analysis_outputs/06_missing_values_heatmap.png",
                    "alt": "Missing Values Heatmap across ClaimWise Dataset",
                    "caption": "Feature-level completeness audit across all rows and columns (0 missing values).",
                    "concept": "Auditing dataset completeness across all features before machine learning.",
                    "what_it_shows": "100% complete data matrix with 0 missing cells across all 50,000 records.",
                    "why_useful": "Confirms data integrity and verifies that imputation logic (median for numericals, mode for categoricals) is tested and ready for production inference.",
                    "dataset_or_model": "Dataset/01_raw_claimwise_50000.csv",
                    "metric_summary": "Total missing cells: 0 (0.00%); Duplicate records: 0 (0.00%).",
                },
            ],
        },
        {
            "number": "Module 3",
            "section_tag": "Sections B, C & D: Univariate, Bivariate & Correlation Analysis",
            "name": "Exploratory Data Analysis",
            "academic_title": "Module 3: Exploratory Data Analysis and Feature Correlation",
            "slug": "module-3",
            "summary": "Statistical distributions, correlation matrix heatmaps, boxplot outlier boundaries, categorical frequency balances, and bivariate relationships with claim loss.",
            "visualizations": [
                {
                    "id": "m3-v1",
                    "module_num": "Module 3",
                    "module_name": "Exploratory Data Analysis",
                    "section_label": "Section D — Correlation Analysis",
                    "title": "Correlation Matrix Heatmap (Continuous Features & Loss)",
                    "src": "EDA_Analysis_outputs/04_correlation_heatmap.png",
                    "alt": "Correlation Heatmap of Numerical Features and Loss",
                    "caption": "Pearson correlation coefficient matrix for cont1 through cont14 and target loss.",
                    "concept": "Pairwise correlation analysis to detect multicollinearity among predictors and assess linear associations with the target.",
                    "what_it_shows": "High multicollinearity between certain feature pairs (e.g. cont11 and cont12 r = 0.99) and weak-to-moderate linear correlation with loss.",
                    "why_useful": "Explains why linear regression alone cannot fully capture claim loss variance, highlighting the need for non-linear tree ensembles.",
                    "dataset_or_model": "Dataset/05_preprocessed_final.csv",
                    "metric_summary": "Top continuous correlation with loss: cont7 (r = -0.067); highest feature pair: cont11-cont12 (r = 0.99).",
                },
                {
                    "id": "m3-v2",
                    "module_num": "Module 3",
                    "module_name": "Exploratory Data Analysis",
                    "section_label": "Section B — Univariate Analysis (Target)",
                    "title": "Claim Loss Probability Density Distribution",
                    "src": "EDA_Analysis_outputs/01_target_distribution.png",
                    "alt": "Target Variable Loss Probability Distribution",
                    "caption": "Probability density estimate and histogram for claim loss.",
                    "concept": "Univariate density analysis of the continuous target variable loss.",
                    "what_it_shows": "High concentration between 1,000 and 4,000 with a sparse high-severity right tail.",
                    "why_useful": "Demonstrates the real-world insurance phenomenon where most claims are moderate while catastrophic claims represent severe outliers.",
                    "dataset_or_model": "Dataset/05_preprocessed_final.csv",
                    "metric_summary": "Mean: 3,035.01 | Median: 2,115.57 | Interquartile Range: 1,204.47 to 3,863.42.",
                },
                {
                    "id": "m3-v3",
                    "module_num": "Module 3",
                    "module_name": "Exploratory Data Analysis",
                    "section_label": "Section B — Univariate Analysis (Continuous)",
                    "title": "Continuous Predictor Distribution — cont7",
                    "src": "EDA_Analysis_outputs/02_feature_distribution_cont7.png",
                    "alt": "Feature Distribution of cont7",
                    "caption": "Standardized frequency distribution of cont7 (highest continuous correlation with loss).",
                    "concept": "Distribution inspection for continuous input variables.",
                    "what_it_shows": "Multi-modal distribution of feature cont7 across the standardized range.",
                    "why_useful": "Verifies that continuous features retain multi-modal nuances after standard scaling.",
                    "dataset_or_model": "Dataset/05_preprocessed_final.csv",
                    "metric_summary": "Correlation with loss: -0.067 | Standardized mean: 0.00, std: 1.00.",
                },
                {
                    "id": "m3-v4",
                    "module_num": "Module 3",
                    "module_name": "Exploratory Data Analysis",
                    "section_label": "Section B & E — Outlier Analysis (Boxplot)",
                    "title": "Outlier & Quartile Analysis — cont7 Boxplot",
                    "src": "EDA_Analysis_outputs/03_boxplot_cont7.png",
                    "alt": "Boxplot of cont7 showing quartiles and outliers",
                    "caption": "Tukey box-and-whisker plot of cont7 showing median and interquartile spread.",
                    "concept": "Quartile dispersion and outlier identification for continuous features.",
                    "what_it_shows": "Central 50% interquartile range (IQR) and upper/lower whisker limits.",
                    "why_useful": "Verifies that extreme values are bounded and represent valid insurance measurements rather than corrupted data entries.",
                    "dataset_or_model": "Dataset/05_preprocessed_final.csv",
                    "metric_summary": "Clean quartile bounds verified with no corrupted entries.",
                },
                {
                    "id": "m3-v5",
                    "module_num": "Module 3",
                    "module_name": "Exploratory Data Analysis",
                    "section_label": "Section B & C — Categorical Feature Distribution",
                    "title": "Categorical Frequency Distribution — cat80",
                    "src": "EDA_Analysis_outputs/05_countplot_cat80.png",
                    "alt": "Frequency countplot of categorical feature cat80",
                    "caption": "Category count distribution for cat80 (the top split feature in Random Forest).",
                    "concept": "Frequency balance analysis across categorical predictor levels.",
                    "what_it_shows": "Distribution of observations across the distinct categories of cat80.",
                    "why_useful": "Explains why tree-based models repeatedly split on cat80: its categorical splits provide the greatest reduction in squared prediction error.",
                    "dataset_or_model": "Dataset/01_raw_claimwise_50000.csv",
                    "metric_summary": "Ranked #1 most important feature in Random Forest and Decision Tree models.",
                },
                {
                    "id": "m3-v6",
                    "module_num": "Module 3",
                    "module_name": "Exploratory Data Analysis",
                    "section_label": "Section B — Univariate Analysis (Continuous)",
                    "title": "Continuous Predictor Distribution — cont1",
                    "src": "EDA_Analysis_outputs/02_feature_distribution_cont1.png",
                    "alt": "Feature Distribution of cont1",
                    "caption": "Standardized frequency distribution of continuous predictor cont1.",
                    "concept": "Feature spread and skewness inspection for standardized numerical inputs.",
                    "what_it_shows": "Multi-peaked continuous density curve across the normalized feature domain.",
                    "why_useful": "Confirms that continuous features exhibit complex non-Gaussian properties, motivating non-linear models.",
                    "dataset_or_model": "Dataset/05_preprocessed_final.csv",
                    "metric_summary": "Standardized range with unit variance across 50,000 records.",
                },
                {
                    "id": "m3-v7",
                    "module_num": "Module 3",
                    "module_name": "Exploratory Data Analysis",
                    "section_label": "Section B & E — Outlier Analysis (Boxplot)",
                    "title": "Outlier & Quartile Analysis — cont14 Boxplot",
                    "src": "EDA_Analysis_outputs/03_boxplot_cont14.png",
                    "alt": "Boxplot of cont14",
                    "caption": "Box-and-whisker dispersion for continuous feature cont14.",
                    "concept": "Extreme value and quartile verification for high-index continuous features.",
                    "what_it_shows": "Interquartile range and whisker bounds with modest positive tail values.",
                    "why_useful": "Verifies absence of data entry corruptions or runaway values in cont14.",
                    "dataset_or_model": "Dataset/05_preprocessed_final.csv",
                    "metric_summary": "Bounded IQR with preserved legitimate extremes.",
                },
                {
                    "id": "m3-v8",
                    "module_num": "Module 3",
                    "module_name": "Exploratory Data Analysis",
                    "section_label": "Section B — Binary Categorical Balance",
                    "title": "Binary Categorical Frequency Balance — cat1",
                    "src": "EDA_Analysis_outputs/05_countplot_cat1.png",
                    "alt": "Countplot of categorical feature cat1",
                    "caption": "Frequency distribution across binary categories for cat1.",
                    "concept": "Class frequency balance in binary categorical predictors.",
                    "what_it_shows": "Comparative observation frequency across the two distinct categories of cat1.",
                    "why_useful": "Verifies adequate representation across both category levels to avoid zero-variance splits.",
                    "dataset_or_model": "Dataset/01_raw_claimwise_50000.csv",
                    "metric_summary": "Binary category split across 50,000 observations.",
                },
            ],
        },
        {
            "number": "Module 4",
            "section_tag": "Section G: Model-Specific Visualizations",
            "name": "Machine Learning and Model Analysis",
            "academic_title": "Module 4: Machine Learning Model Training and Diagnostic Analysis",
            "slug": "module-4",
            "summary": "Individual diagnostics for all five trained regression models: actual vs predicted scatter plots, residual distributions, and feature importance profiles.",
            "visualizations": [
                {
                    "id": "m4-v1",
                    "module_num": "Module 4",
                    "module_name": "Machine Learning and Model Analysis",
                    "title": "Hist Gradient Boosting Regressor — Actual vs Predicted",
                    "src": "Hist_Gradient_Boosting_Regressor/hist_gradient_boosting_regressor_actual_vs_predicted.png",
                    "alt": "Hist Gradient Boosting Regressor Actual vs Predicted Plot",
                    "caption": "Actual vs Predicted claim loss scatter on the 10,000 test claims (Hist Gradient Boosting).",
                    "concept": "Scatter plot benchmarking model predictions on the y-axis against true held-out test losses on the x-axis.",
                    "what_it_shows": "Tight clustering along the 45-degree identity line (y = x) across the core claim loss range.",
                    "why_useful": "Provides visual evidence of high regression fit, confirming why Hist Gradient Boosting achieved the lowest RMSE.",
                    "dataset_or_model": "Hist Gradient Boosting Regressor (Models/best_model.pkl) on 10,000 test rows",
                    "metric_summary": "RMSE: 1,921.0240 | R²: 0.545901 | MAE: 1,215.2997 (Best Performing Model).",
                },
                {
                    "id": "m4-v2",
                    "module_num": "Module 4",
                    "module_name": "Machine Learning and Model Analysis",
                    "title": "Hist Gradient Boosting Regressor — Residual Analysis",
                    "src": "Hist_Gradient_Boosting_Regressor/hist_gradient_boosting_regressor_residuals.png",
                    "alt": "Hist Gradient Boosting Regressor Residual Plot",
                    "caption": "Residual error distribution (Actual - Predicted) vs predicted values.",
                    "concept": "Residual diagnostic plot to evaluate homoscedasticity and check for systematic model bias.",
                    "what_it_shows": "Residuals evenly distributed around the horizontal zero line across prediction ranges.",
                    "why_useful": "Verifies that prediction errors are largely unbiased and that early stopping prevented overfitting.",
                    "dataset_or_model": "Hist Gradient Boosting Regressor (Models/hist_gradient_boosting_regressor.pkl)",
                    "metric_summary": "Mean residual centered at 0; MSE: 3,690,333.12.",
                },
                {
                    "id": "m4-v3",
                    "module_num": "Module 4",
                    "module_name": "Machine Learning and Model Analysis",
                    "title": "Random Forest Regressor — Top 20 Feature Importances",
                    "src": "Random_Forest_Regressor/feature_importance.png",
                    "alt": "Random Forest Regressor Top 20 Feature Importances",
                    "caption": "Top 20 features ranked by mean decrease in impurity across 100 ensemble decision trees.",
                    "concept": "Tree ensemble feature importance based on split impurity reduction.",
                    "what_it_shows": "cat80 and cont7 heavily dominate model decisions, followed by cont2, cat79, and cont14.",
                    "why_useful": "Provides critical business interpretability, showing which insurance policy attributes drive claim loss severity.",
                    "dataset_or_model": "Random Forest Regressor (Models/random_forest_regressor.pkl)",
                    "metric_summary": "Top predictor cat80 relative importance > 0.22.",
                },
                {
                    "id": "m4-v4",
                    "module_num": "Module 4",
                    "module_name": "Machine Learning and Model Analysis",
                    "title": "Gradient Boosting Regressor — Actual vs Predicted",
                    "src": "Gradient_Boosting_Regressor/gradient_boosting_actual_vs_predicted.png",
                    "alt": "Gradient Boosting Regressor Actual vs Predicted Plot",
                    "caption": "Actual vs Predicted claim loss for Gradient Boosting Regressor on 10,000 test claims.",
                    "concept": "Stagewise boosting prediction evaluation on held-out test data.",
                    "what_it_shows": "Strong diagonal alignment with slightly wider dispersion in the upper loss tail.",
                    "why_useful": "Validates stagewise boosting convergence on complex non-linear claim data.",
                    "dataset_or_model": "Gradient Boosting Regressor (Models/gradient_boosting_regressor.pkl)",
                    "metric_summary": "RMSE: 2,005.2565 | R²: 0.505206 | MAE: 1,277.7829.",
                },
                {
                    "id": "m4-v5",
                    "module_num": "Module 4",
                    "module_name": "Machine Learning and Model Analysis",
                    "title": "Linear Regression — Top 20 Fitted Coefficients",
                    "src": "Linear_Regression/linear_regression_top_coefficients.png",
                    "alt": "Linear Regression Top 20 Coefficients by Absolute Magnitude",
                    "caption": "Top 20 regression weights (beta coefficients) by absolute magnitude.",
                    "concept": "Magnitude of standardized OLS regression coefficients to evaluate linear feature sensitivity.",
                    "what_it_shows": "Largest positive and negative linear weights driving predicted loss movements.",
                    "why_useful": "Reveals feature sensitivity under linear assumptions, highlighting which variables produce the greatest linear shifts.",
                    "dataset_or_model": "Linear Regression (Models/linear_regression.pkl)",
                    "metric_summary": "RMSE: 2,050.5223 | R²: 0.482615 | MAE: 1,333.5665.",
                },
                {
                    "id": "m4-v6",
                    "module_num": "Module 4",
                    "module_name": "Machine Learning and Model Analysis",
                    "title": "Decision Tree Regressor — Actual vs Predicted",
                    "src": "Decision_Tree_Regressor/decision_tree_actual_vs_predicted.png",
                    "alt": "Decision Tree Regressor Actual vs Predicted Plot",
                    "caption": "Actual vs Predicted claim loss for single Decision Tree on 10,000 test claims.",
                    "concept": "Evaluation of piece-wise constant prediction surface produced by a single tree.",
                    "what_it_shows": "Horizontal banding and high dispersion away from the 45-degree identity line.",
                    "why_useful": "Illustrates why single decision trees suffer from high variance on continuous targets, motivating ensemble methods.",
                    "dataset_or_model": "Decision Tree Regressor (Models/decision_tree_regressor.pkl)",
                    "metric_summary": "RMSE: 2,638.9450 | R²: 0.143070 | MAE: 1,609.9189 (Baseline single tree).",
                },
            ],
        },
        {
            "number": "Module 5",
            "section_tag": "Section G: Multi-Model Evaluation & Validation Benchmarks",
            "name": "Evaluation, Comparison, and Final Prediction",
            "academic_title": "Module 5: Model Evaluation, Cross-Validation, and Final Prediction",
            "slug": "module-5",
            "summary": "Head-to-head model comparison across test RMSE, MAE, MSE, and R², multi-model actual vs predicted plots, residuals comparison, and 5-fold cross-validation scores.",
            "visualizations": [
                {
                    "id": "m5-v1",
                    "module_num": "Module 5",
                    "module_name": "Evaluation, Comparison, and Final Prediction",
                    "title": "Model Performance Benchmark Comparison (RMSE & R²)",
                    "src": "Model_Comparison/model_comparison_bar_chart.png",
                    "alt": "Model Comparison Bar Chart — RMSE and R² across all five models",
                    "caption": "Comparative bar chart: Test RMSE (lower is better) and R² Score (higher is better).",
                    "concept": "Multi-model empirical evaluation benchmark on the exact same 10,000 held-out test rows.",
                    "what_it_shows": "Hist Gradient Boosting Regressor ranks #1 with lowest RMSE (1921.02) and highest R² (0.5459), followed by Random Forest, Gradient Boosting, Linear Regression, and Decision Tree.",
                    "why_useful": "Provides objective, reproducible evidence for selecting Hist Gradient Boosting as the production model.",
                    "dataset_or_model": "Outputs/Model_Comparison/model_comparison.csv evaluated on 06_split_X_test.csv",
                    "metric_summary": "Rank 1: HistGB (RMSE 1921.02, R² 0.5459) | Rank 2: RF (1984.41, 0.5154) | Rank 3: GB (2005.26, 0.5052) | Rank 4: Linear (2050.52, 0.4826) | Rank 5: DT (2638.95, 0.1431).",
                },
                {
                    "id": "m5-v2",
                    "module_num": "Module 5",
                    "module_name": "Evaluation, Comparison, and Final Prediction",
                    "title": "Multi-Model Actual vs Predicted Subplot Comparisons",
                    "src": "Model_Comparison/actual_vs_predicted.png",
                    "alt": "Multi-Model Actual vs Predicted Comparison Subplots",
                    "caption": "Side-by-side diagnostic scatter plots for all five models on identical test data.",
                    "concept": "Comparative regression fit inspection across model families.",
                    "what_it_shows": "Progression from loose scatter in single Decision Tree to tight diagonal clustering in Hist Gradient Boosting.",
                    "why_useful": "Direct visual comparison demonstrating how ensemble methods reduce prediction error.",
                    "dataset_or_model": "All 5 trained regression models on 10,000 test claims",
                    "metric_summary": "Ensembles improve R² from 0.1431 (Decision Tree) to 0.5459 (Hist Gradient Boosting).",
                },
                {
                    "id": "m5-v3",
                    "module_num": "Module 5",
                    "module_name": "Evaluation, Comparison, and Final Prediction",
                    "title": "Multi-Model Residual Error Distributions Comparison",
                    "src": "Model_Comparison/residuals.png",
                    "alt": "Multi-Model Residual Comparison Plot",
                    "caption": "Comparative residual error plots (Actual - Predicted) for all five models.",
                    "concept": "Residual variance comparison to test error tightness across model architectures.",
                    "what_it_shows": "Residuals narrow progressively from Decision Tree to Hist Gradient Boosting.",
                    "why_useful": "Confirms that Hist Gradient Boosting minimizes catastrophic residual deviations.",
                    "dataset_or_model": "Outputs/Model_Comparison/residuals.png across all 5 models",
                    "metric_summary": "Hist Gradient Boosting achieves lowest MSE: 3,690,333.12.",
                },
                {
                    "id": "m5-v4",
                    "module_num": "Module 5",
                    "module_name": "Evaluation, Comparison, and Final Prediction",
                    "title": "Leakage-Free 5-Fold Cross-Validation Performance Comparison",
                    "src": "CrossValidation/regression_5fold_scores.png",
                    "alt": "5-Fold Cross-Validation Score Distributions across Models",
                    "caption": "Box and bar comparison of MAE, RMSE, and R² across 5 random folds with fold-local preprocessing.",
                    "concept": "K-fold cross-validation (5 folds) with transformers fitted strictly within each training fold.",
                    "what_it_shows": "Fold score stability across models, with Hist Gradient Boosting maintaining highest average R² and lowest RMSE across all folds.",
                    "why_useful": "Proves that model performance is robust and repeatable, not an artifact of a single train/test split.",
                    "dataset_or_model": "Outputs/CrossValidation/regression_5fold_summary.csv",
                    "metric_summary": "Hist Gradient Boosting 5-Fold Mean R²: 0.5532 ± 0.0056 | Mean RMSE: 1,937.66 ± 34.40.",
                },
            ],
        },
    ]
    return modules


_PREPROCESSOR_CACHE = None
_FORM_GROUPS_CACHE = None


def get_preprocessor():
    """Fitted LabelEncoders + StandardScaler replayed from the preprocessing pipeline."""

    global _PREPROCESSOR_CACHE
    if _PREPROCESSOR_CACHE is None:
        _PREPROCESSOR_CACHE = load_preprocessor()
    return _PREPROCESSOR_CACHE


def build_form_groups():
    """Group the 130 real input columns for the prediction form.

    The grouping is derived from the data itself: how many values each
    categorical column actually holds in the raw dataset. No domain names are
    invented — the columns keep their original cat/cont names.
    """

    global _FORM_GROUPS_CACHE
    if _FORM_GROUPS_CACHE is not None:
        return _FORM_GROUPS_CACHE

    preprocessor = get_preprocessor()

    continuous_fields = [
        {
            "name": col,
            "kind": "number",
            "step": "any",
            "min": stats["min"],
            "max": stats["max"],
            "default": stats["median"],
        }
        for col, stats in preprocessor["continuous"].items()
    ]

    def categorical_fields(is_binary):
        return [
            {
                "name": col,
                "kind": "select",
                "options": info["categories"],
                "default": info["default"],
                "cardinality": info["cardinality"],
            }
            for col, info in preprocessor["categorical"].items()
            if info["is_binary"] is is_binary
        ]

    binary_fields = categorical_fields(True)
    multilevel_fields = categorical_fields(False)

    _FORM_GROUPS_CACHE = [
        {
            "key": "continuous",
            "title": f"Continuous features — {len(continuous_fields)} columns (cont1–cont14)",
            "description": (
                "Numeric measurements in their raw, unscaled form. The saved StandardScaler from the "
                "training pipeline converts them before the model sees them; the min/max shown are the "
                "ranges found in the raw dataset and the defaults are the dataset medians."
            ),
            "open": True,
            "fields": continuous_fields,
        },
        {
            "key": "binary",
            "title": f"Binary categorical features — {len(binary_fields)} columns",
            "description": (
                "Categorical columns that hold exactly two possible values in the raw dataset. The saved "
                "LabelEncoder maps the chosen value to the same integer used during training."
            ),
            "open": False,
            "fields": binary_fields,
        },
        {
            "key": "multilevel",
            "title": f"Multi-level categorical features — {len(multilevel_fields)} columns",
            "description": (
                "Categorical columns with three or more possible values (up to 266 for cat116). Every "
                "option in the lists below is a value that really occurs in the raw dataset."
            ),
            "open": False,
            "fields": multilevel_fields,
        },
    ]
    return _FORM_GROUPS_CACHE


def _read_output_csv(relative_path):
    """Read a CSV produced by the notebooks, or return None if it is missing."""

    path = OUTPUTS_DIR / relative_path
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return None


def _output_image(folder, filename, alt, caption=None):
    """Return an image reference only if the notebook actually wrote the file."""

    if not (OUTPUTS_DIR / folder / filename).exists():
        return None
    return {
        "src": f"{folder}/{filename}",
        "alt": alt,
        "caption": caption or alt,
    }


def _fmt(value, digits=4):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_academic_data():
    """Build the Academic ML page from the real CSV/PNG files of notebooks 11-17."""

    sections = []

    # ---------- 11. PCA -------------------------------------------------
    pca_table = _read_output_csv("PCA/pca_explained_variance_table.csv")
    pca_experiment = _read_output_csv("PCA/pca_modeling_experiment.csv")
    pca_cards, pca_tables = [], []
    if pca_table is not None and not pca_table.empty:
        last = pca_table.iloc[-1]
        at_20 = pca_table[pca_table["components"] == 20]
        pca_cards = [
            {
                "label": "Components fitted",
                "value": f"{int(last['components'])}",
                "note": "all features of the ClaimWise matrix",
            },
            {
                "label": "Cumulative variance at 20 components",
                "value": _fmt(float(at_20["cumulative_explained_variance"].iloc[0]) * 100, 1) + "%",
                "note": "roughly half of the information in 15% of the width",
            },
            {
                "label": "Cumulative variance at 130 components",
                "value": _fmt(float(last["cumulative_explained_variance"]) * 100, 1) + "%",
                "note": "the full component set spans the whole feature space",
            },
            {
                "label": "Components needed for 95% variance",
                "value": "89",
                "note": "a 31.5% reduction (notebook 11, executed run)",
            },
        ]
        pca_tables.append(
            {
                "title": "Cumulative explained variance (selected component counts)",
                "headers": ["Components", "Cumulative variance", "Remaining reduction"],
                "rows": [
                    [
                        f"{int(row['components'])}",
                        _fmt(float(row["cumulative_explained_variance"]) * 100, 1) + "%",
                        str(row["reduction_vs_130_features"]),
                    ]
                    for _, row in pca_table.iterrows()
                ],
            }
        )
    if pca_experiment is not None and not pca_experiment.empty:
        pca_tables.append(
            {
                "title": "Modelling experiment — original 130 features vs PCA space",
                "headers": ["Feature space", "Model", "MAE", "RMSE", "R²"],
                "rows": [
                    [
                        str(row["feature_space"]),
                        str(row["model"]),
                        _fmt(row["MAE"], 4),
                        _fmt(row["RMSE"], 4),
                        _fmt(row["R2"], 6),
                    ]
                    for _, row in pca_experiment.iterrows()
                ],
            }
        )
    sections.append(
        {
            "id": "pca",
            "label": "PCA",
            "title": "Principal Component Analysis",
            "notebook": "11_ClaimWise_PCA_Feature_Analysis.ipynb",
            "intro": (
                "PCA re-expresses the 130 ClaimWise features as orthogonal components ordered by the "
                "variance they explain. It produces components — it does not pick a subset of the "
                "original cat/cont columns."
            ),
            "cards": pca_cards,
            "tables": pca_tables,
            "images": [
                img
                for img in (
                    _output_image(
                        "PCA", "pca_cumulative_explained_variance.png",
                        "Cumulative explained variance of the principal components",
                    ),
                    _output_image(
                        "PCA", "pca_explained_variance_ratio.png",
                        "Explained variance ratio of each principal component",
                    ),
                    _output_image(
                        "PCA", "pca_pc1_top_loadings.png",
                        "Features with the largest loadings on the first principal component",
                    ),
                    _output_image(
                        "PCA", "pca_projection_2d.png",
                        "Records projected onto the first two principal components, coloured by loss",
                    ),
                    _output_image(
                        "PCA", "pca_modeling_experiment.png",
                        "Regression performance of the original feature space versus the PCA space",
                    ),
                )
                if img
            ],
            "interpretation": (
                "The first two components carry only part of the variance, so a 2D projection is used "
                "for visualisation, not for modelling. Keeping 95% of the variance needs 89 of 130 "
                "components, and the executed experiment shows the PCA space is worse than the "
                "original features for both models (Hist Gradient Boosting R² 0.509912 with PCA vs "
                "0.545901 without). The production regression model therefore keeps the original "
                "130-feature space."
            ),
        }
    )

    # ---------- 12/13. Classification -----------------------------------
    cls_summary = _read_output_csv("classification/classification_metrics_summary.csv")
    cls_cards, cls_tables = [], []
    if cls_summary is not None and not cls_summary.empty:
        cls_tables.append(
            {
                "title": "Test-set classification metrics (10,000 held-out rows)",
                "headers": ["Model", "Accuracy", "Precision", "Recall", "F1"],
                "rows": [
                    [
                        str(row["model"]),
                        _fmt(row["accuracy"]),
                        _fmt(row["precision"]),
                        _fmt(row["recall"]),
                        _fmt(row["f1"]),
                    ]
                    for _, row in cls_summary.iterrows()
                ],
            }
        )
        for _, row in cls_summary.iterrows():
            cls_cards.append(
                {
                    "label": f"{row['model']} — F1",
                    "value": _fmt(row["f1"]),
                    "note": f"accuracy {_fmt(row['accuracy'])}, "
                            f"precision {_fmt(row['precision'])}, recall {_fmt(row['recall'])}",
                }
            )
    cls_cards.append(
        {
            "label": "Derived target threshold",
            "value": "2118.86",
            "note": "median of the 40,000 training losses — positive class = high-loss claim",
        }
    )
    cls_cards.append(
        {
            "label": "Test split",
            "value": "5037 / 4963",
            "note": "typical-loss / high-loss rows in the 10,000 test records",
        }
    )
    sections.append(
        {
            "id": "classification",
            "label": "Classification",
            "title": "Classification — Logistic Regression & Decision Tree",
            "notebook": "12_ClaimWise_Logistic_Regression_Classification.ipynb",
            "intro": (
                "ClaimWise has no categorical target, so notebooks 12 and 13 derive one that is "
                "defensible and reproducible: loss at or above the median of the training rows is "
                "labelled 1 (high-loss claim). Both classifiers are evaluated with a confusion "
                "matrix, precision, recall, F1 and a ROC curve."
            ),
            "cards": cls_cards,
            "tables": cls_tables,
            "images": [
                img
                for img in (
                    _output_image(
                        "classification", "classification_class_distribution.png",
                        "Class distribution of the derived high-loss target",
                    ),
                    _output_image(
                        "classification", "logistic_regression_confusion_matrix.png",
                        "Confusion matrix of the logistic regression classifier",
                    ),
                    _output_image(
                        "classification", "decision_tree_confusion_matrix.png",
                        "Confusion matrix of the decision tree classifier",
                    ),
                    _output_image(
                        "classification", "logistic_regression_roc_curve.png",
                        "ROC curve of the logistic regression classifier",
                    ),
                    _output_image(
                        "classification", "classification_metric_comparison.png",
                        "Comparison of the classification metrics of both classifiers",
                    ),
                )
                if img
            ],
            "interpretation": (
                "Logistic Regression wins every metric on the held-out test rows, and it needs a "
                "leakage-free StandardScaler layer (fitted on the training rows only) because the "
                "integer category codes otherwise prevent the solver from converging — the executed "
                "notebook prints n_iter_ = 59 and reports no warnings. The Decision Tree classifier "
                "is kept because a single interpretable tree is a required academic comparison. "
                "These classification results are an extension; the regression system of notebooks "
                "04-10 is untouched."
            ),
        }
    )

    # ---------- 14. Clustering ------------------------------------------
    km_summary = _read_output_csv("clustering/kmeans_summary.csv")
    km_profiles = _read_output_csv("clustering/cluster_profiles.csv")
    km_silhouette = _read_output_csv("clustering/kmeans_silhouette_by_k.csv")
    km_cards, km_tables = [], []
    if km_summary is not None and not km_summary.empty:
        row = km_summary.iloc[0]
        km_cards = [
            {
                "label": "Selected number of clusters",
                "value": f"k = {int(row['selected_k'])}",
                "note": "highest silhouette across k = 2…8 (selection rule declared in advance)",
            },
            {
                "label": "Silhouette at selected k",
                "value": _fmt(row["silhouette_selected_k"]),
                "note": "computed on a 3,000-row sample of the k-search data",
            },
            {
                "label": "Rows clustered",
                "value": f"{int(row['rows_clustered']):,}",
                "note": "every ClaimWise record, using the 130 features only",
            },
            {
                "label": "Inertia of the final fit",
                "value": f"{row['inertia_full']:,.0f}",
                "note": "within-cluster sum of squares on the full data",
            },
        ]
    if km_profiles is not None and not km_profiles.empty:
        km_tables.append(
            {
                "title": "Cluster profiles (loss read back afterwards — it was never an input)",
                "headers": ["Cluster", "Records", "Share", "Mean loss", "Median loss", "Min", "Max"],
                "rows": [
                    [
                        f"{int(row['cluster'])}",
                        f"{int(row['count']):,}",
                        _fmt(row["share_%"], 2) + "%",
                        _fmt(row["mean"], 2),
                        _fmt(row["median"], 2),
                        _fmt(row["min"], 2),
                        _fmt(row["max"], 2),
                    ]
                    for _, row in km_profiles.iterrows()
                ],
            }
        )
    if km_silhouette is not None and not km_silhouette.empty:
        km_tables.append(
            {
                "title": "Silhouette score for every k that was tested",
                "headers": ["k", "Silhouette"],
                "rows": [
                    [f"{int(row['k'])}", _fmt(row["silhouette"])]
                    for _, row in km_silhouette.iterrows()
                ],
            }
        )
    sections.append(
        {
            "id": "clustering",
            "label": "Clustering",
            "title": "Clustering — K-Means (Unsupervised)",
            "notebook": "14_ClaimWise_Clustering.ipynb",
            "intro": (
                "K-Means groups the 50,000 records by similarity using the 130 features only. The "
                "target loss is dropped before fitting and read back afterwards purely to describe "
                "the clusters."
            ),
            "cards": km_cards,
            "tables": km_tables,
            "images": [
                img
                for img in (
                    _output_image(
                        "clustering", "kmeans_elbow_silhouette.png",
                        "Elbow curve and silhouette scores used to choose the number of clusters",
                    ),
                    _output_image(
                        "clustering", "kmeans_cluster_sizes.png",
                        "Number of records in each k-means cluster",
                    ),
                    _output_image(
                        "clustering", "kmeans_pca_visualization.png",
                        "K-means cluster labels drawn in a 2D PCA projection",
                    ),
                )
                if img
            ],
            "interpretation": (
                "The silhouette scores are low across every k that was tested (best 0.0836 at k = 2), "
                "which is the honest result for 130-dimensional data: the clusters exist numerically "
                "but are diffuse rather than well separated. The two clusters also show very similar "
                "claim amounts (mean loss 2982.89 vs 3074.16), so the partition does not separate "
                "cheap from expensive claims. The 2D picture is a PCA projection of the labels — the "
                "algorithm itself used all 130 features."
            ),
        }
    )

    # ---------- 15. DBSCAN ----------------------------------------------
    dbscan_summary = _read_output_csv("dbscan/dbscan_summary.csv")
    dbscan_cards, dbscan_tables = [], []
    if dbscan_summary is not None and not dbscan_summary.empty:
        primary = dbscan_summary.iloc[0]
        secondary = dbscan_summary.iloc[1] if len(dbscan_summary) > 1 else None
        dbscan_cards = [
            {
                "label": "Clusters (130 scaled features)",
                "value": f"{int(primary['clusters'])}",
                "note": f"eps = {_fmt(primary['eps'], 3)}, min_samples = {int(primary['min_samples'])}",
            },
            {
                "label": "Noise points (130 features)",
                "value": f"{int(primary['noise_points']):,}",
                "note": f"{_fmt(primary['noise_%'], 2)}% of the 5,000-row sample fits no cluster",
            },
            {
                "label": "Silhouette (130 features)",
                "value": _fmt(primary["silhouette"]),
                "note": "computed on the clustered sample, noise points included as a label",
            },
            {
                "label": "Clusters after PCA (20 components)",
                "value": f"{int(secondary['clusters'])}" if secondary is not None else "—",
                "note": f"eps = {_fmt(secondary['eps'], 3)}, "
                        f"{_fmt(secondary['noise_%'], 2)}% noise"
                        if secondary is not None
                        else "second experiment of the same notebook",
            },
        ]
        dbscan_tables.append(
            {
                "title": "DBSCAN results for both experiments",
                "headers": ["Representation", "Clusters", "Noise points", "Noise %", "eps", "min_samples", "Silhouette"],
                "rows": [
                    [
                        str(row["representation"]),
                        f"{int(row['clusters'])}",
                        f"{int(row['noise_points']):,}",
                        _fmt(row["noise_%"], 2) + "%",
                        _fmt(row["eps"], 4),
                        f"{int(row['min_samples'])}",
                        _fmt(row["silhouette"]),
                    ]
                    for _, row in dbscan_summary.iterrows()
                ],
            }
        )
    sections.append(
        {
            "id": "dbscan",
            "label": "DBSCAN",
            "title": "DBSCAN — Density-Based Clustering",
            "notebook": "15_ClaimWise_DBSCAN.ipynb",
            "intro": (
                "DBSCAN finds clusters by density and marks everything that fits no dense region as "
                "noise. eps is not guessed: it comes from the knee of the k-distance curve, computed "
                "by a fixed rule on a seeded 5,000-row sample (the target loss is again excluded)."
            ),
            "cards": dbscan_cards,
            "tables": dbscan_tables,
            "images": [
                img
                for img in (
                    _output_image(
                        "dbscan", "dbscan_k_distance_130d.png",
                        "k-distance curve used to select eps for the 130-feature run",
                    ),
                    _output_image(
                        "dbscan", "dbscan_k_distance_pca20.png",
                        "k-distance curve used to select eps for the PCA-20 run",
                    ),
                    _output_image(
                        "dbscan", "dbscan_cluster_distribution.png",
                        "Cluster sizes and the noise versus clustered split of DBSCAN",
                    ),
                    _output_image(
                        "dbscan", "dbscan_clusters_pca_visualization.png",
                        "DBSCAN clusters and noise points in a 2D PCA projection",
                    ),
                )
                if img
            ],
            "interpretation": (
                "Density-based clustering behaves differently from k-means: instead of forcing every "
                "record into a group, it isolates a small share of records as noise. Noise here means "
                "'unusual in the feature space' — ClaimWise carries no fraud label, so no claim about "
                "fraud can or is made. The PCA-20 run needs a much smaller eps because distances "
                "shrink once the highest-variance components are kept."
            ),
        }
    )

    # ---------- 16. Anomaly detection -----------------------------------
    anomaly_summary = _read_output_csv("anomaly/anomaly_summary.csv")
    anomaly_cards, anomaly_tables = [], []
    if anomaly_summary is not None and not anomaly_summary.empty:
        row = anomaly_summary.iloc[0]
        anomaly_cards = [
            {
                "label": "Records scored",
                "value": f"{int(row['records']):,}",
                "note": "the full ClaimWise feature matrix, target excluded",
            },
            {
                "label": "Flagged as anomalies",
                "value": f"{int(row['anomalies']):,}",
                "note": f"{_fmt(row['anomaly_%'], 2)}% of all records (contamination = 'auto')",
            },
            {
                "label": "Normal observations",
                "value": f"{int(row['normal']):,}",
                "note": f"{_fmt(100 - row['anomaly_%'], 2)}% of all records",
            },
            {
                "label": "Mean score — normal vs anomaly",
                "value": f"{_fmt(row['mean_score_normal'])} / {_fmt(row['mean_score_anomaly'])}",
                "note": "lower (more negative) scores are the flagged ones",
            },
        ]
        anomaly_tables.append(
            {
                "title": "Isolation Forest summary",
                "headers": ["Records", "Normal", "Anomalies", "Anomaly %", "Mean score (normal)", "Mean score (anomaly)"],
                "rows": [
                    [
                        f"{int(row['records']):,}",
                        f"{int(row['normal']):,}",
                        f"{int(row['anomalies']):,}",
                        _fmt(row["anomaly_%"], 2) + "%",
                        _fmt(row["mean_score_normal"]),
                        _fmt(row["mean_score_anomaly"]),
                    ]
                    for _, row in anomaly_summary.iterrows()
                ],
            }
        )
    sections.append(
        {
            "id": "anomaly",
            "label": "Anomaly Detection",
            "title": "Anomaly Detection — Isolation Forest",
            "notebook": "16_ClaimWise_Anomaly_Detection.ipynb",
            "intro": (
                "Isolation Forest isolates observations by random feature splits: records that are "
                "cut out quickly score as anomalies. The detector sees the 130 features only — loss "
                "is never an input."
            ),
            "cards": anomaly_cards,
            "tables": anomaly_tables,
            "images": [
                img
                for img in (
                    _output_image(
                        "anomaly", "anomaly_score_distribution.png",
                        "Distribution of isolation-forest scores with the decision threshold",
                    ),
                    _output_image(
                        "anomaly", "anomaly_pca_visualization.png",
                        "Flagged anomalies in a 2D PCA projection",
                    ),
                    _output_image(
                        "anomaly", "anomaly_loss_context.png",
                        "Loss of normal versus flagged records — context only, loss was not an input",
                    ),
                )
                if img
            ],
            "interpretation": (
                "An anomaly is an unusual observation in the feature space. It does not mean the "
                "claim is fraudulent: ClaimWise contains no verified fraud labels, so the notebook "
                "states this explicitly and so does this page. Read afterwards purely as context, "
                "the flagged records carry a higher claim amount than the normal ones (mean loss "
                "7397.88 vs 2886.96) — an analysis of the result, not evidence used to build it."
            ),
        }
    )

    # ---------- 17. 5-fold cross-validation ------------------------------
    reg_cv = _read_output_csv("CrossValidation/regression_5fold_summary.csv")
    cls_cv = _read_output_csv("CrossValidation/classification_5fold_summary.csv")
    cv_cards, cv_tables = [], []
    if reg_cv is not None and not reg_cv.empty:
        cv_tables.append(
            {
                "title": "Regression — 5-fold cross-validation (mean and standard deviation)",
                "headers": ["Model", "MAE mean (std)", "RMSE mean (std)", "R² mean (std)"],
                "rows": [
                    [
                        str(row["model"]),
                        f"{_fmt(row['MAE_mean'], 2)} ({_fmt(row['MAE_std'], 2)})",
                        f"{_fmt(row['RMSE_mean'], 2)} ({_fmt(row['RMSE_std'], 2)})",
                        f"{_fmt(row['R2_mean'], 6)} ({_fmt(row['R2_std'], 6)})",
                    ]
                    for _, row in reg_cv.iterrows()
                ],
            }
        )
        best = reg_cv.loc[reg_cv["R2_mean"].idxmax()]
        cv_cards.append(
            {
                "label": "Best regressor (mean R²)",
                "value": _fmt(best["R2_mean"], 6),
                "note": f"{best['model']} — std {_fmt(best['R2_std'], 6)} over the 5 folds",
            }
        )
    if cls_cv is not None and not cls_cv.empty:
        cv_tables.append(
            {
                "title": "Classification — 5-fold cross-validation (mean and standard deviation)",
                "headers": ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"],
                "rows": [
                    [
                        str(row["model"]),
                        f"{_fmt(row['accuracy_mean'])} ({_fmt(row['accuracy_std'])})",
                        f"{_fmt(row['precision_mean'])} ({_fmt(row['precision_std'])})",
                        f"{_fmt(row['recall_mean'])} ({_fmt(row['recall_std'])})",
                        f"{_fmt(row['f1_mean'])} ({_fmt(row['f1_std'])})",
                        f"{_fmt(row['roc_auc_mean'])} ({_fmt(row['roc_auc_std'])})",
                    ]
                    for _, row in cls_cv.iterrows()
                ],
            }
        )
        cv_cards.append(
            {
                "label": "Best classifier (mean F1)",
                "value": _fmt(cls_cv.loc[cls_cv["f1_mean"].idxmax(), "f1_mean"]),
                "note": "averaged over 5 folds with fold-local preprocessing",
            }
        )
    cv_cards.insert(
        0,
        {
            "label": "Cross-validation protocol",
            "value": "5 folds",
            "note": "KFold(5, shuffle=True, random_state=42), preprocessing fitted inside every fold",
        },
    )
    cv_cards.append(
        {
            "label": "Per-fold loss thresholds",
            "value": "2106.31 – 2122.90",
            "note": "training median of each fold — the validation rows never define their own label",
        },
    )
    sections.append(
        {
            "id": "cross-validation",
            "label": "5-Fold CV",
            "title": "Leakage-Free 5-Fold Cross-Validation",
            "notebook": "17_ClaimWise_5Fold_Cross_Validation.ipynb",
            "intro": (
                "One train/test split can be lucky. Notebook 17 evaluates all five regression models "
                "and both classifiers with 5 shuffled folds in which the encoding, scaling and even "
                "the classification threshold are learned from each fold's training rows only."
            ),
            "cards": cv_cards,
            "tables": cv_tables,
            "images": [
                img
                for img in (
                    _output_image(
                        "CrossValidation", "regression_5fold_scores.png",
                        "R² of every regression model on each of the five folds",
                    ),
                    _output_image(
                        "CrossValidation", "classification_5fold_scores.png",
                        "Accuracy and F1 of both classifiers on each of the five folds",
                    ),
                )
                if img
            ],
            "interpretation": (
                "The cross-validated scores agree with the single-split results (Hist Gradient "
                "Boosting remains the strongest regressor at R² 0.553189 ± 0.005585, Logistic "
                "Regression the strongest classifier at accuracy 0.7635 ± 0.0039), which means the "
                "production numbers were not an artefact of one favourable split. Because every "
                "fold refits its own preprocessor, none of these scores is inflated by leakage."
            ),
        }
    )

    available = [section for section in sections if section["cards"] or section["tables"]]
    return available


@app.route("/")
@app.route("/index")
def index():
    model_rows, best_name = load_model_results()
    best_metrics = model_rows[0] if model_rows else None
    return render_template(
        "index.html",
        stats=load_dataset_stats(),
        best_name=best_name,
        best_metrics=best_metrics,
        model_rows=model_rows,
    )

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/dataset", methods=["GET", "POST"])
def dataset():
    upload_error = None
    upload_success = None
    target_column = request.args.get("target_column") or request.form.get("target_column")

    if request.method == "POST" and "file" in request.files:
        file_storage = request.files.get("file")
        if not file_storage or not file_storage.filename:
            upload_error = "Please choose a valid CSV file to upload."
        else:
            saved_path, err = save_uploaded_file(file_storage)
            if err:
                upload_error = err
            else:
                upload_success = f"File '{file_storage.filename}' uploaded successfully into sandbox storage."

    current_upload = get_current_upload_path()
    dataset_csvs = [c for c in get_csv_inventory() if c["page"] == "dataset"]
    dataset_charts = [v for v in get_visualization_inventory() if v["page"] == "dataset"]

    if current_upload is not None:
        # Mode B: Uploaded dataset inspection & model compatibility verification
        df_upload, read_err = read_csv_safe(current_upload)
        if df_upload is None:
            audit = None
            compatibility = None
            upload_error = upload_error or read_err
        else:
            audit = audit_dataframe(
                df_upload,
                file_name=current_upload.name.replace("current_", ""),
                file_size_bytes=current_upload.stat().st_size,
                selected_target=target_column,
            )
            preprocessor = get_preprocessor()
            compatibility = verify_model_compatibility(df_upload, preprocessor)

        mode = "uploaded"
        return render_template(
            "dataset.html",
            mode=mode,
            audit=audit,
            compatibility=compatibility,
            upload_error=upload_error,
            upload_success=upload_success,
            stats=load_dataset_stats(),
            dataset_csvs=dataset_csvs,
            dataset_charts=dataset_charts,
        )
    else:
        # Mode A: Original ClaimWise dataset
        raw_df, _ = read_csv_safe(DATASET_DIR / "01_raw_claimwise_50000.csv")
        audit = (
            audit_dataframe(
                raw_df,
                file_name="01_raw_claimwise_50000.csv",
                file_size_bytes=(DATASET_DIR / "01_raw_claimwise_50000.csv").stat().st_size
                if (DATASET_DIR / "01_raw_claimwise_50000.csv").exists()
                else 0,
                selected_target="loss",
            )
            if raw_df is not None
            else None
        )

        mode = "original"
        return render_template(
            "dataset.html",
            mode=mode,
            audit=audit,
            stats=load_dataset_stats(),
            upload_error=upload_error,
            upload_success=upload_success,
            dataset_csvs=dataset_csvs,
            dataset_charts=dataset_charts,
            target_distribution={
                "src": "Data_Inspection/claimwise_raw_target_distribution.png",
                "alt": "Distribution of Claim Loss (raw dataset)",
                "caption": "Distribution of Claim Loss (raw dataset) — generated by 01_ClaimWise_Data_Inspection.ipynb",
            },
            feature_composition_chart={
                "src": "Data_Inspection/claimwise_feature_types_composition.png",
                "alt": "ClaimWise Feature Types & Schema Composition",
                "caption": "ClaimWise Feature Schema Composition (116 categorical, 14 continuous, 1 target, 1 ID).",
            },
        )


@app.route("/dataset/reset", methods=["GET", "POST"])
def dataset_reset():
    clear_uploaded_files()
    flash("Switched back to the original ClaimWise dataset.")
    return redirect(url_for("dataset"))


@app.route("/preprocessing")
def preprocessing():
    preprocessing_csvs = [c for c in get_csv_inventory() if c["page"] == "preprocessing"]
    preprocessing_charts = [v for v in get_visualization_inventory() if v["page"] == "preprocessing"]
    return render_template(
        "preprocessing.html",
        stats=load_dataset_stats(),
        stage_rows=load_stage_rows(),
        leakage=load_leakage_stats(),
        preprocessing_csvs=preprocessing_csvs,
        preprocessing_charts=preprocessing_charts,
    )


@app.route("/visualization")
def visualization():
    modules = load_visualization_modules()
    eda_csvs = [c for c in get_csv_inventory() if c["page"] == "visualization"]
    eda_charts = [v for v in get_visualization_inventory() if v["page"] == "visualization"]
    return render_template(
        "visualization.html",
        modules=modules,
        eda_csvs=eda_csvs,
        eda_charts=eda_charts,
    )


@app.route("/models")
def models():
    model_rows, best_name = load_model_results()
    core_notebook_map = {
        "Hist Gradient Boosting Regressor": ("08", "08_hist_gradient_boosting.ipynb"),
        "Random Forest Regressor": ("06", "06_random_forest.ipynb"),
        "Gradient Boosting Regressor": ("07", "07_gradient_boosting.ipynb"),
        "Linear Regression": ("04", "04_linear_regression.ipynb"),
        "Decision Tree Regressor": ("05", "05_decision_tree.ipynb"),
    }
    core_models = []
    for r in model_rows:
        name = str(r.get("Model"))
        if name in core_notebook_map:
            row_copy = dict(r)
            row_copy["notebook_num"], row_copy["notebook_file"] = core_notebook_map[name]
            core_models.append(row_copy)

    gallery, _ = load_model_gallery()
    leakage = load_leakage_stats()
    kpi_cards = load_model_kpis()
    model_csvs = [c for c in get_csv_inventory() if c["page"] == "models"]
    model_charts = [v for v in get_visualization_inventory() if v["page"] == "models"]
    return render_template(
        "models.html",
        model_rows=model_rows,
        core_models=core_models,
        best_name=best_name,
        model_ready=bool(model_rows),
        gallery=gallery,
        kpi_cards=kpi_cards,
        comparison_chart=load_comparison_chart(),
        leakage=leakage,
        notebooks=load_notebooks(),
        model_csvs=model_csvs,
        model_charts=model_charts,
    )


@app.route("/academic")
def academic():
    """Academic ML extension: PCA, classification, clustering, DBSCAN, anomalies, 5-fold CV."""
    sections = load_academic_data()
    academic_csvs = [c for c in get_csv_inventory() if c["page"] == "academic"]
    academic_charts = [v for v in get_visualization_inventory() if v["page"] == "academic"]
    return render_template(
        "academic.html",
        sections=sections,
        academic_csvs=academic_csvs,
        academic_charts=academic_charts,
        notebooks=[nb for nb in load_notebooks() if nb["number"] in {"11", "12", "13", "14", "15", "16", "17"}],
    )


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    """Single Primary Prediction Workflow for ClaimWise Insurance Loss.

    Accepts continuous numeric features (cont1-cont14) and categorical attributes,
    applies the authoritative fitted preprocessor (LabelEncoder + StandardScaler),
    and executes inference using the production HistGradientBoosting Regressor.
    """
    model_rows, best_name = load_model_results()
    result = None
    manual = None
    error = None
    warnings = []
    x_test, y_test = load_prediction_data()

    try:
        form_groups = build_form_groups()
    except Exception as exc:  # pragma: no cover
        form_groups = []
        if request.method == "GET":
            error = f"Prediction form could not be built: {exc}"

    if request.method == "POST":
        mode = request.form.get("mode", "manual")
        if not model_is_ready():
            error = "Train the regression models first. The saved best model is unavailable."
        elif mode == "test":
            # Retain backwards compatibility for automated demo test-row evaluations
            if x_test is None or y_test is None:
                error = "The prepared test data is unavailable."
            else:
                try:
                    row_index = int(request.form.get("row_index", ""))
                    if row_index < 0 or row_index >= len(x_test):
                        raise ValueError(f"Choose a test-row number from 0 to {len(x_test) - 1}.")

                    model = joblib.load(MODELS_DIR / "best_model.pkl")
                    row = x_test.iloc[[row_index]]
                    predicted_loss = float(model.predict(row)[0])
                    result = {
                        "row_index": row_index,
                        "predicted_loss": predicted_loss,
                        "actual_loss": float(y_test.iloc[row_index]),
                        "feature_count": row.shape[1],
                        "model": best_name,
                        "model_name_exact": "HistGradientBoosting Regressor",
                        "status": "success",
                    }
                except (TypeError, ValueError) as exc:
                    error = str(exc)
                except Exception as exc:  # pragma: no cover
                    error = f"Prediction could not be completed: {exc}"
        else:
            try:
                preprocessor = get_preprocessor()

                # Build row mapping from form inputs or training defaults
                row_dict = {}
                missing_continuous = []

                # 1. Continuous features (cont1-cont14) MUST be provided and finite
                for col in preprocessor["continuous_columns"]:
                    raw_val = request.form.get(col, "")
                    if str(raw_val).strip() == "":
                        missing_continuous.append(col)
                    else:
                        try:
                            num_val = float(raw_val)
                            if not (-1e12 < num_val < 1e12):
                                raise ValueError(f"Value for {col} is out of numeric limits.")
                            row_dict[col] = num_val
                        except (ValueError, TypeError):
                            raise ValueError(f"Field '{col}' must be a valid finite number.")

                if missing_continuous:
                    raise ValueError(
                        f"{len(missing_continuous)} feature values are empty (first: {missing_continuous[0]}). "
                        "Every field of the form is needed."
                    )

                # 2. Categorical features: Use user input if provided, else use training default mode
                entered_count = len(preprocessor["continuous_columns"])
                defaulted_count = 0
                for col in preprocessor["categorical_columns"]:
                    val = request.form.get(col, "")
                    if str(val).strip() != "":
                        row_dict[col] = str(val).strip()
                        entered_count += 1
                    else:
                        row_dict[col] = str(preprocessor["categorical"][col]["default"])
                        defaulted_count += 1

                raw_row = pd.DataFrame([row_dict])
                features = transform_raw_row(raw_row, preprocessor)

                model = joblib.load(MODELS_DIR / "best_model.pkl")
                predicted_loss = float(model.predict(features)[0])
                warnings = find_out_of_range(raw_row, preprocessor)

                continuous_summary = []
                for col in preprocessor["continuous_columns"]:
                    val_num = float(raw_row[col].iloc[0])
                    stats = preprocessor["continuous"][col]
                    continuous_summary.append({
                        "name": col,
                        "value": val_num,
                        "median": stats["median"],
                        "min": stats["min"],
                        "max": stats["max"],
                        "in_range": (stats["min"] <= val_num <= stats["max"]),
                    })

                categorical_summary = []
                for col in preprocessor["categorical_columns"]:
                    val_str = raw_row[col].iloc[0]
                    info = preprocessor["categorical"][col]
                    categorical_summary.append({
                        "name": col,
                        "value": val_str,
                        "default": info["default"],
                        "cardinality": info["cardinality"],
                        "is_binary": info["is_binary"],
                    })

                timestamp_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

                manual = {
                    "predicted_loss": predicted_loss,
                    "feature_count": int(features.shape[1]),
                    "model": best_name,
                    "model_name_exact": "HistGradientBoosting Regressor",
                    "model_class": type(model).__name__,
                    "timestamp": timestamp_now,
                    "warnings": warnings,
                    "status": "success",
                    "changed": sum(
                        1
                        for col in preprocessor["continuous"]
                        if abs(float(raw_row[col].iloc[0]) - float(preprocessor["continuous"][col]["median"])) > 1e-6
                    ),
                    "entered_count": entered_count,
                    "defaulted_count": defaulted_count,
                    "continuous_summary": continuous_summary,
                    "categorical_summary": categorical_summary,
                }
                if y_test is not None:
                    manual["test_mean_loss"] = float(y_test.mean())
            except (TypeError, ValueError) as exc:
                error = str(exc)
            except Exception as exc:  # pragma: no cover
                error = f"Prediction could not be completed: {exc}"

    # Retain submitted values so "Edit Information" can pre-populate form fields
    submitted_values = None
    if request.method == "POST":
        submitted_values = {
            col: request.form.get(col, "")
            for col in request.form
            if col != "mode"
        }

    upload_path = get_current_upload_path()
    upload_info = None
    if upload_path is not None:
        upload_info = {
            "active": True,
            "filename": upload_path.name.replace("current_", ""),
        }

    feature_count = int(x_test.shape[1]) if x_test is not None else 0
    return render_template(
        "prediction.html",
        model_rows=model_rows,
        best_name=best_name,
        model_ready=model_is_ready(),
        test_row_count=len(x_test) if x_test is not None else 0,
        feature_count=feature_count,
        form_groups=form_groups,
        manual=manual,
        result=result,
        error=error,
        upload_info=upload_info,
        submitted_values=submitted_values,
    )


@app.route("/dashboard")
def dashboard():
    model_rows, best_name = load_model_results()
    best_metrics = model_rows[0] if model_rows else None
    kpi_cards = load_model_kpis()
    all_csvs = get_csv_inventory()
    recent_csvs = [c for c in all_csvs if c["page"] in ["models", "dataset", "preprocessing"]][:8]
    return render_template(
        "dashboard.html",
        model_rows=model_rows,
        best_name=best_name,
        best_metrics=best_metrics,
        kpi_cards=kpi_cards,
        recent_csvs=recent_csvs,
        comparison_chart=load_comparison_chart(),
        stats=load_dataset_stats(),
        model_ready=bool(model_rows),
    )


@app.route("/reports")
def reports():
    model_rows, best_name = load_model_results()
    best_metrics = model_rows[0] if model_rows else None
    all_csvs = get_csv_inventory()
    all_charts = get_visualization_inventory()
    stats = load_dataset_stats()
    kpi_cards = load_model_kpis()
    return render_template(
        "reports.html",
        model_rows=model_rows,
        best_name=best_name,
        best_metrics=best_metrics,
        all_csvs=all_csvs,
        all_charts=all_charts,
        stats=stats,
        kpi_cards=kpi_cards,
        model_ready=bool(model_rows),
    )


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/csv/preview")
def csv_preview():
    """Secure endpoint returning paginated, searchable rows for CSV modal/preview."""
    path = request.args.get("path", "").strip()
    page = request.args.get("page", 1, type=int)
    page_size = request.args.get("page_size", 20, type=int)
    search = request.args.get("search", "").strip()
    preview_data = get_csv_preview(path, page=page, page_size=page_size, search_query=search)
    return jsonify(preview_data)


@app.route("/csv/download/<path:filepath>")
def csv_download(filepath):
    """Secure CSV download endpoint strictly validating path traversal against inventory."""
    safe_path = resolve_safe_csv_path(filepath)
    if safe_path is None or not safe_path.exists():
        flash("Requested file does not exist or access is restricted.")
        return redirect(url_for("reports"))
    return send_from_directory(safe_path.parent, safe_path.name, as_attachment=True)


@app.route("/outputs-gallery", endpoint="outputs_gallery")
@app.route("/gallery")
def outputs_gallery():
    """Display the vertical scrolling Project Outputs gallery discovering all files recursively."""
    summary = get_outputs_summary()
    items = get_outputs_inventory(use_cache=True)
    selected_topic = request.args.get("topic", "all").strip().lower()
    search_query = request.args.get("search", "").strip().lower()

    if selected_topic and selected_topic != "all":
        items = [x for x in items if x["topic_key"] == selected_topic]

    if search_query:
        items = [
            x
            for x in items
            if search_query in x["filename"].lower()
            or search_query in x["topic"].lower()
            or search_query in x.get("description", "").lower()
        ]

    return render_template(
        "outputs_gallery.html",
        active_page="outputs_gallery",
        summary=summary,
        items=items,
        selected_topic=selected_topic,
        search_query=search_query,
    )


@app.route("/outputs/<path:filename>", endpoint="outputs")
def output_file(filename):
    """Serve charts and figures generated by the ML pipeline with strict path traversal validation."""
    safe_target = resolve_safe_output_path(filename)
    if safe_target is None or not safe_target.exists():
        return "Requested output file not found or access restricted.", 404
    return send_from_directory(safe_target.parent, safe_target.name)



@app.route("/notebooks/<path:filename>")
def notebook_file(filename):
    """Download the real Jupyter notebook used for this stage of the project."""
    return send_from_directory(NOTEBOOKS_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    app.run(debug=True, port=port)


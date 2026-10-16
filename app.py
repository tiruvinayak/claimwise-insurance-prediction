import logging
import os
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List

import joblib
import pandas as pd
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, send_from_directory, url_for

app = Flask(__name__)

# Secrets and environment-driven configuration (never hard-code keys in production).
app.secret_key = os.environ.get("CLAIMWISE_SECRET_KEY", "claimwise-dev-only-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload ceiling
FLASK_DEBUG = os.environ.get("CLAIMWISE_DEBUG", "0").lower() in ("1", "true", "yes")
logging.basicConfig(
    level=logging.DEBUG if FLASK_DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

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
from dataset_explorer import (  # noqa: E402
    ALLOWED_PAGE_SIZES,
    DEFAULT_PAGE_SIZE,
    column_statistics,
    explorer_meta,
    export_filtered_csv,
    load_explorer_frame,
    query_records,
    validate_dataset,
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
# Metrics come from the validation leaderboard (Outputs/Model_Comparison/
# model_comparison.csv); the champion also carries a single untouched-test
# evaluation recorded in Outputs/ExperimentSuite/final_test_metrics.csv.
MODEL_GALLERY = (
    {
        "name": "CatBoost Regressor",
        "slug": "catboost",
        "folder": "Final_Model_Evaluation",
        "feature_note": (
            "CatBoost utilizes oblivious (symmetric) decision trees that partition the 130-feature space evenly to avoid overfitting. "
            "Trained under a log1p target transformation to tame extreme positive claim loss skewness, predictions are automatically "
            "inverted via expm1 to produce calibrated dollar claim severity values."
        ),
        "summary": "Symmetric gradient boosted decision trees with log1p target transformation — ClaimWise production champion.",
        "interpretation": (
            "Rank #1 lowest validation MAE ($1,155.67, RMSE $1,940.66, R² 0.5276) across all 15 evaluated machine learning families. "
            "Refitted on 40,000 train+val rows, it reaches holdout MAE $1,154.95, RMSE $2,001.08, and R² 0.5396 on the untouched "
            "10,000-row holdout (seed 2026)."
        ),
        "images": (
            ("final_actual_vs_predicted.png", "CatBoost Regressor (Log1p) — Actual vs Predicted Loss (10,000 holdout claims)"),
            ("final_residuals_vs_predicted.png", "CatBoost Regressor (Log1p) — Residual Analysis vs Predicted Values"),
            ("final_residuals_and_errors.png", "CatBoost Regressor (Log1p) — Residual Error Distribution and Skewness"),
        ),
    },
    {
        "name": "XGBoost Regressor",
        "slug": "xgboost",
        "folder": "Model_Comparison",
        "feature_note": (
            "Extreme Gradient Boosting with exact greedy split finding, column subsampling, and L2 leaf regularization "
            "(tuned n_estimators=1500, learning_rate=0.02, max_depth=8, min_child_weight=15, reg_lambda=1.0)."
        ),
        "summary": "Scalable tree boosting with second-order gradient loss approximation and L2 regularization.",
        "interpretation": (
            "Ranks #2 by validation MAE ($1,199.52, RMSE $1,903.68, R² 0.5454) on the wave-3 validation split, "
            "closely competitive with CatBoost and HistGradientBoosting."
        ),
        "images": (
            ("actual_vs_predicted.png", "Multi-Model Actual vs Predicted Comparison"),
            ("residuals.png", "Multi-Model Residual Distribution Comparison"),
        ),
    },
    {
        "name": "Hist Gradient Boosting Regressor",
        "slug": "hist-gradient-boosting",
        "folder": "Hist_Gradient_Boosting_Regressor",
        "feature_note": (
            "Histogram-based gradient boosting discretizes continuous features into 256 bins for fast split "
            "evaluation. The tuned winner uses lr=0.03, up to 1,500 stages with early stopping, 31 leaves, "
            "min_samples_leaf=60 and L2=0.5. Full search log: Outputs/ExperimentSuite/tuning_results.csv."
        ),
        "summary": "Histogram-based gradient boosting — strong baseline family in the wave-3 search.",
        "interpretation": (
            "A consistent top-tier family across the wave-3 validation split (seed 2026). "
            "Its best configuration and live metrics are shown in the leaderboard table above, "
            "generated directly from Outputs/Model_Comparison/model_comparison.csv."
        ),
        "images": (
            ("hist_gradient_boosting_regressor_actual_vs_predicted.png", "Hist Gradient Boosting Regressor — Actual vs Predicted (validation)"),
            ("hist_gradient_boosting_regressor_residuals.png", "Hist Gradient Boosting Regressor — Residual Analysis (validation)"),
        ),
    },
    {
        "name": "Extra Trees Regressor",
        "slug": "extra-trees",
        "folder": "Extra_Trees_Regressor",
        "feature_note": (
            "Extremely randomized trees draw random cut-points for each candidate feature at every node split, "
            "reducing variance on continuous and encoded inputs. Feature importances are stored in feature_importance.csv."
        ),
        "summary": "Ensemble of 200 extremely randomized trees (max_depth=16, min_samples_leaf=2, max_features=0.5).",
        "interpretation": (
            "Ranks 5th by validation MAE (1,282.74; RMSE 2,094.98, R² 0.5040), slightly behind Random Forest "
            "on this dataset despite the extra randomization."
        ),
        "images": (
            ("extra_trees_regressor_actual_vs_predicted.png", "Extra Trees Regressor — Actual vs Predicted (validation)"),
            ("extra_trees_regressor_residuals.png", "Extra Trees Regressor — Residual Analysis (validation)"),
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
        "summary": "Bootstrap-aggregated ensemble of 200 trees (max_depth=16, min_samples_leaf=2, max_features=0.5).",
        "interpretation": (
            "Ranks 3rd by validation MAE (1,279.26; RMSE 2,042.45, R² 0.5286), confirming bagged trees remain "
            "competitive with boosting on this feature set."
        ),
        "images": (
            ("random_forest_actual_vs_predicted.png", "Random Forest Regressor — Actual vs Predicted (validation)"),
            ("random_forest_residuals.png", "Random Forest Regressor — Residual Analysis (validation)"),
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
        "summary": "400 sequential boosting stages (lr=0.08, max_depth=3, min_samples_leaf=3).",
        "interpretation": (
            "Close runner-up: rank #2 by validation MAE (1,233.38) and the best validation R² of all 13 models "
            "(0.5719; RMSE 1,946.26). Loses the MAE-based selection by only 2.16 MAE points to HistGB."
        ),
        "images": (
            ("gradient_boosting_actual_vs_predicted.png", "Gradient Boosting Regressor — Actual vs Predicted (validation)"),
            ("gradient_boosting_residuals.png", "Gradient Boosting Regressor — Residual Analysis (validation)"),
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
        "summary": "Linear regression with L2 weight regularization (tuned alpha=10).",
        "interpretation": (
            "Best linear-family model at validation RMSE 2,174.26 and R² 0.4658, essentially tied with OLS — "
            "the 130 features are already regularized enough that L2 adds little."
        ),
        "images": (
            ("ridge_regression_actual_vs_predicted.png", "Ridge Regression — Actual vs Predicted (validation)"),
            ("ridge_regression_residuals.png", "Ridge Regression — Residual Analysis (validation)"),
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
            "Explains about 46.5% of validation variance (RMSE 2,175.02, R² 0.4654). The linear constraint caps "
            "performance because insurance claim loss has strong non-linear interactions."
        ),
        "images": (
            ("linear_regression_actual_vs_predicted.png", "Linear Regression — Actual vs Predicted (validation)"),
            ("linear_regression_residuals.png", "Linear Regression — Residual Analysis (validation)"),
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
        "summary": "Linear regression with L1 norm penalty (tuned alpha=0.5).",
        "interpretation": (
            "Validation RMSE 2,174.24 and R² 0.4658 — matches Ridge while shrinking weak coefficients toward zero."
        ),
        "images": (
            ("lasso_regression_actual_vs_predicted.png", "Lasso Regression — Actual vs Predicted (validation)"),
            ("lasso_regression_residuals.png", "Lasso Regression — Residual Analysis (validation)"),
            ("top_coefficients.png", "Lasso Regression — Top 20 Features by Absolute Coefficient"),
        ),
    },
    {
        "name": "Elastic Net Regression",
        "slug": "elastic-net",
        "folder": "Elastic_Net_Regression",
        "feature_note": (
            "Combines L1 and L2 penalties (tuned alpha=0.1, l1_ratio=0.5) to balance sparsity and grouping of correlated variables."
        ),
        "summary": "Regularized linear regression combining L1 (Lasso) and L2 (Ridge) penalties.",
        "interpretation": (
            "Validation RMSE 2,204.38 and R² 0.4509 — trading a little peak accuracy for stronger shrinkage."
        ),
        "images": (
            ("elastic_net_regression_actual_vs_predicted.png", "Elastic Net Regression — Actual vs Predicted (validation)"),
            ("elastic_net_regression_residuals.png", "Elastic Net Regression — Residual Analysis (validation)"),
            ("top_coefficients.png", "Elastic Net Regression — Top 20 Features by Absolute Coefficient"),
        ),
    },
    {
        "name": "Support Vector Regression (Linear)",
        "slug": "support-vector-regression",
        "folder": "Support_Vector_Regression_Linear",
        "feature_note": (
            "Linear SVR fits a margin-based hyperplane with an epsilon-insensitive loss tube. Tuned C=10 with a "
            "wide ε=100 tube that ignores errors below the tube width."
        ),
        "summary": "Linear support vector machine optimizing epsilon-insensitive loss (tuned C=10, ε=100).",
        "interpretation": (
            "Validation RMSE 2,272.49 and R² 0.4164 — the wide ε-tube shrinks residual sensitivity but caps "
            "overall explained variance."
        ),
        "images": (
            ("support_vector_regression_linear_actual_vs_predicted.png", "Linear SVR — Actual vs Predicted (validation)"),
            ("support_vector_regression_linear_residuals.png", "Linear SVR — Residual Analysis (validation)"),
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
        "summary": "A single CART decision tree (tuned max_depth=12) recursively partitioning the 130-feature space.",
        "interpretation": (
            "Predicts piece-wise constant values with high variance: validation RMSE 2,326.44 and R² 0.3884 — "
            "a clear illustration of why single trees lose to ensembles."
        ),
        "images": (
            ("decision_tree_actual_vs_predicted.png", "Decision Tree Regressor — Actual vs Predicted (validation)"),
            ("decision_tree_residuals.png", "Decision Tree Regressor — Residual Analysis (validation)"),
            ("decision_tree_feature_importance.png", "Decision Tree Regressor — Top 20 Feature Importances"),
        ),
    },
    {
        "name": "K-Nearest Neighbors Regressor",
        "slug": "k-nearest-neighbors",
        "folder": "K_Nearest_Neighbors_Regressor",
        "feature_note": (
            "Non-parametric local regression predicting the distance-weighted mean loss of the k=15 nearest "
            "neighbors in 130-dimensional space."
        ),
        "summary": "Instance-based regression using Euclidean distance across 15 nearest neighbors.",
        "interpretation": (
            "Suffers from the curse of dimensionality in 130D space (validation RMSE 2,785.95, R² 0.1229), "
            "confirming that tree ensembles handle high dimensionality far better."
        ),
        "images": (
            ("k_nearest_neighbors_regressor_actual_vs_predicted.png", "KNN Regressor — Actual vs Predicted (validation)"),
            ("k_nearest_neighbors_regressor_residuals.png", "KNN Regressor — Residual Analysis (validation)"),
        ),
    },
    {
        "name": "Dummy Regressor (Median Baseline)",
        "slug": "dummy-regressor",
        "folder": "Dummy_Regressor_Median_Baseline",
        "feature_note": (
            "Zero-feature constant predictor: always predicts the training-set median claim loss without reading any features."
        ),
        "summary": "Non-learning baseline model predicting the training set median.",
        "interpretation": (
            "Sanity check: validation RMSE 3,119.07, MAE 1,827.59, R² -0.0994. Every machine-learning candidate "
            "beats it decisively, confirming the features carry real signal."
        ),
        "images": (
            ("dummy_regressor_median_baseline_actual_vs_predicted.png", "Dummy Regressor — Actual vs Predicted (validation)"),
            ("dummy_regressor_median_baseline_residuals.png", "Dummy Regressor — Residual Analysis (validation)"),
        ),
    },
    {
        "name": "AdaBoost Regressor",
        "slug": "adaboost",
        "folder": "AdaBoost_Regressor",
        "feature_note": (
            "AdaBoost iteratively adjusts instance weights based on prediction errors on the previous estimator "
            "(tuned n_estimators=100, learning_rate=0.1, loss='square')."
        ),
        "summary": "Adaptive boosting ensemble using decision stump base estimators.",
        "interpretation": (
            "Heavy-tailed claim losses destabilize AdaBoost's stage reweighting even with square loss "
            "(validation RMSE 2,424.10, R² 0.3360), demonstrating why squared-loss gradient boosting is necessary here."
        ),
        "images": (
            ("adaboost_regressor_actual_vs_predicted.png", "AdaBoost Regressor — Actual vs Predicted (validation)"),
            ("adaboost_regressor_residuals.png", "AdaBoost Regressor — Residual Analysis (validation)"),
            ("feature_importance.png", "AdaBoost Regressor — Top 20 Feature Importances"),
        ),
    },
)

MODEL_METADATA = {
    "CatBoost Regressor": {
        "model_type": "Gradient Boosted Oblivious Decision Trees with Log1p Target Transformation",
        "family": "Boosting Ensemble",
        "transformation": "Log1p",
        "algorithm": "CatBoost regressor with oblivious (symmetric) decision trees (iterations=2000, lr=0.04, depth=6) optimizing RMSE on log1p(claim_loss), inverted via expm1.",
        "cv_summary": "Wave-3 validation split (7,500 rows, seed 2026): lowest validation MAE ($1,155.67). Refitted on 40,000 train+val rows and evaluated on the untouched 10,000 holdout (MAE $1,154.95, R² 0.5396).",
    },
    "XGBoost Regressor": {
        "model_type": "Extreme Gradient Boosted Decision Trees (XGBoost) with Log1p Target Transformation",
        "family": "Boosting Ensemble",
        "transformation": "Log1p",
        "algorithm": "XGBoost regressor trained on log1p(claim_loss) via TransformedTargetRegressor (n_estimators=800, lr=0.05, max_depth=6, min_child_weight=5, reg_lambda=1.0); predictions inverted with expm1.",
        "cv_summary": "Wave-3 validation split (7,500 rows, seed 2026): MAE $1,160.44, RMSE $1,949.97, R² 0.5230.",
    },
    "Hist Gradient Boosting Regressor": {
        "model_type": "Histogram-based Gradient Boosted Trees with Early Stopping and Log1p Target Transformation",
        "family": "Boosting Ensemble",
        "transformation": "Log1p",
        "algorithm": "Gradient boosting over 256-bin feature histograms trained on log1p(claim_loss) via TransformedTargetRegressor (learning_rate=0.03, max_iter=1500, max_leaf_nodes=31, min_samples_leaf=60, l2_regularization=0.5); predictions inverted with expm1.",
        "cv_summary": "Wave-3 validation split (7,500 rows, seed 2026): MAE $1,163.21, RMSE $1,957.43, R² 0.5194.",
    },
    "Extra Trees Regressor": {
        "model_type": "Extremely Randomized Trees Ensemble",
        "family": "Bagging Ensemble",
        "algorithm": "Ensemble of 200 randomized decision trees drawing random thresholds for each candidate split (max_depth=16, min_samples_leaf=2, max_features=0.5).",
        "cv_summary": "Wave-3 validation split (7,500 rows, seed 2026). Live values come from Outputs/Model_Comparison/model_comparison.csv."
    },
    "Random Forest Regressor": {
        "model_type": "Bootstrap Aggregated (Bagging) Ensemble of Decision Trees",
        "family": "Bagging Ensemble",
        "algorithm": "Ensemble of 200 trees fitted on bootstrap samples with random feature subsampling (max_depth=16, min_samples_leaf=2, max_features=0.5).",
        "cv_summary": "Wave-3 validation split (7,500 rows, seed 2026). Live values come from Outputs/Model_Comparison/model_comparison.csv."
    },
    "Gradient Boosting Regressor": {
        "model_type": "Sequential Stagewise Gradient Boosted Decision Trees",
        "family": "Boosting Ensemble",
        "algorithm": "400 sequential boosting stages adding shallow regression trees (max_depth=3, min_samples_leaf=3) with shrinkage learning_rate=0.08.",
        "cv_summary": "Wave-3 validation split (7,500 rows, seed 2026). Live values come from Outputs/Model_Comparison/model_comparison.csv."
    },
    "Ridge Regression": {
        "model_type": "L2-Regularized Linear Regression (Tikhonov)",
        "family": "Linear Model",
        "algorithm": "Ridge regression imposing an L2 penalty on beta coefficients; tuned alpha=10 from a 6-point grid.",
        "cv_summary": "Wave-3 validation split (7,500 rows, seed 2026). Live values come from Outputs/Model_Comparison/model_comparison.csv."
    },
    "Linear Regression": {
        "model_type": "Ordinary Least Squares (OLS) Linear Regression",
        "family": "Linear Model",
        "algorithm": "Closed-form analytical linear regression baseline estimating 130 beta coefficients to minimize squared residuals.",
        "cv_summary": "Wave-3 validation split (7,500 rows, seed 2026). Live values come from Outputs/Model_Comparison/model_comparison.csv."
    },
    "Lasso Regression": {
        "model_type": "L1-Regularized Linear Regression (Lasso)",
        "family": "Linear Model",
        "algorithm": "L1-regularized regression promoting sparse coefficients; tuned alpha=0.5 from a 5-point grid.",
        "cv_summary": "Wave-3 validation split (7,500 rows, seed 2026). Live values come from Outputs/Model_Comparison/model_comparison.csv."
    },
    "Elastic Net Regression": {
        "model_type": "L1 + L2 Regularized Linear Regression",
        "family": "Linear Model",
        "algorithm": "Linearly combined L1/L2 penalty; tuned alpha=0.1 with l1_ratio=0.5 from a 5-point grid.",
        "cv_summary": "Wave-3 validation split (7,500 rows, seed 2026). Live values come from Outputs/Model_Comparison/model_comparison.csv."
    },
    "Support Vector Regression (Linear)": {
        "model_type": "Linear Support Vector Regression (Linear SVR)",
        "family": "Margin-Based",
        "algorithm": "Epsilon-insensitive linear SVR; tuned C=10 with ε=100 from a 4-point grid. Kernel SVR was excluded as computationally impractical on 32,500×130 inputs.",
        "cv_summary": "Wave-3 validation split (7,500 rows, seed 2026). Live values come from Outputs/Model_Comparison/model_comparison.csv."
    },
    "Decision Tree Regressor": {
        "model_type": "Single Recursive Partitioning Decision Tree",
        "family": "Decision Tree",
        "algorithm": "A single CART regression tree; tuned max_depth=12 from a 4-point grid.",
        "cv_summary": "Wave-3 validation split (7,500 rows, seed 2026). Live values come from Outputs/Model_Comparison/model_comparison.csv."
    },
    "K-Nearest Neighbors Regressor": {
        "model_type": "K-Nearest Neighbors Regression (k=15, distance-weighted)",
        "family": "Instance-Based",
        "algorithm": "Distance-weighted average of the 15 nearest points in 130-dimensional Euclidean space; tuned over 4 k/weight configurations.",
        "cv_summary": "Wave-3 validation split (7,500 rows, seed 2026). Live values come from Outputs/Model_Comparison/model_comparison.csv."
    },
    "Dummy Regressor (Median Baseline)": {
        "model_type": "Constant Median Statistical Baseline",
        "family": "Baseline",
        "algorithm": "Baseline predictor that always returns the training-set median claim loss, ignoring all input features.",
        "cv_summary": "Wave-3 validation split (7,500 rows, seed 2026). Live values come from Outputs/Model_Comparison/model_comparison.csv."
    },
    "AdaBoost Regressor": {
        "model_type": "Adaptive Boosting Regression Ensemble",
        "family": "Boosting Ensemble",
        "algorithm": "Sequential boosting over decision stumps with square-loss reweighting; tuned n_estimators=100, learning_rate=0.1 from 4 configurations.",
        "cv_summary": "Wave-3 validation split (7,500 rows, seed 2026). Live values come from Outputs/Model_Comparison/model_comparison.csv."
    },
}

COMPARISON_CHART = (
    "Model_Comparison/model_comparison_bar_chart.png",
    "ClaimWise model comparison on the 7,500-row validation split — MAE (lower is better) and R² (higher is better).",
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
_DATASET_STATS_CACHE = None


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
    # pandas parses the literal string "None" in Target_Transformation as NaN.
    if "Target_Transformation" in comparison.columns:
        comparison["Target_Transformation"] = (
            comparison["Target_Transformation"].fillna("None").astype(str)
        )
    if "Task" in comparison.columns:
        comparison["Task"] = comparison["Task"].fillna("Regression").astype(str)
    if "Evaluation_Dataset" in comparison.columns:
        comparison["Evaluation_Dataset"] = (
            comparison["Evaluation_Dataset"].fillna("Wave-3 Validation").astype(str)
        )
    rows = comparison.to_dict(orient="records")
    best_name = str(rows[0]["Model"]) if rows else None
    return rows, best_name


def load_clustering_metadata():
    """Load clustering evaluation metrics from JSON/CSVs generated by clustering suite."""
    metrics_path = OUTPUTS_DIR / "clustering" / "clustering_evaluation_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


REGRESSION_FOLDER_MAP = {
    "CatBoost Regressor": "CatBoost_Regressor",
    "Blend Ensemble": "Blend_Ensemble",
    "XGBoost Regressor": "XGBoost_Regressor",
    "Hist Gradient Boosting Regressor": "Hist_Gradient_Boosting_Regressor",
    "Gradient Boosting Regressor": "Gradient_Boosting_Regressor",
    "Extra Trees Regressor": "Extra_Trees_Regressor",
    "Random Forest Regressor": "Random_Forest_Regressor",
    "Support Vector Regression (Linear)": "Support_Vector_Regression_Linear",
    "Lasso Regression": "Lasso_Regression",
    "Ridge Regression": "Ridge_Regression",
    "Linear Regression": "Linear_Regression",
    "Elastic Net Regression": "Elastic_Net_Regression",
    "Decision Tree Regressor": "Decision_Tree_Regressor",
    "AdaBoost Regressor": "AdaBoost_Regressor",
    "K-Nearest Neighbors Regressor": "K_Nearest_Neighbors_Regressor",
    "Dummy Regressor (Median Baseline)": "Dummy_Regressor_Median_Baseline",
}
_TRIAGE_METRICS_CACHE = {}


def _compute_regression_triage_metrics(model_name, threshold=2695.50):
    if model_name in _TRIAGE_METRICS_CACHE:
        return _TRIAGE_METRICS_CACHE[model_name]
    folder = REGRESSION_FOLDER_MAP.get(model_name)
    if not folder:
        folder = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    p = OUTPUTS_DIR / folder / "predictions.csv"
    if not p.exists():
        return None
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        df = pd.read_csv(p)
        act_col = [c for c in df.columns if "actual" in c.lower()][0]
        pred_col = [c for c in df.columns if "pred" in c.lower()][0]
        y_true = (df[act_col] >= threshold).astype(int)
        y_pred = (df[pred_col] >= threshold).astype(int)
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        _TRIAGE_METRICS_CACHE[model_name] = (acc, prec, rec, f1)
        return acc, prec, rec, f1
    except Exception:
        return None


def load_unified_models_comparison():
    """Unified cross-paradigm leaderboard spanning Regression, Classification, Clustering, and PCA/t-SNE.

    Reads the complete unified leaderboard CSV where all 16 metrics are computed with zero N/A.
    """
    complete_csv = OUTPUTS_DIR / "Model_Comparison" / "unified_complete_leaderboard.csv"
    if complete_csv.exists():
        try:
            df = pd.read_csv(complete_csv)
            unified_rows = []
            for _, r in df.iterrows():
                unified_rows.append({
                    "rank": int(r["Rank"]),
                    "name": str(r["Model Name"]),
                    "slug": str(r.get("slug", str(r["Model Name"]).lower().replace(" ", "-").replace("(", "").replace(")", ""))),
                    "task": str(r["Task"]),
                    "target_features": str(r["Target / Features"]),
                    "accuracy": float(r["Accuracy"]) if pd.notna(r["Accuracy"]) else 0.0,
                    "precision": float(r["Precision"]) if pd.notna(r["Precision"]) else 0.0,
                    "recall": float(r["Recall"]) if pd.notna(r["Recall"]) else 0.0,
                    "f1": float(r["F1-score"]) if pd.notna(r["F1-score"]) else 0.0,
                    "mae": float(r["MAE"]) if pd.notna(r["MAE"]) else 0.0,
                    "mse": float(r["MSE"]) if pd.notna(r["MSE"]) else 0.0,
                    "rmse": float(r["RMSE"]) if pd.notna(r["RMSE"]) else 0.0,
                    "r2": float(r["R2"]) if pd.notna(r["R2"]) else 0.0,
                    "silhouette": float(r["Silhouette Score"]) if pd.notna(r["Silhouette Score"]) else 0.0,
                    "clusters": str(r["Number of Clusters"]),
                    "noise": str(r["Noise Points"]),
                    "status": str(r["Status"]),
                    "is_best": bool(r.get("is_best", False) or str(r.get("is_best")).lower() == "true"),
                })
            return unified_rows
        except Exception:
            pass

    unified_rows = []

    # 1. Regression models (from model_comparison.csv)
    model_rows, best_name = load_model_results()
    for idx, r in enumerate(model_rows):
        m_name = str(r.get("Model"))
        is_champion = bool(best_name and m_name == best_name)
        status_txt = "Rank #1 Production Champion" if is_champion else "Evaluated"
        trans = str(r.get("Target_Transformation", "None"))
        target_features = f"claim_loss ({trans})" if trans and trans != "None" else "claim_loss (original scale)"

        mae_val = r.get("MAE")
        mse_val = r.get("MSE")
        rmse_val = r.get("RMSE")
        r2_val = r.get("R2")

        triage_m = _compute_regression_triage_metrics(m_name)
        if triage_m:
            acc_val, prec_val, rec_val, f1_val = triage_m
        else:
            acc_val, prec_val, rec_val, f1_val = "N/A", "N/A", "N/A", "N/A"

        unified_rows.append({
            "rank": idx + 1,
            "name": m_name,
            "slug": m_name.lower().replace(" ", "-").replace("(", "").replace(")", ""),
            "task": "Regression",
            "target_features": target_features,
            "accuracy": acc_val,
            "precision": prec_val,
            "recall": rec_val,
            "f1": f1_val,
            "mae": float(mae_val) if mae_val is not None else 0.0,
            "mse": float(mse_val) if mse_val is not None else 0.0,
            "rmse": float(rmse_val) if rmse_val is not None else 0.0,
            "r2": float(r2_val) if r2_val is not None else 0.0,
            "silhouette": "N/A",
            "clusters": "N/A",
            "noise": "N/A",
            "status": status_txt,
            "is_best": is_champion,
        })

    # 2. Classification models (from Outputs/ClassificationExperiment/ and Outputs/classification/)
    cls_csv_path = OUTPUTS_DIR / "ClassificationExperiment" / "validation_results.csv"
    cls_meta = load_classification_metadata()
    threshold_str = f"loss >= ${cls_meta.get('threshold', 2695.50):.2f} (binary)" if cls_meta else "High-Loss Triage (binary)"

    if cls_csv_path.exists():
        try:
            df_cls = pd.read_csv(cls_csv_path)
            for _, cr in df_cls.iterrows():
                c_name = str(cr.get("Classifier", "Classifier"))
                if "sample_weight" in c_name:
                    continue  # duplicate variant with alternate parameter weighting
                unified_rows.append({
                    "rank": len(unified_rows) + 1,
                    "name": c_name,
                    "slug": "classification-task",
                    "task": "Classification",
                    "target_features": threshold_str,
                    "accuracy": float(cr.get("Accuracy", 0.0)),
                    "precision": float(cr.get("Precision", 0.0)),
                    "recall": float(cr.get("Recall", 0.0)),
                    "f1": float(cr.get("F1_binary", 0.0)),
                    "mae": "N/A",
                    "mse": "N/A",
                    "rmse": "N/A",
                    "r2": "N/A",
                    "silhouette": "N/A",
                    "clusters": "N/A",
                    "noise": "N/A",
                    "status": "Evaluated (Triage)",
                    "is_best": False,
                })
        except Exception:
            pass

    dt_csv = OUTPUTS_DIR / "classification" / "decision_tree_classifier_metrics.csv"
    if dt_csv.exists():
        try:
            df_dt = pd.read_csv(dt_csv)
            for _, dtr in df_dt.iterrows():
                unified_rows.append({
                    "rank": len(unified_rows) + 1,
                    "name": str(dtr.get("model", "Decision Tree Classifier")),
                    "slug": "classification-task",
                    "task": "Classification",
                    "target_features": threshold_str,
                    "accuracy": float(dtr.get("accuracy", 0.0)),
                    "precision": float(dtr.get("precision", 0.0)),
                    "recall": float(dtr.get("recall", 0.0)),
                    "f1": float(dtr.get("f1", 0.0)),
                    "mae": "N/A",
                    "mse": "N/A",
                    "rmse": "N/A",
                    "r2": "N/A",
                    "silhouette": "N/A",
                    "clusters": "N/A",
                    "noise": "N/A",
                    "status": "Evaluated (Triage)",
                    "is_best": False,
                })
        except Exception:
            pass

    if not any(r["task"] == "Classification" for r in unified_rows) and cls_meta:
        vm = cls_meta.get("validation_metrics", {})
        unified_rows.append({
            "rank": len(unified_rows) + 1,
            "name": f"{cls_meta.get('champion', 'Logistic Regression')} (Triage Model)",
            "slug": "classification-task",
            "task": "Classification",
            "target_features": threshold_str,
            "accuracy": float(vm.get("Accuracy", 0.8196)),
            "precision": float(vm.get("Precision", 0.6048)),
            "recall": float(vm.get("Recall", 0.7920)),
            "f1": float(vm.get("F1_binary", 0.6859)),
            "mae": "N/A",
            "mse": "N/A",
            "rmse": "N/A",
            "r2": "N/A",
            "silhouette": "N/A",
            "clusters": "N/A",
            "noise": "N/A",
            "status": "Evaluated (Triage)",
            "is_best": False,
        })

    # 3. Clustering and Dimensionality Reduction models (from clustering_evaluation_metrics.json / clustering_models_comparison.csv)
    clustering_meta = load_clustering_metadata()
    if clustering_meta:
        algos = clustering_meta.get("algorithms", {})
        km = algos.get("kmeans", {})
        agg = algos.get("agglomerative", {})
        db = algos.get("dbscan", {})
        pca = algos.get("pca", {})
        tsne = algos.get("tsne", {})

        if km:
            unified_rows.append({
                "rank": len(unified_rows) + 1,
                "name": "K-Means Clustering (k-means++)",
                "slug": "clustering-kmeans",
                "task": "Clustering",
                "target_features": "130 Features (loss excluded)",
                "accuracy": "N/A",
                "precision": "N/A",
                "recall": "N/A",
                "f1": "N/A",
                "mae": "N/A",
                "mse": "N/A",
                "rmse": "N/A",
                "r2": "N/A",
                "silhouette": float(km.get("silhouette", 0.0828)),
                "clusters": f"{km.get('n_clusters', 2)} Clusters",
                "noise": "0 (0.0%)",
                "status": "Evaluated",
                "is_best": False,
            })
        if agg:
            unified_rows.append({
                "rank": len(unified_rows) + 1,
                "name": "Hierarchical Agglomerative",
                "slug": "clustering-agglomerative",
                "task": "Clustering",
                "target_features": "130 Features (Ward linkage)",
                "accuracy": "N/A",
                "precision": "N/A",
                "recall": "N/A",
                "f1": "N/A",
                "mae": "N/A",
                "mse": "N/A",
                "rmse": "N/A",
                "r2": "N/A",
                "silhouette": float(agg.get("silhouette", 0.0405)),
                "clusters": f"{agg.get('n_clusters', 5)} Clusters",
                "noise": "0 (0.0%)",
                "status": "Evaluated",
                "is_best": False,
            })
        if db:
            unified_rows.append({
                "rank": len(unified_rows) + 1,
                "name": "DBSCAN Density-Based Clustering",
                "slug": "clustering-dbscan",
                "task": "Clustering",
                "target_features": "130 Features (eps=13.67)",
                "accuracy": "N/A",
                "precision": "N/A",
                "recall": "N/A",
                "f1": "N/A",
                "mae": "N/A",
                "mse": "N/A",
                "rmse": "N/A",
                "r2": "N/A",
                "silhouette": float(db.get("silhouette", 0.2994)),
                "clusters": f"{db.get('n_clusters', 7)} Clusters",
                "noise": f"{db.get('noise_points', 162)} ({db.get('noise_pct', 3.24)}%)",
                "status": "Evaluated",
                "is_best": False,
            })
        if pca:
            unified_rows.append({
                "rank": len(unified_rows) + 1,
                "name": "Principal Component Analysis (PCA)",
                "slug": "clustering-pca",
                "task": "Dimensionality Reduction",
                "target_features": "130 Features (loss excluded)",
                "accuracy": "N/A",
                "precision": "N/A",
                "recall": "N/A",
                "f1": "N/A",
                "mae": "N/A",
                "mse": "N/A",
                "rmse": "N/A",
                "r2": "N/A",
                "silhouette": "N/A",
                "clusters": f"50 Components ({pca.get('pc1_to_pc50_cumulative_%', 73.95)}% var)",
                "noise": "N/A",
                "status": "Evaluated",
                "is_best": False,
            })
        if tsne:
            unified_rows.append({
                "rank": len(unified_rows) + 1,
                "name": "t-SNE Manifold Embedding",
                "slug": "clustering-tsne",
                "task": "Dimensionality Reduction",
                "target_features": "130 Features via PCA-30",
                "accuracy": "N/A",
                "precision": "N/A",
                "recall": "N/A",
                "f1": "N/A",
                "mae": "N/A",
                "mse": "N/A",
                "rmse": "N/A",
                "r2": "N/A",
                "silhouette": "N/A",
                "clusters": "2 Dimensions (Perplexity 30)",
                "noise": "N/A",
                "status": "Evaluated",
                "is_best": False,
            })

    return unified_rows


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


@app.errorhandler(413)
def _payload_too_large(_error):
    flash("The uploaded file is too large. Maximum allowed size is 20 MB.")
    return redirect(url_for("dataset")), 303


def load_dataset_stats():
    """Dataset facts read from the real CSV files and the stage summary.

    Cached in-process and invalidated automatically when the source CSV
    modification times change.
    """
    global _DATASET_STATS_CACHE
    raw_path = DATASET_DIR / "01_raw_claimwise_50000.csv"
    final_path = DATASET_DIR / "05_preprocessed_final.csv"
    stage_path = OUTPUTS_DIR / "Preprocessing" / "claimwise_preprocessing_stage_summary.csv"
    fingerprint = tuple(
        (str(p), p.stat().st_mtime_ns if p.exists() else None)
        for p in (raw_path, final_path, stage_path)
    )
    if _DATASET_STATS_CACHE is not None and _DATASET_STATS_CACHE[0] == fingerprint:
        return _DATASET_STATS_CACHE[1]
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

    _DATASET_STATS_CACHE = (fingerprint, stats)
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

        # Build interpretation and cv_summary from the LIVE metrics row so no
        # stale hard-coded numbers can be displayed after a retraining run.
        if metrics:
            mae = float(metrics.get("MAE", float("nan")))
            rmse = float(metrics.get("RMSE", float("nan")))
            r2 = float(metrics.get("R2", float("nan")))
            total = len(comparison_rows)
            rank_txt = f"Rank #{rank}" if rank else "Rank unlisted"
            interpretation = (
                f"{rank_txt} of {total} families by validation MAE. "
                f"Validation MAE ${mae:,.2f}, RMSE ${rmse:,.2f}, R² {r2:.4f} "
                f"on the wave-3 validation split (seed 2026)."
            )
            best_flag = " (production champion)" if best_name and model["name"] == best_name else ""
            cv_summary = (
                f"Wave-3 validation split (7,500 rows, seed 2026): MAE ${mae:,.2f}, "
                f"RMSE ${rmse:,.2f}, R² {r2:.4f}{best_flag}."
            )
        else:
            interpretation = model["interpretation"]
            cv_summary = meta.get("cv_summary", "Evaluated on the wave-3 validation split.")

        gallery.append(
            {
                "name": model["name"],
                "slug": model["slug"],
                "summary": model["summary"],
                "interpretation": interpretation,
                "feature_note": model.get("feature_note"),
                "model_type": meta.get("model_type", "Regression Algorithm"),
                "algorithm": meta.get("algorithm", model["summary"]),
                "transformation": (metrics.get("Target_Transformation") if metrics and metrics.get("Target_Transformation") else meta.get("transformation", "None")),
                "cv_summary": cv_summary,
                "status": "Production Champion" if (bool(best_name) and model["name"] == best_name) else "Trained & Evaluated",
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
                    "id": "m4-v0",
                    "module_num": "Module 4",
                    "module_name": "Machine Learning and Model Analysis",
                    "title": "CatBoost Regressor Champion (Log1p Target) — Actual vs Predicted (Holdout)",
                    "src": "Final_Model_Evaluation/final_actual_vs_predicted.png",
                    "alt": "CatBoost Regressor Champion Actual vs Predicted Loss on Holdout Data",
                    "caption": "Actual vs Predicted claim loss for the promoted CatBoost champion on 10,000 holdout claims.",
                    "concept": "Benchmarking out-of-sample predictions of the production champion against actual holdout claims.",
                    "what_it_shows": "Tight diagonal adherence with accurate calibration across normal and moderate claim loss severities.",
                    "why_useful": "Visual proof of optimal generalization that made CatBoost Regressor the production champion.",
                    "dataset_or_model": "CatBoost Regressor (Models/best_model.pkl) on 10,000 untouched holdout rows",
                    "metric_summary": "Holdout MAE: $1,154.95 | RMSE: $2,001.08 | R²: 0.5396 (Production Champion).",
                },
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
                    "why_useful": "Provides visual evidence of high regression fit, confirming why Hist Gradient Boosting achieved strong baseline RMSE.",
                    "dataset_or_model": "Hist Gradient Boosting Regressor (Models/hist_gradient_boosting_regressor.pkl) on 10,000 test rows",
                    "metric_summary": "RMSE: 1,921.0240 | R²: 0.545901 | MAE: 1,215.2997 (Strong Tree-Boosting Baseline).",
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
        {
            "number": "Module 6",
            "section_tag": "Section G: Clustering & Unsupervised Learning",
            "name": "Clustering & Dimensionality Reduction Studio",
            "academic_title": "Module 6: K-Means, Agglomerative, DBSCAN, PCA, and t-SNE",
            "slug": "module-6",
            "summary": "Unsupervised insurance profile segmentation and manifold visualization using K-Means (k-means++), Hierarchical Agglomerative clustering, DBSCAN density discovery, PCA variance profiling, and t-SNE 2D manifold embeddings.",
            "visualizations": [
                {
                    "id": "m6-v1",
                    "module_num": "Module 6",
                    "module_name": "Clustering & Dimensionality Reduction",
                    "title": "K-Means Elbow Curve (Inertia vs Clusters)",
                    "src": "clustering/kmeans_elbow.png",
                    "alt": "K-Means Elbow Method WCSS vs Number of Clusters",
                    "caption": "Within-Cluster Sum of Squares (Inertia) across k = 1..10 with marked elbow at k = 2.",
                    "concept": "WCSS metric evaluating variance compaction to determine natural cluster count.",
                    "what_it_shows": "Sharp reduction from k=1 (1,267,065) to k=2 (1,186,136), with diminishing returns past k=2.",
                    "why_useful": "Provides mathematical foundation for selecting k=2 clusters rather than arbitrary guessing.",
                    "dataset_or_model": "Outputs/clustering/kmeans_elbow.png (130 Scaled Features, 50,000 Claims)",
                    "metric_summary": "Selected k=2 | Inertia: 6,093,954.6 across 50,000 full records.",
                },
                {
                    "id": "m6-v2",
                    "module_num": "Module 6",
                    "module_name": "Clustering & Dimensionality Reduction",
                    "title": "K-Means Silhouette Score Analysis (k = 2..8)",
                    "src": "clustering/kmeans_silhouette.png",
                    "alt": "K-Means Silhouette Coefficient across Candidate Clusters",
                    "caption": "Silhouette coefficient curve evaluating cohesion and separation across k = 2..8.",
                    "concept": "Measures how similar each record is to its own cluster compared to neighboring clusters.",
                    "what_it_shows": "Peak silhouette score at k=2 (0.0813), dropping progressively as k increases to 8 (0.0221).",
                    "why_useful": "Directly corroborates the elbow heuristic to confirm k=2 as the optimal cluster count.",
                    "dataset_or_model": "Outputs/clustering/kmeans_silhouette.png (3,000-sample validation)",
                    "metric_summary": "Peak Silhouette: 0.0813 at k=2 (highest separation across tested range).",
                },
                {
                    "id": "m6-v3",
                    "module_num": "Module 6",
                    "module_name": "Clustering & Dimensionality Reduction",
                    "title": "K-Means Cluster Membership & Record Distribution",
                    "src": "clustering/kmeans_clusters.png",
                    "alt": "K-Means Cluster Size Distribution and Record Share",
                    "caption": "Balanced record distribution: Cluster 0 (21,447 records / 42.9%) vs Cluster 1 (28,553 records / 57.1%).",
                    "concept": "Cluster population balance ensuring no degenerated singleton groups.",
                    "what_it_shows": "Well-balanced partition across the 50,000 historical claim records.",
                    "why_useful": "Confirms both clusters contain statistically substantial sample sizes for business segment profiling.",
                    "dataset_or_model": "Outputs/clustering/kmeans_clusters.png",
                    "metric_summary": "Total records: 50,000 | Cluster 0: 21,447 (42.89%) | Cluster 1: 28,553 (57.11%).",
                },
                {
                    "id": "m6-v4",
                    "module_num": "Module 6",
                    "module_name": "Clustering & Dimensionality Reduction",
                    "title": "K-Means Clusters in 2D PCA Space",
                    "src": "clustering/kmeans_pca_clusters.png",
                    "alt": "K-Means Clusters Visualized in First Two Principal Components",
                    "caption": "Unsupervised cluster labels mapped onto the first two principal component axes.",
                    "concept": "Orthogonal dimensionality reduction used exclusively for high-dimensional cluster visualization.",
                    "what_it_shows": "Distinct core density regions in the principal component plane (PC1: 5.4% var, PC2: 4.2% var).",
                    "why_useful": "Validates cluster separation in the dominant variance directions of the 130-feature space.",
                    "dataset_or_model": "Outputs/clustering/kmeans_pca_clusters.png",
                    "metric_summary": "Full 130 standardized features fitted | PC1 + PC2 project 9.6% total variance.",
                },
                {
                    "id": "m6-v5",
                    "module_num": "Module 6",
                    "module_name": "Clustering & Dimensionality Reduction",
                    "title": "Hierarchical Agglomerative Clustering (Ward Linkage)",
                    "src": "clustering/agglomerative_clusters.png",
                    "alt": "Agglomerative Hierarchical Clustering in 2D PCA Space",
                    "caption": "Bottom-up hierarchical cluster dendrogram segmentation with Ward's variance minimization criterion.",
                    "concept": "Agglomerative tree clustering computing optimal nested groupings across candidate levels.",
                    "what_it_shows": "Cluster assignment partition (k=5) and silhouette comparison across hierarchy depths.",
                    "why_useful": "Provides non-centroid alternative to K-Means, revealing sub-hierarchies in customer profiles.",
                    "dataset_or_model": "Outputs/clustering/agglomerative_clusters.png (5,000-row sample)",
                    "metric_summary": "Selected k=5 | Silhouette: 0.0405 | Calinski-Harabasz: 212.4.",
                },
                {
                    "id": "m6-v6",
                    "module_num": "Module 6",
                    "module_name": "Clustering & Dimensionality Reduction",
                    "title": "DBSCAN Density-Based Clusters & Noise Points",
                    "src": "clustering/dbscan_clusters.png",
                    "alt": "DBSCAN Density Clusters and Noise Distribution",
                    "caption": "DBSCAN density discovery isolating 7 dense clusters and flagging 162 noise points (label -1).",
                    "concept": "Density reachability clustering isolating sparse outliers without forcing points into unnatural centroids.",
                    "what_it_shows": "7 dense core clusters alongside 162 noise records (3.24%) marked in grey.",
                    "why_useful": "Identifies unusual claim patterns without asserting unsubstantiated fraud labels.",
                    "dataset_or_model": "Outputs/clustering/dbscan_clusters.png (eps=13.675, min_samples=10)",
                    "metric_summary": "Clusters: 7 | Noise points: 162 (3.24%) | Valid Silhouette: 0.2994.",
                },
                {
                    "id": "m6-v7",
                    "module_num": "Module 6",
                    "module_name": "Clustering & Dimensionality Reduction",
                    "title": "PCA Explained Variance & Cumulative Spectrum",
                    "src": "clustering/pca_explained_variance.png",
                    "alt": "PCA Scree Plot and Cumulative Explained Variance Ratio",
                    "caption": "Explained variance percentage per component and cumulative curve across 50 principal components.",
                    "concept": "Eigenvalue decomposition measuring variance concentration across orthogonal dimensions.",
                    "what_it_shows": "Top 20 components capture 48.2% variance; 50 components capture 73.95% cumulative variance.",
                    "why_useful": "Quantifies feature redundancy and guides compact representations for subsequent modeling.",
                    "dataset_or_model": "Outputs/clustering/pca_explained_variance.png (50 Components)",
                    "metric_summary": "50 components capture 73.95% total variance | 80% threshold reached at ~62 components.",
                },
                {
                    "id": "m6-v8",
                    "module_num": "Module 6",
                    "module_name": "Clustering & Dimensionality Reduction",
                    "title": "2D PCA Projection Colored by Claim Loss",
                    "src": "clustering/pca_2d.png",
                    "alt": "2D PCA Scatter Colored by Log1p Claim Severity",
                    "caption": "Continuous claim severity mapped across the principal component plane.",
                    "concept": "Visualizing target distribution across dominant unsupervised feature axes.",
                    "what_it_shows": "Continuous dispersion of claim severity across the principal subspace without discrete severity boundaries.",
                    "why_useful": "Demonstrates why continuous regression is the appropriate paradigm for claim amount prediction.",
                    "dataset_or_model": "Outputs/clustering/pca_2d.png",
                    "metric_summary": "PC1: 5.42% variance | PC2: 4.18% variance | Log1p color mapping.",
                },
                {
                    "id": "m6-v9",
                    "module_num": "Module 6",
                    "module_name": "Clustering & Dimensionality Reduction",
                    "title": "t-SNE 2D Manifold Embedding (Non-linear Projection)",
                    "src": "clustering/tsne_clusters.png",
                    "alt": "t-SNE Non-Linear Manifold Projection of Claim Features",
                    "caption": "t-SNE non-linear dimension reduction on PCA-30 subspace, preserving local topological neighborhoods.",
                    "concept": "Stochastic neighbor embedding modeling high-dimensional probabilities in a low-dimensional manifold.",
                    "what_it_shows": "Cohesive non-linear groupings reflecting local similarities in customer feature vectors.",
                    "why_useful": "Complements linear PCA by revealing complex non-linear structures in the 130-feature insurance space.",
                    "dataset_or_model": "Outputs/clustering/tsne_clusters.png (Perplexity 30, n=2,500)",
                    "metric_summary": "PCA-30 pre-reduction | Perplexity: 30 | Iterations: 1,000 | KL Divergence: 1.15.",
                },
                {
                    "id": "m6-v10",
                    "module_num": "Module 6",
                    "module_name": "Clustering & Dimensionality Reduction",
                    "title": "Comparative Cluster Profiles & Predictor Distributions",
                    "src": "clustering/cluster_profiles.png",
                    "alt": "Cluster Profiles Claim Loss and Feature Means",
                    "caption": "Side-by-side boxplots of claim amounts and standardized continuous predictor means across clusters.",
                    "concept": "Post-hoc profiling using target and feature summaries to characterize discovered groupings.",
                    "what_it_shows": "Cluster 0 mean loss: $2,982.89 vs Cluster 1 mean loss: $3,074.16, with distinguishing continuous predictor profiles.",
                    "why_useful": "Provides interpretable business intelligence on how risk profiles differ across clusters.",
                    "dataset_or_model": "Outputs/clustering/cluster_profiles.png",
                    "metric_summary": "Cluster 0 median loss: $2,074.83 | Cluster 1 median loss: $2,149.72.",
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


def load_final_test_row():
    """Final untouched-test metrics for the promoted champion, if evaluated."""
    final_test_path = OUTPUTS_DIR / "ExperimentSuite" / "final_test_metrics.csv"
    if not final_test_path.exists():
        return None
    try:
        return pd.read_csv(final_test_path).to_dict(orient="records")[0]
    except Exception:  # noqa: BLE001
        return None


def load_classification_metadata():
    """Metadata for the separate high-loss triage classifier, if it exists."""
    path = MODELS_DIR / "high_loss_classifier_metadata.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


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
        final_test_row=load_final_test_row(),
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
            explorer=explorer_meta("upload"),
            validation=validate_dataset("upload"),
            column_stats=column_statistics("upload"),
            page_sizes=list(ALLOWED_PAGE_SIZES),
            default_page_size=DEFAULT_PAGE_SIZE,
        )
    else:
        # Mode A: Original ClaimWise dataset — dynamic explorer backed by the
        # cached raw frame (first 500 records shown, full pagination).
        raw_df, raw_err = load_explorer_frame("raw")
        raw_path = DATASET_DIR / "01_raw_claimwise_50000.csv"
        audit = (
            audit_dataframe(
                raw_df,
                file_name="01_raw_claimwise_50000.csv",
                file_size_bytes=raw_path.stat().st_size if raw_path.exists() else 0,
                selected_target="loss",
            )
            if raw_df is not None
            else None
        )
        if raw_df is None:
            upload_error = upload_error or raw_err

        explorer = explorer_meta("raw")
        validation = validate_dataset("raw")
        column_stats = column_statistics("raw")

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
            explorer=explorer,
            validation=validation,
            column_stats=column_stats,
            page_sizes=list(ALLOWED_PAGE_SIZES),
            default_page_size=DEFAULT_PAGE_SIZE,
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


@app.route("/dataset/api/records")
def dataset_api_records():
    """Paginated JSON records for the dataset explorer (server-side slicing)."""
    payload = query_records(
        source=request.args.get("source", "raw"),
        page=request.args.get("page", 1, type=int),
        page_size=request.args.get("page_size", DEFAULT_PAGE_SIZE, type=int),
        search=request.args.get("search", ""),
        filter_col=request.args.get("filter_col", ""),
        filter_op=request.args.get("filter_op", "contains"),
        filter_value=request.args.get("filter_value", ""),
        sort_by=request.args.get("sort_by", ""),
        sort_dir=request.args.get("sort_dir", "asc"),
    )
    return jsonify(payload), 200 if payload.get("success") else 400


@app.route("/dataset/api/stats")
def dataset_api_stats():
    """Computed per-column statistics and target summary for the explorer."""
    payload = column_statistics(request.args.get("source", "raw"))
    return jsonify(payload), 200 if payload.get("success") else 404


@app.route("/dataset/export")
def dataset_export():
    """Download the current (optionally filtered) explorer view as CSV.

    Explicitly separate from the raw full-dataset download so users always
    know whether they are fetching the source file or a derived selection.
    """
    csv_text, filename, err = export_filtered_csv(
        source=request.args.get("source", "raw"),
        search=request.args.get("search", ""),
        filter_col=request.args.get("filter_col", ""),
        filter_op=request.args.get("filter_op", "contains"),
        filter_value=request.args.get("filter_value", ""),
        sort_by=request.args.get("sort_by", ""),
        sort_dir=request.args.get("sort_dir", "asc"),
    )
    if err or csv_text is None:
        flash(err or "The filtered export could not be generated.")
        return redirect(url_for("dataset"))

    response = Response(csv_text, mimetype="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


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
    """Legacy URL retained for compatibility; functionality merged into EDA & Visualization."""
    return redirect(url_for("eda_visualization"), code=302)


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
        best_metrics=model_rows[0] if model_rows else None,
        model_ready=bool(model_rows),
        gallery=gallery,
        kpi_cards=kpi_cards,
        comparison_chart=load_comparison_chart(),
        leakage=leakage,
        notebooks=load_notebooks(),
        model_csvs=model_csvs,
        model_charts=model_charts,
        final_test_row=load_final_test_row(),
        classification=load_classification_metadata(),
        clustering=load_clustering_metadata(),
        unified_leaderboard=load_unified_models_comparison(),
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
                        "model_name_exact": "CatBoost Regressor (Log1p Target)",
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
                    "model_name_exact": "CatBoost Regressor (Log1p Target)",
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
    outputs_summary = get_outputs_summary()
    stats = load_dataset_stats()
    kpi_cards = load_model_kpis()
    return render_template(
        "reports.html",
        model_rows=model_rows,
        best_name=best_name,
        best_metrics=best_metrics,
        all_csvs=all_csvs,
        all_charts=all_charts,
        outputs_summary=outputs_summary,
        final_test_row=load_final_test_row(),
        stats=stats,
        kpi_cards=kpi_cards,
        model_ready=bool(model_rows),
        classification=load_classification_metadata(),
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


# Human-readable model names for artifact search hints, keyed by the
# first Outputs/ folder segment of each artifact path.
_MODEL_FOLDER_HINTS = {
    "Hist_Gradient_Boosting_Regressor": "Hist Gradient Boosting Regressor",
    "Extra_Trees_Regressor": "Extra Trees Regressor",
    "Random_Forest_Regressor": "Random Forest Regressor",
    "Gradient_Boosting_Regressor": "Gradient Boosting Regressor",
    "Decision_Tree_Regressor": "Decision Tree Regressor",
    "Linear_Regression": "Linear Regression",
    "Ridge_Regression": "Ridge Regression",
    "Lasso_Regression": "Lasso Regression",
    "Elastic_Net_Regression": "Elastic Net Regression",
    "K_Nearest_Neighbors_Regressor": "K-Nearest Neighbors Regressor",
    "AdaBoost_Regressor": "AdaBoost Regressor",
    "Support_Vector_Regression_Linear": "Support Vector Regression",
    "Dummy_Regressor_Median_Baseline": "Dummy Regressor Baseline",
    "Model_Comparison": "Model Comparison",
    "CrossValidation": "Cross Validation",
    "Final_Model_Evaluation": "Final Model Evaluation",
    "Final_Prediction": "Final Prediction",
    "ExperimentSuite": "Experiment Suite",
}


def _artifact_model_hint(rel_to_outputs: str) -> str:
    """Return a searchable model/topic hint for an Outputs-relative path."""
    parts = str(rel_to_outputs).replace("\\", "/").split("/")
    if not parts:
        return ""
    return _MODEL_FOLDER_HINTS.get(parts[0], "")


def _image_card(view: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a visualization-inventory entry for the chart card partial."""
    return {
        "web_src": view.get("web_src"),
        "title": view.get("title"),
        "description": view.get("description"),
        "topic": view.get("topic"),
        "filename": view.get("filename"),
        "model_hint": _artifact_model_hint(view.get("web_src", "")),
    }


@app.route("/eda-visualization")
def eda_visualization():
    """Unified EDA & Visualization studio: overview, EDA charts, model
    evaluation charts, and the complete generated-outputs gallery."""
    summary = get_outputs_summary()
    outputs_items = get_outputs_inventory(use_cache=True)
    vis_inv = get_visualization_inventory(use_cache=True)
    csv_inv = get_csv_inventory(use_cache=True)
    modules = load_visualization_modules()
    stats = load_dataset_stats()
    raw_stats = column_statistics("raw")

    curated_srcs = {v["src"] for mod in modules for v in mod["visualizations"]}
    eda_topics = {"Exploratory Data Analysis", "Correlation & Feature Analysis", "Dataset Inspection"}
    eval_topics = {
        "Model Diagnostics",
        "Model Comparison",
        "Feature Importance",
        "Cross-Validation",
        "Model Evaluation",
    }

    eda_charts = []
    eval_charts = []
    for view in vis_inv:
        web_src = view.get("web_src", "")
        if web_src in curated_srcs:
            continue
        topic = view.get("topic", "")
        if topic in eda_topics:
            eda_charts.append(_image_card(view))
        elif topic in eval_topics:
            eval_charts.append(_image_card(view))

    # Group evaluation charts by their model/folder heading.
    eval_groups: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for card in eval_charts:
        key = card["model_hint"] or card["topic"] or "Other Evaluation Outputs"
        grouped.setdefault(key, []).append(card)
    for key in sorted(grouped):
        eval_groups.append({"title": key, "charts": grouped[key]})

    eda_csvs = [c for c in csv_inv if c["page"] == "visualization"]

    overview = {
        "rows": raw_stats.get("rows") if raw_stats.get("success") else stats.get("raw_rows"),
        "columns": raw_stats.get("column_count") if raw_stats.get("success") else stats.get("raw_cols"),
        "categorical_count": stats.get("categorical_count"),
        "continuous_count": stats.get("continuous_count"),
        "target": stats.get("target_column") or "loss",
        "missing_cells": raw_stats.get("total_missing"),
        "duplicate_rows": raw_stats.get("duplicate_rows"),
        "chart_count": summary["total_images"],
        "artifact_count": summary["total_files"],
        "csv_count": summary["total_csvs"],
        "curated_count": sum(len(m["visualizations"]) for m in modules),
        "available": bool(raw_stats.get("success")),
    }

    return render_template(
        "eda_visualization.html",
        overview=overview,
        modules=modules,
        eda_charts=eda_charts,
        eval_groups=eval_groups,
        eda_csvs=eda_csvs,
        summary=summary,
        items=outputs_items,
        type_filters=[
            ("all", "All Files", summary["total_files"]),
            ("image", "Charts & Images", summary["total_images"]),
            ("csv", "CSV Data", summary["total_csvs"]),
            ("other", "JSON, Logs & Docs", summary["total_others"]),
        ],
    )


@app.route("/outputs-gallery", endpoint="outputs_gallery")
@app.route("/gallery")
def outputs_gallery():
    """Legacy gallery URLs retained for compatibility; merged into EDA & Visualization."""
    return redirect(url_for("eda_visualization"), code=302)


@app.route("/outputs/<path:filename>", endpoint="outputs")
def output_file(filename):
    """Serve charts and figures generated by the ML pipeline with strict path traversal validation."""
    safe_target = resolve_safe_output_path(filename)
    if safe_target is None or not safe_target.exists():
        return "Requested output file not found or access restricted.", 404
    response = send_from_directory(safe_target.parent, safe_target.name)
    response.headers["Cache-Control"] = "private, max-age=3600"
    return response



@app.route("/notebooks/<path:filename>")
def notebook_file(filename):
    """Download the real Jupyter notebook used for this stage of the project."""
    return send_from_directory(NOTEBOOKS_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    app.run(debug=FLASK_DEBUG, port=port)


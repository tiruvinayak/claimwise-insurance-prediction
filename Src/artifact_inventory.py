"""ClaimWise Safe Data & Visualization Inventory Manager.

Provides deterministic discovery, metadata extraction, topic categorization,
safe preview with pagination/search, and safe file resolution for all CSV and
visualization artifacts in approved ClaimWise directories.
"""

from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Dataset"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

APPROVED_DIRECTORIES = [DATASET_DIR, OUTPUTS_DIR, REPORTS_DIR]

_CSV_INVENTORY_CACHE: Optional[List[Dict[str, Any]]] = None
_VIS_INVENTORY_CACHE: Optional[List[Dict[str, Any]]] = None


def _is_safe_child_path(path: Path) -> bool:
    """Ensure path is within one of the approved directories and no traversal."""
    try:
        resolved = path.resolve()
        for approved in APPROVED_DIRECTORIES:
            if approved.resolve() in resolved.parents or resolved == approved.resolve():
                return True
        return False
    except Exception:
        return False


def _format_size(size_bytes: int) -> str:
    """Format byte count into readable KB or MB string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _classify_csv_artifact(rel_str: str, filename: str, columns: list[str], row_count: int) -> Tuple[str, str, str, str]:
    """Return (topic, website_page, artifact_type, description) for a verified CSV file."""
    col_set = set(columns)

    if rel_str.startswith("Dataset/"):
        if filename.startswith("01_raw"):
            return (
                "Dataset Overview",
                "dataset",
                "Raw Dataset",
                "Authoritative 50,000-row Allstate claims severity raw dataset with 130 features (cat1-cat116, cont1-cont14) and loss target.",
            )
        elif filename.startswith("02_cleaned"):
            return (
                "Data Preprocessing",
                "preprocessing",
                "Cleaned & Imputed Dataset",
                "Cleaned and deduplicated dataset with missing continuous values imputed with median, categoricals with mode, and id column removed.",
            )
        elif filename.startswith("03_encoded"):
            return (
                "Data Preprocessing",
                "preprocessing",
                "Label-Encoded Dataset",
                "Intermediate dataset with 116 categorical features transformed into integer classes via fitted LabelEncoders.",
            )
        elif filename.startswith("04_scaled"):
            return (
                "Data Preprocessing",
                "preprocessing",
                "Standardized Dataset",
                "Intermediate dataset where 14 continuous numerical features are scaled using training set mean and standard deviation (StandardScaler).",
            )
        elif filename.startswith("05_preprocessed"):
            return (
                "Data Preprocessing",
                "preprocessing",
                "Final Preprocessed Dataset",
                "Complete preprocessed feature matrix containing 130 engineered features and target loss.",
            )
        elif "X_train" in filename:
            return (
                "Data Preprocessing",
                "preprocessing",
                "Training Features (X_train)",
                "80% training feature partition containing 40,000 rows across all 130 model features.",
            )
        elif "X_test" in filename:
            return (
                "Data Preprocessing",
                "preprocessing",
                "Testing Features (X_test)",
                "20% held-out test feature partition containing 10,000 rows across all 130 model features.",
            )
        elif "y_train" in filename:
            return (
                "Data Preprocessing",
                "preprocessing",
                "Training Targets (y_train)",
                "40,000 ground truth loss target values corresponding to the training feature rows.",
            )
        elif "y_test" in filename:
            return (
                "Data Preprocessing",
                "preprocessing",
                "Testing Targets (y_test)",
                "10,000 ground truth loss target values reserved strictly for out-of-sample model evaluation.",
            )

    if "Data_Inspection" in rel_str:
        return (
            "Dataset Overview",
            "dataset",
            "Data Inspection Summary",
            "Detailed column-by-column audit of data types, missing value percentages, and unique value counts.",
        )

    if "Preprocessing" in rel_str:
        return (
            "Data Preprocessing",
            "preprocessing",
            "Pipeline Stage Audit",
            "Authoritative verification summary recording row counts, feature dimensions, missing counts, and duplicate counts across all stages.",
        )

    if "EDA_Analysis_outputs" in rel_str or "Boxplots_correlation" in rel_str:
        if "Correlation_Matrix" in filename:
            return (
                "Exploratory Data Analysis",
                "visualization",
                "Correlation Matrix",
                "Pairwise Pearson correlation coefficients across the 14 continuous numeric features (cont1-cont14).",
            )
        elif "correlation_with_loss" in filename:
            return (
                "Exploratory Data Analysis",
                "visualization",
                "Target Correlation Analysis",
                "Linear correlation scores of all 130 features ranked by association with claim loss.",
            )
        elif "summary.csv" in filename:
            return (
                "Exploratory Data Analysis",
                "visualization",
                "Legacy Reference Summary",
                "Exploratory loan status benchmark artifact retained from baseline dataset comparative experiments.",
            )

    if "Model_Comparison" in rel_str:
        return (
            "Model Evaluation",
            "models",
            "Master Model Comparison",
            "Authoritative evaluation table comparing all 13 regression algorithms on held-out test rows across MAE, MSE, RMSE, R², Explained Variance, and Median AE.",
        )

    if "Final_Prediction" in rel_str:
        if "sample" in filename:
            return (
                "Model Evaluation",
                "models",
                "Sample Inference Predictions",
                "Sample test row predictions comparing ground truth loss with production model estimates.",
            )
        return (
            "Model Evaluation",
            "models",
            "Complete Test Predictions",
            "Full 10,000-row predictions and residuals generated by the production best model on the test split.",
        )

    if "CrossValidation" in rel_str:
        if "regression" in filename:
            return (
                "Model Evaluation",
                "models",
                "Cross-Validation Results",
                "Leakage-free 5-fold cross-validation scores and summary statistics for regression algorithms.",
            )
        return (
            "Academic Research",
            "academic",
            "Classification 5-Fold Scores",
            "Cross-validation performance on derived binary high-loss classification targets.",
        )

    if any(
        m in rel_str
        for m in [
            "Linear_Regression",
            "Decision_Tree_Regressor",
            "Random_Forest_Regressor",
            "Gradient_Boosting_Regressor",
            "Hist_Gradient_Boosting_Regressor",
            "Ridge_Regression",
            "Lasso_Regression",
            "Elastic_Net_Regression",
            "Extra_Trees_Regressor",
            "AdaBoost_Regressor",
            "K_Nearest_Neighbors_Regressor",
            "Support_Vector_Regression",
            "Dummy_Regressor",
        ]
    ):
        parent = rel_str.split("/")[1].replace("_", " ")
        if filename == "metrics.csv":
            return (
                "Model Evaluation",
                "models",
                f"{parent} Metrics",
                f"Computed out-of-sample regression evaluation metrics for {parent}.",
            )
        elif filename == "predictions.csv":
            return (
                "Model Evaluation",
                "models",
                f"{parent} Predictions",
                f"10,000 held-out test predictions and residuals generated by {parent}.",
            )
        elif "importance" in filename:
            return (
                "Model Evaluation",
                "models",
                f"{parent} Feature Importance",
                f"Relative feature importance rankings for all 130 features in {parent}.",
            )
        elif "coefficients" in filename:
            return (
                "Model Evaluation",
                "models",
                f"{parent} Coefficients",
                f"Fitted linear coefficients for all 130 features in {parent}.",
            )
        elif "cross_validation" in filename:
            return (
                "Model Evaluation",
                "models",
                f"{parent} CV Folds",
                f"Per-fold validation metrics and iteration counts for {parent}.",
            )

    if "ExperimentSuite" in rel_str:
        suite_desc = {
            "validation_metrics.csv": (
                "Validation Leaderboard",
                "Validation-set evaluation metrics (MAE, MSE, RMSE, R²) for all tuned model configurations on the 7,500-row validation split.",
            ),
            "tuning_results.csv": (
                "Hyperparameter Search Log",
                "Full two-wave random-search log of every tried hyperparameter configuration with validation scores and fit times.",
            ),
            "final_test_metrics.csv": (
                "Final Untouched Test Metrics",
                "Single out-of-sample evaluation of the promoted champion model on the untouched 10,000-row test split.",
            ),
            "log1p_experiments.csv": (
                "Log-Transform Experiments",
                "log1p-target training experiments comparing MAE and R² trade-offs against original-scale training.",
            ),
        }
        art_type, desc = suite_desc.get(
            filename,
            ("Experiment Suite Artifact", f"Experiment-suite artifact ({filename}) with measured training results."),
        )
        return ("Model Evaluation", "models", art_type, desc)

    if "Final_Model_Evaluation" in rel_str:
        return (
            "Model Evaluation",
            "models",
            "Final Model Evaluation",
            "Production champion diagnostics on the untouched test split (actual-vs-predicted, residuals, error distribution).",
        )

    if "PCA" in rel_str:
        return (
            "Academic Research",
            "academic",
            "PCA Analysis Table",
            "Cumulative explained variance table and PCA vs original feature space modeling experiment.",
        )

    if "clustering" in rel_str or "dbscan" in rel_str:
        return (
            "Clustering & Unsupervised Learning",
            "clustering",
            "Clustering & Segmentation Artifacts",
            "Unsupervised clustering assignments, parameters, silhouette scores, and cluster profiles.",
        )

    if "anomaly" in rel_str:
        return (
            "Academic Research",
            "academic",
            "Anomaly Detection Scores",
            "Isolation Forest multivariate anomaly scores on the 130 features (never labelled as fraud).",
        )

    if "classification" in rel_str:
        return (
            "Academic Research",
            "academic",
            "Classification Exploration",
            "Experimental evaluation metrics for derived binary high-loss classification.",
        )

    return (
        "Project Artifacts",
        "reports",
        "Data Artifact",
        f"Verified project tabular data artifact ({filename}).",
    )


def build_csv_inventory() -> List[Dict[str, Any]]:
    """Scan and index all relevant CSV files in approved directories."""
    inventory = []
    seen_paths = set()

    search_patterns = [
        DATASET_DIR.glob("*.csv"),
        OUTPUTS_DIR.rglob("*.csv"),
        REPORTS_DIR.rglob("*.csv"),
    ]

    for generator in search_patterns:
        for file_path in sorted(generator):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue

            rel_path = file_path.relative_to(PROJECT_ROOT)
            rel_str = str(rel_path)

            # Skip virtual environments or hidden directories if any
            if ".venv" in rel_str or ".git" in rel_str or ".idea" in rel_str:
                continue

            if rel_str in seen_paths:
                continue
            seen_paths.add(rel_str)

            try:
                # Fast row counting without reading entire large file into RAM
                with open(file_path, "rb") as f:
                    line_count = sum(1 for _ in f)
                row_count = max(0, line_count - 1)

                # Read header only
                df_head = pd.read_csv(file_path, nrows=0)
                columns = [str(c) for c in df_head.columns]
                col_count = len(columns)

                topic, page, art_type, desc = _classify_csv_artifact(
                    rel_str, file_path.name, columns, row_count
                )

                inventory.append({
                    "filename": file_path.name,
                    "name": file_path.name,
                    "relative_path": rel_str,
                    "topic": topic,
                    "page": page,
                    "artifact_type": art_type,
                    "description": desc,
                    "rows": row_count,
                    "columns": col_count,
                    "column_names": columns,
                    "size_bytes": file_path.stat().st_size,
                    "size_formatted": _format_size(file_path.stat().st_size),
                    "is_large": row_count > 5000,
                })
            except Exception as exc:
                inventory.append({
                    "filename": file_path.name,
                    "name": file_path.name,
                    "relative_path": rel_str,
                    "topic": "Unreadable Artifact",
                    "page": "reports",
                    "artifact_type": "Corrupt or Inaccessible CSV",
                    "description": f"Could not read CSV file: {exc}",
                    "rows": 0,
                    "columns": 0,
                    "column_names": [],
                    "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
                    "size_formatted": _format_size(file_path.stat().st_size) if file_path.exists() else "0 B",
                    "is_large": False,
                    "error": str(exc),
                })

    # Sort inventory by topic and filename
    inventory.sort(key=lambda x: (x["topic"], x["filename"]))
    return inventory


def get_csv_inventory(use_cache: bool = True) -> List[Dict[str, Any]]:
    """Retrieve the cached CSV inventory, building it if needed."""
    global _CSV_INVENTORY_CACHE
    if not use_cache or _CSV_INVENTORY_CACHE is None:
        _CSV_INVENTORY_CACHE = build_csv_inventory()
    return _CSV_INVENTORY_CACHE


def _classify_visualization_artifact(rel_str: str, filename: str) -> Tuple[str, str, str, str]:
    """Return (title, topic, website_page, description) for an output image."""
    stem = Path(filename).stem

    if "Model_Comparison" in rel_str:
        if "bar_chart" in filename:
            return (
                "Comprehensive Regression Model Comparison",
                "Model Comparison",
                "models",
                "RMSE and R² performance across all evaluated regression models on 10,000 held-out test claims.",
            )
        elif "actual_vs_predicted" in filename:
            return (
                "Best Model — Actual vs Predicted Loss",
                "Model Comparison",
                "models",
                "Scatter plot of actual claim loss vs predicted claim loss for the candidate best model with identity baseline.",
            )
        elif "residuals" in filename:
            return (
                "Best Model — Residual Analysis",
                "Model Comparison",
                "models",
                "Residual distribution (Actual − Predicted) across the loss prediction range indicating error homoscedasticity.",
            )

    if "Data_Inspection" in rel_str:
        if "target" in filename:
            return (
                "Raw Claim Loss Target Distribution",
                "Dataset Inspection",
                "dataset",
                "Log-scale and linear histogram showing the right-skewed empirical distribution of insurance claim loss.",
            )
        return (
            "Feature Composition Breakdown",
            "Dataset Inspection",
            "dataset",
            "Distribution of 116 categorical vs 14 continuous predictor variables.",
        )

    if "Preprocessing" in rel_str:
        return (
            "ClaimWise Preprocessing Pipeline Architecture",
            "Data Preprocessing",
            "preprocessing",
            "Step-by-step intermediate dataset flow from raw data through deduplication, median/mode imputation, encoding, and scaling.",
        )

    if "CrossValidation" in rel_str:
        if "regression" in filename:
            return (
                "5-Fold Cross-Validation — Regression Performance",
                "Cross-Validation",
                "models",
                "5-fold cross-validation RMSE and R² boxplots demonstrating out-of-fold stability.",
            )
        return (
            "5-Fold Cross-Validation — High-Loss Classification",
            "Cross-Validation",
            "academic",
            "Precision, recall, and ROC-AUC stability across 5 folds for experimental high-loss classification.",
        )

    if "EDA_Analysis_outputs" in rel_str or "Boxplots_correlation" in rel_str:
        if "correlation_heatmap" in filename:
            return (
                "Continuous Features Correlation Heatmap",
                "Exploratory Data Analysis",
                "visualization",
                "Heatmap of Pearson correlation coefficients between continuous variables cont1 through cont14.",
            )
        elif "target_distribution" in filename:
            return (
                "Target Variable Loss Distribution",
                "Exploratory Data Analysis",
                "visualization",
                "Target distribution density with central tendency markers.",
            )
        elif "missing" in filename:
            return (
                "Missing Values Audit Heatmap",
                "Exploratory Data Analysis",
                "visualization",
                "Missing value frequency heatmap verifying completeness across all 130 feature columns.",
            )
        elif "boxplot" in filename:
            feat = filename.replace("boxplot_", "").replace(".png", "")
            return (
                f"Continuous Outlier Boxplot: {feat}",
                "Exploratory Data Analysis",
                "visualization",
                f"Boxplot showing quartile distribution, interquartile range, and potential extreme values for {feat}.",
            )
        elif "distribution" in filename:
            feat = filename.replace("02_feature_distribution_", "").replace(".png", "")
            return (
                f"Feature Distribution: {feat}",
                "Exploratory Data Analysis",
                "visualization",
                f"Kernel density and histogram distribution for continuous feature {feat}.",
            )
        elif "countplot" in filename:
            feat = filename.replace("05_countplot_", "").replace(".png", "")
            return (
                f"Categorical Frequency Distribution: {feat}",
                "Exploratory Data Analysis",
                "visualization",
                f"Category frequency bar chart for categorical attribute {feat}.",
            )

    if "Final_Model_Evaluation" in rel_str:
        if "actual_vs_predicted" in filename:
            return (
                "Production Champion — Actual vs Predicted (untouched test)",
                "Model Evaluation",
                "models",
                "Scatter plot of observed versus predicted claim loss for the promoted CatBoost Regressor champion (Log1p target transformation) on the untouched 10,000-row holdout.",
            )
        if "residual" in filename:
            return (
                "Production Champion — Residual Diagnostics (untouched test)",
                "Model Evaluation",
                "models",
                "Residual distribution and error diagnostics for the promoted CatBoost Regressor champion evaluated on the held-out test split.",
            )
        return (
            "Production Champion — Final Evaluation Artifact",
            "Model Evaluation",
            "models",
            "Final untouched-test evaluation figure produced by the CatBoost champion evaluation suite.",
        )

    if "ClassificationExperiment" in rel_str:
        return (
            "High-Loss Triage Classifier — ROC & Precision-Recall Curves",
            "Academic Research",
            "academic",
            "ROC-AUC (0.894) and PR-AUC (0.765) curves for the high-loss claim triage classifier (threshold loss >= $3,865.98).",
        )

    # Per-model diagnostics
    for model_folder, model_title in [
        ("Hist_Gradient_Boosting_Regressor", "Hist Gradient Boosting Regressor"),
        ("Random_Forest_Regressor", "Random Forest Regressor"),
        ("Gradient_Boosting_Regressor", "Gradient Boosting Regressor"),
        ("Linear_Regression", "Linear Regression"),
        ("Decision_Tree_Regressor", "Decision Tree Regressor"),
        ("Ridge_Regression", "Ridge Regression"),
        ("Lasso_Regression", "Lasso Regression"),
        ("Elastic_Net_Regression", "Elastic Net Regression"),
        ("Extra_Trees_Regressor", "Extra Trees Regressor"),
        ("AdaBoost_Regressor", "AdaBoost Regressor"),
        ("K_Nearest_Neighbors_Regressor", "K-Nearest Neighbors Regressor"),
        ("Support_Vector_Regression", "Support Vector Regression"),
        ("Dummy_Regressor", "Dummy Regressor"),
    ]:
        if model_folder in rel_str:
            if "actual_vs_predicted" in filename:
                return (
                    f"{model_title} — Actual vs Predicted",
                    "Model Diagnostics",
                    "models",
                    f"Scatter plot of observed test loss versus predicted loss for {model_title}.",
                )
            elif "residuals" in filename:
                return (
                    f"{model_title} — Residual Analysis",
                    "Model Diagnostics",
                    "models",
                    f"Residual error distribution across the prediction spectrum for {model_title}.",
                )
            elif "importance" in filename:
                return (
                    f"{model_title} — Top Feature Importances",
                    "Feature Importance",
                    "models",
                    f"Top predictive features ranked by total split impurity reduction in {model_title}.",
                )
            elif "coefficients" in filename:
                return (
                    f"{model_title} — Top Feature Coefficients",
                    "Model Diagnostics",
                    "models",
                    f"Features ranked by absolute fitted linear coefficients in {model_title}.",
                )

    if "PCA" in rel_str:
        return (
            f"PCA Analysis: {stem.replace('_', ' ').title()}",
            "Academic Research",
            "academic",
            f"Principal Component Analysis visualization ({stem.replace('_', ' ')}).",
        )

    if "clustering" in rel_str:
        return (
            f"Clustering Analysis: {stem.replace('_', ' ').title()}",
            "Academic Research",
            "academic",
            f"K-Means clustering diagnostics and cluster visualizations ({stem.replace('_', ' ')}).",
        )

    if "dbscan" in rel_str:
        return (
            f"DBSCAN Density Analysis: {stem.replace('_', ' ').title()}",
            "Academic Research",
            "academic",
            f"DBSCAN density clustering evaluation ({stem.replace('_', ' ')}).",
        )

    if "anomaly" in rel_str:
        return (
            f"Multivariate Anomaly Analysis: {stem.replace('_', ' ').title()}",
            "Academic Research",
            "academic",
            f"Isolation Forest feature outlier distribution ({stem.replace('_', ' ')}).",
        )

    if "classification" in rel_str:
        return (
            f"Classification Experiment: {stem.replace('_', ' ').title()}",
            "Academic Research",
            "academic",
            f"Derived high-loss classification metric and confusion matrix analysis.",
        )

    return (
        stem.replace("_", " ").title(),
        "Visualizations",
        "visualization",
        f"ClaimWise visualization artifact ({filename}).",
    )


def build_visualization_inventory() -> List[Dict[str, Any]]:
    """Scan and index all visual artifacts in Outputs/."""
    inventory = []
    seen = set()

    for file_path in sorted(OUTPUTS_DIR.rglob("*")):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in [".png", ".jpg", ".jpeg", ".svg"]:
            continue

        rel_path = file_path.relative_to(PROJECT_ROOT)
        rel_str = str(rel_path)

        if rel_str in seen:
            continue
        seen.add(rel_str)

        # Build clean web path relative to Outputs
        rel_to_outputs = file_path.relative_to(OUTPUTS_DIR)
        web_src = str(rel_to_outputs).replace("\\", "/")

        title, topic, page, desc = _classify_visualization_artifact(rel_str, file_path.name)

        inventory.append({
            "filename": file_path.name,
            "relative_path": rel_str,
            "web_src": web_src,
            "title": title,
            "topic": topic,
            "page": page,
            "description": desc,
            "size_bytes": file_path.stat().st_size,
        })

    inventory.sort(key=lambda x: (x["page"], x["topic"], x["title"]))
    return inventory


def get_visualization_inventory(use_cache: bool = True) -> List[Dict[str, Any]]:
    """Retrieve the cached visualization inventory, building it if needed."""
    global _VIS_INVENTORY_CACHE
    if not use_cache or _VIS_INVENTORY_CACHE is None:
        _VIS_INVENTORY_CACHE = build_visualization_inventory()
    return _VIS_INVENTORY_CACHE


def resolve_safe_csv_path(relative_path: str) -> Optional[Path]:
    """Validate and resolve a relative path to an approved, indexed CSV file."""
    try:
        clean_rel = str(relative_path).strip().replace("\\", "/")
        # Reject traversal attempts
        if ".." in clean_rel or clean_rel.startswith("/"):
            return None

        # Verify that the file is in our indexed CSV inventory (allowing optional Outputs/ prefix)
        inv = get_csv_inventory()
        valid_paths = {item["relative_path"] for item in inv}
        if clean_rel not in valid_paths:
            if f"Outputs/{clean_rel}" in valid_paths:
                clean_rel = f"Outputs/{clean_rel}"
            elif clean_rel.startswith("Outputs/") and clean_rel[8:] in valid_paths:
                clean_rel = clean_rel[8:]
            else:
                return None

        target = (PROJECT_ROOT / clean_rel).resolve()
        if not _is_safe_child_path(target):
            return None

        if target.exists() and target.is_file():
            return target
        return None
    except Exception:
        return None


def get_csv_preview(
    relative_path: str,
    page: int = 1,
    page_size: int = 20,
    search_query: str = "",
) -> Dict[str, Any]:
    """Return a safe, paginated, searchable preview slice of an approved CSV file."""
    file_path = resolve_safe_csv_path(relative_path)
    if file_path is None:
        return {
            "success": False,
            "error": "The requested CSV file does not exist or access is restricted.",
            "columns": [],
            "rows": [],
            "page": 1,
            "page_size": page_size,
            "total_rows": 0,
            "total_pages": 0,
        }

    try:
        page = max(1, int(page))
        page_size = min(100, max(5, int(page_size)))

        file_size = file_path.stat().st_size
        search_query = str(search_query or "").strip().lower()
        clean_rel = str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/")

        # Retrieve authoritative total rows from inventory or quick line count
        inv = {item["relative_path"]: item for item in get_csv_inventory()}
        known_rows = inv.get(clean_rel, {}).get("rows")
        if known_rows is not None:
            total_rows = int(known_rows)
        else:
            with open(file_path, "rb") as f:
                total_rows = max(0, sum(1 for _ in f) - 1)

        if not search_query:
            # Memory-safe bounded read directly using skiprows and nrows
            skip = (page - 1) * page_size
            df_slice = pd.read_csv(file_path, skiprows=range(1, skip + 1) if skip > 0 else None, nrows=page_size)
            columns = list(df_slice.columns)
            rows = df_slice.fillna("").astype(str).to_dict(orient="records")
            total_pages = max(1, (total_rows + page_size - 1) // page_size)
            return {
                "success": True,
                "columns": columns,
                "rows": rows,
                "page": page,
                "page_size": page_size,
                "total_rows": total_rows,
                "filtered_rows": total_rows,
                "total_pages": total_pages,
                "filename": file_path.name,
                "relative_path": clean_rel,
                "file_size_formatted": _format_size(file_size),
            }

        # Search filter branch: stream chunks to collect all matching rows
        header_df = pd.read_csv(file_path, nrows=0)
        columns = list(header_df.columns)

        matched_rows: list[dict] = []
        for chunk in pd.read_csv(file_path, chunksize=2000):
            mask = chunk.astype(str).apply(lambda row: row.str.lower().str.contains(search_query, regex=False).any(), axis=1)
            filtered = chunk[mask]
            for record in filtered.to_dict(orient="records"):
                clean_rec = {k: ("" if pd.isna(v) else str(v)) for k, v in record.items()}
                matched_rows.append(clean_rec)
                if len(matched_rows) >= 500:  # Cap search results at 500 matches for web responsiveness
                    break
            if len(matched_rows) >= 500:
                break

        filtered_rows = len(matched_rows)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_rows = matched_rows[start_idx:end_idx]
        total_pages = max(1, (filtered_rows + page_size - 1) // page_size)

        return {
            "success": True,
            "columns": columns,
            "rows": page_rows,
            "page": page,
            "page_size": page_size,
            "total_rows": total_rows,
            "filtered_rows": filtered_rows,
            "total_pages": total_pages,
            "filename": file_path.name,
            "relative_path": clean_rel,
            "file_size_formatted": _format_size(file_size),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to preview CSV: {exc}",
            "columns": [],
            "rows": [],
            "page": 1,
            "page_size": page_size,
            "total_rows": 0,
            "total_pages": 0,
        }


# =========================================================================
# RECURSIVE OUTPUTS DIRECTORY INVENTORY (PHASE 2 & PHASE 8)
# =========================================================================

_ALL_OUTPUTS_CACHE: Optional[List[Dict[str, Any]]] = None

TOPIC_CONFIG = {
    "dataset_preprocessing": {
        "title": "Dataset & Preprocessing",
        "badge_class": "badge-dataset",
        "order": 1,
        "description": "Raw dataset structure, stage dimensions progression, and preprocessing pipeline diagrams.",
    },
    "eda": {
        "title": "Exploratory Data Analysis",
        "badge_class": "badge-eda",
        "order": 2,
        "description": "Continuous feature distributions, target density, categorical frequency plots, and missing value heatmaps.",
    },
    "correlation_features": {
        "title": "Correlation & Feature Analysis",
        "badge_class": "badge-corr",
        "order": 3,
        "description": "Bivariate correlation heatmaps, feature-to-loss rankings, and continuous feature boxplots.",
    },
    "model_comparison": {
        "title": "Model Training & Comparison",
        "badge_class": "badge-comparison",
        "order": 4,
        "description": "Multi-model benchmarking leaderboards, comparative error bar charts, and combined residual analyses.",
    },
    "model_evaluation": {
        "title": "Model Evaluation & Diagnostics",
        "badge_class": "badge-models",
        "order": 5,
        "description": "Individual model diagnostics: actual vs predicted scatter plots, residual curves, feature importances, and metric sheets.",
    },
    "prediction_diagnostics": {
        "title": "Prediction Diagnostics",
        "badge_class": "badge-prediction",
        "order": 6,
        "description": "Sample inference records, full test partition prediction arrays, and batch validation logs.",
    },
    "clustering_unsupervised": {
        "title": "Clustering & Unsupervised Learning",
        "badge_class": "badge-clustering",
        "order": 7,
        "description": "K-Means (k-means++), Hierarchical Agglomerative clustering, DBSCAN density analysis, PCA, and t-SNE manifold projections.",
    },
    "academic_research": {
        "title": "Academic Research & Extensions",
        "badge_class": "badge-academic",
        "order": 8,
        "description": "Principal Component Analysis (PCA), binary classification benchmarks, cross-validation runs, and anomaly detection.",
    },
    "other_outputs": {
        "title": "Other Project Outputs",
        "badge_class": "badge-other",
        "order": 9,
        "description": "Supplementary data exports, logs, and technical artifacts.",
    },
}


def _classify_output_file(rel_to_outputs: Path, file_size: int) -> Tuple[str, str, str, str]:
    """Classify an output file into topic_key, topic_title, file_type, and description."""
    rel_str = str(rel_to_outputs).replace("\\", "/")
    filename = rel_to_outputs.name
    stem = rel_to_outputs.stem
    ext = rel_to_outputs.suffix.lower()

    # Determine file type label
    if ext in [".png", ".jpg", ".jpeg", ".svg", ".webp"]:
        file_type = "Image / Plot"
    elif ext == ".csv":
        file_type = "CSV Dataset"
    elif ext == ".json":
        file_type = "JSON Data"
    elif ext in [".txt", ".log", ".md"]:
        file_type = "Document / Text"
    else:
        file_type = f"{ext[1:].upper()} File" if ext else "Binary File"

    # Topic classification based on folder structure
    parts = rel_to_outputs.parts
    first_dir = parts[0] if len(parts) > 1 else ""

    if first_dir in ["Preprocessing", "Data_Inspection"]:
        topic_key = "dataset_preprocessing"
        desc = f"Preprocessing data artifact: {filename}."
        if "pipeline" in filename:
            desc = "Authoritative 10-stage ClaimWise feature engineering and scaling architecture flow."
        elif "composition" in filename:
            desc = "Dataset schema composition: 116 categorical vs 14 continuous numeric features."
        elif "summary" in filename:
            desc = "Stage-by-stage dimensions, missing count, and data leakage audit metrics."

    elif first_dir == "EDA_Analysis_outputs":
        topic_key = "eda"
        if "target" in filename:
            desc = "Empirical distribution density of target claim loss variable."
        elif "correlation" in filename:
            desc = "Pearson correlation heatmap across all 14 continuous numerical features."
        elif "missing" in filename:
            desc = "Missing data density heatmap confirming 0 null values."
        elif "feature_distribution" in filename:
            desc = f"Continuous feature density distribution for {stem.replace('02_feature_distribution_', '')}."
        elif "boxplot" in filename:
            desc = f"Outlier quartile boxplot for {stem.replace('03_boxplot_', '')}."
        elif "countplot" in filename:
            desc = f"Category level frequency counts for {stem.replace('05_countplot_', '')}."
        else:
            desc = f"Exploratory data analysis artifact: {filename}."

    elif first_dir == "Boxplots_correlation":
        topic_key = "correlation_features"
        if "heatmap" in filename:
            desc = "Correlation matrix heatmap verifying feature colinearities."
        elif "boxplot" in filename:
            desc = f"Continuous predictor distribution and quartile ranges ({stem})."
        elif "correlation" in filename:
            desc = "Tabular correlation coefficients between continuous features and target loss."
        else:
            desc = f"Feature correlation and distribution analysis: {filename}."

    elif first_dir in ["Model_Comparison", "CrossValidation"]:
        topic_key = "model_comparison"
        if "bar_chart" in filename:
            desc = "Comprehensive comparative benchmark bar chart across evaluated regression models."
        elif "actual_vs_predicted" in filename:
            desc = "Multi-model comparative scatter plots of predicted versus actual claim loss."
        elif "residuals" in filename:
            desc = "Comparative residual error distribution curves across tested models."
        elif "comparison.csv" in filename:
            desc = "Full 13-model regression comparison table with MAE, MSE, RMSE, R², EV, MedAE, and MAPE."
        elif "comparison.json" in filename:
            desc = "Structured machine-readable benchmark JSON serialization of all 13 models."
        else:
            desc = f"Model training and cross-validation comparison artifact: {filename}."

    elif first_dir in [
        "Hist_Gradient_Boosting_Regressor",
        "Extra_Trees_Regressor",
        "Random_Forest_Regressor",
        "Gradient_Boosting_Regressor",
        "Ridge_Regression",
        "Linear_Regression",
        "Lasso_Regression",
        "Elastic_Net_Regression",
        "Support_Vector_Regression_Linear",
        "Decision_Tree_Regressor",
        "K_Nearest_Neighbors_Regressor",
        "Dummy_Regressor_Median_Baseline",
        "AdaBoost_Regressor",
    ]:
        topic_key = "model_evaluation"
        model_name = first_dir.replace("_", " ")
        if "actual_vs_predicted" in filename:
            desc = f"Actual versus predicted loss evaluation scatter plot for {model_name} on 10,000 test claims."
        elif "residuals" in filename:
            desc = f"Residual error distribution and homoscedasticity plot for {model_name}."
        elif "importance" in filename:
            desc = f"Top feature importances ranked by split impurity reduction in {model_name}."
        elif "coefficients" in filename:
            desc = f"Feature coefficients ranked by absolute fitted magnitude in {model_name}."
        elif filename == "metrics.csv":
            desc = f"Computed out-of-sample regression metrics (MAE, RMSE, R², etc.) for {model_name}."
        elif filename == "predictions.csv":
            desc = f"10,000 held-out test predictions and residual values generated by {model_name}."
        elif "cross_validation" in filename:
            desc = f"Per-fold validation metrics and iteration counts for {model_name}."
        else:
            desc = f"Diagnostic evaluation artifact for {model_name}: {filename}."

    elif first_dir == "Final_Prediction":
        topic_key = "prediction_diagnostics"
        if "sample" in filename:
            desc = "Sample claim loss prediction outputs across benchmark archetypes."
        elif "test_set" in filename:
            desc = "Complete 10,000-claim test partition model predictions and ground-truth comparisons."
        else:
            desc = f"Final model inference diagnostic: {filename}."

    elif first_dir in ["clustering", "dbscan"]:
        topic_key = "clustering_unsupervised"
        if "kmeans_elbow" in filename:
            desc = "K-Means elbow curve showing within-cluster sum of squares (inertia) across k=1..10 with optimal elbow at k=2."
        elif "kmeans_silhouette" in filename:
            desc = "K-Means silhouette score analysis across candidate cluster counts k=2..8 confirming peak separation at k=2."
        elif "kmeans_clusters" in filename:
            desc = "K-Means cluster size distribution and record share across clusters (50,000 records, k=2)."
        elif "kmeans_pca" in filename:
            desc = "2D PCA projection of the 130-feature space colored by unsupervised K-Means cluster assignments."
        elif "agglomerative" in filename:
            desc = "Hierarchical Agglomerative Clustering (Ward linkage) in 2D PCA space with silhouette quality analysis."
        elif "dbscan_clusters" in filename:
            desc = "DBSCAN density-based clusters and noise points (label -1) in 2D PCA space with noise percentage."
        elif "cluster_profiles" in filename:
            desc = "Comparative cluster profiles showing claim loss distributions and standardized continuous predictor means."
        elif "pca_explained_variance" in filename:
            desc = "PCA individual explained variance ratio and cumulative variance across principal components with 80%/90% thresholds."
        elif "pca_2d" in filename:
            desc = "2D Principal Component Analysis projection colored by log1p claim severity."
        elif "tsne" in filename:
            desc = "t-SNE 2D non-linear manifold embedding on PCA-30 pre-reduction colored by K-Means cluster assignments."
        elif first_dir == "clustering":
            desc = f"Unsupervised K-Means clustering evaluation artifact ({stem.replace('kmeans_', '')})."
        elif first_dir == "dbscan":
            desc = f"DBSCAN density clustering and noise analysis ({stem.replace('dbscan_', '')})."
        else:
            desc = f"Clustering and unsupervised learning output: {filename}."

    elif first_dir in ["PCA", "classification", "anomaly"]:
        topic_key = "academic_research"
        module_name = first_dir.title()
        if first_dir == "PCA":
            desc = f"Principal Component Analysis feature variance artifact ({stem.replace('pca_', '')})."
        elif first_dir == "classification":
            desc = f"High-loss classification experiment diagnostic ({stem.replace('_', ' ')})."
        elif first_dir == "anomaly":
            desc = f"Isolation Forest multivariate anomaly detection artifact ({stem.replace('anomaly_', '')})."
        else:
            desc = f"Academic research module output: {filename}."

    elif first_dir == "Final_Model_Evaluation":
        topic_key = "model_evaluation"
        if "actual_vs_predicted" in filename:
            desc = "Actual versus predicted claim loss scatter for the production champion on the untouched test split."
        elif "residual" in filename:
            desc = "Residual diagnostics for the production champion evaluated once on the held-out test split."
        else:
            desc = f"Final untouched-test evaluation artifact: {filename}."

    elif first_dir == "ExperimentSuite":
        topic_key = "model_evaluation"
        if "tuning_results" in filename:
            desc = "Complete hyperparameter random-search log with validation scores for every tried configuration."
        elif "validation_metrics" in filename:
            desc = "Validation-split leaderboard metrics for all tuned model candidates."
        elif "final_test_metrics" in filename:
            desc = "Single untouched-test evaluation of the promoted production champion model."
        elif "log1p" in filename:
            desc = "log1p-target experiment results comparing MAE and R² trade-offs."
        elif "data_audit" in filename:
            desc = "Measured dataset audit: skewness, variance concentration, duplicate features, and split-shift diagnostics."
        elif filename.endswith(".log"):
            desc = f"Experiment-suite execution log: {filename}."
        else:
            desc = f"Experiment-suite artifact: {filename}."

    else:
        topic_key = "other_outputs"
        desc = f"ClaimWise project output file: {filename}."

    topic_title = TOPIC_CONFIG.get(topic_key, {}).get("title", "Other Outputs")
    return topic_key, topic_title, file_type, desc


def build_all_outputs_inventory() -> List[Dict[str, Any]]:
    """Recursively scan Outputs/ and index every valid file with metadata.

    Guarantees:
    - Recursively traverses all subdirectories.
    - Excludes hidden files (.DS_Store, .git, etc.).
    - Never executes or loads pickle files.
    - Computes safe web paths, file sizes, and classification.
    - Sorts deterministically by topic order, then relative path.
    """
    inventory: List[Dict[str, Any]] = []

    if not OUTPUTS_DIR.exists() or not OUTPUTS_DIR.is_dir():
        return inventory

    for file_path in sorted(OUTPUTS_DIR.rglob("*")):
        if not file_path.is_file():
            continue
        # Skip hidden files
        if file_path.name.startswith(".") or file_path.name.endswith("~"):
            continue

        try:
            rel_to_outputs = file_path.relative_to(OUTPUTS_DIR)
            rel_to_project = file_path.relative_to(PROJECT_ROOT)
            web_src = str(rel_to_outputs).replace("\\", "/")
            project_rel_path = str(rel_to_project).replace("\\", "/")

            ext = file_path.suffix.lower()
            stat = file_path.stat()
            file_size = stat.st_size
            size_formatted = _format_size(file_size)
            modified_utc = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )

            is_image = ext in [".png", ".jpg", ".jpeg", ".svg", ".webp"]
            is_csv = ext == ".csv"
            is_json = ext == ".json"
            is_text = ext in [".txt", ".log", ".md"]

            topic_key, topic_title, file_type, description = _classify_output_file(rel_to_outputs, file_size)

            item: Dict[str, Any] = {
                "filename": file_path.name,
                "stem": file_path.stem,
                "extension": ext,
                "relative_path": project_rel_path,
                "outputs_rel_path": web_src,
                "web_src": web_src,
                "size_bytes": file_size,
                "size_formatted": size_formatted,
                "modified_utc": modified_utc,
                "topic_key": topic_key,
                "topic": topic_title,
                "topic_order": TOPIC_CONFIG.get(topic_key, {}).get("order", 99),
                "badge_class": TOPIC_CONFIG.get(topic_key, {}).get("badge_class", "badge-other"),
                "file_type": file_type,
                "description": description,
                "is_image": is_image,
                "is_csv": is_csv,
                "is_json": is_json,
                "is_text": is_text,
                "can_preview": is_image or is_csv or is_json or is_text,
                "can_download": True,
            }

            # If it's a CSV, check if we have row/col counts from CSV inventory
            if is_csv:
                inv_map = {c["relative_path"]: c for c in get_csv_inventory()}
                csv_info = inv_map.get(project_rel_path)
                if csv_info:
                    item["rows"] = csv_info.get("rows", 0)
                    item["columns"] = csv_info.get("columns", 0)
                else:
                    item["rows"] = None
                    item["columns"] = None

            inventory.append(item)
        except Exception:
            continue

    # Sort deterministically by topic order, then directory, then filename
    inventory.sort(key=lambda x: (x["topic_order"], x["web_src"]))
    return inventory


def get_outputs_inventory(use_cache: bool = True) -> List[Dict[str, Any]]:
    """Retrieve the cached complete outputs inventory, scanning if needed."""
    global _ALL_OUTPUTS_CACHE
    if not use_cache or _ALL_OUTPUTS_CACHE is None:
        _ALL_OUTPUTS_CACHE = build_all_outputs_inventory()
    return _ALL_OUTPUTS_CACHE


def get_outputs_summary() -> Dict[str, Any]:
    """Return aggregated counts and topic breakdowns for all discovered outputs."""
    items = get_outputs_inventory(use_cache=True)
    images_count = sum(1 for x in items if x["is_image"])
    csvs_count = sum(1 for x in items if x["is_csv"])
    other_count = sum(1 for x in items if not (x["is_image"] or x["is_csv"]))

    topics_summary: List[Dict[str, Any]] = []
    for key, conf in sorted(TOPIC_CONFIG.items(), key=lambda t: t[1]["order"]):
        topic_items = [x for x in items if x["topic_key"] == key]
        if topic_items:
            topics_summary.append({
                "key": key,
                "title": conf["title"],
                "badge_class": conf["badge_class"],
                "description": conf["description"],
                "count": len(topic_items),
                "image_count": sum(1 for x in topic_items if x["is_image"]),
                "csv_count": sum(1 for x in topic_items if x["is_csv"]),
                "items": topic_items,
            })

    return {
        "total_files": len(items),
        "total_images": images_count,
        "total_csvs": csvs_count,
        "total_others": other_count,
        "topics": topics_summary,
    }


def resolve_safe_output_path(web_src: str) -> Optional[Path]:
    """Validate that requested relative path is strictly within OUTPUTS_DIR.

    Handles duplicated prefixes (e.g. Outputs/Outputs/...), leading slashes,
    and case-insensitive matching while strictly enforcing directory containment.
    """
    try:
        clean = str(web_src).strip().replace("\\", "/")
        if ".." in clean:
            return None
        clean = clean.lstrip("/")
        # Strip repeated leading Outputs/ or outputs/ prefixes
        while clean.lower().startswith("outputs/"):
            clean = clean[len("outputs/"):].lstrip("/")
        if not clean:
            return None
        target = (OUTPUTS_DIR / clean).resolve()
        # Verify strict containment inside OUTPUTS_DIR
        if OUTPUTS_DIR.resolve() not in target.parents:
            return None
        if target.exists() and target.is_file():
            return target

        # Case-insensitive path resolution fallback within OUTPUTS_DIR
        parts = [p for p in clean.split("/") if p]
        curr = OUTPUTS_DIR
        for part in parts:
            if not curr.is_dir():
                return None
            matched = None
            part_lower = part.lower()
            for child in curr.iterdir():
                if child.name.lower() == part_lower:
                    matched = child
                    break
            if matched is None:
                return None
            curr = matched
        if curr.is_file() and OUTPUTS_DIR.resolve() in curr.resolve().parents:
            return curr.resolve()
        return None
    except Exception:
        return None


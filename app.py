from pathlib import Path
import sys

import joblib
import pandas as pd
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "Models"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
DATASET_DIR = PROJECT_ROOT / "Dataset"
NOTEBOOKS_DIR = PROJECT_ROOT / "Notebooks"
SRC_DIR = PROJECT_ROOT / "Src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prediction_preprocessor import (  # noqa: E402  (Src must be on the path first)
    load_preprocessor,
    out_of_range as find_out_of_range,
    transform as transform_raw_row,
)

# Display order of the five regression models on the Models page.
# Images are the exact files produced by the ClaimWise notebooks / training scripts;
# nothing is generated or renamed for the website.
MODEL_GALLERY = (
    {
        "name": "Linear Regression",
        "slug": "linear-regression",
        "folder": "Linear_Regression",
        "feature_note": (
            "The coefficient chart ranks features by the absolute size of their fitted linear coefficient — "
            "the larger the absolute coefficient, the more that feature moves the predicted loss for a "
            "one-unit change in it."
        ),
        "summary": "Ordinary least-squares baseline that predicts loss as a weighted combination of the 130 encoded features.",
        "interpretation": (
            "As the linear baseline it explains about half of the variance on the test rows (R2 0.4826). "
            "Claim loss is right-skewed and the feature effects are not linear, so the ensemble models "
            "reach a lower error with the same inputs."
        ),
        "images": (
            ("linear_regression_actual_vs_predicted.png", "Linear Regression — Actual vs Predicted"),
            ("linear_regression_residuals.png", "Linear Regression — Residual Analysis"),
            ("linear_regression_top_coefficients.png", "Linear Regression — Top 20 Features by Absolute Coefficient"),
        ),
    },
    {
        "name": "Decision Tree Regressor",
        "slug": "decision-tree",
        "folder": "Decision_Tree_Regressor",
        "feature_note": (
            "Decision-tree feature importance is the total reduction in split impurity that the feature "
            "produced across the tree, normalised so all importances sum to 1."
        ),
        "summary": "A single decision tree that recursively splits the feature space to reduce squared prediction error.",
        "interpretation": (
            "One tree predicts a piece-wise constant value, which is a hard fit for a continuous, right-skewed "
            "target. It records the highest RMSE (2638.94) and the lowest R2 (0.1431) of the five models."
        ),
        "images": (
            ("decision_tree_actual_vs_predicted.png", "Decision Tree Regressor — Actual vs Predicted"),
            ("decision_tree_residuals.png", "Decision Tree Regressor — Residual Analysis"),
            ("decision_tree_feature_importance.png", "Decision Tree Regressor — Top 20 Feature Importances"),
        ),
    },
    {
        "name": "Random Forest Regressor",
        "slug": "random-forest",
        "folder": "Random_Forest_Regressor",
        "feature_note": (
            "Feature importance shows which input variables contributed most to the Random Forest model's "
            "predictions. The bar chart is the top 20 of the full list saved to "
            "Outputs/Random_Forest_Regressor/feature_importance.csv — cat80 and cont7 dominate the ranking."
        ),
        "summary": "An ensemble of decision trees fitted on bootstrap samples with random feature subsets, averaging their predictions.",
        "interpretation": (
            "Averaging many trees reduces variance while keeping the fit flexible, which gives the second "
            "lowest RMSE (1984.41) and the second highest R2 (0.5154) on the 10,000 test rows — it was the "
            "best model until the histogram gradient boosting model from the provided code was added."
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
            "Gradient Boosting feature importance is the total reduction in split impurity attributed to "
            "that feature across the boosting stages, normalised to sum to 1."
        ),
        "summary": "Trees are added sequentially, each new stage correcting the residual errors of the previous ensemble.",
        "interpretation": (
            "Gradient Boosting lands very close to the forest (RMSE 2005.26, R2 0.5052) and ranks third, "
            "about 21 RMSE points behind it. Its feature-importance profile also highlights cat80 and cat79."
        ),
        "images": (
            ("gradient_boosting_actual_vs_predicted.png", "Gradient Boosting Regressor — Actual vs Predicted"),
            ("gradient_boosting_residuals.png", "Gradient Boosting Regressor — Residual Analysis"),
            ("gradient_boosting_feature_importance.png", "Gradient Boosting Regressor — Top 20 Feature Importances"),
        ),
    },
    {
        "name": "Hist Gradient Boosting Regressor",
        "slug": "hist-gradient-boosting",
        "folder": "Hist_Gradient_Boosting_Regressor",
        "feature_note": (
            "This model has no separate importance chart; it is a histogram gradient boosting regressor, "
            "so the diagnostic charts below (actual vs predicted and residuals) are its model report. "
            "Its per-fold training history is saved in Outputs/Hist_Gradient_Boosting_Regressor/"
            "cross_validation_folds.csv."
        ),
        "summary": (
            "Histogram-based gradient boosting with early stopping — the algorithm adapted from the "
            "provided Google Drive code, mapped onto sklearn's HistGradientBoostingRegressor."
        ),
        "interpretation": (
            "It gives the lowest error of the five models (RMSE 1921.02, R2 0.5459) after 830 boosting "
            "iterations chosen by 3-fold cross-validation with early stopping, so Models/best_model.pkl "
            "currently holds this model."
        ),
        "images": (
            (
                "hist_gradient_boosting_regressor_actual_vs_predicted.png",
                "Hist Gradient Boosting Regressor — Actual vs Predicted",
            ),
            (
                "hist_gradient_boosting_regressor_residuals.png",
                "Hist Gradient Boosting Regressor — Residual Analysis",
            ),
        ),
    },
)

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
    """Read the real model-comparison CSV generated by train_models.py."""

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

        gallery.append(
            {
                "name": model["name"],
                "slug": model["slug"],
                "summary": model["summary"],
                "interpretation": model["interpretation"],
                "feature_note": model.get("feature_note"),
                "metrics": metrics,
                "images": images,
                "is_best": bool(best_name) and model["name"] == best_name,
            }
        )
    return gallery, best_name


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
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/dataset")
def dataset():
    return render_template(
        "dataset.html",
        stats=load_dataset_stats(),
        target_distribution={
            "src": "Data_Inspection/claimwise_raw_target_distribution.png",
            "alt": "Distribution of Claim Loss (raw dataset)",
            "caption": "Distribution of Claim Loss (raw dataset) — generated by 01_ClaimWise_Data_Inspection.ipynb",
        },
    )

@app.route("/preprocessing")
def preprocessing():
    return render_template(
        "preprocessing.html",
        stats=load_dataset_stats(),
        stage_rows=load_stage_rows(),
        leakage=load_leakage_stats(),
    )

@app.route("/visualization")
def visualization():
    return render_template("visualization.html")

@app.route("/models")
def models():
    model_rows, best_name = load_model_results()
    gallery, _ = load_model_gallery()
    leakage = load_leakage_stats()
    return render_template(
        "models.html",
        model_rows=model_rows,
        best_name=best_name,
        model_ready=bool(model_rows),
        gallery=gallery,
        comparison_chart=load_comparison_chart(),
        leakage=leakage,
        notebooks=load_notebooks(),
    )

@app.route("/academic")
def academic():
    """Academic ML extension: PCA, classification, clustering, DBSCAN, anomalies, 5-fold CV."""

    sections = load_academic_data()
    return render_template(
        "academic.html",
        sections=sections,
        notebooks=[nb for nb in load_notebooks() if nb["number"] in {"11", "12", "13", "14", "15", "16", "17"}],
    )

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    """Two prediction modes on one page.

    1. ``manual`` — the real feature form: the user enters raw ClaimWise values
       (cont1-cont14 numbers, cat1-cat116 categories), the saved preprocessing
       transforms them exactly like training did, and the saved best model
       predicts the loss.
    2. ``test`` — the original demo: pick a prepared test row by number.
    """

    model_rows, best_name = load_model_results()
    result = None
    manual = None
    error = None
    warnings = []
    x_test, y_test = load_prediction_data()

    try:
        form_groups = build_form_groups()
    except Exception as exc:  # pragma: no cover - preprocessing artifact missing
        form_groups = []
        if request.method == "GET":
            error = f"Prediction form could not be built: {exc}"

    if request.method == "POST":
        mode = request.form.get("mode", "manual")
        if not model_is_ready():
            error = "Train the regression models first. The saved best model or test data is unavailable."
        elif mode == "test":
            if x_test is None or y_test is None:
                error = "The prepared test data is unavailable."
            else:
                try:
                    row_index = int(request.form.get("row_index", ""))
                    if row_index < 0 or row_index >= len(x_test):
                        raise ValueError(f"Choose a test-row number from 0 to {len(x_test) - 1}.")

                    model = joblib.load(MODELS_DIR / "best_model.pkl")
                    row = x_test.iloc[[row_index]]
                    expected_columns = list(getattr(model, "feature_names_in_", row.columns))
                    if list(row.columns) != expected_columns:
                        raise ValueError("The selected test row does not match the trained model feature structure.")

                    predicted_loss = float(model.predict(row)[0])
                    result = {
                        "row_index": row_index,
                        "predicted_loss": predicted_loss,
                        "actual_loss": float(y_test.iloc[row_index]),
                        "feature_count": row.shape[1],
                    }
                except (TypeError, ValueError) as exc:
                    error = str(exc)
                except Exception as exc:  # pragma: no cover - protects the web page from model-load errors
                    error = f"Prediction could not be completed: {exc}"
        else:
            try:
                preprocessor = get_preprocessor()
                missing = [
                    col
                    for col in preprocessor["feature_order"]
                    if str(request.form.get(col, "")).strip() == ""
                ]
                if missing:
                    raise ValueError(
                        f"{len(missing)} feature values are empty (first: {missing[0]}). "
                        "Every field of the form is needed."
                    )

                raw_row = pd.DataFrame(
                    [
                        {
                            col: str(request.form.get(col, "")).strip()
                            for col in preprocessor["feature_order"]
                        }
                    ]
                )
                features = transform_raw_row(raw_row, preprocessor)

                model = joblib.load(MODELS_DIR / "best_model.pkl")
                expected_columns = list(
                    getattr(model, "feature_names_in_", features.columns)
                )
                if list(features.columns) != expected_columns:
                    raise ValueError(
                        "The form columns do not match the trained model feature structure."
                    )

                predicted_loss = float(model.predict(features)[0])
                warnings = find_out_of_range(raw_row, preprocessor)
                manual = {
                    "predicted_loss": predicted_loss,
                    "feature_count": int(features.shape[1]),
                    "model": best_name,
                    "warnings": warnings,
                    "changed": sum(
                        1
                        for col in preprocessor["continuous"]
                        if str(raw_row[col].iloc[0])
                        != f"{preprocessor['continuous'][col]['median']:g}"
                    ),
                }
                if y_test is not None:
                    manual["test_mean_loss"] = float(y_test.mean())
            except (TypeError, ValueError) as exc:
                error = str(exc)
            except Exception as exc:  # pragma: no cover - protects the web page from model-load errors
                error = f"Prediction could not be completed: {exc}"

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
    )

@app.route("/dashboard")
def dashboard():
    model_rows, best_name = load_model_results()
    best_metrics = model_rows[0] if model_rows else None
    return render_template(
        "dashboard.html",
        model_rows=model_rows,
        best_name=best_name,
        best_metrics=best_metrics,
        model_ready=bool(model_rows),
    )

@app.route("/reports")
def reports():
    model_rows, best_name = load_model_results()
    best_metrics = model_rows[0] if model_rows else None
    return render_template(
        "reports.html",
        model_rows=model_rows,
        best_name=best_name,
        best_metrics=best_metrics,
        model_ready=bool(model_rows),
    )

@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/outputs/<path:filename>", endpoint="outputs")
def output_file(filename):
    """Serve the charts and figures already generated by the ML pipeline."""

    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/notebooks/<path:filename>")
def notebook_file(filename):
    """Download the real Jupyter notebook used for this stage of the project."""

    return send_from_directory(NOTEBOOKS_DIR, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, port=5000)

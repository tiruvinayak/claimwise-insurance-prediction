"""Dataset management, validation, and schema compatibility inspection for ClaimWise.

Handles:
- Safe file uploads (saved in `uploads/`, never overwriting `Dataset/`).
- CSV validation (extension, size, encoding, readability, empty/malformed checks).
- Dataset audit (shape, missing counts, duplicate counts, dtypes, column summary).
- Data preview (first N rows).
- Model schema compatibility verification:
  Checks uploaded data against the trained ClaimWise regression model and
  the fitted preprocessing pipeline (feature names, counts, types, categories, target).
"""

from __future__ import annotations

import io
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Dataset"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
MODELS_DIR = PROJECT_ROOT / "Models"
RAW_DATASET_PATH = DATASET_DIR / "01_raw_claimwise_50000.csv"

# Configuration
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {".csv"}
TARGET_COLUMN_DEFAULT = "loss"
ID_COLUMN = "id"


def ensure_upload_dir() -> Path:
    """Ensure upload directory exists and is separate from the raw dataset."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOADS_DIR


def is_allowed_file(filename: str) -> bool:
    """Verify that file has an allowed CSV extension."""
    if not filename:
        return False
    _, ext = os.path.splitext(filename)
    return ext.lower() in ALLOWED_EXTENSIONS


def read_csv_safe(file_path_or_buffer: Any) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Read CSV with encoding detection and safe error handling.

    Tries utf-8, then utf-8-sig, then latin-1.
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_error = ""

    for enc in encodings:
        try:
            if hasattr(file_path_or_buffer, "seek"):
                file_path_or_buffer.seek(0)
            df = pd.read_csv(file_path_or_buffer, encoding=enc)
            if df.empty:
                return None, "The uploaded CSV file is empty."
            return df, None
        except UnicodeDecodeError as exc:
            last_error = f"Encoding error ({enc}): {exc}"
        except pd.errors.EmptyDataError:
            return None, "The uploaded CSV file contains no data."
        except pd.errors.ParserError as exc:
            return None, f"Malformed CSV structure: {exc}"
        except Exception as exc:
            last_error = str(exc)

    return None, f"Failed to parse CSV file: {last_error}"


def save_uploaded_file(file_storage) -> Tuple[Optional[Path], Optional[str]]:
    """Save an uploaded file safely to the uploads directory.

    Never touches Dataset/01_raw_claimwise_50000.csv.
    """
    if not file_storage or not file_storage.filename:
        return None, "No file was selected for upload."

    original_filename = file_storage.filename
    if not is_allowed_file(original_filename):
        return None, f"Invalid file format: '{original_filename}'. Only CSV (.csv) files are permitted."

    # Check file size safely
    file_storage.seek(0, os.SEEK_END)
    size_bytes = file_storage.tell()
    file_storage.seek(0)

    if size_bytes == 0:
        return None, "The uploaded file is empty (0 bytes)."
    if size_bytes > MAX_FILE_SIZE_BYTES:
        return None, f"File exceeds maximum allowed size ({MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB)."

    ensure_upload_dir()
    clean_name = secure_filename(original_filename)
    if not clean_name:
        clean_name = "uploaded_dataset.csv"

    # Prefix with uploaded_ to prevent collisions
    target_path = UPLOADS_DIR / f"current_{clean_name}"
    try:
        file_storage.save(str(target_path))
        return target_path, None
    except Exception as exc:
        return None, f"Failed to save uploaded file: {exc}"


def get_current_upload_path() -> Optional[Path]:
    """Find currently active uploaded file in uploads/ if one exists."""
    if not UPLOADS_DIR.exists():
        return None
    files = list(UPLOADS_DIR.glob("current_*.csv"))
    if files:
        # Return most recently modified
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0]
    return None


def clear_uploaded_files() -> None:
    """Remove current uploads from uploads/ directory."""
    if UPLOADS_DIR.exists():
        for f in UPLOADS_DIR.glob("current_*.csv"):
            try:
                f.unlink()
            except OSError:
                pass


def audit_dataframe(
    df: pd.DataFrame,
    file_name: str = "",
    file_size_bytes: int = 0,
    selected_target: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculate descriptive metadata and health statistics for a dataset."""
    rows, cols = df.shape
    total_cells = rows * cols
    missing_by_col = df.isnull().sum()
    total_missing = int(missing_by_col.sum())
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0.0

    duplicate_rows = int(df.duplicated().sum())

    # Column level statistics
    columns_info = []
    for col in df.columns:
        col_series = df[col]
        missing_count = int(col_series.isnull().sum())
        unique_count = int(col_series.nunique(dropna=True))
        dtype_str = str(col_series.dtype)

        # Categorize column
        if pd.api.types.is_numeric_dtype(col_series):
            kind = "numeric"
        else:
            kind = "categorical"

        columns_info.append({
            "name": str(col),
            "dtype": dtype_str,
            "kind": kind,
            "missing_count": missing_count,
            "missing_pct": round((missing_count / rows * 100), 2) if rows > 0 else 0.0,
            "unique_count": unique_count,
            "sample_values": [str(x) for x in col_series.dropna().iloc[:3].tolist()],
        })

    # Data preview: first 8 rows converted to dict
    preview_df = df.head(8)
    preview_rows = preview_df.to_dict(orient="records")

    # Detect candidate target column
    has_loss_target = TARGET_COLUMN_DEFAULT in df.columns
    if selected_target and selected_target in df.columns:
        detected_target = selected_target
    elif has_loss_target:
        detected_target = TARGET_COLUMN_DEFAULT
    else:
        detected_target = df.columns[-1] if cols > 0 else None

    return {
        "filename": file_name,
        "filesize_kb": round(file_size_bytes / 1024, 2) if file_size_bytes > 0 else 0,
        "rows": rows,
        "columns": cols,
        "column_names": [str(c) for c in df.columns],
        "total_missing": total_missing,
        "missing_pct": round(missing_pct, 2),
        "columns_with_missing": int((missing_by_col > 0).sum()),
        "duplicate_rows": duplicate_rows,
        "has_loss_target": has_loss_target,
        "detected_target": detected_target,
        "columns_info": columns_info,
        "preview_rows": preview_rows,
    }


def verify_model_compatibility(df: pd.DataFrame, preprocessor: Dict[str, Any]) -> Dict[str, Any]:
    """Verify whether an uploaded dataset is compatible with the trained ClaimWise model.

    Checks:
    1. Feature Count: Requires 130 features (cat1-cat116, cont1-cont14).
    2. Required Feature Names: Checks exact presence of each feature in feature_order.
    3. Categorical Encoding Compatibility: Checks for unseen category levels.
    4. Numerical Types: Checks that continuous attributes are numeric and finite.
    5. Missing Values: Identifies if imputation is required.
    6. Target Availability: Identifies if 'loss' is present for model evaluation.
    7. Overall Verdict: COMPATIBLE vs INCOMPATIBLE with clear actionable details.
    """
    feature_order = preprocessor.get("feature_order", [])
    categorical_columns = preprocessor.get("categorical_columns", [])
    continuous_columns = preprocessor.get("continuous_columns", [])
    encoders = preprocessor.get("encoders", {})

    expected_feature_count = len(feature_order)  # 130
    df_cols = set(df.columns)

    checks: List[Dict[str, Any]] = []
    incompatibility_reasons: List[str] = []

    # 1. Check feature names
    missing_features = [col for col in feature_order if col not in df_cols]
    extra_features = [col for col in df.columns if col not in set(feature_order) and col not in {ID_COLUMN, TARGET_COLUMN_DEFAULT}]

    if not missing_features:
        checks.append({
            "name": "Feature Schema & Column Names",
            "passed": True,
            "status": "PASSED",
            "message": f"All {expected_feature_count} required features (116 categorical, 14 continuous) are present in the dataset.",
        })
    else:
        incompatibility_reasons.append(
            f"Missing {len(missing_features)} required features: {', '.join(missing_features[:5])}"
            + (f" and {len(missing_features) - 5} more." if len(missing_features) > 5 else ".")
        )
        checks.append({
            "name": "Feature Schema & Column Names",
            "passed": False,
            "status": "FAILED",
            "message": f"Dataset is missing {len(missing_features)} of the 130 required feature columns.",
            "details": missing_features[:10],
        })

    # 2. Check feature count
    present_required_count = expected_feature_count - len(missing_features)
    checks.append({
        "name": "Feature Count",
        "passed": len(missing_features) == 0,
        "status": "PASSED" if len(missing_features) == 0 else "FAILED",
        "message": f"Dataset contains {present_required_count}/{expected_feature_count} required model features.",
    })

    # 3. Check continuous numeric types & finiteness
    numeric_issues: List[str] = []
    for col in continuous_columns:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            nan_count = int(series.isna().sum())
            if nan_count > 0:
                numeric_issues.append(f"{col} has {nan_count} non-numeric or missing values")
            elif not np.isfinite(series.to_numpy(dtype=float)).all():
                numeric_issues.append(f"{col} contains infinite values")

    if not numeric_issues:
        checks.append({
            "name": "Continuous Variables (cont1–cont14) Data Types",
            "passed": True,
            "status": "PASSED",
            "message": "All continuous features contain valid, finite numeric values.",
        })
    else:
        incompatibility_reasons.append(f"Numeric validation issues in continuous columns: {', '.join(numeric_issues[:3])}.")
        checks.append({
            "name": "Continuous Variables (cont1–cont14) Data Types",
            "passed": False,
            "status": "FAILED",
            "message": f"{len(numeric_issues)} continuous features contain non-numeric or non-finite values.",
            "details": numeric_issues[:10],
        })

    # 4. Check categorical encoding compatibility (unseen categories)
    unseen_category_issues: List[str] = []
    unseen_summary_count = 0

    for col in categorical_columns:
        if col in df.columns and col in encoders:
            known_classes = set(encoders[col].classes_)
            actual_values = set(df[col].dropna().astype(str).str.strip().unique())
            unseen = actual_values - known_classes
            if unseen:
                unseen_summary_count += len(unseen)
                sample_unseen = list(unseen)[:3]
                unseen_category_issues.append(
                    f"{col} has {len(unseen)} unseen categories (e.g. {sample_unseen})"
                )

    if not unseen_category_issues:
        checks.append({
            "name": "Categorical Encoding Compatibility",
            "passed": True,
            "status": "PASSED",
            "message": "All categorical values match the classes fitted during production training.",
        })
    else:
        incompatibility_reasons.append(
            f"Categorical encoding conflict: {len(unseen_category_issues)} columns contain unseen categories "
            f"that cannot be encoded with the trained LabelEncoders."
        )
        checks.append({
            "name": "Categorical Encoding Compatibility",
            "passed": False,
            "status": "FAILED",
            "message": f"Found unseen category levels across {len(unseen_category_issues)} columns ({unseen_summary_count} total new categories).",
            "details": unseen_category_issues[:10],
        })

    # 5. Check missing values
    missing_count = int(df[[c for c in feature_order if c in df.columns]].isnull().sum().sum())
    if missing_count == 0:
        checks.append({
            "name": "Missing Value Integrity",
            "passed": True,
            "status": "PASSED",
            "message": "No missing values found among predictor features.",
        })
    else:
        checks.append({
            "name": "Missing Value Integrity",
            "passed": True,  # Pipeline can impute, but warning
            "status": "WARNING",
            "message": f"Found {missing_count} missing values in predictor features; pipeline median/mode imputation will be required.",
        })

    # 6. Check target column availability
    has_target = TARGET_COLUMN_DEFAULT in df.columns
    if has_target:
        checks.append({
            "name": "Target Column (loss) Availability",
            "passed": True,
            "status": "PASSED",
            "message": "Target column 'loss' is present. Model evaluation (MAE, RMSE, R²) can be performed.",
        })
    else:
        checks.append({
            "name": "Target Column (loss) Availability",
            "passed": True,
            "status": "INFO",
            "message": "Target column 'loss' is absent. Feature inference can be performed, but evaluation metrics require target labels.",
        })

    # Overall compatibility verdict
    is_compatible = len(incompatibility_reasons) == 0

    return {
        "is_compatible": is_compatible,
        "verdict": "COMPATIBLE WITH TRAINED MODEL" if is_compatible else "INCOMPATIBLE WITH TRAINED MODEL",
        "badge_class": "status-ready" if is_compatible else "status-error",
        "incompatibility_reasons": incompatibility_reasons,
        "checks": checks,
        "missing_features_count": len(missing_features),
        "extra_features_count": len(extra_features),
        "unseen_categorical_columns": len(unseen_category_issues),
    }

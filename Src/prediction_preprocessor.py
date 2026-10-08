"""Rebuild the ClaimWise preprocessing so single rows can be predicted.

The training pipeline (``Src/claimwise_preprocessing_pipeline.py``) fits one
``LabelEncoder`` per categorical column and one ``StandardScaler`` on the
continuous columns of the training rows, but it does not persist those
transformers. This module replays exactly the same steps in memory — same
cleaning, same encoders, same scaler fitted on the same
``train_test_split(..., test_size=0.20, random_state=42)`` row indices — and
verifies the result against ``Dataset/05_preprocessed_final.csv`` before
anything is used for a prediction.

The fitted transformers are then stored in ``Models/claimwise_preprocessor.pkl``
so the Flask app can transform user input without re-reading the raw dataset.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Dataset"
MODELS_DIR = PROJECT_ROOT / "Models"
RAW_PATH = DATASET_DIR / "01_raw_claimwise_50000.csv"
FINAL_PATH = DATASET_DIR / "05_preprocessed_final.csv"
ARTIFACT_PATH = MODELS_DIR / "claimwise_preprocessor.pkl"

TARGET_COLUMN = "loss"
ID_COLUMN = "id"
VERIFICATION_ATOL = 1e-9


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning as step 2 of the authoritative pipeline."""

    cleaned = df.drop_duplicates()

    numeric_cols = cleaned.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        if cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    text_cols = cleaned.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    for col in text_cols:
        cleaned[col] = cleaned[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        if cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])

    if ID_COLUMN in cleaned.columns:
        cleaned = cleaned.drop(columns=[ID_COLUMN])

    return cleaned


def build_preprocessor() -> dict:
    """Replay pipeline steps 2-4 and verify them against the saved final dataset."""

    raw = pd.read_csv(RAW_PATH)
    cleaned = _clean_frame(raw.copy())

    categorical_columns = cleaned.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()
    continuous_columns = [col for col in cleaned.columns if col.startswith("cont")]

    encoders = {}
    for col in categorical_columns:
        encoder = LabelEncoder()
        cleaned[col] = encoder.fit_transform(cleaned[col])
        encoders[col] = encoder

    row_indices = np.arange(len(cleaned))
    train_indices, _ = train_test_split(row_indices, test_size=0.20, random_state=42)
    scaler = StandardScaler()
    scaler.fit(cleaned.iloc[train_indices][continuous_columns])
    cleaned[continuous_columns] = scaler.transform(cleaned[continuous_columns])

    reference = pd.read_csv(FINAL_PATH)
    column_match = list(cleaned.columns) == list(reference.columns)
    if not column_match:
        raise AssertionError(
            "Replayed pipeline columns differ from 05_preprocessed_final.csv: "
            f"{list(cleaned.columns)[:5]}... vs {list(reference.columns)[:5]}..."
        )

    max_abs_diff = float(
        np.abs(cleaned.to_numpy(dtype=float) - reference.to_numpy(dtype=float)).max()
    )
    if not max_abs_diff <= VERIFICATION_ATOL:
        raise AssertionError(
            "Replayed preprocessing does not reproduce 05_preprocessed_final.csv "
            f"(max abs diff {max_abs_diff} > {VERIFICATION_ATOL})."
        )

    raw_stats = _clean_frame(raw.copy())
    categorical = {}
    for col in categorical_columns:
        value_counts = raw_stats[col].value_counts()
        categories = list(encoders[col].classes_)
        categorical[col] = {
            "categories": categories,
            "cardinality": int(len(categories)),
            "default": str(value_counts.index[0]),
            "is_binary": bool(len(categories) == 2),
        }

    continuous = {}
    for col in continuous_columns:
        raw_values = raw_stats[col].astype(float)
        continuous[col] = {
            "min": float(raw_values.min()),
            "max": float(raw_values.max()),
            "median": float(raw_values.median()),
            "mean": float(raw_values.mean()),
            "scaler_mean": float(scaler.mean_[continuous_columns.index(col)]),
            "scaler_scale": float(scaler.scale_[continuous_columns.index(col)]),
        }

    feature_order = [col for col in cleaned.columns if col != TARGET_COLUMN]

    return {
        "encoders": encoders,
        "scaler": scaler,
        "categorical_columns": categorical_columns,
        "continuous_columns": continuous_columns,
        "categorical": categorical,
        "continuous": continuous,
        "feature_order": feature_order,
        "verification": {
            "raw_rows": int(len(raw)),
            "verified_rows": int(len(cleaned)),
            "verified_columns": int(len(cleaned.columns)),
            "feature_count": int(len(feature_order)),
            "max_abs_diff": max_abs_diff,
            "verified_against": str(FINAL_PATH.name),
        },
    }


def save_preprocessor(preprocessor: dict) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, ARTIFACT_PATH)
    return ARTIFACT_PATH


def load_preprocessor(rebuild: bool = False) -> dict:
    """Load the persisted transformers, rebuilding them if they are missing."""

    if rebuild or not ARTIFACT_PATH.exists():
        preprocessor = build_preprocessor()
        save_preprocessor(preprocessor)
        return preprocessor
    preprocessor = joblib.load(ARTIFACT_PATH)
    if int(preprocessor["verification"]["max_abs_diff"] > VERIFICATION_ATOL):
        preprocessor = build_preprocessor()
        save_preprocessor(preprocessor)
    return preprocessor


def transform(raw_row: pd.DataFrame, preprocessor: dict) -> pd.DataFrame:
    """Turn raw (unscaled) ClaimWise columns into the model-ready feature matrix."""

    frame = raw_row.copy()

    for col in preprocessor["categorical_columns"]:
        if col not in frame.columns:
            raise ValueError(f"Missing categorical column: {col}")
        values = frame[col].tolist()
        cleaned_values = [v.strip() if isinstance(v, str) else v for v in values]
        unknown = [v for v in cleaned_values if v not in set(preprocessor["encoders"][col].classes_)]
        if unknown:
            raise ValueError(
                f"{col}: unknown category {unknown[0]!r}. "
                "Choose one of the values offered by the form."
            )
        frame[col] = preprocessor["encoders"][col].transform(cleaned_values)

    cont_cols = preprocessor["continuous_columns"]
    for col in cont_cols:
        if col not in frame.columns:
            raise ValueError(f"Missing continuous column: {col}")
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"{col}: enter a finite number.")
        frame[col] = numeric.to_numpy(dtype=float)

    if ID_COLUMN in frame.columns:
        frame = frame.drop(columns=[ID_COLUMN])
    if TARGET_COLUMN in frame.columns:
        raise ValueError("The target column 'loss' must not be sent to the model.")

    frame[cont_cols] = preprocessor["scaler"].transform(frame[cont_cols])

    missing = [col for col in preprocessor["feature_order"] if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing input columns: {', '.join(missing[:5])}")

    model_input = frame[preprocessor["feature_order"]]
    if list(model_input.columns) != list(preprocessor["feature_order"]):
        raise ValueError("Feature order does not match the trained model.")
    return model_input


def out_of_range(raw_row: pd.DataFrame, preprocessor: dict) -> list:
    """Continuous values outside the range seen in the raw dataset (warning only)."""

    warnings = []
    for col, stats in preprocessor["continuous"].items():
        value = pd.to_numeric(pd.Series([raw_row[col].iloc[0]]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        if float(value) < stats["min"] or float(value) > stats["max"]:
            warnings.append(
                f"{col} = {float(value):g} is outside the dataset range "
                f"[{stats['min']:g}, {stats['max']:g}]"
            )
    return warnings


def main() -> None:
    print("Replaying the ClaimWise preprocessing pipeline for prediction...")
    preprocessor = build_preprocessor()
    path = save_preprocessor(preprocessor)
    verification = preprocessor["verification"]
    print("Verification against", verification["verified_against"])
    print("  rows              :", verification["raw_rows"])
    print("  columns           :", verification["verified_columns"])
    print("  model features    :", verification["feature_count"])
    print("  max abs difference:", verification["max_abs_diff"])
    print("  categorical cols  :", len(preprocessor["categorical_columns"]))
    print("  continuous cols   :", len(preprocessor["continuous_columns"]))
    print("Saved:", path)


if __name__ == "__main__":
    main()

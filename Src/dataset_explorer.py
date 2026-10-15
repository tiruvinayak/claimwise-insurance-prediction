"""ClaimWise dynamic raw-dataset explorer.

Provides memory-efficient, server-side pagination, global search, column
filtering, sorting, column statistics, dataset validation, and safe filtered
CSV export for the authoritative raw dataset (or an active uploaded sandbox
dataset). The raw source file is never modified or rewritten.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dataset_manager import get_current_upload_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_RELATIVE_PATH = "Dataset/01_raw_claimwise_50000.csv"
RAW_PATH = PROJECT_ROOT / RAW_RELATIVE_PATH

ALLOWED_PAGE_SIZES = (100, 250, 500)
DEFAULT_PAGE_SIZE = 500
MAX_PAGE_SIZE = 500
MAX_EXPORT_ROWS = 50000
SEARCH_MIN_CHARS = 1

FILTER_OPS = ("contains", "eq", "ne", "gt", "gte", "lt", "lte")
_FILTER_OP_LABELS = {
    "contains": "contains",
    "eq": "equals",
    "ne": "not equal to",
    "gt": "greater than",
    "gte": "greater than or equal to",
    "lt": "less than",
    "lte": "less than or equal to",
}

_EXPECTED_RAW_COLUMNS = (
    ["id"]
    + [f"cat{i}" for i in range(1, 117)]
    + [f"cont{i}" for i in range(1, 15)]
    + ["loss"]
)

# (fingerprint, payload) caches keyed by logical name.
_CACHE: Dict[str, Tuple[Tuple[Any, ...], Any]] = {}


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _fingerprint(path: Path) -> Tuple[Any, ...]:
    stat = path.stat()
    return (str(path), stat.st_size, stat.st_mtime_ns)


def _resolve_source(source: str) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve the explorer source file ("raw" or "upload")."""
    source = (source or "raw").strip().lower()
    if source == "upload":
        upload = get_current_upload_path()
        if upload is None or not upload.exists():
            return None, "No uploaded dataset is active. Switch to an uploaded CSV first."
        return upload, None
    if not RAW_PATH.exists():
        return None, (
            "The authoritative raw dataset is missing from Dataset/. "
            "Restore Dataset/01_raw_claimwise_50000.csv to use the dataset explorer."
        )
    return RAW_PATH, None


def _cached(name: str, fingerprint: Tuple[Any, ...], builder):
    entry = _CACHE.get(name)
    if entry is not None and entry[0] == fingerprint and entry[1] is not None:
        return entry[1]
    payload = builder()
    _CACHE[name] = (fingerprint, payload)
    return payload


def load_explorer_frame(source: str = "raw") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load (and cache) the explorer DataFrame. Categorical columns use
    category dtype to keep the 50,000 x 132 frame near 12 MB of RAM."""
    path, err = _resolve_source(source)
    if err:
        return None, err

    def _build() -> pd.DataFrame:
        header = pd.read_csv(path, nrows=0).columns
        dtypes = {c: "category" for c in header if str(c).startswith("cat")}
        return pd.read_csv(path, dtype=dtypes)

    try:
        frame = _cached(f"frame::{path}", _fingerprint(path), _build)
    except Exception as exc:
        return None, f"CSV could not be parsed: {exc}"
    return frame, None


def explorer_meta(source: str = "raw") -> Dict[str, Any]:
    """File-level metadata for the explorer source."""
    path, err = _resolve_source(source)
    if err:
        return {"success": False, "error": err, "source": source}
    frame, frame_err = load_explorer_frame(source)
    if frame_err:
        return {"success": False, "error": frame_err, "source": source, "filename": path.name}
    return {
        "success": True,
        "source": source,
        "filename": path.name,
        "relative_path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "size_bytes": path.stat().st_size,
        "size_formatted": _format_size(path.stat().st_size),
        "total_rows": int(len(frame)),
        "total_columns": int(frame.shape[1]),
        "columns": [str(c) for c in frame.columns],
    }


def _is_categorical(series: pd.Series) -> bool:
    return str(series.dtype) == "category"


def _search_mask(frame: pd.DataFrame, query: str) -> np.ndarray:
    """Vectorized global search across every column."""
    q = query.lower()
    mask = np.zeros(len(frame), dtype=bool)
    categorical = [c for c in frame.columns if _is_categorical(frame[c])]
    cat_set = set(categorical)

    for col in categorical:
        categories = frame[col].cat.categories.astype(str)
        matching_idx = [i for i, label in enumerate(categories) if q in label.lower()]
        if matching_idx:
            mask |= np.isin(frame[col].cat.codes.to_numpy(), matching_idx)
        if mask.all():
            return mask

    for col in frame.columns:
        if col in cat_set:
            continue
        mask |= (
            frame[col]
            .astype(str)
            .str.lower()
            .str.contains(q, regex=False, na=False)
            .to_numpy()
        )
    return mask


def _filter_mask(series: pd.Series, op: str, value: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    op = (op or "contains").strip().lower()
    value = "" if value is None else str(value).strip()
    if op not in FILTER_OPS:
        return None, f"Unsupported filter operator '{op}'."
    if not value:
        return None, "A filter value is required."

    if op in ("gt", "gte", "lt", "lte"):
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            return None, f"Operator '{_FILTER_OP_LABELS[op]}' requires a numeric value."
        numeric = pd.to_numeric(series, errors="coerce")
        comparison = {
            "gt": numeric > threshold,
            "gte": numeric >= threshold,
            "lt": numeric < threshold,
            "lte": numeric <= threshold,
        }[op]
        return comparison.fillna(False).to_numpy(), None

    as_text = series.astype(str)
    if op == "contains":
        return as_text.str.contains(value, case=False, regex=False, na=False).to_numpy(), None
    if op == "eq":
        return as_text.str.lower().eq(value.lower()).to_numpy(), None
    return as_text.str.lower().ne(value.lower()).to_numpy(), None


def _cell_to_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, float):
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text if text else "0"
    return str(value)


def query_records(
    source: str = "raw",
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    search: str = "",
    filter_col: str = "",
    filter_op: str = "contains",
    filter_value: str = "",
    sort_by: str = "",
    sort_dir: str = "asc",
) -> Dict[str, Any]:
    """Server-side pagination over the explorer source with optional
    global search, column filter, and sort. Returns only one page of
    records so the browser never holds the full 50,000-row dataset."""
    frame, err = load_explorer_frame(source)
    if err:
        return {"success": False, "error": err}

    meta = explorer_meta(source)
    if not meta.get("success"):
        return {"success": False, "error": meta.get("error", "Dataset is unavailable.")}

    try:
        page = max(1, int(page))
    except (TypeError, ValueError):
        page = 1
    try:
        page_size = int(page_size)
    except (TypeError, ValueError):
        page_size = DEFAULT_PAGE_SIZE
    if page_size not in ALLOWED_PAGE_SIZES:
        page_size = DEFAULT_PAGE_SIZE

    search = str(search or "").strip()
    filter_col = str(filter_col or "").strip()
    filter_value = str(filter_value or "").strip()
    sort_by = str(sort_by or "").strip()
    sort_dir = "desc" if str(sort_dir or "").strip().lower() == "desc" else "asc"

    working = frame
    errors: List[str] = []
    applied_filters: List[Dict[str, str]] = []

    if filter_col:
        if filter_col not in frame.columns:
            errors.append(f"Column '{filter_col}' does not exist in this dataset.")
        else:
            mask, mask_err = _filter_mask(frame[filter_col], filter_op, filter_value)
            if mask_err:
                errors.append(mask_err)
            else:
                working = working[mask]
                applied_filters.append(
                    {
                        "column": filter_col,
                        "op": filter_op,
                        "op_label": _FILTER_OP_LABELS.get(filter_op, filter_op),
                        "value": filter_value,
                    }
                )

    if search and len(search) >= SEARCH_MIN_CHARS and not errors:
        sub_mask = _search_mask(working, search)
        working = working[sub_mask]

    if errors:
        return {
            "success": False,
            "error": " ".join(errors),
            "columns": [str(c) for c in frame.columns],
            "rows": [],
            "page": 1,
            "page_size": page_size,
            "total_rows": int(len(frame)),
            "filtered_rows": 0,
            "total_pages": 0,
            "start_row": 0,
            "end_row": 0,
        }

    if sort_by:
        if sort_by not in frame.columns:
            return {
                "success": False,
                "error": f"Cannot sort by '{sort_by}': the column does not exist.",
                "columns": [str(c) for c in frame.columns],
                "rows": [],
                "page": 1,
                "page_size": page_size,
                "total_rows": int(len(frame)),
                "filtered_rows": 0,
                "total_pages": 0,
                "start_row": 0,
                "end_row": 0,
            }
        working = working.sort_values(sort_by, ascending=(sort_dir == "asc"), kind="stable")

    total_rows = int(len(frame))
    filtered_rows = int(len(working))
    total_pages = max(1, (filtered_rows + page_size - 1) // page_size) if filtered_rows else 1
    page = min(page, total_pages)

    start = (page - 1) * page_size
    end = min(start + page_size, filtered_rows)
    slice_df = working.iloc[start:end]

    rows = []
    columns = [str(c) for c in frame.columns]
    for record in slice_df.to_dict(orient="records"):
        rows.append({k: _cell_to_text(v) for k, v in record.items()})

    return {
        "success": True,
        "columns": columns,
        "rows": rows,
        "page": page,
        "page_size": page_size,
        "total_rows": total_rows,
        "filtered_rows": filtered_rows,
        "total_pages": total_pages,
        "start_row": (start + 1) if filtered_rows else 0,
        "end_row": end,
        "search": search,
        "sort_by": sort_by,
        "sort_dir": sort_dir,
        "applied_filters": applied_filters,
        "filename": meta["filename"],
        "relative_path": meta["relative_path"],
        "file_size_formatted": meta["size_formatted"],
        "source": source,
        "is_filtered": bool(search or applied_filters),
    }


def column_statistics(source: str = "raw") -> Dict[str, Any]:
    """Computed per-column statistics, duplicate audit, and target summary.

    All values are measured from the real file and cached against its
    modification fingerprint. Nothing is hard-coded.
    """
    path, err = _resolve_source(source)
    if err:
        return {"success": False, "error": err}

    frame, frame_err = load_explorer_frame(source)
    if frame_err:
        return {"success": False, "error": frame_err}

    def _build() -> Dict[str, Any]:
        columns_info: List[Dict[str, Any]] = []
        total_cells = int(frame.shape[0] * frame.shape[1])
        total_missing = int(frame.isna().sum().sum())

        for position, col in enumerate(frame.columns):
            series = frame[col]
            non_null = int(series.notna().sum())
            missing = int(series.isna().sum())
            info: Dict[str, Any] = {
                "position": position + 1,
                "name": str(col),
                "dtype": str(series.dtype),
                "non_null": non_null,
                "missing": missing,
                "missing_pct": round((missing / len(frame) * 100), 4) if len(frame) else 0.0,
                "unique": int(series.nunique(dropna=True)),
                "kind": "categorical",
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
            }
            name = str(col)
            if name == "id":
                info["kind"] = "identifier"
            elif name == "loss":
                info["kind"] = "target"
            elif name.startswith("cont"):
                info["kind"] = "continuous"

            if info["kind"] in ("identifier", "continuous", "target"):
                numeric = pd.to_numeric(series, errors="coerce")
                if numeric.notna().any():
                    info["min"] = float(numeric.min())
                    info["max"] = float(numeric.max())
                    info["mean"] = float(numeric.mean())
                    info["median"] = float(numeric.median())
                    info["std"] = float(numeric.std())
            else:
                values = series.astype(str)
                top = values.value_counts(dropna=True).head(3)
                info["top_values"] = [
                    {"value": str(idx), "count": int(cnt)} for idx, cnt in top.items()
                ]
            columns_info.append(info)

        target_summary = None
        if "loss" in frame.columns:
            numeric = pd.to_numeric(frame["loss"], errors="coerce")
            if numeric.notna().any():
                target_summary = {
                    "name": "loss",
                    "count": int(numeric.notna().sum()),
                    "mean": round(float(numeric.mean()), 4),
                    "std": round(float(numeric.std()), 4),
                    "min": round(float(numeric.min()), 4),
                    "q1": round(float(numeric.quantile(0.25)), 4),
                    "median": round(float(numeric.median()), 4),
                    "q3": round(float(numeric.quantile(0.75)), 4),
                    "max": round(float(numeric.max()), 4),
                }

        return {
            "columns": columns_info,
            "total_cells": total_cells,
            "total_missing": total_missing,
            "duplicate_rows": int(frame.duplicated().sum()),
            "target_summary": target_summary,
            "rows": int(frame.shape[0]),
            "column_count": int(frame.shape[1]),
        }

    payload = _cached(f"stats::{path}", _fingerprint(path), _build)
    return {"success": True, "source": source, "filename": path.name, **payload}


def validate_dataset(source: str = "raw") -> Dict[str, Any]:
    """Dataset integrity checks: existence, parsing, shape, schema, and
    missing/duplicate audits measured from the real file."""
    path, err = _resolve_source(source)
    if err:
        return {
            "success": False,
            "error": err,
            "file_exists": False,
            "issues": [err],
        }

    stats = column_statistics(source)
    if not stats.get("success"):
        return {
            "success": False,
            "error": stats.get("error"),
            "file_exists": True,
            "issues": [stats.get("error", "Dataset could not be analyzed.")],
        }

    frame, _ = load_explorer_frame(source)
    columns = [str(c) for c in frame.columns] if frame is not None else []
    issues: List[str] = []

    schema_ok = None
    missing_expected: List[str] = []
    if source != "upload" and columns:
        missing_expected = [c for c in _EXPECTED_RAW_COLUMNS if c not in columns]
        schema_ok = not missing_expected and len(columns) == len(_EXPECTED_RAW_COLUMNS)
        if missing_expected:
            issues.append(
                f"{len(missing_expected)} expected columns are missing "
                f"(first: {missing_expected[0]})."
            )
        extra = [c for c in columns if c not in _EXPECTED_RAW_COLUMNS]
        if extra:
            issues.append(f"{len(extra)} unexpected columns present (first: {extra[0]}).")

    if stats["total_missing"] > 0:
        issues.append(f"{stats['total_missing']:,} missing cells detected.")
    if stats["duplicate_rows"] > 0:
        issues.append(f"{stats['duplicate_rows']:,} duplicate rows detected.")
    if "loss" not in columns:
        issues.append("Target column 'loss' is not present.")

    return {
        "success": True,
        "source": source,
        "file_exists": True,
        "filename": path.name,
        "relative_path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "rows": stats["rows"],
        "columns": stats["column_count"],
        "column_names": columns,
        "target_present": "loss" in columns,
        "expected_schema_ok": schema_ok,
        "missing_expected_columns": missing_expected,
        "total_missing_cells": stats["total_missing"],
        "duplicate_rows": stats["duplicate_rows"],
        "parse_errors": [],
        "issues": issues,
        "is_valid": not issues,
    }


def export_filtered_csv(
    source: str = "raw",
    search: str = "",
    filter_col: str = "",
    filter_op: str = "contains",
    filter_value: str = "",
    sort_by: str = "",
    sort_dir: str = "asc",
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (csv_text, filename, error) for the currently filtered view.

    The export is always explicitly separate from the raw download and is
    capped at MAX_EXPORT_ROWS so a browser request cannot exhaust memory.
    """
    frame, err = load_explorer_frame(source)
    if err:
        return None, None, err

    working = frame
    if filter_col:
        if filter_col not in frame.columns:
            return None, None, f"Column '{filter_col}' does not exist in this dataset."
        mask, mask_err = _filter_mask(frame[filter_col], filter_op, filter_value)
        if mask_err:
            return None, None, mask_err
        working = working[mask]

    search = str(search or "").strip()
    if search and len(search) >= SEARCH_MIN_CHARS:
        working = working[_search_mask(working, search)]

    sort_by = str(sort_by or "").strip()
    if sort_by and sort_by in frame.columns:
        working = working.sort_values(
            sort_by,
            ascending=(str(sort_dir).strip().lower() != "desc"),
            kind="stable",
        )

    working = working.head(MAX_EXPORT_ROWS)

    buffer = io.StringIO()
    working.to_csv(buffer, index=False)
    suffix = "filtered" if (search or filter_col or sort_by) else "full"
    filename = f"claimwise_dataset_{suffix}_export.csv"
    return buffer.getvalue(), filename, None


def clear_explorer_cache() -> None:
    """Drop cached frames and statistics (used by tests and uploads)."""
    _CACHE.clear()

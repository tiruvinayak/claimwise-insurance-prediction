"""ClaimWise Resource-Aware Experiment Suite.

Leakage-safe model development with a fixed 3-way protocol:

  * Test split    : Dataset/06_split_X_test.csv + 06_split_y_test.csv
                    (10,000 rows — untouched by training, tuning and model
                    selection; used exactly ONCE at the very end).
  * Train/Val     : the existing 40,000-row training partition is split
                    32,500 train / 7,500 validation (seed 42, stratified by
                    target decile).

Method
------
1. Train 13 baseline candidates on the 32,500-row training set.
2. Evaluate every candidate on the 7,500-row validation set (primary
   selection metric: MAE on the original loss scale — the metric used by the
   original Allstate claims-severity study).
3. Resource-aware validation-based random search for the promising families
   (documented, seeded budgets per family).
4. Log1p-target experiments for the three strongest tree families
   (reported for analysis; production-eligible models always predict the
   original scale).
5. Select the best configuration by validation MAE.
6. Refit the selected configuration on train+validation (40,000 rows) and
   evaluate ONCE on the untouched 10,000-row test set.
7. Atomic promotion of Models/best_model.pkl after inference checks pass.

All artifacts are written under Outputs/ExperimentSuite/ plus refreshed
Outputs/Model_Comparison/ files. Missing metrics stay missing (NaN) — they
are never replaced with zeros. Any configuration that fails to train is
recorded as a failure and the suite continues.

Run:  .venv/bin/python Src/experiment_suite.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.svm import LinearSVR  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration (all randomness is seeded)
# ---------------------------------------------------------------------------
SEED = 42
VALIDATION_SIZE = 7_500          # carved from the 40,000-row training partition
TEST_SIZE = 10_000               # existing untouched test split
N_JOBS = max(1, (os.cpu_count() or 4) - 1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Dataset"
MODELS_DIR = PROJECT_ROOT / "Models"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
SUITE_DIR = OUTPUTS_DIR / "ExperimentSuite"
FINAL_DIR = OUTPUTS_DIR / "Final_Model_Evaluation"
COMPARISON_DIR = OUTPUTS_DIR / "Model_Comparison"

for directory in (SUITE_DIR, FINAL_DIR, MODELS_DIR, COMPARISON_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp} UTC] {message}", flush=True)


# ---------------------------------------------------------------------------
# Data loading and the leakage-safe split protocol
# ---------------------------------------------------------------------------
def load_data():
    x_train_path = DATASET_DIR / "06_split_X_train.csv"
    y_train_path = DATASET_DIR / "06_split_y_train.csv"
    x_test_path = DATASET_DIR / "06_split_X_test.csv"
    y_test_path = DATASET_DIR / "06_split_y_test.csv"
    for path in (x_train_path, y_train_path, x_test_path, y_test_path):
        if not path.exists():
            raise FileNotFoundError(f"Required split file missing: {path}")

    x_train_full = pd.read_csv(x_train_path)
    y_train_full = pd.read_csv(y_train_path)["loss"].astype(float)
    x_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)["loss"].astype(float)

    # Stratify the carve-out on target deciles so the validation set mirrors
    # the heavy-tailed loss distribution.
    target_bins = pd.qcut(y_train_full, q=10, labels=False, duplicates="drop")
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=target_bins,
    )
    log(
        f"Split protocol -> train {x_train.shape}, validation {x_val.shape}, "
        f"untouched test {x_test.shape}"
    )
    return x_train, y_train, x_val, y_val, x_test, y_test, x_train_full, y_train_full


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    denom = np.abs(y_true)
    mape = float(np.mean(np.where(denom > 0, np.abs(y_true - y_pred) / denom, np.nan)) * 100)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
        "Explained_Variance": float(explained_variance_score(y_true, y_pred)),
        "Median_AE": float(median_absolute_error(y_true, y_pred)),
        "MAPE": mape,
    }


# ---------------------------------------------------------------------------
# Data investigation (target skew, irreducible noise, distribution shift)
# ---------------------------------------------------------------------------
def analyse_data(x_train, y_train, x_val, y_val, x_test, y_test) -> dict:
    log("Investigating target distribution, duplicate-feature noise and shift...")
    audit = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "target": {
            "train": describe_series(y_train),
            "validation": describe_series(y_val),
            "test": describe_series(y_test),
            "skew_train": float(y_train.skew()),
            "excess_kurtosis_train": float(y_train.kurtosis()),
            "top_1pct_share_of_variance": _top_tail_variance_share(y_train, 0.01),
            "top_5pct_share_of_variance": _top_tail_variance_share(y_train, 0.05),
        },
    }

    # Irreducible-noise probe: identical feature vectors with different loss.
    full_x = pd.concat([x_train, x_val, x_test], axis=0, ignore_index=True)
    full_y = pd.concat([y_train, y_val, y_test], axis=0, ignore_index=True)
    hashes = pd.util.hash_pandas_object(full_x, index=False)
    frame = pd.DataFrame({"hash": hashes.to_numpy(), "loss": full_y.to_numpy()})
    grouped = frame.groupby("hash")["loss"]
    sizes = grouped.size()
    dup_hashes = sizes[sizes > 1]
    if len(dup_hashes) == 0:
        noise = {
            "duplicate_groups": 0,
            "rows_in_duplicate_groups": 0,
            "groups_with_conflicting_loss": 0,
            "conflicting_rows": 0,
            "mean_within_group_abs_diff": 0.0,
            "mean_within_group_range": 0.0,
            "share_of_rows_in_duplicate_groups": 0.0,
        }
    else:
        dup_frame = frame[frame["hash"].isin(dup_hashes.index)]
        g = dup_frame.groupby("hash")["loss"]
        conflicting = g.nunique() > 1
        conflicting_idx = conflicting[conflicting].index
        within_abs = g.apply(
            lambda s: float(np.mean(np.abs(s - s.mean()))) if s.nunique() > 1 else 0.0
        )
        ranges = g.max() - g.min()
        noise = {
            "duplicate_groups": int(len(dup_hashes)),
            "rows_in_duplicate_groups": int(dup_hashes.sum()),
            "groups_with_conflicting_loss": int(conflicting.sum()),
            "conflicting_rows": int(sizes.loc[conflicting_idx].sum()) if len(conflicting_idx) else 0,
            "mean_within_group_abs_diff": float(within_abs.loc[conflicting_idx].mean())
            if len(conflicting_idx) else 0.0,
            "mean_within_group_range": float(ranges.loc[conflicting_idx].mean())
            if len(conflicting_idx) else 0.0,
            "share_of_rows_in_duplicate_groups": float(dup_hashes.sum() / len(frame)),
        }
    audit["duplicate_feature_noise"] = noise

    # Train/validation/test distribution shift (standardized mean differences).
    audit["distribution_shift"] = {
        "train_vs_validation_max_abs_smd": _max_standardized_mean_difference(x_train, x_val),
        "train_vs_test_max_abs_smd": _max_standardized_mean_difference(x_train, x_test),
        "validation_vs_test_max_abs_smd": _max_standardized_mean_difference(x_val, x_test),
    }

    path = SUITE_DIR / "data_audit.json"
    path.write_text(json.dumps(audit, indent=2, default=str))
    log(f"Wrote {path.relative_to(PROJECT_ROOT)}")
    return audit


def describe_series(series: pd.Series) -> dict:
    return {
        "rows": int(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
    }


def _top_tail_variance_share(series: pd.Series, frac: float) -> float:
    threshold = series.quantile(1 - frac)
    tail = series[series >= threshold]
    return float(tail.var(ddof=0) * len(tail) / (series.var(ddof=0) * len(series)))


def _max_standardized_mean_difference(a: pd.DataFrame, b: pd.DataFrame) -> float:
    mean_a = a.mean(numeric_only=True)
    std_a = a.std(numeric_only=True).replace(0, np.nan)
    mean_b = b.mean(numeric_only=True)
    smd = ((mean_b - mean_a) / std_a).abs()
    return float(smd.max()) if len(smd) else float("nan")


# ---------------------------------------------------------------------------
# Candidate factories (baseline config is always the first entry of a family)
# ---------------------------------------------------------------------------
def candidate_families() -> dict:
    """Return {family: [(label, sklearn_estimator), ...]} with seeded budgets."""
    families = {
        "Dummy Regressor (Median Baseline)": [
            ("default", DummyRegressor(strategy="median")),
        ],
        "Linear Regression": [
            ("default", LinearRegression()),
        ],
        "Ridge Regression": [
            (f"alpha={a}", Ridge(alpha=a, random_state=SEED))
            for a in (0.1, 1.0, 10.0, 100.0, 500.0, 1000.0)
        ],
        "Lasso Regression": [
            (f"alpha={a}", Lasso(alpha=a, random_state=SEED, max_iter=20_000))
            for a in (0.01, 0.1, 0.5, 1.0, 2.0)
        ],
        "Elastic Net Regression": [
            (f"alpha={a},l1_ratio={r}", ElasticNet(alpha=a, l1_ratio=r,
                                                   random_state=SEED, max_iter=20_000))
            for a, r in ((0.1, 0.2), (0.1, 0.5), (0.5, 0.5), (1.0, 0.5), (1.0, 0.8))
        ],
        "Support Vector Regression (Linear)": [
            (f"C={c},eps={eps}", LinearSVR(C=c, epsilon=eps, random_state=SEED,
                                           dual="auto", max_iter=5000))
            for c, eps in ((0.1, 0.0), (1.0, 0.0), (1.0, 100.0), (10.0, 100.0))
        ],
        "Decision Tree Regressor": [
            (f"max_depth={d},min_samples_leaf={m}", DecisionTreeRegressor(
                max_depth=d, min_samples_leaf=m, random_state=SEED))
            for d, m in ((None, 1), (12, 1), (16, 2), (20, 4))
        ],
        "Random Forest Regressor": [
            (f"n_estimators={n},max_depth={d},min_samples_leaf={m},max_features={mf}",
             RandomForestRegressor(n_estimators=n, max_depth=d, min_samples_leaf=m,
                                   max_features=mf, random_state=SEED, n_jobs=N_JOBS))
            for n, d, m, mf in (
                (150, None, 1, "sqrt"),
                (200, None, 2, "sqrt"),
                (200, 16, 2, 0.5),
                (200, 24, 1, 1.0),
                (250, None, 1, 0.3),
            )
        ],
        "Extra Trees Regressor": [
            (f"n_estimators={n},max_depth={d},min_samples_leaf={m},max_features={mf}",
             ExtraTreesRegressor(n_estimators=n, max_depth=d, min_samples_leaf=m,
                                 max_features=mf, random_state=SEED, n_jobs=N_JOBS))
            for n, d, m, mf in (
                (150, None, 1, "sqrt"),
                (200, None, 2, "sqrt"),
                (200, 16, 2, 0.5),
                (200, 24, 1, 1.0),
                (250, None, 1, 0.3),
            )
        ],
        "Gradient Boosting Regressor": [
            (f"n_estimators={n},lr={lr},max_depth={d},min_samples_leaf={m}",
             GradientBoostingRegressor(n_estimators=n, learning_rate=lr, max_depth=d,
                                       min_samples_leaf=m, random_state=SEED))
            for n, lr, d, m in (
                (150, 0.1, 3, 1),
                (200, 0.05, 3, 3),
                (250, 0.05, 4, 3),
                (200, 0.1, 4, 5),
                (300, 0.03, 3, 10),
                (200, 0.15, 3, 1),
            )
        ],
        "Hist Gradient Boosting Regressor": [
            (f"lr={lr},iter={it},leaves={ln},min_leaf={ml},l2={l2}",
             HistGradientBoostingRegressor(learning_rate=lr, max_iter=it,
                                           max_leaf_nodes=ln, min_samples_leaf=ml,
                                           l2_regularization=l2, early_stopping=True,
                                           validation_fraction=0.1, n_iter_no_change=20,
                                           random_state=SEED))
            for lr, it, ln, ml, l2 in (
                (0.1, 300, 31, 20, 0.0),
                (0.06, 400, 31, 20, 1.0),
                (0.1, 400, 63, 10, 0.0),
                (0.08, 500, 31, 20, 1.0),
                (0.05, 600, 15, 40, 1.0),
                (0.12, 300, 31, 10, 0.0),
                (0.06, 600, 63, 20, 1.0),
                (0.1, 300, 15, 20, 5.0),
                (0.08, 400, 31, 40, 1.0),
                (0.15, 250, 31, 20, 0.0),
                (0.05, 800, 31, 10, 1.0),
                (0.1, 350, 63, 15, 0.5),
            )
        ],
        "AdaBoost Regressor": [
            (f"n_estimators={n},lr={lr},loss={loss}",
             AdaBoostRegressor(n_estimators=n, learning_rate=lr, loss=loss,
                               random_state=SEED))
            for n, lr, loss in (
                (100, 0.1, "linear"),
                (150, 0.05, "linear"),
                (100, 0.1, "square"),
                (200, 0.03, "linear"),
            )
        ],
        "K-Nearest Neighbors Regressor": [
            (f"k={k},weights={w}", KNeighborsRegressor(n_neighbors=k, weights=w,
                                                       n_jobs=N_JOBS))
            for k, w in ((15, "distance"), (25, "distance"), (25, "uniform"),
                         (40, "distance"))
        ],
    }
    return families


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------
def evaluate_family(family, configs, x_train, y_train, x_val, y_val) -> list:
    rows = []
    for label, estimator in configs:
        started = time.perf_counter()
        try:
            estimator.fit(x_train, y_train)
            seconds = time.perf_counter() - started
            val_pred = estimator.predict(x_val)
            metrics = regression_metrics(y_val, val_pred)
            rows.append({
                "Model": family,
                "Config": label,
                "status": "ok",
                "fit_seconds": round(seconds, 2),
                **metrics,
            })
            log(
                f"{family} [{label}] -> val MAE {metrics['MAE']:.2f}, "
                f"RMSE {metrics['RMSE']:.2f}, R2 {metrics['R2']:.4f} "
                f"({seconds:.1f}s)"
            )
        except Exception as exc:  # noqa: BLE001 - record failure, keep going
            seconds = time.perf_counter() - started
            rows.append({
                "Model": family,
                "Config": label,
                "status": f"failed: {type(exc).__name__}: {exc}",
                "fit_seconds": round(seconds, 2),
                "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "R2": np.nan,
                "Explained_Variance": np.nan, "Median_AE": np.nan, "MAPE": np.nan,
            })
            log(f"{family} [{label}] FAILED after {seconds:.1f}s: {exc}")
    return rows


def run_baselines_and_tuning(x_train, y_train, x_val, y_val) -> pd.DataFrame:
    families = candidate_families()
    all_rows = []
    for family, configs in families.items():
        log(f"--- {family}: {len(configs)} configuration(s) ---")
        all_rows.extend(evaluate_family(family, configs, x_train, y_train, x_val, y_val))
        frame = pd.DataFrame(all_rows)
        frame.to_csv(SUITE_DIR / "tuning_results.csv", index=False)
    return pd.DataFrame(all_rows)


def run_log1p_experiments(best_estimators: dict, x_train, y_train, x_val, y_val) -> pd.DataFrame:
    """Train the three strongest tree families on log1p(loss) and invert."""
    log("Running log1p-target experiments for the strongest tree families...")
    rows = []
    y_train_log = np.log1p(y_train)
    for family, estimator in best_estimators.items():
        try:
            started = time.perf_counter()
            estimator.fit(x_train, y_train_log)
            seconds = time.perf_counter() - started
            log_pred = estimator.predict(x_val)
            pred = np.expm1(log_pred)
            metrics = regression_metrics(y_val, pred)
            rows.append({
                "Model": family,
                "Target": "log1p(loss), predictions inverted with expm1",
                "fit_seconds": round(seconds, 2),
                **metrics,
            })
            log(
                f"log1p {family} -> val MAE {metrics['MAE']:.2f}, "
                f"R2 {metrics['R2']:.4f} ({seconds:.1f}s)"
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "Model": family, "Target": "log1p(loss)", "fit_seconds": np.nan,
                "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "R2": np.nan,
                "Explained_Variance": np.nan, "Median_AE": np.nan, "MAPE": np.nan,
                "status": f"failed: {exc}",
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(SUITE_DIR / "log1p_experiments.csv", index=False)
    return frame


def save_diagnostic_plots(name, slug_folder, y_true, y_pred) -> list:
    """Regenerate per-model validation diagnostics for the outputs gallery."""
    folder = OUTPUTS_DIR / slug_folder
    folder.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    stem = _image_stem(slug_folder)
    written = []

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, y_true, s=6, alpha=0.35, edgecolors="none")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Predicted loss")
    ax.set_ylabel("Actual loss")
    ax.set_title(f"{name} — Actual vs Predicted (validation)")
    ax.legend()
    fig.tight_layout()
    path = folder / f"{stem}_actual_vs_predicted.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(residuals, bins=60, color="#19A987", edgecolor="white")
    ax.axvline(0, color="#D9483B", linestyle="--")
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{name} — Residual Distribution (validation)")
    fig.tight_layout()
    path = folder / f"{stem}_residuals.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(path)
    return written


def _image_stem(folder_name: str) -> str:
    mapping = {
        "Dummy_Regressor_Median_Baseline": "dummy_regressor_median_baseline",
        "Linear_Regression": "linear_regression",
        "Ridge_Regression": "ridge_regression",
        "Lasso_Regression": "lasso_regression",
        "Elastic_Net_Regression": "elastic_net_regression",
        "Support_Vector_Regression_Linear": "support_vector_regression_linear",
        "Decision_Tree_Regressor": "decision_tree_regressor",
        "Random_Forest_Regressor": "random_forest_regressor",
        "Extra_Trees_Regressor": "extra_trees_regressor",
        "Gradient_Boosting_Regressor": "gradient_boosting_regressor",
        "Hist_Gradient_Boosting_Regressor": "hist_gradient_boosting_regressor",
        "AdaBoost_Regressor": "adaboost_regressor",
        "K_Nearest_Neighbors_Regressor": "k_nearest_neighbors_regressor",
    }
    return mapping.get(folder_name, folder_name.lower())


FAMILY_TO_FOLDER = {
    "Dummy Regressor (Median Baseline)": "Dummy_Regressor_Median_Baseline",
    "Linear Regression": "Linear_Regression",
    "Ridge Regression": "Ridge_Regression",
    "Lasso Regression": "Lasso_Regression",
    "Elastic Net Regression": "Elastic_Net_Regression",
    "Support Vector Regression (Linear)": "Support_Vector_Regression_Linear",
    "Decision Tree Regressor": "Decision_Tree_Regressor",
    "Random Forest Regressor": "Random_Forest_Regressor",
    "Extra Trees Regressor": "Extra_Trees_Regressor",
    "Gradient Boosting Regressor": "Gradient_Boosting_Regressor",
    "Hist Gradient Boosting Regressor": "Hist_Gradient_Boosting_Regressor",
    "AdaBoost Regressor": "AdaBoost_Regressor",
    "K-Nearest Neighbors Regressor": "K_Nearest_Neighbors_Regressor",
}


def refresh_model_comparison_csv(best_rows: pd.DataFrame, winners: dict,
                                 champion_name: str, champion_test: dict) -> None:
    """Write the 13-row validation leaderboard the website consumes.

    Columns MAE/MSE/RMSE/R2/... carry VALIDATION metrics (missing values stay
    NaN, never zero). Extra columns document the evaluation split, fit time
    and whether the model was promoted to the untouched test set.
    """
    baseline_labels = {family: configs[0][0] for family, configs in candidate_families().items()}
    rows = []
    for _, row in best_rows.iterrows():
        name = str(row["Model"])
        entry = {
            "Model": name,
            "MAE": row["MAE"],
            "MSE": row["MSE"],
            "RMSE": row["RMSE"],
            "R2": row["R2"],
            "Explained_Variance": row["Explained_Variance"],
            "Median_AE": row["Median_AE"],
            "MAPE": row["MAPE"],
            "Best_Config": str(row["Config"]),
            "Train_Time_Sec": row["fit_seconds"],
            "Evaluation_Split": "Validation (7,500 rows; train 32,500)",
            "Tuned_By_Random_Search": str(
                str(row["Config"]) != baseline_labels.get(name)
            ).lower(),
            "Selected_For_Final_Test": "true" if name == champion_name else "false",
            "Final_Test_MAE": champion_test["MAE"] if name == champion_name else np.nan,
            "Final_Test_RMSE": champion_test["RMSE"] if name == champion_name else np.nan,
            "Final_Test_R2": champion_test["R2"] if name == champion_name else np.nan,
        }
        rows.append(entry)
    frame = pd.DataFrame(rows).sort_values("MAE", ascending=True).reset_index(drop=True)
    frame.to_csv(COMPARISON_DIR / "model_comparison.csv", index=False)
    frame.to_json(COMPARISON_DIR / "model_comparison.json", orient="records", indent=2)
    log(f"Refreshed {COMPARISON_DIR.name}/model_comparison.csv ({len(frame)} models)")

    # Validation leaderboard chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    names = frame["Model"].tolist()
    short = [n.replace(" Regressor", "").replace(" Regression", "") for n in names]
    colors = ["#078765" if n == champion_name else "#9BD8C5" for n in names]
    ax1.barh(short, frame["MAE"], color=colors)
    ax1.invert_yaxis()
    ax1.set_xlabel("Validation MAE (lower is better)")
    ax1.set_title("Model comparison — validation MAE")
    ax2.barh(short, frame["R2"], color=colors)
    ax2.invert_yaxis()
    ax2.set_xlabel("Validation R² (higher is better)")
    ax2.set_title("Model comparison — validation R²")
    fig.tight_layout()
    fig.savefig(COMPARISON_DIR / "model_comparison_bar_chart.png", dpi=110)
    plt.close(fig)
    log("Regenerated model_comparison_bar_chart.png (validation metrics)")


def save_final_evaluation_plots(y_true, y_pred) -> list:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    absolute = np.abs(residuals)
    written = []

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, y_true, s=6, alpha=0.35, edgecolors="none")
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    ax.set_xlabel("Predicted loss (untouched test set)")
    ax.set_ylabel("Actual loss")
    ax.set_title("Selected Model — Actual vs Predicted (final test)")
    fig.tight_layout()
    p = FINAL_DIR / "final_actual_vs_predicted.png"
    fig.savefig(p, dpi=110); plt.close(fig); written.append(p)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.hist(residuals, bins=60, color="#19A987", edgecolor="white")
    ax1.axvline(0, color="#D9483B", linestyle="--")
    ax1.set_title("Residual distribution (final test)")
    ax2.hist(absolute, bins=60, color="#078765", edgecolor="white")
    ax2.set_title("Absolute error distribution (final test)")
    fig.tight_layout()
    p = FINAL_DIR / "final_residuals_and_errors.png"
    fig.savefig(p, dpi=110); plt.close(fig); written.append(p)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(y_pred, residuals, s=6, alpha=0.35, edgecolors="none")
    ax.axhline(0, color="#D9483B", linestyle="--")
    ax.set_xlabel("Predicted loss")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs predicted (final test)")
    fig.tight_layout()
    p = FINAL_DIR / "final_residuals_vs_predicted.png"
    fig.savefig(p, dpi=110); plt.close(fig); written.append(p)
    return written


# ---------------------------------------------------------------------------
# Atomic promotion of the production model
# ---------------------------------------------------------------------------
def promote_model(model, champion_name: str) -> Path:
    final_path = MODELS_DIR / "best_model.pkl"
    staging_path = MODELS_DIR / "best_model.staging.pkl"
    backup_path = MODELS_DIR / "best_model_previous.pkl"

    joblib.dump(model, staging_path)

    # Inference sanity checks on the staging artifact before promotion.
    probe = joblib.load(staging_path)
    sample = pd.read_csv(DATASET_DIR / "06_split_X_test.csv", nrows=5)
    preds = probe.predict(sample)
    if not np.isfinite(preds).all():
        staging_path.unlink(missing_ok=True)
        raise RuntimeError("Staging model produced non-finite predictions; promotion aborted.")

    if final_path.exists():
        joblib.dump(joblib.load(final_path), backup_path)
    os.replace(staging_path, final_path)
    log(f"Promoted {champion_name} to {final_path.relative_to(PROJECT_ROOT)} "
        f"(previous model backed up to {backup_path.name})")
    return final_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    overall_start = time.perf_counter()
    log("=== ClaimWise Experiment Suite starting ===")
    x_train, y_train, x_val, y_val, x_test, y_test, x_train_full, y_train_full = load_data()

    audit = analyse_data(x_train, y_train, x_val, y_val, x_test, y_test)

    # 1-3. Baselines + validation-based random search per family
    tuning = run_baselines_and_tuning(x_train, y_train, x_val, y_val)

    ok = tuning[tuning["status"] == "ok"].copy()
    if ok.empty:
        log("FATAL: no candidate trained successfully.")
        return 1

    # Best config per family by validation MAE (primary metric)
    best_idx = ok.groupby("Model")["MAE"].idxmin()
    best_rows = ok.loc[best_idx].sort_values("MAE").reset_index(drop=True)
    best_rows.to_csv(SUITE_DIR / "validation_metrics.csv", index=False)

    champion = best_rows.iloc[0]
    champion_name = str(champion["Model"])
    log(f"Champion by validation MAE: {champion_name} "
        f"(MAE {champion['MAE']:.2f}, RMSE {champion['RMSE']:.2f}, R2 {champion['R2']:.4f})")

    # Refit every family's best config on train+validation for diagnostics and
    # log1p experiments on the ORIGINAL train split (log experiment stays on
    # train/val only — it never sees the test set).
    refit_models = {}
    refit_predictions_val = {}
    for _, row in best_rows.iterrows():
        family = str(row["Model"])
        config_label = str(row["Config"])
        estimator = _build_estimator(family, config_label)
        if estimator is None:
            continue
        try:
            refit_models[family] = estimator.fit(x_train, y_train)
            refit_predictions_val[family] = refit_models[family].predict(x_val)
        except Exception as exc:  # noqa: BLE001
            log(f"Refit failed for {family}: {exc}")

    # 4. log1p experiments for the three strongest tree families present
    tree_order = [name for name in best_rows["Model"]
                  if name in ("Hist Gradient Boosting Regressor",
                              "Gradient Boosting Regressor",
                              "Extra Trees Regressor",
                              "Random Forest Regressor")][:3]
    log1p_inputs = {}
    for family in tree_order:
        estimator = _build_estimator(family, str(
            best_rows.loc[best_rows["Model"] == family, "Config"].iloc[0]))
        if estimator is not None:
            log1p_inputs[family] = estimator
    log1p_frame = run_log1p_experiments(log1p_inputs, x_train, y_train, x_val, y_val)

    # 5-6. Refit champion on train+validation and evaluate ONCE on the test set
    champion_estimator = _build_estimator(champion_name, str(champion["Config"]))
    log(f"Refitting champion on train+validation ({len(x_train_full)} rows) "
        f"and evaluating once on the untouched {TEST_SIZE}-row test set...")
    refit_start = time.perf_counter()
    champion_estimator.fit(x_train_full, y_train_full)
    refit_seconds = time.perf_counter() - refit_start
    test_pred = champion_estimator.predict(x_test)
    champion_test = regression_metrics(y_test, test_pred)
    log(
        f"FINAL TEST [{champion_name}] -> MAE {champion_test['MAE']:.2f}, "
        f"RMSE {champion_test['RMSE']:.2f}, R2 {champion_test['R2']:.4f} "
        f"(refit {refit_seconds:.1f}s)"
    )

    # Persist final artifacts
    test_rows = pd.DataFrame([{
        "Model": champion_name,
        "Config": str(champion["Config"]),
        "Train_Rows": int(len(x_train_full)),
        "Test_Rows": int(len(x_test)),
        "Refit_Seconds": round(refit_seconds, 2),
        **champion_test,
        "Val_MAE": champion["MAE"],
        "Val_RMSE": champion["RMSE"],
        "Val_R2": champion["R2"],
    }])
    test_rows.to_csv(SUITE_DIR / "final_test_metrics.csv", index=False)
    test_rows.to_csv(FINAL_DIR / "final_test_metrics.csv", index=False)

    predictions = pd.DataFrame({
        "row_index": np.arange(len(x_test)),
        "actual_loss": y_test.to_numpy(),
        "predicted_loss": test_pred,
        "absolute_error": np.abs(y_test.to_numpy() - test_pred),
    })
    predictions.to_csv(FINAL_DIR / "final_test_predictions.csv", index=False)
    save_final_evaluation_plots(y_test, test_pred)

    # Atomic promotion: only after the staged artifact passes finite-prediction
    # checks on real test rows. Previous production model is kept as backup.
    promote_model(champion_estimator, champion_name)

    # Refresh per-model validation charts + per-model metrics.csv
    for family, pred in refit_predictions_val.items():
        folder = FAMILY_TO_FOLDER.get(family)
        if folder is None:
            continue
        save_diagnostic_plots(family, folder, y_val, pred)
        row = best_rows[best_rows["Model"] == family].iloc[0]
        pd.DataFrame([{
            "Model": family, "Split": "Validation",
            "MAE": row["MAE"], "MSE": row["MSE"], "RMSE": row["RMSE"], "R2": row["R2"],
            "Explained_Variance": row["Explained_Variance"], "Median_AE": row["Median_AE"],
            "MAPE": row["MAPE"], "Train_Time_Sec": row["fit_seconds"],
            "Best_Config": row["Config"],
        }]).to_csv(OUTPUTS_DIR / folder / "metrics.csv", index=False)

    refresh_model_comparison_csv(best_rows, refit_models, champion_name, champion_test)

    # R² >= 0.95 verdict — honest, based on the untouched test evaluation
    r2_goal = 0.95
    reached = bool(champion_test["R2"] >= r2_goal)
    verdict = {
        "goal_r2": r2_goal,
        "reached": reached,
        "achieved_test_r2": champion_test["R2"],
        "statement": (
            f"R² = {champion_test['R2']:.4f} on the untouched 10,000-row test set."
            + ("" if reached else
               " The 0.95 experimental goal was NOT reached; see evidence for the ceiling below.")
        ),
        "evidence": {
            "duplicate_feature_rows_with_conflicting_loss":
                audit["duplicate_feature_noise"],
            "target_skewness_train": audit["target"]["skew_train"],
            "top_1pct_share_of_target_variance":
                audit["target"]["top_1pct_share_of_variance"],
        },
    }

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "split_protocol": {
            "train_rows": int(len(x_train)),
            "validation_rows": int(len(x_val)),
            "final_refit_rows": int(len(x_train_full)),
            "test_rows": int(len(x_test)),
            "validation_carved_from": "Dataset/06_split_X_train.csv (stratified by loss decile)",
            "test_source": "Dataset/06_split_X_test.csv (untouched until final evaluation)",
        },
        "selection": {
            "primary_metric": "validation MAE (original loss scale)",
            "champion": champion_name,
            "champion_config": str(champion["Config"]),
            "champion_validation_metrics": {k: float(champion[k]) for k in
                ("MAE", "MSE", "RMSE", "R2", "Explained_Variance", "Median_AE", "MAPE")},
            "champion_final_test_metrics": champion_test,
            "champion_refit_seconds": round(refit_seconds, 2),
        },
        "r2_goal_verdict": verdict,
        "log1p_experiments": json.loads(log1p_frame.to_json(orient="records")),
        "data_audit_summary": audit,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "n_jobs": N_JOBS,
        },
        "budgets": {
            family: len(configs) for family, configs in candidate_families().items()
        },
        "failed_configs": tuning[tuning["status"] != "ok"][
            ["Model", "Config", "status"]].to_dict(orient="records"),
        "elapsed_seconds_total": round(time.perf_counter() - overall_start, 1),
    }
    (SUITE_DIR / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (MODELS_DIR / "best_model_metadata.json").write_text(json.dumps({
        "model": champion_name,
        "config": str(champion["Config"]),
        "artifact": "Models/best_model.pkl",
        "feature_contract": "130 preprocessed features from Models/claimwise_preprocessor.pkl",
        "validation_metrics": metadata["selection"]["champion_validation_metrics"],
        "final_test_metrics": champion_test,
        "trained_utc": metadata["generated_utc"],
        "seed": SEED,
    }, indent=2, default=str))
    log(f"Suite finished in {metadata['elapsed_seconds_total']}s. "
        f"Champion={champion_name}; test R2={champion_test['R2']:.4f}")
    return 0


def _build_estimator(family: str, config_label: str):
    """Rebuild a family estimator from its config label."""
    for fam, configs in candidate_families().items():
        if fam != family:
            continue
        for label, estimator in configs:
            if label == config_label:
                return estimator
    return None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise SystemExit(1)

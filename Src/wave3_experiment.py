"""ClaimWise Wave-3 experiment: corrected protocol with a fresh, genuinely
untouched holdout.

Why a new split exists
----------------------
The historical 10,000-row test split (Dataset/06_split_*.csv, seed 42) was
reused across earlier project phases: every candidate model was scored on it
and `Src/model_training_utils.py::write_model_comparison` originally ranked
models by TEST RMSE. A test set that has already driven model selection is no
longer an untouched holdout. Wave 3 therefore re-partitions all 50,000
preprocessed rows with a NEW seed (2026) into:

    train 32,500 | validation 7,500 | holdout 10,000

(stratified by loss deciles). Within wave 3 the holdout is scored exactly
once, for the promoted champion only, after selection is locked on the
validation set. All metrics are computed on the ORIGINAL loss scale.

Protocol
--------
1. Verify the preprocessed matrix aligns row-for-row with the raw dataset
   (loss column must be bit-identical — loss is never transformed).
2. Persist the new splits (the previous files are archived, not deleted) and
   an id-level split-membership audit file.
3. Train every candidate family on the 32,500-row train split — the existing
   13 sklearn families (reused from `experiment_suite.py`) plus XGBoost,
   CatBoost, and log1p-target variants wrapped in TransformedTargetRegressor
   so predictions are always returned on the original scale.
4. Score every configuration on the 7,500-row validation split (selection
   metric: MAE on the original loss scale). The current champion's exact
   recipe is re-trained here as an explicit incumbent baseline for a fair,
   same-data comparison.
5. Select the best configuration by validation MAE; refit it on train+validation
   (40,000 rows); evaluate ONCE on the new 10,000-row holdout.
6. Gated promotion: the incumbent artifact is backed up, the staging model is
   reloaded and schema/finite-prediction checked, then promoted atomically.
   Because the deployed app serves predictions against the new X_test file,
   the promoted model is always trained on the new train+validation rows so
   the demo rows can never be training rows of the served model.
7. Update Outputs/Model_Comparison/, Outputs/Final_Model_Evaluation/,
   Outputs/ExperimentSuite/wave3_* artifacts, and
   Models/best_model_metadata.json atomically.

Run:  .venv/bin/python Src/wave3_experiment.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import TransformedTargetRegressor  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_suite import (  # noqa: E402
    candidate_families,
    regression_metrics,
    save_final_evaluation_plots,
)

try:  # CatBoost is optional at import time; failures are recorded per-config.
    from catboost import CatBoostRegressor  # noqa: E402

    CATBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    CATBOOST_AVAILABLE = False

SEED = 2026
WAVE = 3
N_JOBS = max(1, (os.cpu_count() or 4) - 1)
HOLDOUT_ROWS = 10_000
VALIDATION_ROWS = 7_500

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Dataset"
MODELS_DIR = PROJECT_ROOT / "Models"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
SUITE_DIR = OUTPUTS_DIR / "ExperimentSuite"
FINAL_DIR = OUTPUTS_DIR / "Final_Model_Evaluation"
COMPARISON_DIR = OUTPUTS_DIR / "Model_Comparison"
ARCHIVE_DIR = DATASET_DIR / "archive_pre_wave3"
FINAL_ARCHIVE_DIR = FINAL_DIR / "archive_pre_wave3"

RAW_PATH = DATASET_DIR / "01_raw_claimwise_50000.csv"
PREPROCESSED_PATH = DATASET_DIR / "05_preprocessed_final.csv"
MEMBERSHIP_PATH = DATASET_DIR / "wave3_split_membership.csv"

INCUMBENT_CONFIG = HistGradientBoostingRegressor(
    learning_rate=0.03,
    max_iter=1500,
    max_leaf_nodes=31,
    min_samples_leaf=60,
    l2_regularization=0.5,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=SEED,
)
INCUMBENT_LABEL = "incumbent champion recipe (lr=0.03,iter=1500,leaves=31,min_leaf=60,l2=0.5)"


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp} UTC] {message}", flush=True)


# ---------------------------------------------------------------------------
# Data loading, alignment verification and the fresh split protocol
# ---------------------------------------------------------------------------
def load_and_verify() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X_all, y_all, ids) after proving row alignment with the raw file."""

    raw = pd.read_csv(RAW_PATH, usecols=["id", "loss"])
    pre = pd.read_csv(PREPROCESSED_PATH)

    if list(pre.columns)[-1] != "loss":
        raise AssertionError("Preprocessed matrix must end with the 'loss' column.")
    feature_columns = [c for c in pre.columns if c != "loss"]
    if len(feature_columns) != 130:
        raise AssertionError(f"Expected 130 features, found {len(feature_columns)}.")
    if pre.isna().any().any():
        raise AssertionError("Preprocessed matrix contains missing values.")
    if not np.allclose(raw["loss"].to_numpy(), pre["loss"].to_numpy(), atol=1e-9):
        raise AssertionError(
            "Row alignment broken: raw loss differs from preprocessed loss. "
            "The preprocessed file is not a faithful row-order transform of the raw data."
        )

    ids = raw["id"].astype(np.int64)
    X_all = pre[feature_columns]
    y_all = pre["loss"].astype(float)
    log(f"Loaded and verified {X_all.shape[0]} rows x {X_all.shape[1]} features.")
    return X_all, y_all, ids


def make_fresh_splits(
    X_all: pd.DataFrame, y_all: pd.Series, ids: pd.Series
) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]]:
    """Stratified 3-way split (seed 2026): train / validation / holdout."""

    bins_all = pd.qcut(y_all, q=10, labels=False, duplicates="drop")
    X_dev, X_hold, y_dev, y_hold, id_dev, id_hold = train_test_split(
        X_all, y_all, ids, test_size=HOLDOUT_ROWS, random_state=SEED, stratify=bins_all
    )
    bins_dev = pd.qcut(y_dev, q=10, labels=False, duplicates="drop")
    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X_dev, y_dev, id_dev, test_size=VALIDATION_ROWS, random_state=SEED,
        stratify=bins_dev,
    )
    log(
        f"Fresh split (seed {SEED}): train {X_train.shape[0]}, "
        f"validation {X_val.shape[0]}, holdout {X_hold.shape[0]}"
    )
    for name, frame in (("train", y_train), ("validation", y_val), ("holdout", y_hold)):
        log(
            f"  {name:11s} loss mean {frame.mean():9.2f}  median {frame.median():9.2f}  "
            f"max {frame.max():10.2f}  skew {frame.skew():.3f}"
        )
    return {
        "train": (X_train, y_train, id_train),
        "validation": (X_val, y_val, id_val),
        "holdout": (X_hold, y_hold, id_hold),
    }


def splits_already_persisted() -> bool:
    """True when a previous wave-3 run already archived the old splits and wrote new ones."""

    needed = [
        DATASET_DIR / "06_split_X_train.csv", DATASET_DIR / "06_split_y_train.csv",
        DATASET_DIR / "06_split_X_val.csv", DATASET_DIR / "06_split_y_val.csv",
        DATASET_DIR / "06_split_X_test.csv", DATASET_DIR / "06_split_y_test.csv",
        MEMBERSHIP_PATH,
    ]
    if not all(p.exists() for p in needed):
        return False
    train_rows = sum(1 for _ in open(needed[0], "r", encoding="utf-8")) - 1
    val_rows = sum(1 for _ in open(needed[2], "r", encoding="utf-8")) - 1
    hold_rows = sum(1 for _ in open(needed[4], "r", encoding="utf-8")) - 1
    ok = (train_rows, val_rows, hold_rows) == (
        50_000 - HOLDOUT_ROWS - VALIDATION_ROWS, VALIDATION_ROWS, HOLDOUT_ROWS
    )
    if ok:
        log("Existing wave-3 split files found and verified — reusing them (no re-split).")
    return ok


def load_persisted_splits() -> dict:
    """Rebuild the in-memory split dict from the persisted wave-3 CSV files."""

    x_train = pd.read_csv(DATASET_DIR / "06_split_X_train.csv")
    y_train = pd.read_csv(DATASET_DIR / "06_split_y_train.csv")["loss"].astype(float)
    x_val = pd.read_csv(DATASET_DIR / "06_split_X_val.csv")
    y_val = pd.read_csv(DATASET_DIR / "06_split_y_val.csv")["loss"].astype(float)
    x_hold = pd.read_csv(DATASET_DIR / "06_split_X_test.csv")
    y_hold = pd.read_csv(DATASET_DIR / "06_split_y_test.csv")["loss"].astype(float)
    return {
        "train": (x_train, y_train, np.arange(len(x_train))),
        "validation": (x_val, y_val, np.arange(len(x_val))),
        "holdout": (x_hold, y_hold, np.arange(len(x_hold))),
    }


def persist_splits(splits: dict) -> None:
    """Archive the old split files and persist the new ones plus the membership audit."""

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for name in (
        "06_split_X_train.csv", "06_split_y_train.csv",
        "06_split_X_test.csv", "06_split_y_test.csv",
    ):
        src = DATASET_DIR / name
        if src.exists():
            shutil.move(str(src), str(ARCHIVE_DIR / f"pre_wave3_{name}"))

    x_train, y_train, id_train = splits["train"]
    x_val, y_val, id_val = splits["validation"]
    x_hold, y_hold, id_hold = splits["holdout"]

    x_train.to_csv(DATASET_DIR / "06_split_X_train.csv", index=False)
    y_train.to_frame("loss").to_csv(DATASET_DIR / "06_split_y_train.csv", index=False)
    x_val.to_csv(DATASET_DIR / "06_split_X_val.csv", index=False)
    y_val.to_frame("loss").to_csv(DATASET_DIR / "06_split_y_val.csv", index=False)
    x_hold.to_csv(DATASET_DIR / "06_split_X_test.csv", index=False)
    y_hold.to_frame("loss").to_csv(DATASET_DIR / "06_split_y_test.csv", index=False)

    membership = pd.concat(
        [
            pd.DataFrame({"id": id_train.to_numpy(), "split": "train"}),
            pd.DataFrame({"id": id_val.to_numpy(), "split": "validation"}),
            pd.DataFrame({"id": id_hold.to_numpy(), "split": "holdout"}),
        ],
        ignore_index=True,
    ).sort_values("id")
    membership.to_csv(MEMBERSHIP_PATH, index=False)
    log(f"Persisted new splits; old files archived under {ARCHIVE_DIR.name}/.")
    log(f"Wrote split membership audit: {MEMBERSHIP_PATH.name} ({len(membership)} rows).")


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------
def xgboost_configs() -> list[tuple[str, object]]:
    return [
        (
            f"n_estimators={n},lr={lr},depth={d},min_child_weight={m},reg_lambda={l2}",
            XGBRegressor(
                n_estimators=n, learning_rate=lr, max_depth=d,
                min_child_weight=m, reg_lambda=l2, subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", random_state=SEED, n_jobs=N_JOBS,
                early_stopping_rounds=50,
            ),
        )
        for n, lr, d, m, l2 in (
            (800, 0.05, 6, 5, 1.0),
            (1200, 0.03, 6, 10, 1.0),
            (800, 0.05, 8, 5, 2.0),
            (1500, 0.02, 8, 15, 1.0),
            (1000, 0.04, 6, 1, 0.5),
        )
    ]


def catboost_configs() -> list[tuple[str, object]]:
    if not CATBOOST_AVAILABLE:
        return []
    return [
        (
            f"iterations={it},lr={lr},depth={d},l2={l2}",
            CatBoostRegressor(
                iterations=it, learning_rate=lr, depth=d, l2_leaf_reg=l2,
                random_seed=SEED, early_stopping_rounds=50, verbose=0,
                allow_writing_files=False,
            ),
        )
        for it, lr, d, l2 in (
            (2000, 0.04, 6, 3.0),
            (2500, 0.03, 8, 3.0),
            (1500, 0.05, 6, 1.0),
            (3000, 0.02, 6, 5.0),
        )
    ]


def log1p_wrapped_configs() -> list[tuple[str, object]]:
    """log1p-target variants; TransformedTargetRegressor handles expm1 inversion."""

    base = [
        ("HistGB lr=0.03,iter=1500,leaves=31,min_leaf=60", INCUMBENT_CONFIG),
        (
            "XGB n_estimators=800,lr=0.05,depth=6",
            XGBRegressor(
                n_estimators=800, learning_rate=0.05, max_depth=6, min_child_weight=5,
                reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", random_state=SEED, n_jobs=N_JOBS,
            ),
        ),
    ]
    if CATBOOST_AVAILABLE:
        base.append(
            (
                "CatBoost iterations=2000,lr=0.04,depth=6",
                CatBoostRegressor(
                    iterations=2000, learning_rate=0.04, depth=6, l2_leaf_reg=3.0,
                    random_seed=SEED, verbose=0, allow_writing_files=False,
                ),
            )
        )
    return [
        (
            f"log1p({label})",
            TransformedTargetRegressor(
                regressor=estimator, func=np.log1p, inverse_func=np.expm1,
                check_inverse=True,
            ),
        )
        for label, estimator in base
    ]


def all_candidates() -> dict[str, list[tuple[str, object]]]:
    """Family -> [(config label, estimator)] including incumbent, XGB, CatBoost, log1p."""

    families = {k: list(v) for k, v in candidate_families().items()}
    families["Hist Gradient Boosting Regressor"].append((INCUMBENT_LABEL, INCUMBENT_CONFIG))
    families["XGBoost Regressor"] = xgboost_configs()
    if CATBOOST_AVAILABLE:
        families["CatBoost Regressor"] = catboost_configs()
    else:
        log("WARNING: CatBoost unavailable — skipping its configs.")
    families["Log1p Target Experiments"] = log1p_wrapped_configs()
    return families


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def fit_and_score(estimator, x_train, y_train, x_val, y_val) -> dict:
    """Fit on train (XGBoost/CatBoost early-stop on the validation split) and score."""

    uses_early_stop = isinstance(estimator, XGBRegressor) or (
        CATBOOST_AVAILABLE and isinstance(estimator, CatBoostRegressor)
    )
    if uses_early_stop:
        estimator.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    else:
        estimator.fit(x_train, y_train)
    return regression_metrics(y_val, estimator.predict(x_val))


def run_all_candidates(splits: dict) -> pd.DataFrame:
    x_train, y_train, _ = splits["train"]
    x_val, y_val, _ = splits["validation"]
    rows: list[dict] = []
    tuning_path = SUITE_DIR / "wave3_tuning_results.csv"

    # Resume support: skip configurations already recorded by a previous run.
    done: set[tuple[str, str]] = set()
    if tuning_path.exists():
        prior = pd.read_csv(tuning_path)
        done = set(zip(prior["Model"].astype(str), prior["Config"].astype(str)))
        rows = prior.to_dict(orient="records")
        log(f"Resuming: {len(done)} configuration(s) already recorded.")

    for family, configs in all_candidates().items():
        todo = [(l, e) for l, e in configs if (family, l) not in done]
        if not todo:
            log(f"--- {family}: all {len(configs)} config(s) already done, skipping ---")
            continue
        log(f"--- {family}: {len(todo)}/{len(configs)} config(s) to run ---")
        for label, estimator in todo:
            t0 = time.perf_counter()
            try:
                metrics = fit_and_score(estimator, x_train, y_train, x_val, y_val)
                seconds = time.perf_counter() - t0
                rows.append(
                    {"Model": family, "Config": label, "status": "ok",
                     "fit_seconds": round(seconds, 2), "wave": WAVE, "seed": SEED, **metrics}
                )
                log(
                    f"  [{label}] val MAE {metrics['MAE']:.2f}  RMSE {metrics['RMSE']:.2f}  "
                    f"R2 {metrics['R2']:.4f}  ({seconds:.1f}s)"
                )
            except Exception as exc:  # noqa: BLE001 - one bad config must not kill the run
                rows.append(
                    {"Model": family, "Config": label, "status": f"failed: {exc}",
                     "fit_seconds": round(time.perf_counter() - t0, 2), "wave": WAVE,
                     "seed": SEED,
                     **{k: np.nan for k in
                        ("MAE", "MSE", "RMSE", "R2", "Explained_Variance", "Median_AE", "MAPE")}}
                )
                log(f"  [{label}] FAILED: {exc}")
            # Incremental write so an interrupted run never loses completed work.
            SUITE_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(tuning_path, index=False)

    frame = pd.DataFrame(rows)
    frame.to_csv(tuning_path, index=False)
    log(f"Wrote {tuning_path} ({len(frame)} rows).")
    return frame


# ---------------------------------------------------------------------------
# Blend candidate (validation-MAE driven, no holdout access)
# ---------------------------------------------------------------------------
def maybe_add_blend(results: pd.DataFrame, splits: dict) -> pd.DataFrame:
    """Test a simple 50/50 blend of the two best distinct families on validation."""

    ok = results[results["status"] == "ok"].copy()
    if len(ok) < 2:
        return results
    if (results["Model"] == "Blend Ensemble").any():
        log("Blend row already recorded — skipping (resume).")
        return results
    best_two = (
        ok.sort_values("MAE").groupby("Model", as_index=False).first()
        .sort_values("MAE").head(2)
    )
    if len(best_two) < 2 or best_two.iloc[0]["Model"] == best_two.iloc[1]["Model"]:
        return results

    x_train, y_train, _ = splits["train"]
    x_val, y_val, _ = splits["validation"]

    # Rebuild the two estimators by label search.
    def build(model_name: str, config_label: str):
        for fam, cfgs in all_candidates().items():
            if fam == model_name:
                for label, est in cfgs:
                    if label == config_label:
                        return est
        return None

    est_a = build(str(best_two.iloc[0]["Model"]), str(best_two.iloc[0]["Config"]))
    est_b = build(str(best_two.iloc[1]["Model"]), str(best_two.iloc[1]["Config"]))
    if est_a is None or est_b is None:
        return results

    t0 = time.perf_counter()
    try:
        fit_and_score(est_a, x_train, y_train, x_val, y_val)
        fit_and_score(est_b, x_train, y_train, x_val, y_val)
        blend_pred = (np.asarray(est_a.predict(x_val)) + np.asarray(est_b.predict(x_val))) / 2.0
        metrics = regression_metrics(y_val, blend_pred)
        label = f"50/50 mean of {best_two.iloc[0]['Model']} + {best_two.iloc[1]['Model']}"
        row = {"Model": "Blend Ensemble", "Config": label, "status": "ok",
               "fit_seconds": round(time.perf_counter() - t0, 2), "wave": WAVE,
               "seed": SEED, **metrics}
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        results.to_csv(SUITE_DIR / "wave3_tuning_results.csv", index=False)
        log(f"Blend [{label}] val MAE {metrics['MAE']:.2f} "
            f"RMSE {metrics['RMSE']:.2f} R2 {metrics['R2']:.4f}")
    except Exception as exc:  # noqa: BLE001
        log(f"Blend failed: {exc}")
    return results


def select_champion(results: pd.DataFrame) -> pd.Series:
    ok = results[results["status"] == "ok"].copy()
    best = ok.sort_values(["MAE", "RMSE", "R2"], ascending=[True, True, False]).iloc[0]
    log(
        f"Champion by validation MAE: {best['Model']} [{best['Config']}] "
        f"val MAE {best['MAE']:.2f}, RMSE {best['RMSE']:.2f}, R2 {best['R2']:.4f}"
    )
    return best


# ---------------------------------------------------------------------------
# Promotion (schema, finite-prediction, round-trip, atomic replace)
# ---------------------------------------------------------------------------
def build_estimator_for_row(model_name: str, config_label: str):
    for fam, cfgs in all_candidates().items():
        if fam == model_name:
            for label, est in cfgs:
                if label == config_label:
                    return est
    return None


def promote_champion(champion: pd.Series, splits: dict) -> dict:
    """Refit on train+validation, evaluate once on the holdout, promote atomically."""

    x_train, y_train, _ = splits["train"]
    x_val, y_val, _ = splits["validation"]
    x_hold, y_hold, _ = splits["holdout"]
    x_full = pd.concat([x_train, x_val], axis=0)
    y_full = pd.concat([y_train, y_val], axis=0)

    estimator = build_estimator_for_row(str(champion["Model"]), str(champion["Config"]))
    if estimator is None:
        raise RuntimeError("Could not rebuild the champion estimator for refit.")

    log(f"Refitting champion on train+validation ({len(x_full)} rows)...")
    t0 = time.perf_counter()
    fit_and_score(estimator, x_train, y_train, x_val, y_val)  # early-stop warm-up where applicable
    estimator.fit(x_full, y_full)
    refit_seconds = time.perf_counter() - t0

    # ---- ONE holdout evaluation for the locked champion --------------------
    hold_pred = np.asarray(estimator.predict(x_hold), dtype=float)
    hold_metrics = regression_metrics(y_hold, hold_pred)
    log(
        f"HOLDOUT (once) [{champion['Model']}] -> MAE {hold_metrics['MAE']:.2f}  "
        f"RMSE {hold_metrics['RMSE']:.2f}  R2 {hold_metrics['R2']:.4f}"
    )

    # ---- Serialization + schema + finite checks on staging -----------------
    staging = MODELS_DIR / "best_model.staging.pkl"
    joblib.dump(estimator, staging)
    reloaded = joblib.load(staging)
    probe = reloaded.predict(x_hold.head(5))
    if not np.isfinite(probe).all():
        staging.unlink(missing_ok=True)
        raise RuntimeError("Staging model produced non-finite predictions; promotion aborted.")
    # TransformedTargetRegressor exposes n_features_in_ as 0; check the inner estimator.
    n_features = int(getattr(reloaded, "n_features_in_", 0) or 0)
    if n_features == 0:
        inner = getattr(reloaded, "regressor_", None) or getattr(reloaded, "regressor", None)
        n_features = int(getattr(inner, "n_features_in_", 130) or 130)
    if n_features != 130:
        staging.unlink(missing_ok=True)
        raise RuntimeError(
            f"Staging model expects {n_features} features, contract requires 130."
        )
    if not np.allclose(probe, estimator.predict(x_hold.head(5)), atol=1e-9):
        staging.unlink(missing_ok=True)
        raise RuntimeError("Round-trip predictions differ; promotion aborted.")

    # ---- Atomic promotion with backups -------------------------------------
    final_path = MODELS_DIR / "best_model.pkl"
    if final_path.exists():
        shutil.copy2(final_path, MODELS_DIR / "best_model_previous.pkl")
        shutil.copy2(final_path, MODELS_DIR / "best_model_pre_wave3.pkl")
    os.replace(staging, final_path)

    # Persist the champion under its family name too (small files only).
    family_slug = (
        str(champion["Model"]).lower().replace(" ", "_").replace("(", "").replace(")", "")
    )
    joblib.dump(estimator, MODELS_DIR / f"wave3_{family_slug}.pkl")

    # ---- Final evaluation artifacts (plots + predictions + metrics) --------
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("final_test_metrics.csv", "final_test_predictions.csv",
                 "final_actual_vs_predicted.png", "final_residuals_vs_predicted.png",
                 "final_residuals_and_errors.png"):
        src = FINAL_DIR / name
        if src.exists():
            shutil.copy2(src, FINAL_ARCHIVE_DIR / name)

    pd.DataFrame([{
        "Model": champion["Model"], "Config": champion["Config"],
        "Train_Rows": int(len(x_full)), "Test_Rows": int(len(x_hold)),
        "Refit_Seconds": round(refit_seconds, 2), "Wave": WAVE, "Seed": SEED,
        "Evaluated_Utc": datetime.now(timezone.utc).isoformat(),
        **hold_metrics,
        "Val_MAE": float(champion["MAE"]), "Val_RMSE": float(champion["RMSE"]),
        "Val_R2": float(champion["R2"]),
    }]).to_csv(FINAL_DIR / "final_test_metrics.csv", index=False)
    (SUITE_DIR / "final_test_metrics.csv").write_text(
        (FINAL_DIR / "final_test_metrics.csv").read_text()
    )

    predictions = pd.DataFrame({
        "row_index": np.arange(len(x_hold)),
        "actual_loss": y_hold.to_numpy(),
        "predicted_loss": hold_pred,
        "absolute_error": np.abs(y_hold.to_numpy() - hold_pred),
    })
    predictions.to_csv(FINAL_DIR / "final_test_predictions.csv", index=False)
    save_final_evaluation_plots(y_hold, hold_pred)

    return {
        "hold_metrics": hold_metrics,
        "refit_seconds": round(refit_seconds, 2),
        "model_name": str(champion["Model"]),
        "config": str(champion["Config"]),
    }


# ---------------------------------------------------------------------------
# Model-comparison refresh + metadata
# ---------------------------------------------------------------------------
def refresh_model_comparison(results: pd.DataFrame, champion: pd.Series) -> None:
    ok = results[results["status"] == "ok"].copy()
    # One best config per family for the leaderboard (validation metrics).
    best_per_family = (
        ok.sort_values("MAE").groupby("Model", as_index=False).first()
        .sort_values(["MAE", "RMSE"], ascending=[True, True]).reset_index(drop=True)
    )
    best_per_family["Selected_For_Final_Test"] = best_per_family["Model"].eq(
        champion["Model"]
    ) & best_per_family["Config"].eq(champion["Config"])
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    best_per_family.to_csv(COMPARISON_DIR / "model_comparison.csv", index=False)
    (COMPARISON_DIR / "model_comparison.json").write_text(
        json.dumps(best_per_family.to_dict(orient="records"), indent=2, default=str)
    )

    ordered = best_per_family.sort_values("MAE", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    axes[0].barh(ordered["Model"], ordered["MAE"], color="#0D9488", edgecolor="#0F766E")
    axes[0].set_title("Validation MAE (Lower is Better)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("MAE (Loss Scale)")
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)
    axes[1].barh(ordered["Model"], ordered["R2"], color="#1D4ED8", edgecolor="#1E3A8A")
    axes[1].set_title("Validation R² (Higher is Better)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("R²")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)
    fig.suptitle(
        f"ClaimWise Wave-{WAVE} Validation Leaderboard (seed {SEED}, 7,500 validation rows)",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(COMPARISON_DIR / "model_comparison_bar_chart.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    log(f"Refreshed {COMPARISON_DIR / 'model_comparison.csv'} ({len(ordered)} families).")


def update_metadata(champion: pd.Series, promotion: dict, results: pd.DataFrame,
                    splits: dict) -> None:
    ok = results[results["status"] == "ok"]
    incumbent_row = ok[ok["Config"] == INCUMBENT_LABEL]
    incumbent_val = (
        {k: float(incumbent_row.iloc[0][k]) for k in
         ("MAE", "MSE", "RMSE", "R2", "Explained_Variance", "Median_AE", "MAPE")}
        if len(incumbent_row) else None
    )

    metadata_path = SUITE_DIR / "experiment_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata[f"wave{WAVE}"] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "why_new_split": (
            "The historical seed-42 test split had already been used for model "
            "comparison and selection in earlier phases (test-RMSE ranking in "
            "Src/model_training_utils.py::write_model_comparison), so it was not a "
            "pristine holdout. Wave 3 re-partitions all 50,000 rows with a fresh "
            "seed; within wave 3 the holdout is scored exactly once for the "
            "promoted champion."
        ),
        "split_protocol": {
            "train_rows": int(splits["train"][0].shape[0]),
            "validation_rows": int(splits["validation"][0].shape[0]),
            "holdout_rows": int(splits["holdout"][0].shape[0]),
            "final_refit_rows": int(splits["train"][0].shape[0] + splits["validation"][0].shape[0]),
            "stratified_by": "loss deciles (pd.qcut, q=10)",
            "membership_file": str(MEMBERSHIP_PATH.name),
        },
        "configs_tried": int(len(results)),
        "configs_ok": int(len(ok)),
        "selection": {
            "primary_metric": "validation MAE (original loss scale)",
            "champion": promotion["model_name"],
            "champion_config": promotion["config"],
            "champion_validation_metrics": {k: float(champion[k]) for k in
                ("MAE", "MSE", "RMSE", "R2", "Explained_Variance", "Median_AE", "MAPE")},
            "incumbent_recipe_retrained_validation_metrics": incumbent_val,
            "champion_final_holdout_metrics": promotion["hold_metrics"],
        },
        "protocol_notes": [
            "All learned preprocessing was fitted on training data only (imputation "
            "and deduplication were no-ops: the raw file has 0 missing values and 0 "
            "duplicate rows; the StandardScaler was fitted on training rows only; "
            "label encoders are target-free vocabulary mappings).",
            "The final holdout was scored exactly once, for the promoted champion, "
            "after selection was locked on the validation set.",
            "log1p-target variants use TransformedTargetRegressor so predictions are "
            "inverted with expm1 and every reported metric is on the original loss scale.",
            "Because the app serves predictions against the new 06_split_X_test.csv "
            "holdout file, the promoted model is refitted on the new train+validation "
            "rows so demo rows can never be training rows of the served model.",
        ],
    }
    metadata["selection"] = metadata[f"wave{WAVE}"]["selection"]
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    best_metrics = promotion["hold_metrics"]
    (MODELS_DIR / "best_model_metadata.json").write_text(json.dumps({
        "model": promotion["model_name"],
        "config": promotion["config"],
        "artifact": "Models/best_model.pkl",
        "feature_contract": "130 preprocessed features from Models/claimwise_preprocessor.pkl",
        "split_protocol": (
            f"wave-{WAVE} fresh re-split (seed {SEED}): "
            f"{metadata[f'wave{WAVE}']['split_protocol']['train_rows']:,} train / "
            f"{metadata[f'wave{WAVE}']['split_protocol']['validation_rows']:,} validation / "
            f"{metadata[f'wave{WAVE}']['split_protocol']['holdout_rows']:,} holdout"
        ),
        "validation_metrics": metadata[f"wave{WAVE}"]["selection"]["champion_validation_metrics"],
        "final_test_metrics": best_metrics,
        "trained_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "wave": WAVE,
        "previous_champion_backup": "Models/best_model_pre_wave3.pkl",
    }, indent=2, default=str))
    log("Updated experiment_metadata.json and Models/best_model_metadata.json.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    started = time.perf_counter()
    log(f"=== Wave-{WAVE} corrected-protocol experiment starting (seed {SEED}) ===")
    if splits_already_persisted():
        splits = load_persisted_splits()
    else:
        X_all, y_all, ids = load_and_verify()
        splits = make_fresh_splits(X_all, y_all, ids)
        persist_splits(splits)

    results = run_all_candidates(splits)
    results = maybe_add_blend(results, splits)
    champion = select_champion(results)
    promotion = promote_champion(champion, splits)
    refresh_model_comparison(results, champion)
    update_metadata(champion, promotion, results, splits)

    log(
        f"Wave {WAVE} finished in {time.perf_counter() - started:.1f}s — "
        f"champion {promotion['model_name']} [{promotion['config']}] "
        f"holdout MAE {promotion['hold_metrics']['MAE']:.2f}, "
        f"R2 {promotion['hold_metrics']['R2']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

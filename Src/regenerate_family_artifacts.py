"""Regenerate per-family evaluation artifacts on the wave-3 validation split.

Reads the authoritative leaderboard (Outputs/Model_Comparison/model_comparison.csv),
rebuilds every leaderboard configuration with the exact same seed/protocol used by
Src/wave3_experiment.py, refits it on the wave-3 training split (32,500 rows),
scores it once on the wave-3 validation split (7,500 rows, seed 2026) and rewrites:

  * Outputs/<Family>/metrics.csv          (schema unchanged, Split = wave-3 validation)
  * Outputs/<Family>/predictions.csv      (Actual_Loss, Predicted_Loss, Residual)
  * every diagnostic PNG in the folder    (actual-vs-predicted, residuals,
                                           feature importance, coefficients)
  * Outputs/Hist_Gradient_Boosting_Regressor/cross_validation_folds.csv (3-fold CV)
  * new folders: CatBoost_Regressor, XGBoost_Regressor, Blend_Ensemble
  * Outputs/Model_Comparison/actual_vs_predicted.png + residuals.png
    (multi-panel comparison for the top five families + Decision Tree baseline)
  * Models/xgboost_regressor.pkl          (best XGBoost config refit on 40,000 rows)

The holdout split is never touched. Metrics are asserted against the leaderboard
within floating-point tolerance so no artifact can drift away from the table the
website renders.

Run:  .venv/bin/python Src/regenerate_family_artifacts.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wave3_experiment as w3  # noqa: E402
from experiment_suite import FAMILY_TO_FOLDER, regression_metrics  # noqa: E402

OUTPUTS_DIR = w3.OUTPUTS_DIR
MODELS_DIR = w3.MODELS_DIR
COMPARISON_DIR = w3.COMPARISON_DIR
MODELS_DIR.mkdir(parents=True, exist_ok=True)

METRIC_KEYS = ("MAE", "MSE", "RMSE", "R2", "Explained_Variance", "Median_AE", "MAPE")
METRICS_SCHEMA = [
    "Model", "Split", "MAE", "MSE", "RMSE", "R2", "Explained_Variance",
    "Median_AE", "MAPE", "Train_Time_Sec", "Best_Config",
]
SPLIT_LABEL = f"Wave-3 Validation ({w3.VALIDATION_ROWS:,} rows, seed {w3.SEED})"

# Families whose estimators expose a linear coefficient vector.
LINEAR_FAMILIES = {
    "Linear Regression", "Ridge Regression", "Lasso Regression",
    "Elastic Net Regression", "Support Vector Regression (Linear)",
}
# Families whose estimators expose impurity/gain feature importances.
IMPORTANCE_FAMILIES = {
    "Random Forest Regressor", "Gradient Boosting Regressor",
    "Decision Tree Regressor", "Extra Trees Regressor", "AdaBoost Regressor",
    "XGBoost Regressor",
}

TOP_PANEL_FAMILIES = [
    "CatBoost Regressor", "XGBoost Regressor",
    "Hist Gradient Boosting Regressor", "Gradient Boosting Regressor",
    "Extra Trees Regressor", "Decision Tree Regressor",
]


def log(message: str) -> None:
    print(f"[regen] {message}", flush=True)


def build_estimator(candidates: dict, family: str, config_label: str):
    """Resolve a leaderboard row to a fresh estimator (log1p rows live in their own family)."""
    for label, est in candidates.get(family, []):
        if label == config_label:
            return est
    for fam, cfgs in candidates.items():
        for label, est in cfgs:
            if label == config_label:
                return est
    raise KeyError(f"No configuration [{config_label}] for family [{family}]")


def inner_estimator(estimator):
    return getattr(estimator, "regressor_", estimator)


def feature_importance_frame(estimator, feature_names) -> pd.DataFrame:
    est = inner_estimator(estimator)
    values = est.feature_importances_
    return (
        pd.DataFrame({"Feature": list(feature_names), "Importance": values})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def coefficient_frame(estimator, feature_names) -> pd.DataFrame:
    est = inner_estimator(estimator)
    coefs = np.asarray(est.coef_, dtype=float).ravel()
    return (
        pd.DataFrame({
            "Feature": list(feature_names),
            "Coefficient": coefs,
            "AbsCoefficient": np.abs(coefs),
        })
        .sort_values("AbsCoefficient", ascending=False)
        .reset_index(drop=True)
    )


def plot_actual_vs_predicted(path: Path, title: str, y_true, y_pred) -> None:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    plt.figure(figsize=(7.5, 6))
    plt.scatter(actual, predicted, alpha=0.35, color="#D4512A", edgecolors="none")
    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    plt.plot([lo, hi], [lo, hi], color="#18181B", linewidth=1.5)
    plt.title(title)
    plt.xlabel("Actual loss")
    plt.ylabel("Predicted loss")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_residuals(path: Path, title: str, y_pred, residuals) -> None:
    predicted = np.asarray(y_pred, dtype=float)
    plt.figure(figsize=(7.5, 5.5))
    plt.scatter(predicted, residuals, alpha=0.35, color="#E58A3A", edgecolors="none")
    plt.axhline(0, color="#18181B", linewidth=1.5)
    plt.title(title)
    plt.xlabel("Predicted loss")
    plt.ylabel("Residual (actual - predicted)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_top_features(path: Path, title: str, frame: pd.DataFrame, value_col: str,
                      xlabel: str) -> None:
    top = frame.head(20).sort_values(value_col)
    plt.figure(figsize=(9, 8))
    plt.barh(top["Feature"], top[value_col], color="#0D9488")
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def assert_matches_leaderboard(row: pd.Series, metrics: dict, context: str) -> None:
    for key in METRIC_KEYS:
        expected = float(row[key])
        actual = float(metrics[key])
        if not np.isclose(actual, expected, rtol=1e-6, atol=1e-4):
            raise AssertionError(
                f"{context}: {key} mismatch — regenerated {actual!r} vs leaderboard {expected!r}"
            )


def write_family_artifacts(
    family: str,
    row: pd.Series,
    estimator,
    y_val: pd.Series,
    y_pred: np.ndarray,
    feature_names: list[str],
    fit_seconds: float,
) -> dict:
    folder_name = FAMILY_TO_FOLDER[family]
    out_dir = OUTPUTS_DIR / folder_name
    folder_existed = out_dir.exists()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = regression_metrics(y_val, y_pred)
    assert_matches_leaderboard(row, metrics, f"{family} [{row['Config']}]")

    pd.DataFrame([{
        "Model": family,
        "Split": SPLIT_LABEL,
        **{k: metrics[k] for k in METRIC_KEYS},
        "Train_Time_Sec": round(float(fit_seconds), 2),
        "Best_Config": str(row["Config"]),
    }])[METRICS_SCHEMA].to_csv(out_dir / "metrics.csv", index=False)

    actual = np.asarray(y_val, dtype=float)
    residual = actual - np.asarray(y_pred, dtype=float)
    pd.DataFrame({
        "Actual_Loss": actual,
        "Predicted_Loss": np.asarray(y_pred, dtype=float),
        "Residual": residual,
    }).to_csv(out_dir / "predictions.csv", index=False)

    slug = folder_name.lower()
    plot_actual_vs_predicted(
        out_dir / f"{slug}_actual_vs_predicted.png",
        f"Actual vs Predicted loss — {family}",
        actual, y_pred,
    )
    plot_residuals(
        out_dir / f"{slug}_residuals.png",
        f"Residual plot — {family}",
        y_pred, residual,
    )

    imp_needed = False
    coef_needed = False
    for existing in sorted(out_dir.iterdir()):
        if not existing.is_file() or existing.suffix != ".png":
            continue
        name = existing.name
        if "actual_vs_predicted" in name and name != f"{slug}_actual_vs_predicted.png":
            plot_actual_vs_predicted(
                existing, f"Actual vs Predicted loss — {family}", actual, y_pred,
            )
        elif "residual" in name and name != f"{slug}_residuals.png":
            plot_residuals(existing, f"Residual plot — {family}", y_pred, residual)
        elif "importance" in name:
            imp_needed = True
        elif "coefficient" in name:
            coef_needed = True

    # CSV side-cars and any PNG found above (folders use two naming generations).
    has_imp_csv = (out_dir / "feature_importance.csv").exists() or any(
        f.suffix == ".csv" and "importance" in f.name for f in out_dir.iterdir() if f.is_file()
    )
    imp_frame = None
    if family in IMPORTANCE_FAMILIES and not folder_existed:
        # Brand-new folder (XGBoost): create the importance artifacts directly.
        imp_frame = feature_importance_frame(estimator, feature_names)
        imp_frame.to_csv(out_dir / "feature_importance.csv", index=False)
        plot_top_features(
            out_dir / "feature_importance.png",
            f"{family} — Top 20 Feature Importances",
            imp_frame, "Importance", "Importance",
        )
    elif family in IMPORTANCE_FAMILIES and (imp_needed or has_imp_csv):
        imp_frame = feature_importance_frame(estimator, feature_names)
        for existing in sorted(out_dir.iterdir()):
            if not existing.is_file():
                continue
            name = existing.name
            if name.endswith(".csv") and "importance" in name:
                imp_frame.to_csv(existing, index=False)
            elif name.endswith(".png") and "importance" in name:
                plot_top_features(
                    existing, f"{family} — Top 20 Feature Importances",
                    imp_frame, "Importance", "Importance",
                )
    elif imp_needed:
        log(f"WARNING: {family} has importance plots but no importance vector — skipped.")

    coef_frame = None
    if family in LINEAR_FAMILIES:
        coef_frame = coefficient_frame(estimator, feature_names)
        for existing in sorted(out_dir.iterdir()):
            if not existing.is_file():
                continue
            name = existing.name
            if name.endswith(".csv") and "coefficient" in name:
                coef_frame.to_csv(existing, index=False)
            elif name.endswith(".png") and "coefficient" in name:
                plot_top_features(
                    existing, f"{family} — Top 20 Features by Absolute Coefficient",
                    coef_frame, "AbsCoefficient", "Absolute Coefficient",
                )

    log(f"{family}: metrics/predictions/plots written to Outputs/{folder_name} "
        f"(MAE {metrics['MAE']:.2f}, RMSE {metrics['RMSE']:.2f}, R2 {metrics['R2']:.4f})")
    return {"metrics": metrics, "actual": actual, "predicted": np.asarray(y_pred, dtype=float)}


def regenerate_histgb_cv_folds(candidates: dict, x_train: pd.DataFrame,
                               y_train: pd.Series) -> None:
    """3-fold CV on the wave-3 train split with the family's best (log1p) config."""
    estimator = build_estimator(
        candidates, "Hist Gradient Boosting Regressor",
        "log1p(HistGB lr=0.03,iter=1500,leaves=31,min_leaf=60)",
    )
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    rows = []
    t0 = time.perf_counter()
    for fold, (tr_idx, va_idx) in enumerate(kfold.split(x_train), start=1):
        x_tr, y_tr = x_train.iloc[tr_idx], y_train.iloc[tr_idx]
        x_va, y_va = x_train.iloc[va_idx], y_train.iloc[va_idx]
        estimator.fit(x_tr, y_tr)
        pred = estimator.predict(x_va)
        metrics = regression_metrics(y_va, pred)
        iterations = int(inner_estimator(estimator).n_iter_)
        rows.append({
            "fold": fold, "iterations": iterations,
            "MAE": metrics["MAE"], "MSE": metrics["MSE"],
            "RMSE": metrics["RMSE"], "R2": metrics["R2"],
        })
        log(f"HistGB CV fold {fold}: iterations={iterations} "
            f"MAE {metrics['MAE']:.2f} RMSE {metrics['RMSE']:.2f} R2 {metrics['R2']:.4f}")
    out = OUTPUTS_DIR / "Hist_Gradient_Boosting_Regressor" / "cross_validation_folds.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log(f"Wrote {out} ({time.perf_counter() - t0:.1f}s).")


def write_comparison_panels(results: dict, y_val: pd.Series) -> None:
    """Multi-panel actual-vs-predicted and residual figures for Model_Comparison."""
    panels = [f for f in TOP_PANEL_FAMILIES if f in results]
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    for ax, name in zip(axes.ravel(), panels):
        res = results[name]
        actual, pred = res["actual"], res["predicted"]
        ax.scatter(actual, pred, alpha=0.3, color="#D4512A", edgecolors="none", s=18)
        lo = min(actual.min(), pred.min())
        hi = max(actual.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], color="#18181B", linewidth=1.3)
        ax.set_title(f"{name}\nMAE {res['metrics']['MAE']:,.0f} · R2 "
                     f"{res['metrics']['R2']:.3f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Actual loss", fontsize=9)
        ax.set_ylabel("Predicted loss", fontsize=9)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(
        "Actual vs Predicted Loss — top-5 leaderboard families + Decision Tree baseline "
        "(wave-3 validation, 7,500 rows)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(COMPARISON_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    for ax, name in zip(axes.ravel(), panels):
        res = results[name]
        resid = res["actual"] - res["predicted"]
        ax.scatter(res["predicted"], resid, alpha=0.3, color="#E58A3A",
                   edgecolors="none", s=18)
        ax.axhline(0, color="#18181B", linewidth=1.3)
        ax.set_title(f"{name}\nMAE {res['metrics']['MAE']:,.0f} · RMSE "
                     f"{res['metrics']['RMSE']:,.0f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted loss", fontsize=9)
        ax.set_ylabel("Residual (actual - predicted)", fontsize=9)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(
        "Residual Analysis — top-5 leaderboard families + Decision Tree baseline "
        "(wave-3 validation, 7,500 rows)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(COMPARISON_DIR / "residuals.png", dpi=160)
    plt.close(fig)
    log("Regenerated Outputs/Model_Comparison/actual_vs_predicted.png and residuals.png.")


def fit_xgboost_artifact(candidates: dict, splits: dict) -> None:
    """Refit the winning log1p XGBoost config on all 40,000 train+val rows."""
    estimator = build_estimator(
        candidates, "XGBoost Regressor", "log1p(XGB n_estimators=800,lr=0.05,depth=6)",
    )
    x_tr, y_tr, _ = splits["train"]
    x_va, y_va, _ = splits["validation"]
    x_full = pd.concat([x_tr, x_va], axis=0)
    y_full = pd.concat([y_tr, y_va], axis=0)
    # Warm-up pass first (identical to the leaderboard protocol), then full refit.
    w3.fit_and_score(estimator, x_tr, y_tr, x_va, y_va)
    estimator.fit(x_full, y_full)
    probe = np.asarray(estimator.predict(x_va.head(5)), dtype=float)
    if not np.isfinite(probe).all():
        raise RuntimeError("XGBoost artifact produced non-finite predictions.")
    staging = MODELS_DIR / "xgboost_regressor.staging.pkl"
    joblib.dump(estimator, staging)
    staging.replace(MODELS_DIR / "xgboost_regressor.pkl")
    log("Saved Models/xgboost_regressor.pkl (log1p best config, refit on 40,000 rows).")


def main() -> int:
    leaderboard_path = COMPARISON_DIR / "model_comparison.csv"
    leaderboard = pd.read_csv(leaderboard_path)
    log(f"Leaderboard rows: {len(leaderboard)}")

    splits = w3.load_persisted_splits()
    x_train, y_train, _ = splits["train"]
    x_val, y_val, _ = splits["validation"]
    feature_names = list(x_train.columns)
    candidates = w3.all_candidates()

    results: dict[str, dict] = {}
    fitted: dict[tuple[str, str], object] = {}

    # Pass 1 — every non-blend leaderboard row.
    for _, row in leaderboard.iterrows():
        family = str(row["Model"])
        if family == "Blend Ensemble":
            continue
        if family not in FAMILY_TO_FOLDER:
            raise KeyError(f"No output folder mapping for family {family!r}")
        config_label = str(row["Config"])
        log(f"Fitting {family} [{config_label}] ...")
        estimator = build_estimator(candidates, family, config_label)
        t0 = time.perf_counter()
        w3.fit_and_score(estimator, x_train, y_train, x_val, y_val)
        fit_seconds = time.perf_counter() - t0
        y_pred = np.asarray(estimator.predict(x_val), dtype=float)
        fitted[(family, config_label)] = estimator
        results[family] = write_family_artifacts(
            family, row, estimator, y_val, y_pred, feature_names, fit_seconds,
        )

    # Pass 2 — the 50/50 blend. Replicate the original maybe_add_blend recipe:
    # the two best distinct families (excluding the blend itself) from the
    # wave-3 tuning log at blend-construction time.
    blend_rows = leaderboard[leaderboard["Model"] == "Blend Ensemble"]
    if len(blend_rows):
        row = blend_rows.iloc[0]
        tuning = pd.read_csv(w3.SUITE_DIR / "wave3_tuning_results.csv")
        ok = tuning[tuning["status"].astype(str) == "ok"]
        ok = ok[ok["Model"] != "Blend Ensemble"]
        best_two = (
            ok.sort_values("MAE").groupby("Model", as_index=False).first()
            .sort_values("MAE").head(2)
        )
        t0 = time.perf_counter()
        constituent_preds = []
        for record in best_two.itertuples():
            key = (str(record.Model), str(record.Config))
            estimator = fitted.get(key)
            if estimator is None:
                log(f"Blend constituent needs fitting: {key[0]} [{key[1]}]")
                estimator = build_estimator(candidates, key[0], key[1])
                w3.fit_and_score(estimator, x_train, y_train, x_val, y_val)
                fitted[key] = estimator
            constituent_preds.append(np.asarray(estimator.predict(x_val), dtype=float))
        blend_pred = sum(constituent_preds) / float(len(constituent_preds))
        fit_seconds = time.perf_counter() - t0
        log("Blend constituents: " + " + ".join(
            f"{m} [{c}]" for m, c in zip(best_two["Model"], best_two["Config"]),
        ))
        results["Blend Ensemble"] = write_family_artifacts(
            "Blend Ensemble", row, fitted[cat_key], y_val, blend_pred,
            feature_names, fit_seconds,
        )

    missing = [str(r["Model"]) for _, r in leaderboard.iterrows() if r["Model"] not in results]
    if missing:
        raise RuntimeError(f"Leaderboard rows without regenerated artifacts: {missing}")

    regenerate_histgb_cv_folds(candidates, x_train, y_train)
    write_comparison_panels(results, y_val)
    fit_xgboost_artifact(candidates, splits)
    log(f"Done — {len(results)} leaderboard families regenerated on wave-3 validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

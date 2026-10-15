"""ClaimWise Wave-3 classification task: high-severity claim triage flag.

Business justification
----------------------
Insurance claims operations routinely triage incoming claims: higher-severity
claims are routed to senior manual adjusters while routine claims are settled
faster. This task models that workflow as a *separate* binary classification
problem. It does NOT measure the regression model's accuracy; it answers a
different operational question ("which claims are likely to be high severity?").

Target definition
-----------------
    high_loss = 1  if  loss >= threshold
    high_loss = 0  otherwise

The threshold is the 75th percentile of the TRAINING split's loss values only
(upper-quartile triage). It is never computed from validation or holdout data,
so the label definition cannot leak holdout distribution information.
Rationale: an upper-quartile cutoff is a standard, interpretable triage band
(~25% of claims flagged for senior review).

Leakage controls
----------------
* The binary label is derived from the target ONLY; it is never appended to
  the feature matrix.
* Class imbalance is handled with training-only class weights.
* Stratified train/validation/holdout splits (the same wave-3 partition as the
  regression task) keep the positive rate consistent across splits.
* Classifiers are compared on validation F1 (binary, positive class =
  high_loss); the winner is refitted on train+validation and evaluated ONCE on
  the untouched holdout.

Run:  .venv/bin/python Src/wave3_classification.py
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

SEED = 2026
WAVE = 3
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Dataset"
MODELS_DIR = PROJECT_ROOT / "Models"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
CLASS_DIR = OUTPUTS_DIR / "ClassificationExperiment"

CLASSIFIER_PATH = MODELS_DIR / "high_loss_classifier.pkl"
METADATA_PATH = MODELS_DIR / "high_loss_classifier_metadata.json"


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp} UTC] {message}", flush=True)


def load_splits():
    x_train = pd.read_csv(DATASET_DIR / "06_split_X_train.csv")
    y_train = pd.read_csv(DATASET_DIR / "06_split_y_train.csv")["loss"].astype(float)
    x_val = pd.read_csv(DATASET_DIR / "06_split_X_val.csv")
    y_val = pd.read_csv(DATASET_DIR / "06_split_y_val.csv")["loss"].astype(float)
    x_hold = pd.read_csv(DATASET_DIR / "06_split_X_test.csv")
    y_hold = pd.read_csv(DATASET_DIR / "06_split_y_test.csv")["loss"].astype(float)
    return x_train, y_train, x_val, y_val, x_hold, y_hold


def binary_metrics(y_true, y_prob, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "F1_binary": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
        "Confusion_TN": int(confusion_matrix(y_true, y_pred).ravel()[0]),
        "Confusion_FP": int(confusion_matrix(y_true, y_pred).ravel()[1]),
        "Confusion_FN": int(confusion_matrix(y_true, y_pred).ravel()[2]),
        "Confusion_TP": int(confusion_matrix(y_true, y_pred).ravel()[3]),
        "Positive_Rate_Pred": float(y_pred.mean()),
    }


def candidate_classifiers(train_pos_rate: float) -> list[tuple[str, object]]:
    """Training-only class weights; no resampling of validation/holdout."""

    ratio = (1 - train_pos_rate) / max(train_pos_rate, 1e-9)
    return [
        (
            "Logistic Regression (class_weight=balanced)",
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=SEED, n_jobs=None
            ),
        ),
        (
            "Random Forest Classifier (class_weight=balanced_subsample)",
            RandomForestClassifier(
                n_estimators=400, min_samples_leaf=5, max_features="sqrt",
                class_weight="balanced_subsample", random_state=SEED, n_jobs=max(1, (os.cpu_count() or 4) - 1),
            ),
        ),
        (
            "Hist Gradient Boosting Classifier (class_weight=balanced)",
            HistGradientBoostingClassifier(
                learning_rate=0.06, max_iter=400, max_leaf_nodes=31,
                min_samples_leaf=40, class_weight="balanced",
                early_stopping=True, random_state=SEED,
            ),
        ),
        (
            f"Hist Gradient Boosting Classifier (scale_pos_weight={ratio:.2f} via sample_weight)",
            HistGradientBoostingClassifier(
                learning_rate=0.06, max_iter=400, max_leaf_nodes=31,
                min_samples_leaf=40, early_stopping=True, random_state=SEED,
            ),
        ),
    ]


def main() -> int:
    started = time.perf_counter()
    log("=== Wave-3 high-loss classification experiment starting ===")
    x_train, y_train_raw, x_val, y_val_raw, x_hold, y_hold_raw = load_splits()

    # Threshold from TRAINING loss only (upper quartile).
    threshold = float(y_train_raw.quantile(0.75))
    log(f"Threshold (train P75) = {threshold:.2f}")

    y_train = (y_train_raw >= threshold).astype(int)
    y_val = (y_val_raw >= threshold).astype(int)
    y_hold = (y_hold_raw >= threshold).astype(int)
    log(
        f"Positive rates — train {y_train.mean():.4f}, validation {y_val.mean():.4f}, "
        f"holdout {y_hold.mean():.4f}"
    )

    CLASS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    fitted = {}
    for name, clf in candidate_classifiers(float(y_train.mean())):
        t0 = time.perf_counter()
        try:
            if "sample_weight" in name:
                w = np.where(y_train == 1, (1 - y_train.mean()) / max(y_train.mean(), 1e-9), 1.0)
                clf.fit(x_train, y_train, sample_weight=w)
            else:
                clf.fit(x_train, y_train)
            proba_val = clf.predict_proba(x_val)[:, 1]
            metrics = binary_metrics(y_val, proba_val, threshold=0.5)
            seconds = time.perf_counter() - t0
            rows.append({"Classifier": name, "status": "ok", "fit_seconds": round(seconds, 2),
                         "wave": WAVE, "seed": SEED, "eval_split": "validation", **metrics})
            fitted[name] = clf
            log(f"  [{name}] val F1 {metrics['F1_binary']:.4f}  "
                f"P {metrics['Precision']:.4f}  R {metrics['Recall']:.4f}  "
                f"ROC-AUC {metrics['ROC_AUC']:.4f}  ({seconds:.1f}s)")
        except Exception as exc:  # noqa: BLE001
            rows.append({"Classifier": name, "status": f"failed: {exc}",
                         "fit_seconds": round(time.perf_counter() - t0, 2), "wave": WAVE,
                         "seed": SEED, "eval_split": "validation",
                         **{k: np.nan for k in ("F1_binary", "Precision", "Recall", "Accuracy",
                                                "ROC_AUC", "PR_AUC", "Confusion_TN", "Confusion_FP",
                                                "Confusion_FN", "Confusion_TP", "Positive_Rate_Pred")}})
            log(f"  [{name}] FAILED: {exc}")

    results = pd.DataFrame(rows)
    results.to_csv(CLASS_DIR / "validation_results.csv", index=False)

    ok = results[results["status"] == "ok"]
    if ok.empty:
        raise RuntimeError("No classifier trained successfully.")
    champion_row = ok.sort_values("F1_binary", ascending=False).iloc[0]
    champion_name = str(champion_row["Classifier"])
    log(f"Champion by validation F1: {champion_name} (F1 {champion_row['F1_binary']:.4f})")

    # Refit champion on train+validation, evaluate once on the untouched holdout.
    champion = fitted[champion_name]
    x_full = pd.concat([x_train, x_val], axis=0)
    y_full = np.concatenate([y_train.to_numpy(), y_val.to_numpy()])
    t0 = time.perf_counter()
    if "sample_weight" in champion_name:
        w = np.where(y_full == 1, (1 - y_train.mean()) / max(y_train.mean(), 1e-9), 1.0)
        champion.fit(x_full, y_full, sample_weight=w)
    else:
        champion.fit(x_full, y_full)
    refit_seconds = time.perf_counter() - t0

    proba_hold = champion.predict_proba(x_hold)[:, 1]
    hold_metrics = binary_metrics(y_hold, proba_hold, threshold=0.5)
    log(
        f"HOLDOUT (once) [{champion_name}] F1 {hold_metrics['F1_binary']:.4f}  "
        f"P {hold_metrics['Precision']:.4f}  R {hold_metrics['Recall']:.4f}  "
        f"Acc {hold_metrics['Accuracy']:.4f}  ROC-AUC {hold_metrics['ROC_AUC']:.4f}  "
        f"PR-AUC {hold_metrics['PR_AUC']:.4f}"
    )

    # Persist artifacts.
    joblib.dump(champion, CLASSIFIER_PATH)
    hold_row = {"Classifier": champion_name, "status": "ok",
                "fit_seconds": round(refit_seconds, 2), "wave": WAVE, "seed": SEED,
                "eval_split": "holdout (untouched)", **hold_metrics}
    pd.DataFrame([hold_row]).to_csv(CLASS_DIR / "final_holdout_metrics.csv", index=False)

    cm = confusion_matrix(y_hold, (proba_hold >= 0.5).astype(int))
    pd.DataFrame(
        cm, index=["Actual 0 (routine)", "Actual 1 (high_loss)"],
        columns=["Predicted 0", "Predicted 1"],
    ).to_csv(CLASS_DIR / "confusion_matrix_holdout.csv")

    predictions = pd.DataFrame({
        "row_index": np.arange(len(y_hold)),
        "actual_high_loss": y_hold.to_numpy(),
        "predicted_probability": proba_hold,
        "predicted_label": (proba_hold >= 0.5).astype(int),
    })
    predictions.to_csv(CLASS_DIR / "holdout_predictions.csv", index=False)

    # ROC and PR curves.
    fpr, tpr, _ = roc_curve(y_hold, proba_hold)
    prec, rec, _ = precision_recall_curve(y_hold, proba_hold)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(fpr, tpr, color="#0D9488", lw=2)
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title(f"ROC Curve (holdout) — AUC {hold_metrics['ROC_AUC']:.4f}")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[1].plot(rec, prec, color="#1D4ED8", lw=2)
    axes[1].set_title(f"Precision-Recall Curve (holdout) — AP {hold_metrics['PR_AUC']:.4f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    fig.suptitle(
        f"High-Loss Triage Classifier — positive class: loss >= {threshold:,.2f} (train P75)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(CLASS_DIR / "roc_pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "task": "binary classification — high-loss claim triage (separate from regression)",
        "positive_class_definition": f"high_loss = 1 when loss >= {threshold:.2f}",
        "threshold": threshold,
        "threshold_source": "75th percentile of the TRAINING split loss only (no validation/holdout leakage)",
        "selection_metric": "validation F1 (binary, positive class = high_loss)",
        "champion": champion_name,
        "validation_metrics": {k: float(champion_row[k]) if k not in
                               ("Confusion_TN", "Confusion_FP", "Confusion_FN", "Confusion_TP")
                               else int(champion_row[k])
                               for k in ("F1_binary", "Precision", "Recall", "Accuracy",
                                         "ROC_AUC", "PR_AUC", "Confusion_TN", "Confusion_FP",
                                         "Confusion_FN", "Confusion_TP")},
        "final_holdout_metrics": hold_metrics,
        "split_protocol": "wave-3 fresh re-split (seed 2026): 32,500 train / 7,500 validation / 10,000 holdout",
        "class_imbalance_handling": "training-only class weights / sample weights",
        "trained_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "wave": WAVE,
        "note": (
            "This classifier's F1 measures high-severity claim triage, NOT the "
            "accuracy of the continuous loss regression model. The two tasks are "
            "reported separately."
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str))
    log(f"Saved {CLASSIFIER_PATH.name} and {METADATA_PATH.name}.")
    log(f"Classification experiment finished in {metadata['elapsed_seconds']}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

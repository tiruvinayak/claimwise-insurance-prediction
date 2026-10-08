"""Train and evaluate the ClaimWise Hist Gradient Boosting Regressor.

Source of the algorithm logic: the provided Google Drive notebook
``Home_Credit_Default_Risk_FINAL_FIXED.ipynb`` (LightGBM + K-fold early stopping).

Adaptation map (provided code -> ClaimWise):

========================================  =========================================================
Provided (Home Credit)                    ClaimWise
========================================  =========================================================
TARGET (binary), ROC-AUC / Gini           ``loss`` (continuous), MAE / MSE / RMSE / R2
LGBMClassifier                            HistGradientBoostingRegressor (sklearn, no network install)
StratifiedKFold(3, shuffle, SEED)         KFold(3, shuffle, SEED) - stratification is meaningless
                                          for a continuous target
num_leaves=31                             max_leaf_nodes=31
max_depth=-1                              max_depth=None (unlimited)
min_child_samples=50                      min_samples_leaf=50
colsample_bytree=0.80                     max_features=0.80
reg_lambda=0.50                           l2_regularization=0.50
n_estimators=2500                         max_iter=2500 (cut by early stopping)
early_stopping(rounds=150)                early_stopping=True, n_iter_no_change=150, scoring="loss"
eval_set per fold                         X_val / y_val per fold
OOF + fold score reporting                preserved verbatim in ``run_cross_validation``
prediction sanity checks                  preserved in ``validate_predictions``
/reg_alpha, /subsample, /n_jobs           no sklearn HistGB equivalent - dropped, documented
6 historical tables + SK_ID key maps      no ClaimWise equivalent (single flat table) - see notebook 08
add_application_features ratio columns    Home Credit named finance fields do not exist in ClaimWise
                                          and must not be invented - not carried over
========================================  =========================================================
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from model_training_utils import (
    OUTPUTS_DIR,
    load_prepared_data,
    reduce_numeric_memory,
    save_model_diagnostic_plots,
    save_model_results,
)

MODEL_NAME = "Hist Gradient Boosting Regressor"
MODEL_FILE = "hist_gradient_boosting_regressor.pkl"
OUTPUT_FOLDER = "Hist_Gradient_Boosting_Regressor"

# Configuration carried over from the provided notebook (SEED / N_SPLITS / params).
SEED = 42
N_SPLITS = 3


def build_model(max_iter: int = 2500) -> HistGradientBoostingRegressor:
    """Create the ClaimWise HistGradientBoosting model (single source of truth).

    Hyper-parameters are the provided LightGBM parameters mapped to the closest
    sklearn HistGradientBoosting equivalents. Used by the training script and by
    the matching Jupyter notebook so they are defined in exactly one place.
    """
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.035,
        max_iter=max_iter,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=50,
        max_features=0.80,
        l2_regularization=0.50,
        random_state=SEED,
    )


def build_cv_model() -> HistGradientBoostingRegressor:
    """Cross-validation variant with the provided early-stopping behaviour."""

    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.035,
        max_iter=2500,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=50,
        max_features=0.80,
        l2_regularization=0.50,
        early_stopping=True,
        n_iter_no_change=150,
        scoring="loss",
        random_state=SEED,
    )


def run_cross_validation(
    x_train: pd.DataFrame, y_train: pd.Series
) -> Dict[str, object]:
    """K-fold training with early stopping, reporting OOF regression metrics.

    This is the provided notebook's Cell 16 (fold loop, OOF array, fold scores,
    early stopping) adapted from binary classification to regression.
    """

    x_train = reduce_numeric_memory(x_train.copy())
    y_array = y_train.to_numpy()

    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_prediction = np.zeros(len(x_train), dtype=np.float64)

    fold_scores: List[Dict[str, float]] = []
    fold_iterations: List[int] = []

    for fold, (train_index, valid_index) in enumerate(kfold.split(x_train), start=1):
        x_tr = x_train.iloc[train_index]
        x_va = x_train.iloc[valid_index]
        y_tr = y_array[train_index]
        y_va = y_array[valid_index]

        model = build_cv_model()
        model.fit(x_tr, y_tr, X_val=x_va, y_val=y_va)
        valid_prediction = model.predict(x_va)

        oof_prediction[valid_index] = valid_prediction
        fold_iterations.append(int(model.n_iter_))
        fold_scores.append(
            {
                "fold": fold,
                "iterations": int(model.n_iter_),
                "MAE": float(mean_absolute_error(y_va, valid_prediction)),
                "MSE": float(mean_squared_error(y_va, valid_prediction)),
                "RMSE": float(mean_squared_error(y_va, valid_prediction) ** 0.5),
                "R2": float(r2_score(y_va, valid_prediction)),
            }
        )
        print(
            f"Fold {fold}/{N_SPLITS}: iterations={model.n_iter_} "
            f"MAE={fold_scores[-1]['MAE']:.4f} RMSE={fold_scores[-1]['RMSE']:.4f} "
            f"R2={fold_scores[-1]['R2']:.4f}"
        )

    oof_metrics = {
        "MAE": float(mean_absolute_error(y_array, oof_prediction)),
        "MSE": float(mean_squared_error(y_array, oof_prediction)),
        "RMSE": float(mean_squared_error(y_array, oof_prediction) ** 0.5),
        "R2": float(r2_score(y_array, oof_prediction)),
    }

    return {
        "fold_scores": pd.DataFrame(fold_scores),
        "fold_iterations": fold_iterations,
        "best_iterations": int(np.median(fold_iterations)),
        "oof_metrics": oof_metrics,
        "oof_prediction": oof_prediction,
    }


def validate_predictions(y_true, predictions) -> None:
    """Provided notebook Cell 17 - prediction sanity checks, regression version."""

    if len(predictions) != len(y_true):
        raise ValueError("Prediction length does not match the target length.")
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain NaN/inf values.")


def main() -> None:
    x_train, x_test, y_train, y_test = load_prepared_data()
    print(f"Loaded prepared data: X_train={x_train.shape}, X_test={x_test.shape}")

    print(f"\n{N_SPLITS}-fold cross-validation with early stopping...")
    cv_result = run_cross_validation(x_train, y_train)
    print("OOF metrics:", cv_result["oof_metrics"])
    print("Best iterations per fold:", cv_result["fold_iterations"])

    best_iter = max(int(cv_result["best_iterations"]), 1)
    print(f"\nFitting final model on all {len(x_train)} training rows "
          f"(max_iter={best_iter}, median best fold iteration)...")

    model = build_model(max_iter=best_iter)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    validate_predictions(y_test, predictions)

    metrics = save_model_results(
        MODEL_NAME,
        MODEL_FILE,
        model,
        x_test,
        y_test,
        predictions,
    )
    save_model_diagnostic_plots(MODEL_NAME, OUTPUT_FOLDER, y_test, predictions)

    cv_result["fold_scores"].to_csv(
        OUTPUTS_DIR / OUTPUT_FOLDER / "cross_validation_folds.csv", index=False
    )

    print(f"{MODEL_NAME} metrics:", metrics)
    print(f"Cross-validation fold metrics:\n{cv_result['fold_scores']}")


if __name__ == "__main__":
    main()

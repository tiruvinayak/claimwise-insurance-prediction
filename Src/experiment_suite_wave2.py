"""Wave-2 targeted search extending the ClaimWise experiment suite.

Adds the previous champion's proven recipe family (low learning rate,
high min_samples_leaf HistGB) and deeper/longer Gradient Boosting configs.
Selection uses the validation set ONLY. The untouched test set is scored
exactly once for the final champion. Results are merged into
Outputs/ExperimentSuite/tuning_results.csv and, if the champion changes,
the production artifacts are re-promoted atomically.

Run:  .venv/bin/python Src/experiment_suite_wave2.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

from experiment_suite import (  # noqa: E402
    COMPARISON_DIR,
    SEED,
    SUITE_DIR,
    analyse_data,
    candidate_families,
    load_data,
    log,
    promote_model,
    regression_metrics,
    refresh_model_comparison_csv,
    run_log1p_experiments,
    save_diagnostic_plots,
    save_final_evaluation_plots,
    FAMILY_TO_FOLDER,
)
from pathlib import Path  # noqa: E402
import json  # noqa: E402

WAVE2_CONFIGS = {
    "Hist Gradient Boosting Regressor": [
        ("lr=0.035,iter=830,leaves=31,min_leaf=50,l2=0.5 (prior champion recipe)",
         HistGradientBoostingRegressor(learning_rate=0.035, max_iter=830, max_leaf_nodes=31,
                                       min_samples_leaf=50, l2_regularization=0.5,
                                       early_stopping=True, validation_fraction=0.1,
                                       n_iter_no_change=10, random_state=SEED)),
        ("lr=0.05,iter=1000,leaves=31,min_leaf=50,l2=0.5",
         HistGradientBoostingRegressor(learning_rate=0.05, max_iter=1000, max_leaf_nodes=31,
                                       min_samples_leaf=50, l2_regularization=0.5,
                                       early_stopping=True, random_state=SEED)),
        ("lr=0.035,iter=1200,leaves=63,min_leaf=40,l2=0.5",
         HistGradientBoostingRegressor(learning_rate=0.035, max_iter=1200, max_leaf_nodes=63,
                                       min_samples_leaf=40, l2_regularization=0.5,
                                       early_stopping=True, random_state=SEED)),
        ("lr=0.05,iter=1000,leaves=31,min_leaf=30,l2=1.0",
         HistGradientBoostingRegressor(learning_rate=0.05, max_iter=1000, max_leaf_nodes=31,
                                       min_samples_leaf=30, l2_regularization=1.0,
                                       early_stopping=True, random_state=SEED)),
        ("lr=0.03,iter=1500,leaves=31,min_leaf=60,l2=0.5",
         HistGradientBoostingRegressor(learning_rate=0.03, max_iter=1500, max_leaf_nodes=31,
                                       min_samples_leaf=60, l2_regularization=0.5,
                                       early_stopping=True, random_state=SEED)),
        ("lr=0.04,iter=1000,leaves=31,min_leaf=50,l2=2.0",
         HistGradientBoostingRegressor(learning_rate=0.04, max_iter=1000, max_leaf_nodes=31,
                                       min_samples_leaf=50, l2_regularization=2.0,
                                       early_stopping=True, random_state=SEED)),
    ],
    "Gradient Boosting Regressor": [
        (f"n_estimators={n},lr={lr},max_depth={d},min_samples_leaf={m}",
         GradientBoostingRegressor(n_estimators=n, learning_rate=lr, max_depth=d,
                                   min_samples_leaf=m, random_state=SEED))
        for n, lr, d, m in (
            (400, 0.05, 3, 1),
            (250, 0.1, 3, 1),
            (400, 0.08, 3, 3),
            (300, 0.07, 4, 5),
        )
    ],
}


def _build_estimator(family: str, config_label: str):
    for fam, configs in candidate_families().items():
        if fam == family:
            for label, estimator in configs:
                if label == config_label:
                    return estimator
    for fam, configs in WAVE2_CONFIGS.items():
        if fam == family:
            for label, estimator in configs:
                if label == config_label:
                    return estimator
    return None


def main() -> int:
    started = time.perf_counter()
    log("=== Wave-2 targeted search starting ===")
    x_train, y_train, x_val, y_val, x_test, y_test, x_train_full, y_train_full = load_data()

    new_rows = []
    for family, configs in WAVE2_CONFIGS.items():
        log(f"--- wave2 {family}: {len(configs)} config(s) ---")
        for label, estimator in configs:
            t0 = time.perf_counter()
            try:
                estimator.fit(x_train, y_train)
                seconds = time.perf_counter() - t0
                metrics = regression_metrics(y_val, estimator.predict(x_val))
                new_rows.append({"Model": family, "Config": label, "status": "ok",
                                 "fit_seconds": round(seconds, 2), "wave": 2, **metrics})
                log(f"{family} [{label}] -> val MAE {metrics['MAE']:.2f}, "
                    f"RMSE {metrics['RMSE']:.2f}, R2 {metrics['R2']:.4f} ({seconds:.1f}s)")
            except Exception as exc:  # noqa: BLE001
                new_rows.append({"Model": family, "Config": label,
                                 "status": f"failed: {exc}", "fit_seconds": round(time.perf_counter() - t0, 2),
                                 "wave": 2, **{k: np.nan for k in
                                    ("MAE", "MSE", "RMSE", "R2", "Explained_Variance", "Median_AE", "MAPE")}})
                log(f"{family} [{label}] FAILED: {exc}")

    tuning_path = SUITE_DIR / "tuning_results.csv"
    merged = pd.concat([pd.read_csv(tuning_path), pd.DataFrame(new_rows)], ignore_index=True)
    merged.to_csv(tuning_path, index=False)

    ok = merged[merged["status"] == "ok"].copy()
    best_idx = ok.groupby("Model")["MAE"].idxmin()
    best_rows = ok.loc[best_idx].sort_values("MAE").reset_index(drop=True)
    best_rows.to_csv(SUITE_DIR / "validation_metrics.csv", index=False)

    champion = best_rows.iloc[0]
    champion_name = str(champion["Model"])
    champion_config = str(champion["Config"])
    log(f"Champion after wave 2: {champion_name} [{champion_config}] "
        f"val MAE {champion['MAE']:.2f}, R2 {champion['R2']:.4f}")

    # Refresh per-model validation charts + metrics.csv for updated champions
    refit = _build_estimator(champion_name, champion_config)
    refit_models, preds_val = {}, {}
    for _, row in best_rows.iterrows():
        fam, cfg = str(row["Model"]), str(row["Config"])
        est = _build_estimator(fam, cfg)
        if est is None:
            continue
        try:
            refit_models[fam] = est.fit(x_train, y_train)
            preds_val[fam] = refit_models[fam].predict(x_val)
        except Exception as exc:  # noqa: BLE001
            log(f"Refit failed for {fam}: {exc}")
    for fam, pred in preds_val.items():
        folder = FAMILY_TO_FOLDER.get(fam)
        if folder:
            save_diagnostic_plots(fam, folder, y_val, pred)
            row = best_rows[best_rows["Model"] == fam].iloc[0]
            pd.DataFrame([{
                "Model": fam, "Split": "Validation", "MAE": row["MAE"], "MSE": row["MSE"],
                "RMSE": row["RMSE"], "R2": row["R2"],
                "Explained_Variance": row["Explained_Variance"],
                "Median_AE": row["Median_AE"], "MAPE": row["MAPE"],
                "Train_Time_Sec": row["fit_seconds"], "Best_Config": row["Config"],
            }]).to_csv((Path("Outputs") / folder) / "metrics.csv", index=False)

    # Final test evaluation for the (possibly new) champion — one pass.
    log(f"Final test evaluation for champion: {champion_name}")
    t0 = time.perf_counter()
    champion_model = refit_models.get(champion_name) or _build_estimator(champion_name, champion_config)
    champion_model.fit(x_train_full, y_train_full)
    refit_seconds = time.perf_counter() - t0
    test_pred = champion_model.predict(x_test)
    champion_test = regression_metrics(y_test, test_pred)
    log(f"FINAL TEST [{champion_name}] -> MAE {champion_test['MAE']:.2f}, "
        f"RMSE {champion_test['RMSE']:.2f}, R2 {champion_test['R2']:.4f}")

    test_rows = pd.DataFrame([{
        "Model": champion_name, "Config": champion_config,
        "Train_Rows": int(len(x_train_full)), "Test_Rows": int(len(x_test)),
        "Refit_Seconds": round(refit_seconds, 2), **champion_test,
        "Val_MAE": champion["MAE"], "Val_RMSE": champion["RMSE"], "Val_R2": champion["R2"],
    }])
    test_rows.to_csv(SUITE_DIR / "final_test_metrics.csv", index=False)
    test_rows.to_csv(Path("Outputs") / "Final_Model_Evaluation" / "final_test_metrics.csv", index=False)

    predictions = pd.DataFrame({
        "row_index": np.arange(len(x_test)),
        "actual_loss": y_test.to_numpy(),
        "predicted_loss": test_pred,
        "absolute_error": np.abs(y_test.to_numpy() - test_pred),
    })
    predictions.to_csv(Path("Outputs") / "Final_Model_Evaluation" / "final_test_predictions.csv", index=False)
    save_final_evaluation_plots(y_test, test_pred)

    refresh_model_comparison_csv(best_rows, refit_models, champion_name, champion_test)
    promote_model(champion_model, champion_name)

    # Merge wave-2 metadata into the suite metadata file
    metadata_path = SUITE_DIR / "experiment_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["wave2"] = {
        "configs_tried": sum(len(v) for v in WAVE2_CONFIGS.values()),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "champion_after_wave2": champion_name,
        "champion_config": champion_config,
        "champion_validation_metrics": {k: float(champion[k]) for k in
            ("MAE", "MSE", "RMSE", "R2", "Explained_Variance", "Median_AE", "MAPE")},
        "champion_final_test_metrics": champion_test,
        "note": (
            "Wave 2 was selected on the validation set only; the untouched test "
            "set is scored once per promoted champion for final reporting."
        ),
    }
    metadata["selection"]["champion"] = champion_name
    metadata["selection"]["champion_config"] = champion_config
    metadata["selection"]["champion_validation_metrics"] = metadata["wave2"]["champion_validation_metrics"]
    metadata["selection"]["champion_final_test_metrics"] = champion_test
    metadata["r2_goal_verdict"]["achieved_test_r2"] = champion_test["R2"]
    metadata["r2_goal_verdict"]["reached"] = bool(champion_test["R2"] >= 0.95)
    metadata["r2_goal_verdict"]["statement"] = (
        f"R² = {champion_test['R2']:.4f} on the untouched 10,000-row test set after "
        f"two validation-selected waves. The 0.95 experimental goal was NOT reached."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    (Path("Models") / "best_model_metadata.json").write_text(json.dumps({
        "model": champion_name, "config": champion_config,
        "artifact": "Models/best_model.pkl",
        "feature_contract": "130 preprocessed features from Models/claimwise_preprocessor.pkl",
        "validation_metrics": metadata["wave2"]["champion_validation_metrics"],
        "final_test_metrics": champion_test,
        "trained_utc": metadata["generated_utc"], "seed": SEED, "wave": 2,
    }, indent=2, default=str))
    log(f"Wave 2 finished in {metadata['wave2']['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

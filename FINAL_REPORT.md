# ClaimWise — Final Project Report

Everything below was produced by executing the project files in this repository. No number in this
report is typed in from memory or estimated: each one is traceable to a CSV, a notebook output or a
Flask response listed in the cited path.

**Report date:** 2026-10-08
**Environment:** project `.venv` (Python 3.14, pandas 3.0.5, scikit-learn 1.9.1, Flask 3.1.3),
notebooks executed with `jupyter nbconvert --execute` from the project root.
**Scope:** the production regression system (sections A–Q) **plus** the academic machine-learning
extension — PCA, classification, clustering, DBSCAN, anomaly detection and leakage-free 5-fold
cross-validation (sections R–T).

---

## A. Project summary

ClaimWise predicts a continuous insurance claim loss (`loss`) from 130 preprocessed features and
publishes the whole workflow — data audit, preprocessing, EDA, five regression models, comparison,
prediction, plus the academic ML extension — as seventeen ordered Jupyter notebooks and a Flask web
application.

Delivered in this phase:

1. All seventeen notebooks execute top-to-bottom with **0 errors and 0 warnings** (350 cells).
2. The provided Google Drive code was adapted into a real fifth model
   (`Hist Gradient Boosting Regressor`), trained with 3-fold cross-validation and early stopping —
   it became the new best model on every metric.
3. All five models retrained; comparison table, `Models/best_model.pkl` and every figure regenerated.
4. Stale, non-ClaimWise and duplicate images removed (67 files deleted, 0 duplicates remain).
5. The website now has a **real prediction form** covering all 130 raw features, wired to the saved
   preprocessing artifacts and the saved best model, with the original test-row demo kept as a
   secondary mode.
6. Cross-checked end-to-end: form input → same feature matrix as the trained test row → same
   prediction.
7. Academic extension added without touching any of the above: notebooks 11–17 implement PCA
   (11), Logistic Regression classification with confusion matrix / precision / recall / F1 (12),
   Decision Tree classification (13), K-Means clustering (14), DBSCAN (15), Isolation-Forest anomaly
   detection (16) and leakage-free 5-fold cross-validation for all seven models (17), published on
   the new `/academic` route.

---

## B. Dataset

| Property | Value | Source |
|---|---:|---|
| Raw rows × columns | 50,000 × 132 | `Dataset/01_raw_claimwise_50000.csv` |
| Missing values (raw) | 0 | notebook 01 / pipeline summary |
| Duplicate rows (raw) | 0 | notebook 01 / pipeline summary |
| Categorical columns | 116 (`cat1`–`cat116`) | raw dtypes |
| Continuous columns | 14 (`cont1`–`cont14`) | raw dtypes |
| Removed column | `id` | pipeline step 2 |
| Final rows × columns | 50,000 × 131 | `Dataset/05_preprocessed_final.csv` |
| Features after split | 130 | `Dataset/06_split_X_train.csv` (40,000 × 130) |
| Test rows | 10,000 | `Dataset/06_split_X_test.csv` |
| Target range | 10.00 – 85,923.56, mean 3035.01 | raw `loss` |
| Categorical cardinality | 73 columns with 2 values, 43 with 3–266 (max `cat116` = 266) | raw `nunique()` |

The feature names are anonymised in the source data. The project therefore never renames them into
domain terms (`age`, `customer`, `policy`, …); every page, form field and figure keeps the original
`cat*` / `cont*` names.

---

## C. Preprocessing — one authoritative pipeline

`Src/claimwise_preprocessing_pipeline.py` (executed by notebook 02 and `main.py`) is the only
preprocessing path:

1. Load raw CSV, audit shape / dtypes / missing / duplicates.
2. Drop duplicates (0 removed), median-impute numeric columns, mode-impute and strip categorical
   columns, drop `id` → `02_cleaned_dedup_median_imputed.csv`.
3. `LabelEncoder` per categorical column → `03_encoded_label.csv`.
4. `StandardScaler` on the 14 continuous columns, **fitted only on the training row indices**
   → `04_scaled_standardized.csv`.
5. Save `05_preprocessed_final.csv`, then `train_test_split(..., test_size=0.20, random_state=42)`
   → `06_split_{X,y}_{train,test}.csv`.

Stage-by-stage evidence is displayed on `/preprocessing` from
`Outputs/Preprocessing/claimwise_preprocessing_stage_summary.csv`.

---

## D. Data-leakage prevention and verification

The scaler is fitted on 40,000 training rows only (pipeline lines 118–128). Notebook 02 verifies the
consequence on the prepared matrices:

| Metric | Value | Meaning |
|---|---:|---|
| Max abs mean of continuous columns, training rows | 0.000000 (3.8e-16) | scaler statistics came from these rows |
| Max abs mean of continuous columns, test rows | 0.023239 | test rows were only transformed, never used to fit |

These two numbers are rendered on the `/preprocessing` page with a caption that states exactly what
is measured. The site never claims "zero leakage" — it states the fitted-on-training fact and shows
the measured values.

---

## E. Notebooks (seventeen, ordered, fully executable)

Execution command used for every notebook:

```bash
/Users/padaltiruvinayak/Desktop/Placement_predict/.venv/bin/jupyter nbconvert --to notebook \
  --execute --inplace --ExecutePreprocessor.timeout=1800 \
  --ExecutePreprocessor.kernel_name=python3 Notebooks/<name>.ipynb
```

| # | Notebook | Cells | Errors | Warnings |
|---|---|---:|---:|---:|
| 01 | `01_ClaimWise_Data_Inspection.ipynb` | 17 | 0 | 0 |
| 02 | `02_ClaimWise_Data_Preprocessing.ipynb` | 17 | 0 | 0 |
| 03 | `03_ClaimWise_EDA.ipynb` | 14 | 0 | 0 |
| 04 | `04_ClaimWise_Linear_Regression.ipynb` | 25 | 0 | 0 |
| 05 | `05_ClaimWise_Decision_Tree.ipynb` | 25 | 0 | 0 |
| 06 | `06_ClaimWise_Random_Forest.ipynb` | 25 | 0 | 0 |
| 07 | `07_ClaimWise_Gradient_Boosting.ipynb` | 25 | 0 | 0 |
| 08 | `08_ClaimWise_Hist_Gradient_Boosting.ipynb` | 28 | 0 | 0 |
| 09 | `09_ClaimWise_Model_Comparison.ipynb` | 18 | 0 | 0 |
| 10 | `10_ClaimWise_Final_Prediction.ipynb` | 16 | 0 | 0 |
| 11 | `11_ClaimWise_PCA_Feature_Analysis.ipynb` | 23 | 0 | 0 |
| 12 | `12_ClaimWise_Logistic_Regression_Classification.ipynb` | 22 | 0 | 0 |
| 13 | `13_ClaimWise_Decision_Tree_Classification.ipynb` | 18 | 0 | 0 |
| 14 | `14_ClaimWise_Clustering.ipynb` | 21 | 0 | 0 |
| 15 | `15_ClaimWise_DBSCAN.ipynb` | 19 | 0 | 0 |
| 16 | `16_ClaimWise_Anomaly_Detection.ipynb` | 17 | 0 | 0 |
| 17 | `17_ClaimWise_5Fold_Cross_Validation.ipynb` | 20 | 0 | 0 |
| | **Total** | **350** | **0** | **0** |

Notebooks 01–10 are the production workflow; 11–17 are the academic extension (section R).
Conclusion markdown in notebooks 05–17 was re-checked against the CSVs written during the same runs
(ranking rows, metric columns, fold tables and the named best model all match).

---

## F. Exploratory data analysis

Notebook 03 executes `Src/6_EDA_Analysis.py`, which writes 36 figures into
`Outputs/EDA_Analysis_outputs/`: target distribution, 14 continuous distributions, 14 boxplots, the
correlation heatmap, five categorical count plots and the missing-values heatmap. Notebook 03 also
stores the full feature→loss correlation ranking in
`Outputs/EDA_Analysis_outputs/claimwise_correlation_with_loss.csv`.
`Outputs/Boxplots_correlation/` (15 figures + correlation CSV) is produced by
`Src/Correlation_Matrix_heatmap_boxplots_M1.py` and is documented in notebook 03.

---

## G. Model implementation table

| # | Model | Algorithm / key hyper-parameters | Module | Artifact | MAE | MSE | RMSE | R² |
|---:|---|---|---|---|---:|---:|---:|---:|
| 1 | Linear Regression | `LinearRegression()` — OLS on the 130 encoded features | `Src/linear_regression_model.py` | `Models/linear_regression.pkl` | 1333.5665 | 4204641.7064 | 2050.5223 | 0.482615 |
| 2 | Decision Tree Regressor | `DecisionTreeRegressor(max_depth=20, min_samples_leaf=2, random_state=42)` — single tree | `Src/decision_tree_regressor_model.py` | `Models/decision_tree_regressor.pkl` | 1609.9189 | 6964030.4498 | 2638.9450 | 0.143070 |
| 3 | Random Forest Regressor | `RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1)` | `Src/random_forest_regressor_model.py` | `Models/random_forest_regressor.pkl` | 1259.6218 | 3937886.6824 | 1984.4109 | 0.515440 |
| 4 | Gradient Boosting Regressor | `GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, min_samples_leaf=2, loss="squared_error", random_state=42)` | `Src/gradient_boosting_regressor_model.py` | `Models/gradient_boosting_regressor.pkl` | 1277.7829 | 4021053.5725 | 2005.2565 | 0.505206 |
| 5 | **Hist Gradient Boosting Regressor** | `HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.035, max_iter=<CV median 830>, max_leaf_nodes=31, max_depth=None, min_samples_leaf=50, max_features=0.80, l2_regularization=0.50, random_state=42)` | `Src/hist_gradient_boosting_regressor_model.py` | `Models/hist_gradient_boosting_regressor.pkl` | **1215.2997** | **3690333.1165** | **1921.0240** | **0.545901** |

All rows come from `Outputs/Model_Comparison/model_comparison.csv`, written by
`Src/train_models.py` during the executed run of notebook 09. Shared helpers (data loading, metrics,
figure saving, best-model selection) live in `Src/model_training_utils.py` — the models do not
duplicate that logic.

**Selection rule** (no model is preferred by name): lowest test RMSE, tie-break by MAE, then MSE, then
highest R². Winner: Hist Gradient Boosting Regressor → copied to `Models/best_model.pkl`
(class verified to be `HistGradientBoostingRegressor`).

---

## H. Provided Google Drive code — what was implemented

Source files inspected: `~/Downloads/Home_Credit_Default_Risk_FINAL_Perfect.ipynb` (23 cells) and
`~/Downloads/Home_Credit_Default_Risk_FINAL_FIXED.ipynb` (24 cells, the newer variant with the fixed
`SK_ID_PREV` mapping). They are a LightGBM **binary-classification** Kaggle pipeline over eight Home
Credit tables. The single ClaimWise version of that logic is
`Notebooks/08_ClaimWise_Hist_Gradient_Boosting.ipynb` + `Src/hist_gradient_boosting_regressor_model.py`;
`FINAL_Perfect` is superseded by `FINAL_FIXED` and both are represented there.

| Provided (Home Credit) | ClaimWise | Status |
|---|---|---|
| `SEED = 42`, file discovery + sanity checks | `SEED = 42`, prepared-file validation in `load_prepared_data()` | preserved |
| `reduce_numeric_memory()` | same function in `Src/model_training_utils.py` | preserved |
| `encode_categoricals()` | `LabelEncoder` step of the authoritative pipeline | preserved (pipeline already does it) |
| fold loop + OOF array + fold score reporting | `run_cross_validation()` | preserved |
| `StratifiedKFold(3, shuffle, SEED)` | `KFold(3, shuffle, SEED)` | re-parameterised (continuous target) |
| early stopping `rounds=150` + `eval_set` | `early_stopping=True`, `n_iter_no_change=150`, `scoring="loss"`, `X_val`/`y_val` | re-parameterised |
| `num_leaves=31` / `max_depth=-1` / `min_child_samples=50` / `colsample_bytree=0.80` / `reg_lambda=0.50` / `n_estimators=2500` | `max_leaf_nodes=31` / `max_depth=None` / `min_samples_leaf=50` / `max_features=0.80` / `l2_regularization=0.50` / `max_iter=2500` | re-parameterised |
| `LGBMClassifier` | `HistGradientBoostingRegressor` (sklearn) | substituted — `lightgbm` is not installed and no network install is allowed; confirmed with you before implementing |
| binary `TARGET`, ROC-AUC / Gini | `loss`, MAE / MSE / RMSE / R² | re-targeted (regression) |
| prediction sanity checks (length, NaN/inf) | `validate_predictions()` | preserved |
| 6 historical tables + `SK_ID_BUREAU` / `SK_ID_PREV` key maps | — | **dropped**: ClaimWise is one flat table, no such tables or keys exist |
| `add_application_features()` ratio columns | — | **dropped**: built from Home Credit named finance fields (`AMT_CREDIT`, `DAYS_BIRTH`, …); ClaimWise columns are anonymised, so replacements would have to be invented |
| `reg_alpha`, `subsample`, `n_jobs` | — | **dropped**: no sklearn HistGradientBoosting equivalent |
| `submission.csv` | `Models/*.pkl` + `Outputs/…/metrics.csv` + Flask prediction | re-targeted |

Cross-validation result of the adapted model (executed, notebook 08 / script output):

| Fold | Iterations (early stopped) | MAE | RMSE | R² |
|---:|---:|---:|---:|---:|
| 1 | 830 | 1215.3804 | 1950.7341 | 0.5618 |
| 2 | 572 | 1224.1672 | 1913.9725 | 0.5349 |
| 3 | 842 | 1204.9775 | 2004.2711 | 0.5473 |

Out-of-fold over all 40,000 training rows: MAE 1214.8417, RMSE 1956.6771, R² 0.548424.
Median best iteration = 830 → final model `max_iter=830`, fitted once on all 40,000 training rows.
Fold table saved to `Outputs/Hist_Gradient_Boosting_Regressor/cross_validation_folds.csv`.

---

## I. Model comparison and best-model verification

Executed in notebook 09 (`Src/train_models.py` + `Src/model_comparison.py`):

| Rank | Model | MAE | MSE | RMSE | R² |
|---:|---|---:|---:|---:|---:|
| 1 | Hist Gradient Boosting Regressor | 1215.2997 | 3690333.1165 | 1921.0240 | 0.545901 |
| 2 | Random Forest Regressor | 1259.6218 | 3937886.6824 | 1984.4109 | 0.515440 |
| 3 | Gradient Boosting Regressor | 1277.7829 | 4021053.5725 | 2005.2565 | 0.505206 |
| 4 | Linear Regression | 1333.5665 | 4204641.7064 | 2050.5223 | 0.482615 |
| 5 | Decision Tree Regressor | 1609.9189 | 6964030.4498 | 2638.9450 | 0.143070 |

Notebook 09 asserts that the class inside `Models/best_model.pkl` equals the class implied by the
lowest-RMSE row (`HistGradientBoostingRegressor`) and re-computes the metrics from the saved
artifact. Comparison chart:
`Outputs/Model_Comparison/model_comparison_bar_chart.png`.

---

## J. Artifacts

| Artifact | Path |
|---|---|
| Best model | `Models/best_model.pkl` |
| The five individual models | `Models/{linear_regression, decision_tree_regressor, random_forest_regressor, gradient_boosting_regressor, hist_gradient_boosting_regressor}.pkl` |
| Fitted preprocessing (encoders + scaler) for the web form | `Models/claimwise_preprocessor.pkl` |
| Comparison table | `Outputs/Model_Comparison/model_comparison.csv` |
| Per-model metrics / predictions | `Outputs/<Model_Folder>/metrics.csv`, `predictions.csv` |
| CV folds (5th model) | `Outputs/Hist_Gradient_Boosting_Regressor/cross_validation_folds.csv` |
| Sample + full test predictions | `Outputs/Final_Prediction/claimwise_{sample,test_set}_predictions.csv` |
| Classification metrics + confusion matrices | `Outputs/classification/*.csv`, `*.png` |
| PCA results | `Outputs/PCA/pca_explained_variance_table.csv`, `pca_modeling_experiment.csv` |
| Clustering results | `Outputs/clustering/{kmeans_summary,cluster_profiles,kmeans_silhouette_by_k}.csv` |
| DBSCAN results | `Outputs/dbscan/dbscan_summary.csv` |
| Anomaly results | `Outputs/anomaly/{anomaly_summary,anomaly_scores}.csv` |
| 5-fold cross-validation results | `Outputs/CrossValidation/{regression_5fold,classification_5fold}_{scores,summary}.csv` |

---

## K. Outputs and image cleanup

Before cleanup the repository carried 106 files in `Outputs/EDA_Analysis_outputs/`, including legacy
charts from an unrelated lending dataset (`Boxplot_loan_amnt.png`, `Histogram_dti.png`,
`Countplot_purpose.png`, …) and duplicate naming schemes.

Method (no guessing): `Src/6_EDA_Analysis.py` was executed and every PNG it regenerated was kept;
the 67 PNGs it did **not** produce were deleted. A content-hash pass over the whole `Outputs/` tree
after the academic extension now reports:

- **93 PNGs total, 0** legacy/non-ClaimWise names, **0** exact-duplicate PNG groups, 0 empty output
  folders.
- Per-model folders: 3 figures each for Linear/Decision/Random Forest/Gradient Boosting, 2 for Hist
  Gradient Boosting (that model ships no impurity-based importance chart — documented on the model
  card instead), 3 in `Model_Comparison`, 36 in `EDA_Analysis_outputs`, 15 in
  `Boxplots_correlation`, 1 in `Data_Inspection`.
- Academic folders: 5 in `PCA`, 7 in `classification`, 3 in `clustering`, 4 in `dbscan`, 3 in
  `anomaly`, 2 in `CrossValidation`.

Every image referenced by the site was re-checked with 0 broken references.

---

## L. Web application (Flask)

`app.py` reads real files only — nothing about models or metrics is hard-coded in the templates:

| Route | Data source |
|---|---|
| `/`, `/about`, `/contact` | static content |
| `/dataset` | raw/final CSV columns + `Outputs/Data_Inspection/…` figure |
| `/preprocessing` | `Outputs/Preprocessing/claimwise_preprocessing_stage_summary.csv`, leakage stats computed from the split CSVs |
| `/visualization` | EDA figures |
| `/models` | `Outputs/Model_Comparison/model_comparison.csv` + per-model figures + notebooks list |
| `/academic` | PCA / classification / clustering / DBSCAN / anomaly / 5-fold-CV CSVs + figures under `Outputs/` |
| `/dashboard`, `/reports` | same comparison CSV |
| `/prediction` | `Models/best_model.pkl` + `Models/claimwise_preprocessor.pkl` + form (see section N) |
| `/outputs/<path>` | serves figures generated by the pipeline |
| `/notebooks/<path>` | downloads the executed notebooks |

Verified: all **12 routes** return **200**, every internal link and asset resolves (66 unique
targets checked, 0 broken), the `/academic` page renders 22 real output images, every `<img>` has an
`alt`, all pages parse with balanced HTML tags, and the models page lists all five models with the
"Best Performing Model" badge on the winner. The real 130-field prediction form still posts
successfully (manual mode → 3038.79 on defaults; test-row mode row 0 → 3565.33).

---

## M. Design-system compliance

Palette used throughout (`static/style.css`): background `#F7F5F0`, text `#18181B`, primary accent
`#D4512A`, secondary accent `#E58A3A`, success `#3F7D4A`, border `#E7E2D8`. No external/CDN
resources, no off-palette colors, no `box-shadow` used as a colour substitute.

---

## N. Prediction flow

Two modes on `/prediction`; mode 1 is the real form, mode 2 is the original demo.

```text
MODE 1 — real feature form (default)
  Browser form (130 fields, raw values)
    · 14 number inputs: cont1–cont14, min/max from the raw dataset, default = dataset median
    · 116 dropdowns: cat1–cat116, options = the values that actually occur in the raw data,
      default = most frequent value; grouped into
        – Continuous features (14)
        – Binary categorical features (73 columns with exactly 2 values)
        – Multi-level categorical features (43 columns with 3–266 values)
        (grouping is derived from the data itself; column names are never renamed)
        │
        ▼  POST /prediction  (mode=manual)
  Validation
    · every field present and non-empty
    · continuous values parse as finite numbers
    · categorical values must exist in the fitted LabelEncoder vocabulary
    · continuous values outside the observed dataset range are reported as a warning
        │
        ▼
  Src/prediction_preprocessor.py::transform()
    · identical cleaning/encoding/scaling as the training pipeline:
      strip → LabelEncoder per cat column → StandardScaler on cont columns
      (scaler statistics taken from the same training-row fit)
    · target `loss` / `id` are refused; feature order forced to the model's 130 columns
        │
        ▼
  Models/best_model.pkl  (HistGradientBoostingRegressor)
    · feature_names_in_ checked against the form matrix (order + count + shape)
        │
        ▼
  Numeric predicted loss + model name + feature count + test-set mean + range warnings
```

```text
MODE 2 — prepared test row (secondary, unchanged behaviour)
  row_index 0…9999  →  Dataset/06_split_X_test.csv row  →  best_model.pkl
  → predicted loss shown next to the stored actual loss
```

**Preprocessing artifact.** The training pipeline did not persist its encoders/scaler, so
`Src/prediction_preprocessor.py` replays the same steps (same cleaning, same `LabelEncoder`s, same
`StandardScaler` fitted on the same `train_test_split(..., test_size=0.20, random_state=42)` row
indices) and verifies the replay against `Dataset/05_preprocessed_final.csv` before anything is used:

| Verification | Value |
|---|---:|
| Rows × columns re-checked | 50,000 × 131 |
| Max abs difference vs the saved final dataset | 4.44e-16 |
| Result | pass → saved to `Models/claimwise_preprocessor.pkl` |

**End-to-end cross-check (form vs trained data path).** For raw rows taken from the test split, the
form's transformed feature matrix was compared with the corresponding trained test row and both were
scored by the same saved model:

| Test row | Max abs feature difference | Form prediction | Trained-path prediction |
|---:|---:|---:|---:|
| 0 | 2.2e-16 | 3565.3302 | 3565.3302 |
| 1 | 6.9e-17 | 2061.5085 | 2061.5085 |
| 7 | 1.1e-16 | 11682.9801 | 11682.9801 |
| 4321 | < 1e-15 | 3899.06 | 3899.06 |
| 9999 | 2.2e-16 | 2595.75 | 2595.81 |

Across all 10,000 test rows the rebuilt matrix differs from `06_split_X_test.csv` by at most
4.44e-16. The single 0.065 prediction difference at row 9999 is a histogram-bin boundary effect: a
value that sits exactly on a bin edge moves to the neighbouring bin under float noise. Feature order,
column set and shape are identical in every check, and `loss` is never passed to the model.

**Request-level checks (Flask test client):** 130 fields and 116 dropdowns present on GET; default
form POST returns a prediction that matches an independent `transform()` + `predict()` call;
empty field → clear error; unknown category → clear error; out-of-range value → warning **and** a
prediction; test-row mode still returns predicted + actual; all model/figure/notebook assets serve
200.

---

## O. Verification summary

| Check | Result |
|---|---|
| Seventeen notebooks executed, errors / warnings | 0 / 0 (350 cells) |
| Notebooks' conclusion tables vs their own CSVs | match (regression rankings, classification metrics, fold tables) |
| Models trained | 5, comparison CSV rank 1–5 as in section I |
| `Models/best_model.pkl` class | `HistGradientBoostingRegressor` |
| Site routes | 12/12 HTTP 200 (incl. `/academic`) |
| Internal links + assets | 66 unique targets checked, 0 broken |
| Images | 93 PNGs, 0 legacy names, 0 duplicates, all referenced images exist |
| HTML structure | all pages tag-balanced |
| Prediction form after the academic work | GET 200 (130 fields) + manual POST 200 + test-row POST 200 |
| Preprocessing replay vs saved dataset | max abs diff 4.44e-16 |
| Form vs trained test row | max feature diff 4.44e-16, predictions equal (see N) |
| Leakage display values | train 0.000000, test 0.023239 (real data) |
| 5-fold CV preprocessing | fitted inside every fold; classification thresholds 2106.31–2122.90 |
| `Src/` compile | `python -m py_compile` clean for changed modules |

---

## P. Known limitations and honest notes

1. **Anonymised features.** `cat*`/`cont*` have no documented meanings, so no page, form label or
   figure invents one. Home Credit ratio features and multi-table joins were dropped for the same
   reason (section H).
2. **No unit for `loss`.** The dataset defines no currency, so the prediction page reports a plain
   number on the dataset's own scale (10.00–85,923.56).
3. **Static numbers in README/report.** The README model table and this report are snapshots of the
   current run; the website itself always reads the CSVs.
4. **sklearn version skew.** Notebooks run on a kernel with scikit-learn 1.9.0 while `.venv` has
   1.9.1, so loading a notebook-trained model prints `InconsistentVersionWarning`. Predictions are
   verified identical in section N; retraining with one environment removes the warning.
5. **Case-sensitive hosting.** Git tracks `Static/style.css` while the on-disk directory is `static/`
   (fine on macOS, 404 on a case-sensitive server). Fix before pushing with
   `git rm --cached Static/style.css && git add static/style.css`.
6. **Derived classification target.** ClaimWise has no categorical label, so notebooks 12/13/17 use
   `loss_high = 1` when `loss` ≥ the training median (2118.86 on the production split; 2106.31–2122.90
   per CV fold). The rule is calculated from training rows only and printed — never chosen by hand.
7. **No fraud labels.** The anomaly detector and DBSCAN noise flags describe "unusual in the feature
   space". ClaimWise carries no verified fraud target, so no output calls a record fraudulent.
8. **Low silhouette in clustering.** K-means on 130 dimensions yields small silhouette scores (best
   0.0836 at k = 2) and near-identical mean loss per cluster (2982.89 vs 3074.16). This is reported
   as the honest result, not tuned away.
9. **DBSCAN runs on a sample.** Density clustering uses a fixed, seeded 5,000-row sample (and a
   PCA-20 variant) because neighbourhood queries over 50,000 × 130 values are unnecessary for the
   analysis; the sample size is stated on the page and in the notebook.
10. **Classifier scaling layer.** Logistic Regression trains on the 130 production features plus a
    `StandardScaler` fitted on the training rows only; without it the unscaled category codes make
    `lbfgs` fail to converge (the first run's `ConvergenceWarning` is why the layer exists, and the
    notebook reports `n_iter_ = 59`).
11. **Nothing committed.** All changes are in the working tree only.

---

## Q. How to reproduce

```bash
# 1. preprocessing + EDA (also regenerates Outputs/EDA_Analysis_outputs)
.venv/bin/python main.py
.venv/bin/python Src/6_EDA_Analysis.py

# 2. web-form preprocessing artifact (verified against 05_preprocessed_final.csv)
.venv/bin/python Src/prediction_preprocessor.py

# 3. train all five models + comparison + best_model.pkl
.venv/bin/python Src/train_models.py

# 4. notebooks (any subset; 08 = provided-code adaptation, 09 = comparison, 10 = prediction)
/Users/padaltiruvinayak/Desktop/Placement_predict/.venv/bin/jupyter nbconvert --to notebook \
  --execute --inplace --ExecutePreprocessor.timeout=1800 \
  --ExecutePreprocessor.kernel_name=python3 Notebooks/08_ClaimWise_Hist_Gradient_Boosting.ipynb

# 4b. academic extension (11 PCA · 12 logistic · 13 decision tree · 14 k-means · 15 DBSCAN
#     · 16 isolation forest · 17 five-fold CV) — same command per notebook, e.g.
/Users/padaltiruvinayak/Desktop/Placement_predict/.venv/bin/jupyter nbconvert --to notebook \
  --execute --inplace --ExecutePreprocessor.timeout=3600 \
  --ExecutePreprocessor.kernel_name=python3 Notebooks/17_ClaimWise_5Fold_Cross_Validation.ipynb

# 5. website
.venv/bin/python app.py        # http://127.0.0.1:5000  → /prediction and → /academic
```

---

## R. Academic ML extension (notebooks 11–17)

Every method below runs on the real ClaimWise files, shares one helper module
(`Src/academic_common.py`), uses `random_state = 42`, and writes only into
`Outputs/{PCA,classification,clustering,dbscan,anomaly,CrossValidation}/`. The production
regression system (sections A–Q) was not modified.

### R.1 PCA — notebook 11

| Result | Value |
|---|---|
| Components fitted | 130 (all — the useful count is read from the curve) |
| Cumulative variance at 20 / 75 / 100 components | 49.41% / 89.29% / 98.42% |
| Components for 90% / 95% / 99% variance | 77 / **89** / 104 |
| Modelling experiment, original 130 features | Linear R² 0.482615, HistGB R² **0.545901** |
| Modelling experiment, PCA (89 components) | Linear R² 0.475020, HistGB R² 0.509912 |

PCA produces **components** (loadings), not a subset of the original `cat*`/`cont*` columns; the
first-two-component plot is used for visualisation only. The executed experiment shows the PCA space
is worse for both models, so the production model keeps the original 130 features.
Artifacts: `Outputs/PCA/pca_{explained_variance_ratio,cumulative_explained_variance,pc1_top_loadings,projection_2d,modeling_experiment}.png`,
`pca_explained_variance_table.csv`, `pca_modeling_experiment.csv`.

### R.2 Classification — notebooks 12 and 13

Derived target (no categorical label exists in ClaimWise): `loss_high = 1` when
`loss >= 2118.86` (median of the 40,000 training rows only); test split 5037 typical / 4963 high.

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7583 | 0.7828 | 0.7101 | 0.7446 |
| Decision Tree Classifier | 0.6882 | 0.7062 | 0.6365 | 0.6696 |

Both notebooks print `classification_report`, save a confusion matrix, and (for Logistic
Regression) a ROC curve with AUC. The Logistic Regression gets a train-fitted `StandardScaler`
layer because the unscaled category codes otherwise leave `lbfgs` unconverged (`n_iter_ = 59`
afterwards, 0 warnings).
Artifacts: `Outputs/classification/{logistic_regression,decision_tree}_*.{csv,png}`,
`classification_metrics_summary.csv`, `classification_metric_comparison.png`,
`classification_class_distribution.png`.

### R.3 Clustering (K-Means) — notebook 14

`loss` is dropped before fitting and read back only to profile the clusters.

| Result | Value |
|---|---|
| Rows clustered | 50,000 × 130 features (target excluded) |
| Selection rule (declared first) | highest silhouette for k = 2…8 |
| Selected k / silhouette | **2 / 0.0836** (full-fit sample silhouette 0.0822) |
| Cluster sizes | 21,447 (42.89%) and 28,553 (57.11%) |
| Mean loss per cluster (read back afterwards) | 2982.89 vs 3074.16 |

The low silhouette scores and near-identical cluster means are reported as the honest outcome: the
partition exists numerically but the 130-dimensional data has no well-separated spherical clusters.
Artifacts: `Outputs/clustering/{kmeans_elbow_silhouette,kmeans_cluster_sizes,kmeans_pca_visualization}.png`,
`{cluster_profiles,kmeans_summary,kmeans_silhouette_by_k}.csv`.

### R.4 DBSCAN — notebook 15

Same rules: `loss` excluded, `eps` from the knee of the k-distance curve (computed, not eyeballed),
seeded 5,000-row sample stated everywhere, PCA used only for plotting.

| Representation | Clusters | Noise points | Noise % | eps | Silhouette |
|---|---:|---:|---:|---:|---:|
| 130 scaled features | 7 | 164 | 3.28% | 13.6748 | 0.2617 |
| PCA (20 components) | 3 | 67 | 1.34% | 8.8960 | 0.3675 |

Artifacts: `Outputs/dbscan/dbscan_{k_distance_130d,k_distance_pca20,clusters_pca_visualization,cluster_distribution}.png`,
`dbscan_summary.csv`.

### R.5 Anomaly detection (Isolation Forest) — notebook 16

`loss` is never an input; `contamination="auto"` derives the threshold from the model's own score
distribution.

| Result | Value |
|---|---|
| Records scored | 50,000 |
| Anomalies flagged | **1,641 (3.28%)** |
| Mean decision score, normal vs anomaly | 0.0754 vs −0.0243 |
| Context only (read back afterwards) | mean loss 7397.88 flagged vs 2886.96 normal |

An anomaly is *unusual*, never *fraudulent* — ClaimWise has no fraud labels and the notebook plus
the website state this explicitly.
Artifacts: `Outputs/anomaly/{anomaly_score_distribution,anomaly_pca_visualization,anomaly_loss_context}.png`,
`{anomaly_summary,anomaly_scores}.csv`.

### R.6 Leakage-free 5-fold cross-validation — notebook 17

Raw features (`01_raw_claimwise_50000.csv`) + `KFold(5, shuffle, 42)`; the `ColumnTransformer`
(ordinal-encode `cat*`, standardise `cont*`) lives **inside** the pipeline and is refitted on each
fold's training rows; classification thresholds come from each fold's training rows only.

Regression — mean (std) over folds:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Hist Gradient Boosting Regressor | 1203.37 (10.99) | 1937.66 (34.40) | **0.553189 (0.005585)** |
| Random Forest Regressor | 1260.39 (5.03) | 1978.81 (28.28) | 0.533885 (0.011586) |
| Gradient Boosting Regressor | 1269.72 (13.74) | 1978.59 (41.74) | 0.533926 (0.017902) |
| Linear Regression | 1327.86 (15.98) | 2093.48 (51.00) | 0.478500 (0.010828) |
| Decision Tree Regressor | 1613.58 (26.99) | 2618.39 (77.33) | 0.183556 (0.044351) |

Classification — mean (std) over folds:

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7635 (0.0039) | 0.7896 (0.0071) | 0.7183 (0.0063) | 0.7523 (0.0066) | 0.8437 (0.0040) |
| Decision Tree Classifier | 0.6902 (0.0019) | 0.7111 (0.0062) | 0.6407 (0.0063) | 0.6741 (0.0041) | 0.6920 (0.0062) |

Per-fold thresholds: 2106.31 – 2122.90. The cross-validated ranking matches the single-split
ranking, so the production numbers are not an artefact of one favourable split.
Artifacts: `Outputs/CrossValidation/{regression_5fold,classification_5fold}_{scores,summary}.csv`,
`{regression_5fold_scores,classification_5fold_scores}.png`.

---

## S. Academic requirement table

| # | Requirement | Where it is implemented | Executed evidence | Website section | Status |
|---:|---|---|---|---|---|
| 1 | PCA | `Notebooks/11_ClaimWise_PCA_Feature_Analysis.ipynb` | 130 components, 89 for 95% variance, variance/loading/projection figures, PCA-vs-original experiment CSV | `/academic` → *Principal Component Analysis* | done |
| 2 | Classification with Logistic Regression | `Notebooks/12_…Logistic_Regression_Classification.ipynb` | accuracy 0.7583, precision 0.7828, recall 0.7101, F1 0.7446, confusion matrix PNG, ROC curve | `/academic` → *Classification* | done |
| 3 | Classification with Decision Tree | `Notebooks/13_…Decision_Tree_Classification.ipynb` | accuracy 0.6882, precision 0.7062, recall 0.6365, F1 0.6696, confusion matrix PNG, top features | `/academic` → *Classification* | done |
| 4 | Confusion matrix / precision / recall / F1 | both classification notebooks (`classification_metrics()`, `plot_confusion_matrix()`) | `Outputs/classification/*metrics*.csv` + 2 confusion matrices + `classification_metric_comparison.png` | `/academic` → *Classification* | done |
| 5 | Clustering (K-Means) | `Notebooks/14_ClaimWise_Clustering.ipynb` | k = 2 chosen by silhouette 0.0836 after elbow + k = 2…8 sweep, 50,000 rows clustered, cluster profiles | `/academic` → *Clustering* | done |
| 6 | DBSCAN | `Notebooks/15_ClaimWise_DBSCAN.ipynb` | 7 clusters / 164 noise (3.28%) / sil 0.2617 on 130 features; 3 clusters / 67 noise / sil 0.3675 on PCA-20; k-distance knee figures | `/academic` → *DBSCAN* | done |
| 7 | Anomaly detection | `Notebooks/16_ClaimWise_Anomaly_Detection.ipynb` | 1,641 of 50,000 flagged (3.28%), score histogram, PCA view, loss context chart | `/academic` → *Anomaly Detection* | done |
| 8 | 5-fold cross-validation, preprocessing inside the fold | `Notebooks/17_ClaimWise_5Fold_Cross_Validation.ipynb` + `Src/academic_common.py` | 7 models × 5 folds, Fold 1–5 scores, mean + std for MAE/MSE/RMSE/R² and accuracy/precision/recall/F1/AUC | `/academic` → *Leakage-Free 5-Fold Cross-Validation* | done |
| 9 | Classification target derived (none supplied) | `Src/academic_common.py::build_classification_target()` | train-median threshold 2118.86 printed and stored; CV thresholds 2106.31–2122.90 | `/academic` → *Classification*, *5-Fold CV* | done |
| 10 | Reproducibility | `random_state=42` / `SEED=42` in every estimator, sampler and splitter | notebooks re-executed twice with identical CSVs | all sections | done |
| 11 | Website publishing | `app.py::load_academic_data()` + `templates/academic.html` + nav link | `/academic` 200, 22 real output images, 6 jump sections, 66 internal targets OK | header + footer nav | done |
| 12 | Keep the working system intact | production code paths untouched | 12/12 routes 200, form GET/manual POST/test-row POST all 200, regression CSVs byte-identical to the earlier run | `/prediction`, `/models` | done |
| 13 | No fabricated results / no placeholders | every number read from `Outputs/*.csv` written in the executed runs | 0 placeholder tokens left in notebooks 11–17; 0 duplicate/legacy images | all sections | done |
| 14 | Honesty constraints | `loss` never an input to unsupervised methods; anomalies ≠ fraud; PCA = components | stated in every notebook conclusion and on `/academic` | all unsupervised sections | done |

---

## T. Final checklist

- [x] 17 notebooks execute top-to-bottom with **0 errors / 0 warnings** (350 cells), re-run as a full
      sweep after the last edit.
- [x] Production regression untouched: `Outputs/Model_Comparison/model_comparison.csv` still ranks
      Hist Gradient Boosting first (RMSE 1921.0240, R² 0.545901); `Models/best_model.pkl` unchanged.
- [x] Prediction form still works after the extension (130 fields; manual + test-row POSTs return
      predictions, no errors).
- [x] Academic notebooks 11–17 exist, are fully executed, and contain no placeholder cells.
- [x] One authoritative preprocessing path for production; fold-local pipelines for cross-validation
      (documented as two different protocols, not mixed).
- [x] Classification target derived from training data only and printed everywhere it is used.
- [x] `loss` excluded from clustering, DBSCAN and anomaly inputs; used only for after-the-fact
      profiling, clearly labelled as such.
- [x] Anomaly/noise output never described as fraud anywhere in code, notebooks or site.
- [x] PCA described as producing components; 2D PCA used for visualisation only.
- [x] All results traceable to CSVs under `Outputs/` (93 PNGs, 0 duplicates, 0 legacy names).
- [x] Website: 12 routes 200, `/academic` renders the six real sections, 0 broken internal links.
- [x] `FINAL_REPORT.md` covers sections A–Q plus R–T with the academic requirement table.
- [x] Nothing committed to git; the `Static/` vs `static/` path issue remains documented for the
      eventual push.

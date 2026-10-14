# ClaimWise Insurance Prediction

A machine learning system that predicts insurance claim loss (`loss`) from 130 preprocessed input
features — 116 label-encoded categorical columns (`cat1`–`cat116`) and 14 standardized continuous
columns (`cont1`–`cont14`) — served through a Flask web application.

---

## 📌 Project Overview

**ClaimWise Insurance Prediction** is a machine learning project developed to analyze the claim dataset and prepare it for predicting insurance claim loss.

The project follows a structured machine learning workflow that includes data understanding, data cleaning, preprocessing, feature preparation, exploratory data analysis, train-test splitting, model development, model evaluation, and web application integration.

The feature columns are anonymised in the dataset, so the project keeps their original names and
never invents domain labels for them.

The primary prediction target of the project is the **`loss`** variable.

---

## 🎯 Objectives

The main objectives of ClaimWise Insurance Prediction are:

- Analyze insurance-related data.
- Understand the structure and characteristics of the dataset.
- Identify and handle missing values.
- Remove duplicate records.
- Encode categorical variables.
- Scale and normalize numerical features.
- Prepare clean and consistent data for machine learning.
- Separate input features from the target variable.
- Split the dataset into training and testing datasets.
- Perform exploratory data analysis.
- Develop suitable regression models.
- Evaluate and compare model performance.
- Apply PCA and evaluate its effect on modelling.
- Build classifiers (Logistic Regression, Decision Tree) with confusion matrix, precision, recall and F1.
- Explore unsupervised structure with K-Means clustering and DBSCAN.
- Detect anomalies with Isolation Forest (unusual records — never called fraud).
- Validate every model with leakage-free 5-fold cross-validation.
- Provide a professional web-based interface.
- Integrate the trained machine learning model for future real-time prediction.

---

## 🧠 Machine Learning Problem

### Problem Type

**Regression**

### Target Variable

`loss`

The target variable `loss` represents the insurance claim loss that the machine learning system is designed to predict.

### Input Features

After preprocessing, the dataset contains:

- **130 input features**
- **1 target variable**
- **Target:** `loss`

The `id` column is removed because it is an identifier and does not provide meaningful predictive information.

---

## 📊 Dataset Information

The project uses an insurance dataset containing **50,000 records** with **132 original columns**.

| Dataset Property | Value |
|---|---:|
| Total Records | 50,000 |
| Original Columns | 132 |
| Final Records | 50,000 |
| Final Columns | 131 |
| Input Features | 130 |
| Target Variable | `loss` |
| Removed Column | `id` |
| Training Records | 40,000 |
| Testing Records | 10,000 |
| Missing Values After Preprocessing | 0 |
| Duplicate Records After Preprocessing | 0 |

---

## 🔄 Machine Learning Workflow

```text
Raw Insurance Dataset
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Duplicate Removal
        ↓
Missing Value Handling
        ↓
Categorical Feature Encoding
        ↓
Feature Scaling
        ↓
Feature Normalization
        ↓
Feature / Target Separation
        ↓
Train-Test Split
        ↓
Exploratory Data Analysis
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Claim Loss Prediction
        ↓
Web Application

---

## 📓 Jupyter Notebooks

The complete machine-learning workflow is organised as seventeen ordered, fully executable notebooks in
`Notebooks/` — 01–10 for the production regression system and 11–17 for the academic extension.
Every notebook runs from top to bottom (**Kernel → Restart Kernel and Run All**) with
no manual steps.

| # | Notebook | Purpose | Model |
|---|----------|---------|-------|
| 01 | `01_ClaimWise_Data_Inspection.ipynb` | Audit the raw dataset: shape, data types, missing values, duplicates, target | — |
| 02 | `02_ClaimWise_Data_Preprocessing.ipynb` | Run the single authoritative preprocessing pipeline and verify every stage + data-leakage checks | — |
| 03 | `03_ClaimWise_EDA.ipynb` | Exploratory data analysis on the preprocessed dataset | — |
| 04 | `04_ClaimWise_Linear_Regression.ipynb` | Train, evaluate, visualize and save the linear baseline | Linear Regression |
| 05 | `05_ClaimWise_Decision_Tree.ipynb` | Train, evaluate, visualize and save the decision tree | Decision Tree Regressor |
| 06 | `06_ClaimWise_Random_Forest.ipynb` | Train, evaluate, visualize and save the random forest + feature importances | Random Forest Regressor |
| 07 | `07_ClaimWise_Gradient_Boosting.ipynb` | Train, evaluate, visualize and save gradient boosting | Gradient Boosting Regressor |
| 08 | `08_ClaimWise_Hist_Gradient_Boosting.ipynb` | Adapt the provided Google Drive notebook: 3-fold CV with early stopping, train and save the histogram GB model | Hist Gradient Boosting Regressor |
| 09 | `09_ClaimWise_Model_Comparison.ipynb` | Train all five models, compare MAE / MSE / RMSE / R², select the best | all five |
| 10 | `10_ClaimWise_Final_Prediction.ipynb` | End-to-end prediction of claim loss with the final saved model | best model |
| 11 | `11_ClaimWise_PCA_Feature_Analysis.ipynb` | PCA: explained variance, component loadings, PCA-space vs original-space experiment | — |
| 12 | `12_ClaimWise_Logistic_Regression_Classification.ipynb` | Derived high-loss target, logistic regression, confusion matrix, precision/recall/F1, ROC | Logistic Regression |
| 13 | `13_ClaimWise_Decision_Tree_Classification.ipynb` | Decision tree classifier, confusion matrix and comparison with logistic regression | Decision Tree Classifier |
| 14 | `14_ClaimWise_Clustering.ipynb` | K-Means clustering on the features only (target excluded), elbow + silhouette selection | — |
| 15 | `15_ClaimWise_DBSCAN.ipynb` | DBSCAN with k-distance eps selection, noise reporting and a PCA-20 variant | — |
| 16 | `16_ClaimWise_Anomaly_Detection.ipynb` | Isolation Forest anomaly detection (features only; anomalies are not fraud) | — |
| 17 | `17_ClaimWise_5Fold_Cross_Validation.ipynb` | Leakage-free 5-fold CV for all five regressors and both classifiers | all seven |

Notes:

- The notebooks **reuse** the existing project code — preprocessing is executed from
  `Src/claimwise_preprocessing_pipeline.py`, EDA from `Src/6_EDA_Analysis.py`, model definitions from
  `Src/<model>_model.py` and the shared helpers in `Src/model_training_utils.py`.
- Notebooks 11–17 additionally share `Src/academic_common.py` (derived classification target,
  metrics, figure helpers, output folders).
- The `StandardScaler` is fitted **only on the 80% training rows** (data-leakage fix, documented and
  verified in notebook 02).
- All metrics, tables and figures are produced by executing the notebooks; nothing is hard-coded.

### Academic extension — results (executed)

| Method | Key result |
|---|---|
| PCA (11) | 89 of 130 components keep 95% of the variance; PCA space is worse than the original features (HistGB R² 0.5099 vs 0.5459), so production keeps 130 features |
| Logistic Regression (12) | accuracy 0.7583, precision 0.7828, recall 0.7101, F1 0.7446 |
| Decision Tree classifier (13) | accuracy 0.6882, precision 0.7062, recall 0.6365, F1 0.6696 |
| K-Means (14) | k = 2 by silhouette (0.0836), 50,000 rows, target excluded from the inputs |
| DBSCAN (15) | 7 clusters / 164 noise (3.28%) on 130 features; 3 clusters / 67 noise on PCA-20 |
| Isolation Forest (16) | 1,641 of 50,000 flagged (3.28%) — unusual, not fraudulent |
| 5-fold CV (17) | HistGB R² 0.553189 ± 0.005585, Logistic accuracy 0.7635 ± 0.0039, preprocessing inside every fold |

**How to run**

```bash
pip install -r requirements.txt
jupyter notebook Notebooks/
```

---

## 🤖 Machine Learning Models (13 Models Evaluated)

Thirteen regression models were trained on the exact 40,000-row training split and evaluated on the held-out 10,000-row test dataset (`Dataset/06_split_X_test.csv`, `06_split_y_test.csv`) across all 7 standard continuous loss metrics:

| Rank | Model | MAE | RMSE | R² | EV | MedAE | MAPE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | **Hist Gradient Boosting Regressor** | **1,215.30** | **1,921.02** | **0.5459** | **0.5459** | **789.34** | **63.68%** |
| 2 | Extra Trees Regressor | 1,254.61 | 1,980.32 | 0.5174 | 0.5174 | 808.38 | 67.20% |
| 3 | Random Forest Regressor | 1,259.62 | 1,984.41 | 0.5154 | 0.5154 | 819.76 | 68.08% |
| 4 | Gradient Boosting Regressor | 1,277.78 | 2,005.26 | 0.5052 | 0.5052 | 856.61 | 69.53% |
| 5 | Ridge Regression | 1,333.49 | 2,050.45 | 0.4826 | 0.4827 | 926.40 | 70.90% |
| 6 | Linear Regression | 1,333.57 | 2,050.52 | 0.4826 | 0.4826 | 926.69 | 70.90% |
| 7 | Lasso Regression | 1,333.58 | 2,051.88 | 0.4819 | 0.4819 | 924.98 | 70.58% |
| 8 | Elastic Net Regression | 1,394.39 | 2,162.80 | 0.4244 | 0.4244 | 997.68 | 76.61% |
| 9 | Support Vector Regression (Linear) | 1,289.04 | 2,174.74 | 0.4180 | 0.4387 | 785.42 | 58.12% |
| 10 | Decision Tree Regressor | 1,609.92 | 2,638.94 | 0.1431 | 0.1431 | 899.17 | 76.17% |
| 11 | K-Nearest Neighbors Regressor | 1,720.33 | 2,673.83 | 0.1203 | 0.1294 | 1,215.07 | 95.17% |
| 12 | Dummy Regressor (Median Baseline) | 1,788.56 | 2,986.61 | -0.0976 | 0.0000 | 1,104.24 | 80.34% |
| 13 | AdaBoost Regressor | 4,278.58 | 4,627.19 | -1.6346 | 0.2775 | 4,580.97 | 334.40% |

The selection rule strictly identifies the model with the **lowest test RMSE** (MAE, MSE, and highest R² as tie-breakers), confirming **Hist Gradient Boosting Regressor** as the champion production model (`Models/best_model.pkl`).

---

## 🌐 Web Application & Features

```bash
.venv/bin/python app.py      # http://127.0.0.1:5000
```

- `/` / `/index` — Modern, dynamic landing page with verified 50,000 dataset records, 130 features, champion model spotlight card, complete 13-model benchmark summary, and direct stage navigation.
- `/dataset` — Dataset management portal with verified Mode A production dataset audit, Mode B sandbox upload analyzer with 6-point schema compatibility validator, and direct modal preview/download of raw and split CSV artifacts.
- `/preprocessing` — Stage-by-stage pipeline audit across all 10 stages (132 → 131 → 130 features), empirical data leakage verification, and verified CSV artifact cards with preview modal.
- `/visualization` — Comprehensive module-wise exploratory data analysis and model diagnostic figures across Modules 1 to 5, featuring interactive tabs, modal expansions, and 8-point academic metadata breakdowns.
- `/models` — Complete 13-model benchmark suite featuring 5 KPI highlight cards (Total Evaluated, Lowest MAE, Lowest RMSE, Highest R², Champion Model), interactive comparison table with 7 continuous metrics, diagnostic bar charts, model evaluation CSV tables, and notebook references.
- `/prediction` — **Exactly ONE primary prediction workflow**:
  1. *Section 1 (Continuous Numeric Predictors)*: Granular inputs for all 14 continuous variables (`cont1`–`cont14`) with historical range tags and median defaults.
  2. *Section 2 (Primary Categorical Factors)*: Top influential categorical attributes (`cat80`, `cat79`, `cat1`, `cat2`, `cat50`, `cat12`) identified through feature importance.
  3. *Section 3 (Advanced Technical Predictors)*: Collapsible drawer providing transparency into all remaining categorical features (`cat1`–`cat116`) pre-filled with observed historical modes.
  4. *Direct Inference Engine*: Evaluates submitted input against the champion `HistGradientBoostingRegressor` model, outputting formatted claim loss, feature verification summary, and legal estimation disclaimer without mock delays or duplicate tabs.
- `/academic` — Full academic extension including PCA variance decomposition, logistic regression & decision tree classification, K-Means clustering, DBSCAN density analysis, Isolation Forest anomaly detection, and leakage-free 5-fold cross-validation.
- `/dashboard` — Executive ML telemetry dashboard with dynamic KPI cards, top benchmark comparison charts, recent model logs, and quick artifact previews.
- `/reports` — Full audit center with system architecture specification, 13-model leaderboard, technical audit matrices, and complete 70+ CSV artifact inventory with universal modal preview and download buttons.
- `/csv/preview` & `/csv/download/<path>` — Universal safe CSV modal viewer endpoint supporting chunked debounced search, server-side pagination, verified row counts, and strict traversal-protected downloads.

---

## 🧪 Testing and Verification

Run the comprehensive automated test suite verifying data integrity, model inference parity, single prediction workflow, route availability, and CSV preview/download security:

```bash
.venv/bin/pytest tests/test_claimwise_system.py -v
```

**Test Suite Results:**
- **17 passed, 0 failed, 0 errors** (100% pass rate).
- Validates all 11 application routes, model file integrity, 130-feature shape compatibility, numerical inference parity, single-form enforcement, and CSV security endpoints.

---

## 📁 Project Structure

```text
Credit ML/
├── Dataset/            # raw + intermediate + final train/test CSV files (01_ ... 06_)
├── Models/             # saved models (*.pkl), best_model.pkl + claimwise_preprocessor.pkl
├── Notebooks/          # 01 ... 17 — the executable ML workflow + academic extension
├── Outputs/            # metrics, predictions, comparisons and all 90+ figures across modules
├── Src/                # preprocessing, EDA, model training, dataset manager, comparison scripts
├── reports/            # dataset lifecycle + audit reports
├── tests/              # automated verification test suite (test_claimwise_system.py)
├── templates/          # Jinja2 templates (index, dataset, preprocessing, visualization, models, etc.)
├── static/             # design-system styles (style.css)
├── uploads/            # isolated sandbox for uploaded CSV datasets (never overwrites Dataset/)
├── app.py              # Flask web application
├── main.py             # runs the preprocessing pipeline
├── FINAL_REPORT.md     # complete traceable project verification report
├── README.md           # project documentation
└── requirements.txt    # verified python dependencies
```

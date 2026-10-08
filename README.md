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

## 🤖 Models

Five regression models are trained on the identical 40,000/10,000 split and compared on the same
10,000 test rows:

| Rank | Model | MAE | MSE | RMSE | R² |
|---:|---|---:|---:|---:|---:|
| 1 | Hist Gradient Boosting Regressor | 1215.2997 | 3690333.1165 | 1921.0240 | 0.545901 |
| 2 | Random Forest Regressor | 1259.6218 | 3937886.6824 | 1984.4109 | 0.515440 |
| 3 | Gradient Boosting Regressor | 1277.7829 | 4021053.5725 | 2005.2565 | 0.505206 |
| 4 | Linear Regression | 1333.5665 | 4204641.7064 | 2050.5223 | 0.482615 |
| 5 | Decision Tree Regressor | 1609.9189 | 6964030.4498 | 2638.9450 | 0.143070 |

The selection rule is the **lowest test RMSE** (MAE, MSE, then highest R² as tie-breakers), so
`Models/best_model.pkl` currently holds the Hist Gradient Boosting Regressor.

### Fifth model — adapted from the provided Google Drive code

The provided notebooks (`Home_Credit_Default_Risk_FINAL_Perfect.ipynb` and
`Home_Credit_Default_Risk_FINAL_FIXED.ipynb`) were a LightGBM binary-classification Kaggle pipeline.
`Notebooks/08_ClaimWise_Hist_Gradient_Boosting.ipynb` keeps their reusable logic — SEED 42,
3-fold cross-validation with patience-based early stopping, out-of-fold scoring, prediction
sanity checks — and maps it onto this regression problem with sklearn's
`HistGradientBoostingRegressor` (LightGBM is not installed and no network install is allowed).
The mapping table (what was preserved, re-parameterised, or dropped, and why) is documented in that
notebook. The multi-table Home Credit feature engineering has no equivalent in the single-table
ClaimWise dataset and is therefore not carried over.

---

## 🌐 Web Application

```bash
.venv/bin/python app.py      # http://127.0.0.1:5000
```

- `/dataset`, `/preprocessing`, `/visualization` — verified dataset facts, pipeline stages and the
  leakage check (scaler fitted on training rows only).
- `/models`, `/dashboard`, `/reports` — real metrics read from `Outputs/Model_Comparison/model_comparison.csv`
  and the model figures produced by the notebooks.
- `/academic` — the academic extension: PCA, classification (confusion matrix, precision/recall/F1),
  clustering, DBSCAN, anomaly detection and 5-fold cross-validation, each section reading its real
  CSVs and figures from `Outputs/`.
- `/prediction` — the real prediction form: all 130 raw features (14 number inputs with the dataset's
  min/max, 116 dropdowns with the values that actually occur in the data), transformed by the saved
  `LabelEncoder`/`StandardScaler` (`Models/claimwise_preprocessor.pkl`, verified to reproduce
  `05_preprocessed_final.csv`) and scored by `Models/best_model.pkl`. A secondary mode predicts a
  prepared test row by index.
- `/notebooks/<file>` — download any of the seventeen executed notebooks.

---

## 📁 Project Structure

```text
Credit ML/
├── Dataset/            # raw + intermediate + final train/test CSV files (01_ ... 06_)
├── Models/             # saved models (*.pkl), best_model.pkl + claimwise_preprocessor.pkl
├── Notebooks/          # 01 ... 17 — the executable ML workflow + academic extension
├── Outputs/            # metrics, predictions, comparisons and all figures
├── Src/                # preprocessing, EDA, model training, comparison and prediction-prep scripts
├── reports/            # dataset lifecycle + audit reports
├── app.py              # Flask web dashboard using the trained model
├── main.py             # runs the preprocessing pipeline
├── README.md
└── requirements.txt
```

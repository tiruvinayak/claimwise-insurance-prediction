import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================
# CLAIMWISE INSURANCE PREDICTION — MASTER PREPROCESSING PIPELINE
# ============================================================
# Sequential Intermediate Dataset Flow:
# 1. 01_raw_claimwise_50000.csv
#    ↓
# 2. 02_cleaned_dedup_median_imputed.csv
#    ↓
# 3. 03_encoded_label.csv
#    ↓
# 4. 04_scaled_standardized.csv
#    ↓
# 5. 05_preprocessed_final.csv + X_train, X_test, y_train, y_test
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Fixed: was "/" causing Read-only error
DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "01_raw_claimwise_50000.csv")
STEP2_PATH = os.path.join(BASE_DIR, "Dataset", "02_cleaned_dedup_median_imputed.csv")
STEP3_PATH = os.path.join(BASE_DIR, "Dataset", "03_encoded_label.csv")
STEP4_PATH = os.path.join(BASE_DIR, "Dataset", "04_scaled_standardized.csv")
PREPROCESSED_DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "05_preprocessed_final.csv")

X_TRAIN_PATH = os.path.join(BASE_DIR, "Dataset", "06_split_X_train.csv")
X_TEST_PATH = os.path.join(BASE_DIR, "Dataset", "06_split_X_test.csv")
Y_TRAIN_PATH = os.path.join(BASE_DIR, "Dataset", "06_split_y_train.csv")
Y_TEST_PATH = os.path.join(BASE_DIR, "Dataset", "06_split_y_test.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "Outputs", "EDA_Analysis_outputs")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

if not os.path.exists(DATASET_PATH):
    print("Dataset file not found:")
    print(DATASET_PATH)
    raise FileNotFoundError(DATASET_PATH)

print("=" * 75)
print("1. STEP 1: LOAD DATASET AND AUDIT (01_raw_claimwise_50000.csv)")
print("=" * 75)
df = pd.read_csv(DATASET_PATH)
print("Original Dataset Shape:", df.shape)

# ============================================================
# Step 2: Cleaning, Imputation & Dropping ID -> 02_cleaned_dedup_median_imputed.csv
# ============================================================
print("\n" + "=" * 75)
print("2. STEP 2: CLEANING & IMPUTATION -> 02_cleaned_dedup_median_imputed.csv")
print("=" * 75)

df_clean = df.copy()

# Remove duplicate rows
df_clean = df_clean.drop_duplicates()

# Impute numerical features with median
numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
for col in numeric_cols:
    if df_clean[col].isnull().any():
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Strip whitespace and impute categorical features with mode
cat_cols = df_clean.select_dtypes(exclude=np.number).columns.tolist()
for col in cat_cols:
    df_clean[col] = df_clean[col].apply(lambda x: x.strip() if isinstance(x, str) else x)  # Fixed: preserve NaN
    if df_clean[col].isnull().any():
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# Drop ID column if present
if "id" in df_clean.columns:
    df_clean = df_clean.drop(columns=["id"])
    print("Removed column: 'id' (Unique identifier with no predictive value).")

# Save Step 2 intermediate CSV
df_clean.to_csv(STEP2_PATH, index=False)
print("Saved Step 2 dataset:", STEP2_PATH)

# ============================================================
# Step 3: Label Encoding -> 03_encoded_label.csv
# ============================================================
print("\n" + "=" * 75)
print("3. STEP 3: CATEGORICAL LABEL ENCODING -> 03_encoded_label.csv")
print("=" * 75)

df_step2 = pd.read_csv(STEP2_PATH)
df_step3 = df_step2.copy()

categorical_columns = df_step3.select_dtypes(include=["object", "str", "category"]).columns.tolist()
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df_step3[col] = le.fit_transform(df_step3[col])
    label_encoders[col] = le

df_step3.to_csv(STEP3_PATH, index=False)
print("Saved Step 3 dataset:", STEP3_PATH)

# ============================================================
# Step 4: Feature Scaling -> 04_scaled_standardized.csv
# ============================================================
print("\n" + "=" * 75)
print("4. STEP 4: FEATURE SCALING -> 04_scaled_standardized.csv")
print("=" * 75)

df_step3_read = pd.read_csv(STEP3_PATH)
df_step4 = df_step3_read.copy()

continuous_features = [col for col in df_step4.columns if col.startswith("cont")]

# DATA LEAKAGE FIX:
# The scaler must only learn statistics (mean/std) from the training rows.
# The 80/20 split below uses test_size=0.20 and random_state=42, and
# train_test_split depends only on the number of rows and that seed, so the
# same training row indices are reproduced here and in Step 5.
row_indices = np.arange(len(df_step4))
train_indices, _ = train_test_split(row_indices, test_size=0.20, random_state=42)

scaler = StandardScaler()
scaler.fit(df_step4.iloc[train_indices][continuous_features])
df_step4[continuous_features] = scaler.transform(df_step4[continuous_features])

df_step4.to_csv(STEP4_PATH, index=False)
print("Saved Step 4 dataset:", STEP4_PATH)

# ============================================================
# Step 5: Final Preprocessing & Train-Test Split -> 05_preprocessed_final.csv
# ============================================================
print("\n" + "=" * 75)
print("5. STEP 5: FINAL DATASET & SPLITTING -> 05_preprocessed_final.csv")
print("=" * 75)

df_final = pd.read_csv(STEP4_PATH)
target_col = "loss"

X = df_final.drop(columns=[target_col])
y = df_final[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Save final outputs
df_final.to_csv(PREPROCESSED_DATASET_PATH, index=False)
X_train.to_csv(X_TRAIN_PATH, index=False)
X_test.to_csv(X_TEST_PATH, index=False)
y_train.to_csv(Y_TRAIN_PATH, index=False)
y_test.to_csv(Y_TEST_PATH, index=False)

print("Saved Preprocessed Dataset:", PREPROCESSED_DATASET_PATH)
print("Saved X_train              :", X_TRAIN_PATH)
print("Saved X_test               :", X_TEST_PATH)
print("Saved y_train              :", Y_TRAIN_PATH)
print("Saved y_test               :", Y_TEST_PATH)

# ============================================================
# Final Summary
# ============================================================
print("\n" + "=" * 75)
print("========== FINAL PREPROCESSING SUMMARY ==========")
print("=" * 75)
print(f"Original dataset rows  : {df.shape[0]}")
print(f"Final dataset rows     : {df_final.shape[0]}")
print(f"Original columns       : {df.shape[1]}")
print(f"Final columns          : {df_final.shape[1]}")
print(f"Training dataset rows  : {X_train.shape[0]}")
print(f"Testing dataset rows   : {X_test.shape[0]}")
print(f"Missing values remaining: {df_final.isnull().sum().sum()}")
print(f"Duplicate rows remaining: {df_final.duplicated().sum()}")
print("-------------------------------------------------")
print("Target Column          : loss")
print("Removed Columns        : ['id']")
print(f"Total Input Features   : {X.shape[1]}")
print("=================================================")
print("ClaimWise Insurance Prediction preprocessing completed successfully.")

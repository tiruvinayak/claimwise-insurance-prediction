import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================================
# CLAIMWISE INSURANCE PREDICTION — STEP 4
# Note: filename legacy 'minmax_stand_norma' but actually uses StandardScaler only (correct for cont features)
# FEATURE SCALING (STANDARD SCALER)
# Reads : 03_encoded_label.csv
# Saves : 04_scaled_standardized.csv
# ==========================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Project root: Credit ML
INPUT_FILE = os.path.join(BASE_DIR, "Dataset", "03_encoded_label.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "Dataset", "04_scaled_standardized.csv")

if not os.path.exists(INPUT_FILE):
    print("Input dataset file not found at:")
    print(INPUT_FILE)
    raise FileNotFoundError(INPUT_FILE)

print("Input dataset file found:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)
data = df.copy()

print("=" * 70)
print("STEP 4: FEATURE SCALING (03_encoded_label.csv -> 04_scaled_standardized.csv)")
print("=" * 70)

# Remove 'id' column if present
if "id" in data.columns:
    data = data.drop(columns=["id"])

continuous_features = [col for col in data.columns if col.startswith("cont")]
target_col = "loss"

print("Continuous Features to Scale:", continuous_features)
print("Target Column              :", target_col)

# Apply StandardScaler
# DATA LEAKAGE FIX: fit the scaler on the 80% training rows only, then
# transform every row. The split uses test_size=0.20 and random_state=42, the
# same values used later in 5_final_preprocess_M2.py, so the training rows
# identified here are exactly the rows used to train the models.
row_indices = np.arange(len(data))
train_indices, _ = train_test_split(row_indices, test_size=0.20, random_state=42)

scaler = StandardScaler()
scaler.fit(data.iloc[train_indices][continuous_features])
scaled_data = data.copy()
scaled_data[continuous_features] = scaler.transform(data[continuous_features])

# Save output
scaled_data.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("STEP 4 COMPLETED SUCCESSFULLY")
print("=" * 70)
print("Scaled Dataset Saved To :", OUTPUT_FILE)
print("Final Scaled Data Shape :", scaled_data.shape)

import os
import sys
import subprocess

# ============================================================
# CLAIMWISE INSURANCE PREDICTION — MAIN ENTRY POINT
# ============================================================
# This is the main entry script for the ClaimWise Insurance
# Prediction project, following the Placement Prediction project structure.
# ============================================================

def main():
    print("=" * 70)
    print("       CLAIMWISE INSURANCE PREDICTION PROJECT       ")
    print("==================================================")
    print("\nProject Structure:")
    print(" - Dataset/              : CSV datasets and preprocessed splits")
    print(" - Src/                  : Preprocessing and EDA modules")
    print(" - Outputs/              : Visualization charts and correlation metrics")
    print(" - Models/               : Machine Learning model artifacts")
    print(" - General_Programs/     : Utility scripts")
    print(" - static/ & templates/  : Web dashboard interface")
    print("=" * 70)

    pipeline_script = os.path.join(os.path.dirname(__file__), "Src", "claimwise_preprocessing_pipeline.py")
    if os.path.exists(pipeline_script):  # Fixed: now uses absolute Src path
        print(f"\nRunning main preprocessing pipeline: {pipeline_script}...\n")
        # Use same python executable (works with venv) instead of hardcoded python3
        subprocess.run([sys.executable, pipeline_script])
    else:
        print(f"Error: {pipeline_script} not found.")

if __name__ == "__main__":
    main()

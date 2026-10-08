"""Compare saved model metrics and select the best regression model."""

import shutil

import joblib
import pandas as pd

from model_training_utils import (
    MODELS_DIR,
    OUTPUTS_DIR,
    load_prepared_data,
    save_best_model_plots,
    write_model_comparison,
)


MODEL_FILES = {
    "Linear Regression": "linear_regression.pkl",
    "Decision Tree Regressor": "decision_tree_regressor.pkl",
    "Random Forest Regressor": "random_forest_regressor.pkl",
    "Gradient Boosting Regressor": "gradient_boosting_regressor.pkl",
    "Hist Gradient Boosting Regressor": "hist_gradient_boosting_regressor.pkl",
}


def main() -> None:
    model_names = [
        "Linear Regression",
        "Decision Tree Regressor",
        "Random Forest Regressor",
        "Gradient Boosting Regressor",
        "Hist Gradient Boosting Regressor",
    ]
    results = {}
    for model_name in model_names:
        folder_name = model_name.replace(" ", "_")
        metrics_path = OUTPUTS_DIR / folder_name / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Missing {metrics_path}. Run the model training scripts first."
            )
        row = pd.read_csv(metrics_path).iloc[0]
        results[model_name] = {
            "MAE": float(row["MAE"]),
            "MSE": float(row["MSE"]),
            "RMSE": float(row["RMSE"]),
            "R2": float(row["R2"]),
        }

    best_name, best_metrics = write_model_comparison(results)
    best_source = MODELS_DIR / MODEL_FILES[best_name]
    best_destination = MODELS_DIR / "best_model.pkl"
    shutil.copyfile(best_source, best_destination)

    _, x_test, _, y_test = load_prepared_data()
    best_model = joblib.load(best_destination)
    save_best_model_plots(best_name, y_test, best_model.predict(x_test))

    print("Model comparison saved to:", OUTPUTS_DIR / "Model_Comparison" / "model_comparison.csv")
    print("Best model selected by lowest RMSE, with MAE/MSE/R2 tie-breakers:", best_name)
    print("Best model metrics:", best_metrics)
    print("Best model saved to:", best_destination)


if __name__ == "__main__":
    main()

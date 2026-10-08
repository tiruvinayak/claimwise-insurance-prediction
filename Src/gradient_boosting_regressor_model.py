"""Train and evaluate the ClaimWise Gradient Boosting Regressor."""

from sklearn.ensemble import GradientBoostingRegressor

from model_training_utils import load_prepared_data, save_model_results


def build_model() -> GradientBoostingRegressor:
    """Create the ClaimWise GradientBoostingRegressor model (single source of truth).

    Used by the training script and by the matching Jupyter notebook so the
    hyperparameters are defined in exactly one place.
    """
    return GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=2,
        random_state=42,
        loss="squared_error",
    )


def main() -> None:
    x_train, x_test, y_train, y_test = load_prepared_data()
    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = save_model_results(
        "Gradient Boosting Regressor",
        "gradient_boosting_regressor.pkl",
        model,
        x_test,
        y_test,
        predictions,
    )
    print("Gradient Boosting Regressor metrics:", metrics)


if __name__ == "__main__":
    main()

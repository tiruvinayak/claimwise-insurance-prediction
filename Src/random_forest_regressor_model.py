"""Train and evaluate the ClaimWise Random Forest Regressor."""

from sklearn.ensemble import RandomForestRegressor

from model_training_utils import (
    load_prepared_data,
    save_model_results,
    save_random_forest_feature_importance,
)


def build_model() -> RandomForestRegressor:
    """Create the ClaimWise RandomForestRegressor model (single source of truth).

    Used by the training script and by the matching Jupyter notebook so the
    hyperparameters are defined in exactly one place.
    """
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


def main() -> None:
    x_train, x_test, y_train, y_test = load_prepared_data()
    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = save_model_results(
        "Random Forest Regressor",
        "random_forest_regressor.pkl",
        model,
        x_test,
        y_test,
        predictions,
    )
    save_random_forest_feature_importance(model, x_train.columns)
    print("Random Forest Regressor metrics:", metrics)


if __name__ == "__main__":
    main()

"""Train and evaluate the ClaimWise Decision Tree Regressor."""

from sklearn.tree import DecisionTreeRegressor

from model_training_utils import load_prepared_data, save_model_results


def build_model() -> DecisionTreeRegressor:
    """Create the ClaimWise DecisionTreeRegressor model (single source of truth).

    Used by the training script and by the matching Jupyter notebook so the
    hyperparameters are defined in exactly one place.
    """
    return DecisionTreeRegressor(
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
    )


def main() -> None:
    x_train, x_test, y_train, y_test = load_prepared_data()
    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = save_model_results(
        "Decision Tree Regressor",
        "decision_tree_regressor.pkl",
        model,
        x_test,
        y_test,
        predictions,
    )
    print("Decision Tree Regressor metrics:", metrics)


if __name__ == "__main__":
    main()

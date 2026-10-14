"""Run the complete ClaimWise regression model development pipeline."""

from __future__ import annotations

from train_all_regression_models import main as train_all_models


def main() -> None:
    train_all_models()


if __name__ == "__main__":
    main()


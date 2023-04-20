import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from typing import Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def calculate_feature_importance(
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        mse_test: float,
        label: str,
        seed: int = 0,
        iterations: int = 10
) -> pd.DataFrame:
    """Iterates over all features in X_test and calculates the corresponding PFI score based on one permutation.

    """
    np.random.seed(seed)

    pfis = []
    for i in range(iterations):
        for feature in X_test:
            if feature == label:
                continue
            X_test_perm = X_test.copy()
            X_test_perm[feature] = np.random.permutation(X_test_perm[feature])

            mse = mean_squared_error(y_test.to_numpy(), model.predict(X_test_perm))
            pfis += [{"feature": feature, "iteration": i, "importance": mse - mse_test}]

    return pd.DataFrame(pfis)


def fit_model(
        train_data: pd.DataFrame,
        features: List[str], target: str,
        verbosity: int = 0
) -> Tuple[float, pd.DataFrame]:

    train, test = train_test_split(train_data, test_size=0.3, random_state=42)

    model = TabularPredictor(label=target, verbosity=verbosity)
    model.fit(train_data)

    mse_test = mean_squared_error(test[target].to_numpy(), model.predict(test[features]))
    pfis = calculate_feature_importance(model, test[features], test[target], mse_test, target)

    return mse_test, pfis





from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import load_data, split_data

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "housing_model.pkl"


def fit_model(X_train, y_train):
    """
    Train a Histogram-based Gradient Boosting Regressor.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    Returns:
        HistGradientBoostingRegressor: The fitted model.
    """
    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.1,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Score the model on the held-out test set.

    Returns:
        dict: R², MAE and RMSE (MAE/RMSE in $100,000s).
    """
    y_pred = model.predict(X_test)
    return {
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
    }


def save_model(model, feature_names, metrics, n_train, n_test):
    """
    Save the model together with the metadata the API serves at /model-info.
    """
    bundle = {
        "model": model,
        "model_type": type(model).__name__,
        "feature_names": feature_names,
        "target": "MedHouseVal (median house value, $100,000s)",
        "metrics": metrics,
        "n_train": n_train,
        "n_test": n_test,
        "n_iterations": int(model.n_iter_),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)


if __name__ == "__main__":
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model, feature_names, metrics, len(X_train), len(X_test))

    print(f"Trained {type(model).__name__} ({model.n_iter_} boosting iterations)")
    print(f"Test R²:   {metrics['r2']}")
    print(f"Test MAE:  {metrics['mae']}  (~${metrics['mae'] * 100_000:,.0f})")
    print(f"Test RMSE: {metrics['rmse']}  (~${metrics['rmse'] * 100_000:,.0f})")
    print(f"Saved to {MODEL_PATH}")

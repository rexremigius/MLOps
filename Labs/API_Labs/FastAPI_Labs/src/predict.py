from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "housing_model.pkl"


@lru_cache(maxsize=1)
def load_bundle():
    """
    Load the saved model bundle once and reuse it for every request.

    Returns:
        dict: The fitted model plus its training metadata (see train.save_model).
    Raises:
        FileNotFoundError: If train.py has not been run yet.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model at {MODEL_PATH}. Run `python train.py` from src/ first."
        )
    return joblib.load(MODEL_PATH)


def predict_data(X):
    """
    Predict median house values for the input rows.

    Args:
        X (array-like): Rows of features in the training column order
            [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude].
    Returns:
        y_pred (numpy.ndarray): Predicted median house values, in $100,000s.
    """
    model = load_bundle()["model"]
    return model.predict(np.asarray(X, dtype=float))

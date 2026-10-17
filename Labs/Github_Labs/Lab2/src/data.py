from pathlib import Path

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

CACHE = Path(__file__).resolve().parent.parent / "data" / "openml"


def load_data():
    # Pima Indians diabetes (OpenML id 37): 768 patients, 8 features
    # target: does the patient have diabetes? yes (1) / no (0)
    ds = fetch_openml(data_id=37, as_frame=True, data_home=str(CACHE))
    X = ds.data.to_numpy(dtype=float)
    y = (ds.target == "tested_positive").astype(int).to_numpy()
    return X, y


def get_splits():
    X, y = load_data()
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

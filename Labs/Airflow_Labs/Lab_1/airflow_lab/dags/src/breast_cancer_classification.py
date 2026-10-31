"""
ML functions for the Airflow Lab 1 pipeline (breast cancer classifier).

Every task in a DAG run works inside the same folder, working_data/runs/<run_id>/.
Datasets and models are passed between tasks as files in that folder. Only small,
JSON-safe values (like the metrics dict) go through XCom.
"""
import json
import os
import re
import shutil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "breast_cancer.csv"
# New tumors with no diagnosis, held out from breast_cancer.csv so the model never trains on them
NEW_SAMPLES_PATH = DATA_PATH.parent / "test.csv"
# Mounted from the host by docker-compose, so outputs show up on your machine
WORKING_DIR = Path(os.environ.get("AIRFLOW_HOME", "/opt/airflow")) / "working_data"
REGISTRY_DIR = WORKING_DIR / "registry"
TARGET = "diagnosis"


def _run_dir(run_id: str) -> Path:
    """Folder for one DAG run. Run ids contain ':' and '+', so swap those out."""
    return WORKING_DIR / "runs" / re.sub(r"[^A-Za-z0-9_.-]", "_", str(run_id))


def load_data(run_id: str):
    """
    Validates the raw CSV and snapshots it into this run's folder,
    so the run can be reproduced even if the source file changes later.
    """
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"{DATA_PATH.name} has no '{TARGET}' column")
    if df[TARGET].nunique() != 2:
        raise ValueError(f"Expected 2 classes in '{TARGET}', found {sorted(df[TARGET].unique())}")

    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "raw.csv", index=False)
    print(f"Loaded {len(df)} rows x {df.shape[1]} columns into {run_dir}")
    print(f"Class counts: {df[TARGET].value_counts().to_dict()}")


def preprocess(run_id: str, test_size: float = 0.2):
    """
    Drops missing and duplicate rows, encodes the label (malignant = 1),
    and writes a stratified train/test split.
    """
    run_dir = _run_dir(run_id)
    df = pd.read_csv(run_dir / "raw.csv").dropna().drop_duplicates()
    df[TARGET] = (df[TARGET] == "malignant").astype(int)

    train, test = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=42)
    train.to_csv(run_dir / "train.csv", index=False)
    test.to_csv(run_dir / "test.csv", index=False)
    print(f"Train: {len(train)} rows, test: {len(test)} rows")


def train_model(run_id: str, n_estimators: int, max_depth: int):
    """Fits a RandomForest on the training split and saves it to model.joblib."""
    run_dir = _run_dir(run_id)
    train = pd.read_csv(run_dir / "train.csv")

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(train.drop(columns=TARGET), train[TARGET])
    joblib.dump(model, run_dir / "model.joblib")
    print(f"Trained RandomForest (n_estimators={n_estimators}, max_depth={max_depth})")


def evaluate_model(run_id: str) -> dict:
    """
    Scores the model on the held-out test split.
    Returns the metrics dict, which Airflow pushes to XCom for the branch task.
    """
    run_dir = _run_dir(run_id)
    test = pd.read_csv(run_dir / "test.csv")
    X, y = test.drop(columns=TARGET), test[TARGET]
    model = joblib.load(run_dir / "model.joblib")

    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Test metrics: {metrics}")
    return metrics


def register_model(run_id: str) -> str:
    """
    Copies an accepted model and its metrics into the registry,
    and points registry/latest.json at it.
    """
    run_dir = _run_dir(run_id)
    dest = REGISTRY_DIR / run_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "model.joblib", dest / "model.joblib")
    shutil.copy2(run_dir / "metrics.json", dest / "metrics.json")

    latest = {
        "run_id": run_id,
        "model_path": str(dest / "model.joblib"),
        "metrics": json.loads((dest / "metrics.json").read_text()),
    }
    (REGISTRY_DIR / "latest.json").write_text(json.dumps(latest, indent=2))
    print(f"Registered model at {dest}")
    return str(dest / "model.joblib")


def reject_model(run_id: str, min_f1: float):
    """Logs why the model was rejected. The registry is left unchanged."""
    metrics = json.loads((_run_dir(run_id) / "metrics.json").read_text())
    print(f"Rejected: test F1 {metrics['f1']} is below the threshold {min_f1}")

    latest = REGISTRY_DIR / "latest.json"
    if latest.exists():
        current = json.loads(latest.read_text())
        print(f"Registry still serves run {current['run_id']} (F1 {current['metrics']['f1']})")
    else:
        print("Registry is empty, no model has been accepted yet")


def predict(run_id: str) -> list:
    """
    Classifies the unlabeled tumors in data/test.csv with the registered model,
    not this run's model, since this run's model may have been rejected.
    Writes predictions.csv to the run folder and returns the predictions for XCom.
    """
    latest = REGISTRY_DIR / "latest.json"
    if not latest.exists():
        raise FileNotFoundError("No registered model yet. Trigger a run whose model passes the F1 threshold first.")
    current = json.loads(latest.read_text())
    model = joblib.load(current["model_path"])
    print(f"Using the registered model from run {current['run_id']} (F1 {current['metrics']['f1']})")

    # Same columns, in the same order, as the model was trained on
    samples = pd.read_csv(NEW_SAMPLES_PATH)[model.feature_names_in_]
    labels = model.predict(samples)
    proba = model.predict_proba(samples)[:, 1]

    predictions = [
        {"sample": i, "prediction": "malignant" if label == 1 else "benign", "malignant_probability": round(float(p), 3)}
        for i, (label, p) in enumerate(zip(labels, proba), start=1)
    ]
    pd.DataFrame(predictions).to_csv(_run_dir(run_id) / "predictions.csv", index=False)
    for p in predictions:
        print(f"Sample {p['sample']}: {p['prediction']} (malignant probability {p['malignant_probability']})")
    return predictions

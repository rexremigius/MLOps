import argparse, json, sys
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data import get_splits

ROOT = Path(__file__).resolve().parent.parent
MIN_AUC = 0.75  # workflow fails if the model drops below this

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamp", required=True)
    ts = ap.parse_args().timestamp

    model = joblib.load(ROOT / "models" / f"model_{ts}_gb.joblib")
    _, X_te, _, y_te = get_splits()

    pred = model.predict(X_te)
    metrics = {
        "accuracy": accuracy_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]),
    }

    (ROOT / "metrics").mkdir(exist_ok=True)
    with open(ROOT / "metrics" / f"{ts}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(metrics)

    if metrics["roc_auc"] < MIN_AUC:
        sys.exit(f"ROC AUC {metrics['roc_auc']:.3f} below {MIN_AUC}, not committing this model")

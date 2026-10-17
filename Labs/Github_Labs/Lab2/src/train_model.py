import argparse
from pathlib import Path

import joblib
import mlflow
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

from data import get_splits

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamp", required=True)
    ts = ap.parse_args().timestamp

    # diabetes yes/no dataset instead of make_classification
    X_tr, X_te, y_tr, y_te = get_splits()

    params = {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3,
              "subsample": 0.8, "random_state": 42}

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("pima_diabetes_gb")
    with mlflow.start_run(run_name=f"train_{ts}"):
        mlflow.log_params({**params, "n_train": len(y_tr), "n_test": len(y_te)})
        model = GradientBoostingClassifier(**params).fit(X_tr, y_tr)
        pred = model.predict(X_tr)
        mlflow.log_metrics({"train_accuracy": accuracy_score(y_tr, pred),
                            "train_f1": f1_score(y_tr, pred)})

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / f"model_{ts}_gb.joblib")
    print(f"saved models/model_{ts}_gb.joblib")

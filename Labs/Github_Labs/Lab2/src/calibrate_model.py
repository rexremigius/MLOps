import argparse, json
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss

from data import get_splits

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestamp", help="defaults to the latest model in models/")
    ap.add_argument("--method", default="isotonic", choices=["isotonic", "sigmoid"])
    args = ap.parse_args()

    if args.timestamp:
        ts = args.timestamp
    else:
        latest = sorted((ROOT / "models").glob("model_*_gb.joblib"))[-1]
        ts = latest.name[len("model_"):-len("_gb.joblib")]
    print(f"calibrating model {ts}")

    base = joblib.load(ROOT / "models" / f"model_{ts}_gb.joblib")
    X_tr, X_te, y_tr, y_te = get_splits()  # same seed as training

    cal = CalibratedClassifierCV(GradientBoostingClassifier(**base.get_params()),
                                 method=args.method, cv=5).fit(X_tr, y_tr)

    before = brier_score_loss(y_te, base.predict_proba(X_te)[:, 1])
    after = brier_score_loss(y_te, cal.predict_proba(X_te)[:, 1])

    joblib.dump(cal, ROOT / "models" / f"model_{ts}_gb_calibrated.joblib")
    (ROOT / "metrics").mkdir(exist_ok=True)
    with open(ROOT / "metrics" / f"{ts}_calibration.json", "w") as f:
        json.dump({"method": args.method, "brier_before": before, "brier_after": after}, f, indent=4)
    print(f"brier: {before:.4f} -> {after:.4f}")

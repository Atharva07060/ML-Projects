from pathlib import Path
import json
import math
import joblib
import pandas as pd
from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
app = Flask(__name__)


def load_artifacts():
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Trained model is missing. Run `python train.py` first.")
        model = joblib.load(MODEL_PATH)
        summary_path = REPORTS / "eda_summary.json"
        metrics_path = REPORTS / "model_comparison.csv"
        if not summary_path.exists() or not metrics_path.exists():
            raise FileNotFoundError("Reports are missing. Run `python train.py` first.")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = pd.read_csv(metrics_path).to_dict(orient="records")
        return model, summary, metrics, None
    except Exception as exc:
        return None, None, [], str(exc)


@app.route("/", methods=["GET", "POST"])
def index():
    model, summary, metrics, error = load_artifacts()
    prediction = None
    defaults = summary.get("prediction_defaults", {}) if summary else {}
    if request.method == "POST" and model is not None:
        try:
            row = {}
            for col in ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]:
                raw_value = request.form.get(col, "").strip()
                if col == "total_bedrooms" and not raw_value:
                    # This source column has missing values; the fitted pipeline imputes them.
                    row[col] = float("nan")
                    continue
                try:
                    row[col] = float(raw_value)
                except (TypeError, ValueError):
                    raise ValueError(f"{col.replace('_', ' ').title()} must be a valid number.")
                if not math.isfinite(row[col]):
                    raise ValueError(f"{col.replace('_', ' ').title()} must be a finite number.")
            if not (-125 <= row["longitude"] <= -113 and 32 <= row["latitude"] <= 43):
                raise ValueError("Enter a location within the California dataset range.")
            for col in ["housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]:
                if row[col] < 0:
                    raise ValueError(f"{col.replace('_', ' ').title()} cannot be negative.")
            category = request.form.get("ocean_proximity", "")
            if category not in {"<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"}:
                raise ValueError("Select an ocean proximity category.")
            row["ocean_proximity"] = category
            value = float(model.predict(pd.DataFrame([row]))[0])
            prediction = {"value": value, "formatted": f"${value:,.0f}"}
        except (ValueError, TypeError) as exc:
            error = f"Check the values and try again: {exc}"
    return render_template("index.html", summary=summary, metrics=metrics, error=error,
                           prediction=prediction, form=request.form, defaults=defaults)


@app.get("/health")
def health():
    _, _, _, error = load_artifacts()
    return ({"status": "ok"} if error is None else {"status": "not_ready", "error": error}, 200 if error is None else 503)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5001)

from pathlib import Path
import json
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
    if request.method == "POST" and model is not None:
        try:
            age = float(request.form.get("age", ""))
            salary = float(request.form.get("salary", ""))
            gender = request.form.get("gender", "")
            if not (1 <= age <= 120): raise ValueError("Age must be between 1 and 120.")
            if not (0 <= salary <= 100000000): raise ValueError("Salary must be a non-negative number.")
            if gender not in {"Male", "Female"}: raise ValueError("Choose Male or Female.")
            row = pd.DataFrame([{"Gender": gender, "Age": age, "EstimatedSalary": salary}])
            label = int(model.predict(row)[0])
            probability = float(model.predict_proba(row)[0][1])
            prediction = {"label": "Likely to purchase" if label else "Unlikely to purchase", "probability": probability}
        except (ValueError, TypeError) as exc:
            error = f"Please check the prediction inputs: {exc}"
    return render_template("index.html", summary=summary, metrics=metrics, error=error,
                           prediction=prediction, form=request.form)


@app.get("/health")
def health():
    _, _, _, error = load_artifacts()
    return ({"status": "ok"} if error is None else {"status": "not_ready", "error": error}, 200 if error is None else 503)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)

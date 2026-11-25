from pathlib import Path
import json
import warnings
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "Social_Network_Ads.csv"
CLEAN = ROOT / "data" / "processed" / "social_network_ads_clean.csv"
MODEL_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TARGET = "Purchased"
FEATURES = ["Gender", "Age", "EstimatedSalary"]


def load_clean_data():
    if not RAW.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW}. Put Social_Network_Ads.csv in data/raw/.")
    df = pd.read_csv(RAW)
    required = {"User ID", "Gender", "Age", "EstimatedSalary", TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    df = df[list(required)].copy()
    df = df.drop_duplicates()
    df["Gender"] = df["Gender"].astype("string").str.strip().str.title()
    for col in ["Age", "EstimatedSalary", TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[~df[TARGET].isin([0, 1]), TARGET] = pd.NA
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    df.loc[~df["Gender"].isin(["Male", "Female"]), "Gender"] = pd.NA
    df.loc[df["Age"] <= 0, "Age"] = pd.NA
    df.loc[df["EstimatedSalary"] < 0, "EstimatedSalary"] = pd.NA
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN, index=False)
    return df


def make_preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric, ["Age", "EstimatedSalary"]),
                              ("categorical", categorical, ["Gender"])])


def make_pipeline(estimator):
    return Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])


def run_eda(df):
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x=TARGET)
    plt.title("Purchase outcomes (0 = no, 1 = yes)")
    plt.tight_layout(); plt.savefig(FIGURES / "purchase_counts.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="Age", y="EstimatedSalary", hue=TARGET, alpha=.7)
    plt.title("Age and estimated salary by purchase outcome")
    plt.tight_layout(); plt.savefig(FIGURES / "age_salary_scatter.png", dpi=150); plt.close()
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="Age", hue=TARGET, bins=20, element="step", stat="density", common_norm=False)
    plt.title("Age distribution by purchase outcome")
    plt.tight_layout(); plt.savefig(FIGURES / "age_distribution.png", dpi=150); plt.close()
    data = {
        "rows": int(len(df)), "columns": list(df.columns),
        "missing_values": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "class_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().to_dict().items()},
        "numeric_summary": df[["Age", "EstimatedSalary"]].describe().round(2).to_dict(),
        "gender_counts": {str(k): int(v) for k, v in df["Gender"].fillna("Missing").value_counts().to_dict().items()},
        "sample": df.drop(columns=["User ID"]).head(250).fillna("").to_dict(orient="records")
    }
    (REPORTS / "eda_summary.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def train_and_evaluate(df):
    X, y = df[FEATURES], df[TARGET]
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Training needs at least two examples in each Purchased class.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows, fitted = [], {}
    for name, estimator in candidates.items():
        pipe = make_pipeline(estimator)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        rows.append({"model": name, "cv_f1_mean": float(scores.mean()), "cv_f1_std": float(scores.std()),
                     "accuracy": float(accuracy_score(y_test, pred)),
                     "precision": float(precision_score(y_test, pred, zero_division=0)),
                     "recall": float(recall_score(y_test, pred, zero_division=0)),
                     "f1": float(f1_score(y_test, pred, zero_division=0)),
                     "roc_auc": float(roc_auc_score(y_test, prob))})
        fitted[name] = (pipe, pred, prob)
    metrics = pd.DataFrame(rows).sort_values(["cv_f1_mean", "f1"], ascending=False).reset_index(drop=True)
    best_name = str(metrics.iloc[0]["model"])
    best, pred, prob = fitted[best_name]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_DIR / "best_model.joblib")
    (MODEL_DIR / "model_info.json").write_text(json.dumps({"best_model": best_name, "features": FEATURES,
        "target": TARGET, "test_size": 0.25, "random_state": 42,
        "class_labels": {"0": "Not purchased", "1": "Purchased"}}, indent=2), encoding="utf-8")
    metrics.to_csv(REPORTS / "model_comparison.csv", index=False)
    (REPORTS / "classification_report.txt").write_text(classification_report(y_test, pred, target_names=["Not purchased", "Purchased"]), encoding="utf-8")
    plt.figure(figsize=(5, 4)); sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix: {best_name}")
    plt.tight_layout(); plt.savefig(FIGURES / "confusion_matrix.png", dpi=150); plt.close()
    return metrics, best_name


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    df = load_clean_data()
    run_eda(df)
    metrics, best = train_and_evaluate(df)
    print(f"Cleaned rows: {len(df)}")
    print(f"Best model by 5-fold training CV F1: {best}")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Reports: {REPORTS}")

if __name__ == "__main__":
    main()

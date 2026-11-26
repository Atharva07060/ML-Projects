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
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "housing.csv"
CLEAN = ROOT / "data" / "processed" / "housing_clean.csv"
MODEL_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TARGET = "median_house_value"
NUMERIC = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms",
           "population", "households", "median_income"]
CATEGORICAL = ["ocean_proximity"]
FEATURES = NUMERIC + CATEGORICAL


def load_clean_data():
    if not RAW.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW}. Place housing.csv in data/raw/.")
    df = pd.read_csv(RAW)
    required = set(FEATURES + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    df = df[FEATURES + [TARGET]].copy().drop_duplicates()
    for col in NUMERIC + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[CATEGORICAL[0]] = df[CATEGORICAL[0]].astype("string").str.strip()
    df.loc[df[CATEGORICAL[0]].eq(""), CATEGORICAL[0]] = pd.NA
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0]
    df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    if df.empty:
        raise ValueError("No usable rows remain after validating the target column.")
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN, index=False)
    return df


def make_preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])


def make_pipeline(estimator):
    return Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])


def run_eda(df):
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    plt.figure(figsize=(8, 5))
    sample = df.sample(min(len(df), 5000), random_state=42)
    points = plt.scatter(sample["longitude"], sample["latitude"], c=sample[TARGET], s=8, alpha=.65, cmap="viridis")
    plt.colorbar(points, label="Median house value ($)")
    plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.title("California house values by location")
    plt.tight_layout(); plt.savefig(FIGURES / "location_values.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 5))
    sns.histplot(df[TARGET] / 1000, bins=45, color="#315efb")
    plt.axvline(500.001, color="#e45c43", linestyle="--", label="Dataset value cap")
    plt.xlabel("Median house value ($ thousands)"); plt.ylabel("Number of districts")
    plt.title("Distribution of median house values"); plt.legend()
    plt.tight_layout(); plt.savefig(FIGURES / "value_distribution.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=sample, x="median_income", y=TARGET, hue="ocean_proximity", alpha=.6, s=18)
    plt.xlabel("Median income (dataset units)"); plt.ylabel("Median house value ($)")
    plt.title("Income and house value by ocean proximity")
    plt.tight_layout(); plt.savefig(FIGURES / "income_vs_value.png", dpi=150); plt.close()
    clean = df.copy()
    overview = {
        "rows": int(len(clean)), "columns": [str(c) for c in clean.columns],
        "missing_values": {str(k): int(v) for k, v in clean.isna().sum().to_dict().items()},
        "duplicate_rows_removed": int(pd.read_csv(RAW).duplicated().sum()),
        "value_cap": 500001,
        "cap_count": int((clean[TARGET] >= 500001).sum()),
        "target_summary": {str(k): float(v) for k, v in clean[TARGET].describe().round(2).to_dict().items()},
        "ocean_counts": {str(k): int(v) for k, v in clean[CATEGORICAL[0]].value_counts(dropna=False).to_dict().items()},
        "sample": clean.sample(min(len(clean), 1800), random_state=42).where(pd.notna(clean), None).to_dict(orient="records"),
        "prediction_defaults": {**{col: float(clean[col].median()) for col in NUMERIC},
                                "ocean_proximity": str(clean[CATEGORICAL[0]].mode().iloc[0])}
    }
    (REPORTS / "eda_summary.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
    return overview


def train_and_evaluate(df):
    X, y = df[FEATURES], df[TARGET]
    missing_bedrooms = X["total_bedrooms"].isna().astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42, stratify=missing_bedrooms)
    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=18, min_samples_leaf=3, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=120, max_depth=22, min_samples_leaf=2,
                                                random_state=42, n_jobs=1),
        "Histogram Gradient Boosting": HistGradientBoostingRegressor(max_iter=150, learning_rate=.08,
                                                                        max_leaf_nodes=25, l2_regularization=1.0,
                                                                        random_state=42)
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_splits = list(cv.split(X_train, X_train["total_bedrooms"].isna().astype(int)))
    rows, fitted = [], {}
    for name, estimator in candidates.items():
        pipe = make_pipeline(estimator)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv_splits,
                                    scoring="neg_root_mean_squared_error", n_jobs=1)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rmse = float(mean_squared_error(y_test, pred) ** .5)
        rows.append({"model": name, "cv_rmse_mean": float(-cv_scores.mean()),
                     "cv_rmse_std": float(cv_scores.std()), "mae": float(mean_absolute_error(y_test, pred)),
                     "rmse": rmse, "r2": float(r2_score(y_test, pred))})
        fitted[name] = (pipe, pred)
    metrics = pd.DataFrame(rows).sort_values(["cv_rmse_mean", "rmse"]).reset_index(drop=True)
    best_name = str(metrics.iloc[0]["model"])
    best, pred = fitted[best_name]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_DIR / "best_model.joblib")
    (MODEL_DIR / "model_info.json").write_text(json.dumps({"best_model": best_name, "features": FEATURES,
        "target": TARGET, "test_size": .20, "random_state": 42, "selection_metric": "5-fold CV RMSE"}, indent=2), encoding="utf-8")
    metrics.to_csv(REPORTS / "model_comparison.csv", index=False)
    (REPORTS / "regression_report.txt").write_text(
        f"Selected model: {best_name}\nTest MAE: ${mean_absolute_error(y_test, pred):,.0f}\n"
        f"Test RMSE: ${mean_squared_error(y_test, pred) ** .5:,.0f}\nTest R-squared: {r2_score(y_test, pred):.3f}\n",
        encoding="utf-8")
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_test, y=pred, alpha=.35, s=18, color="#315efb")
    low, high = float(y_test.min()), float(y_test.max())
    plt.plot([low, high], [low, high], "--", color="#e45c43")
    plt.xlabel("Actual value ($)"); plt.ylabel("Predicted value ($)")
    plt.title(f"Actual vs predicted — {best_name}")
    plt.tight_layout(); plt.savefig(FIGURES / "actual_vs_predicted.png", dpi=150); plt.close()
    return metrics, best_name


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    df = load_clean_data()
    run_eda(df)
    metrics, best = train_and_evaluate(df)
    print(f"Cleaned records: {len(df):,}")
    print(f"Best model by 5-fold CV RMSE: {best}")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print(f"Reports: {REPORTS}")

if __name__ == "__main__":
    main()


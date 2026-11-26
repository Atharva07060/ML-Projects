# California Housing Price Prediction

A machine learning project for exploring California district-level housing data, comparing regression models, and estimating median house values in a Flask dashboard.

## Project overview

The training script cleans the dataset, creates exploratory charts, evaluates several regression models, and saves a fitted model pipeline. The dashboard includes geographic and distribution plots, a model comparison, and a form for estimating a district's median house value.

## Dataset

The project uses `housing.csv`, a California housing dataset with 20,640 district records and 10 columns. The target is `median_house_value` (USD). The model uses eight numeric predictors and `ocean_proximity`; longitude and latitude are included to represent location.

The source data has 207 missing values in `total_bedrooms` and no exact duplicate rows. Missing numeric values are imputed with the training-set median inside the model pipeline, and `ocean_proximity` is one-hot encoded. The target has a maximum value of $500,001; 965 records are at that cap, so high-value estimates are constrained by the source data.

## Models and evaluation

The project compares Linear Regression, Decision Tree, Random Forest, and Histogram Gradient Boosting. It uses a random 80/20 train/test split and five-fold cross-validation on the training portion. The split and folds preserve the distribution of records with missing bedroom counts so the imputation path is represented during training and evaluation. Models are ranked by mean cross-validation RMSE; the test report also includes MAE, RMSE, and R-squared.

## Project structure

```text
California_Housing_Price_Prediction/
├── app.py
├── train.py
├── requirements.txt
├── data/
│   ├── raw/housing.csv
│   └── processed/housing_clean.csv
├── models/
│   ├── best_model.joblib
│   └── model_info.json
├── reports/
│   ├── eda_summary.json
│   ├── model_comparison.csv
│   ├── regression_report.txt
│   └── figures/
└── templates/index.html
```

## Setup on Windows

Open PowerShell in this project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
python app.py
```

If PowerShell blocks activation, allow it for the current terminal session and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Open [http://127.0.0.1:5001](http://127.0.0.1:5001) after Flask starts. The `/health` endpoint reports whether the model and reports are ready. Stop the server with `Ctrl+C`.

## Notes

- The target is district-level median house value, not an individual home's sale price.
- The $500,001 ceiling in the source limits evaluation and predictions for the highest-value areas.
- Results are a learning exercise and should not be used as an appraisal.
- Interactive Plotly charts load from a public CDN and need an internet connection. Flask predictions run locally.

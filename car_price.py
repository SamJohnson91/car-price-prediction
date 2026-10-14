"""
Car price prediction - guided first project.

Run:    python car_price.py
Needs:  pandas, numpy, scikit-learn, matplotlib
Data:   data/used_cars.csv  (unzipped from archive.zip)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------
# STEP 1: Load the data
# One row = one car. 'price' is what we want to predict (the target).
# ---------------------------------------------------------------
df = pd.read_csv(HERE / "data" / "used_cars.csv")
print(f"Loaded {df.shape[0]} cars, {df.shape[1]} columns")

# ---------------------------------------------------------------
# STEP 2: Clean the data
# Models need numbers. Our file has text like "$10,300" and "51,000 mi."
# ---------------------------------------------------------------
# "$10,300" -> 10300.0   (remove $ and commas)
df["price"] = df["price"].str.replace(r"[$,]", "", regex=True).astype(float)

# "51,000 mi." -> 51000.0   (keep only the digits)
df["mileage"] = df["milage"].str.replace(r"[^\d]", "", regex=True).astype(float)

# Year is less useful than age: older car -> usually cheaper.
df["car_age"] = 2024 - df["model_year"]

# Yes/no columns -> 1/0
df["has_accident"] = (df["accident"] == "At least 1 accident or damage reported").astype(int)
df["clean_title"] = (df["clean_title"] == "Yes").astype(int)

# Fuel type has junk values ("-" and "not supported"). Call them "Unknown".
df["fuel_type"] = (
    df["fuel_type"].replace({"–": np.nan, "not supported": np.nan}).fillna("Unknown")
)

# The transmission column has 62 different texts. We only keep: manual or not.
trans = df["transmission"].str.upper()
df["manual"] = (trans.str.contains("M/T") | trans.str.contains("MANUAL")).astype(int)

# The engine column is a long text. Pull out two numbers with regex.
#   "300.0HP 3.7L V6 ..." -> horsepower = 300.0, engine_liters = 3.7
df["horsepower"] = df["engine"].str.extract(r"(\d+\.?\d*)HP")[0].astype(float)
df["engine_liters"] = df["engine"].str.extract(r"(\d\.\d)\s*(?:L|Liter)")[0].astype(float)

# Outliers: a few very expensive cars (up to $2.9 million) would confuse the model.
# We remove the most expensive 1%.
cap = df["price"].quantile(0.99)
df = df[df["price"] <= cap].copy()
print(f"Removed cars above ${cap:,.0f}. {len(df)} cars left.")

# ---------------------------------------------------------------
# STEP 3: Choose features (X) and target (y)
# We skip 'model', 'ext_col', 'int_col': too many unique values / too noisy.
# ---------------------------------------------------------------
numeric = ["mileage", "car_age", "horsepower", "engine_liters",
           "has_accident", "clean_title", "manual"]
categorical = ["brand", "fuel_type"]

X = df[numeric + categorical]
y = df["price"]

# ---------------------------------------------------------------
# STEP 4: Split into train and test
# The model learns from 80% of the cars. We test it on the 20% it has never seen.
# random_state makes the split the same every run.
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)} cars, Test: {len(X_test)} cars")

# ---------------------------------------------------------------
# STEP 5: Prepare the features (inside a pipeline)
#  - numbers: fill missing values with the median (the middle value)
#  - text categories (brand, fuel): one-hot encoding -> one 0/1 column per value
# The pipeline learns the medians from TRAIN only, so the test set stays "unseen".
# ---------------------------------------------------------------
prep = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
])


def make_model(regressor, log_target):
    """Prep + model. If log_target: the model learns log(price), answers come back in dollars."""
    pipe = Pipeline([("prep", prep), ("model", regressor)])
    if log_target:
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


# ---------------------------------------------------------------
# STEP 6: Train the models
# 1) Baseline: always guess the average price. Any real model must beat this.
# 2) Linear regression on raw price.
# 3) Linear regression on log(price): prices are skewed, log makes them more even.
# 4) Random forest (many decision trees voting) on log(price).
# ---------------------------------------------------------------
models = {
    "Baseline (average price)": make_model(DummyRegressor(strategy="mean"), False),
    "Linear regression": make_model(LinearRegression(), False),
    "Linear regression (log price)": make_model(LinearRegression(), True),
    "Random forest (log price)": make_model(
        RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1), True
    ),
}

# ---------------------------------------------------------------
# STEP 7: Evaluate on the test set
#  MAE = average error in dollars (lower is better)
#  R2  = share of price differences the model explains (1.0 = perfect, 0 = same as baseline)
# ---------------------------------------------------------------
rows, preds = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    preds[name] = p
    rows.append({
        "model": name,
        "MAE ($)": round(mean_absolute_error(y_test, p)),
        "R2": round(r2_score(y_test, p), 3),
    })

results = pd.DataFrame(rows)
print("\n" + results.to_string(index=False))
results.to_csv(OUT / "metrics.csv", index=False)

# ---------------------------------------------------------------
# STEP 8: Look at the results
# Plot 1: actual vs predicted price. Points on the diagonal line = perfect.
# Plot 2: which features the random forest used most.
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
for ax, name in zip(axes, ["Linear regression (log price)", "Random forest (log price)"]):
    ax.scatter(y_test, preds[name], s=8, alpha=0.4)
    top = max(y_test.max(), preds[name].max())
    ax.plot([0, top], [0, top], color="red", lw=1)
    ax.set_title(name)
    ax.set_xlabel("Actual price ($)")
axes[0].set_ylabel("Predicted price ($)")
plt.tight_layout()
plt.savefig(OUT / "actual_vs_predicted.png", dpi=120)
plt.close()

forest = models["Random forest (log price)"].regressor_
names = forest.named_steps["prep"].get_feature_names_out()
importance = pd.Series(forest.named_steps["model"].feature_importances_, index=names)
importance = importance.sort_values().tail(10)
importance.plot.barh(figsize=(7, 5), title="Top 10 features (random forest)")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=120)
plt.close()

print("\nTop features:")
print(importance.sort_values(ascending=False).round(3).to_string())
print(f"\nSaved plots and metrics in: {OUT}")

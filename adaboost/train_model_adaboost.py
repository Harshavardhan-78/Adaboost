import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =========================
# Create models folder
# =========================

os.makedirs("models", exist_ok=True)

# =========================
# Load Dataset
# =========================

df = pd.read_csv("data/Cardetailsv3.csv")

# =========================
# Handle Missing Values
# =========================

# Mileage
df['mileage'] = df['mileage'].str.extract(
    r'(\d+\.?\d*)'
).astype(float)

df['mileage'] = df['mileage'].fillna(
    df['mileage'].median()
)

# Engine
df['engine'] = df['engine'].str.extract(
    r'(\d+)'
).astype(float)

df['engine'] = df['engine'].fillna(
    df['engine'].median()
)

# Max Power
df['max_power'] = df['max_power'].str.extract(
    r'(\d+\.?\d*)'
).astype(float)

df['max_power'] = df['max_power'].fillna(
    df['max_power'].median()
)

# Torque
df['torque'] = df['torque'].str.extract(
    r'(\d+)'
).astype(float)

df['torque'] = df['torque'].fillna(
    df['torque'].median()
)

# Seats
df['seats'] = df['seats'].fillna(
    df['seats'].mode()[0]
)

# =========================
# Features & Target
# =========================

x = df.drop(
    ['selling_price', 'name'],
    axis=1
)

y = df['selling_price']

# =========================
# One Hot Encoding
# =========================

x = pd.get_dummies(
    x,
    columns=[
        'fuel',
        'seller_type',
        'transmission',
        'owner'
    ],
    drop_first=True
)

# =========================
# Save Feature Columns
# =========================

model_columns = x.columns

pickle.dump(
    model_columns,
    open("models/adaboost_model_columns.pkl", "wb")
)

# =========================
# Train Test Split
# =========================

xtr, xte, ytr, yte = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# AdaBoost Regressor
# =========================

base_estimator = DecisionTreeRegressor(
    max_depth=4
)

ada = AdaBoostRegressor(
    estimator=base_estimator,
    random_state=42
)

# =========================
# Hyperparameter Grid
# =========================

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'loss': ['linear', 'square', 'exponential']
}

# =========================
# Grid Search
# =========================

grid_search = GridSearchCV(
    estimator=ada,
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)

# =========================
# Train Model
# =========================

grid_search.fit(xtr, ytr)

# =========================
# Best Model
# =========================

best_model = grid_search.best_estimator_

# =========================
# Predictions
# =========================

ypred = best_model.predict(xte)

# =========================
# Metrics
# =========================

r2 = r2_score(yte, ypred)

mae = mean_absolute_error(yte, ypred)

rmse = np.sqrt(
    mean_squared_error(yte, ypred)
)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nR2 Score:")
print(r2)

print("\nMAE:")
print(mae)

print("\nRMSE:")
print(rmse)

# =========================
# Save Metrics
# =========================

metrics = {
    "R2": r2,
    "MAE": mae,
    "RMSE": rmse,
    "best_params": grid_search.best_params_
}

pickle.dump(
    metrics,
    open("models/adaboost_metrics.pkl", "wb")
)

# =========================
# Save Model
# =========================

pickle.dump(
    best_model,
    open("models/adaboost_regressor.pkl", "wb")
)

print("\nAdaBoost Regressor Saved Successfully!")

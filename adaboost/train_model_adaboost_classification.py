import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
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
# Data Cleaning
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
# Create Classification Target
# =========================

df['price_category'] = pd.cut(
    df['selling_price'],
    bins=[0, 500000, 1000000, 10000000],
    labels=['Low', 'Medium', 'High']
)

# =========================
# Features and Target
# =========================

x = df.drop(
    ['selling_price', 'price_category', 'name'],
    axis=1
)

y = df['price_category']

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
    open("models/adaboost_model_columns_classifier.pkl", "wb")
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
# AdaBoost Classifier
# =========================

base_estimator = DecisionTreeClassifier(
    max_depth=2
)

ada = AdaBoostClassifier(
    estimator=base_estimator,
    random_state=42
)

# =========================
# Hyperparameter Grid
# =========================

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'algorithm': ['SAMME']
}

# =========================
# GridSearchCV
# =========================

grid_search = GridSearchCV(
    estimator=ada,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
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
# Prediction
# =========================

ypred = best_model.predict(xte)

# =========================
# Evaluation
# =========================

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nAccuracy Score:")
print(accuracy_score(yte, ypred))

print("\nClassification Report:")
print(classification_report(yte, ypred))

print("\nConfusion Matrix:")
print(confusion_matrix(yte, ypred))

# =========================
# Save Model
# =========================

pickle.dump(
    best_model,
    open("models/adaboost_classifier.pkl", "wb")
)

print("\nAdaBoost Classifier Saved Successfully!")

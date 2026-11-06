import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load dataset
df = pd.read_csv(
    "data/processed/future_accident_dataset.csv",
    low_memory=False
)

df["date"] = pd.to_datetime(df["date"])

# Features
numeric_features = [
    "hour", "minute", "day_of_week", "month", "day",
    "is_weekend", "is_peak_hour", "temperature",
    "rainfall", "visibility", "public_holiday",
    "event_flag", "roadwork_flag", "signal_timing",
    "road_capacity"
]

categorical_features = [
    "road_id", "weather_condition"
]

features = numeric_features + categorical_features

target = "future_accident_occurred"

X = df[features]
y = df[target]

# Same chronological split
unique_dates = sorted(df["date"].unique())

test_start = int(len(unique_dates) * 0.85)

test_dates = unique_dates[test_start:]

test_mask = df["date"].isin(test_dates)

X_test = X[test_mask]
y_test = y[test_mask]

# Load model and threshold
model = joblib.load(
    "models/future_accident_model.pkl"
)

threshold = joblib.load(
    "models/future_accident_threshold.pkl"
)

# Predict probabilities
probabilities = model.predict_proba(X_test)[:, 1]

print("\n========== MODEL EVALUATION ==========")

print("Actual accident rate:",
      round(y_test.mean() * 100, 3), "%")

print("Baseline PR-AUC:",
      round(y_test.mean(), 5))

print("Model PR-AUC:",
      round(average_precision_score(y_test, probabilities), 5))

print("ROC-AUC:",
      round(roc_auc_score(y_test, probabilities), 5))

print("\n========== THRESHOLD COMPARISON ==========")

for t in [0.20, 0.30, 0.40, 0.44, 0.50, 0.60, 0.70]:

    predictions = (probabilities >= t).astype(int)

    print(f"\nThreshold: {t}")

    print("Predicted accidents:", predictions.sum())

    print("Precision:",
          round(precision_score(y_test, predictions, zero_division=0), 4))

    print("Recall:",
          round(recall_score(y_test, predictions, zero_division=0), 4))

    print("F1:",
          round(f1_score(y_test, predictions, zero_division=0), 4))

print("\n========== CONFUSION MATRIX ==========")

predictions = (probabilities >= threshold).astype(int)

print(confusion_matrix(y_test, predictions))

print("\nEvaluation completed!")
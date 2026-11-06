import pandas as pd
import numpy as np
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
accuracy_score,
classification_report,
confusion_matrix
)

# Load dataset

df = pd.read_csv(
"data/processed/flowcast_features.csv",
low_memory=False
)

print("Dataset loaded successfully!")

# Remove missing target labels

df = df.dropna(subset=["congestion_level"]).copy()

df["date"] = pd.to_datetime(df["date"])

# Define input features

numeric_features = [
"hour",
"minute",
"day_of_week",
"month",
"day",
"is_weekend",
"is_peak_hour",
"temperature",
"rainfall",
"visibility",
"public_holiday",
"event_flag",
"roadwork_flag",
"signal_timing",
"road_capacity"
]

categorical_features = ["road_id"]

target = "congestion_level"

X = df[numeric_features + categorical_features]
y = df[target]

# Chronological split

dates = sorted(df["date"].unique())

split_index = int(len(dates) * 0.80)
split_date = dates[split_index]

train_mask = df["date"] < split_date
test_mask = df["date"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("\n========== DATA SPLIT ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training ends:", df.loc[train_mask, "date"].max())
print("Testing starts:", df.loc[test_mask, "date"].min())

# Preprocessing

numeric_transformer = Pipeline(steps=[
("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
("imputer", SimpleImputer(strategy="most_frequent")),
("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
("num", numeric_transformer, numeric_features),
("cat", categorical_transformer, categorical_features)
])

# Random Forest classifier

model = Pipeline(steps=[
("preprocessor", preprocessor),
("classifier", RandomForestClassifier(
n_estimators=100,
max_depth=20,
min_samples_split=5,
class_weight="balanced",
random_state=42,
n_jobs=-1
))
])

# Train model

print("\nTraining congestion classification model...")
model.fit(X_train, y_train)

# Predictions

y_pred = model.predict(X_test)

# Evaluation

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
y_test,
y_pred,
zero_division=0
))

print("\nConfusion Matrix:")
labels = ["Free-flow", "Moderate", "Heavy", "Severe"]

print(pd.DataFrame(
confusion_matrix(y_test, y_pred, labels=labels),
index=labels,
columns=labels
))

# Save model

os.makedirs("models", exist_ok=True)

joblib.dump(
model,
"models/congestion_model.pkl"
)

print("\nCongestion model saved successfully!")

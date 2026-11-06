
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve
)


# 1. Load dataset

df = pd.read_csv(
    "data/processed/flowcast_features.csv",
    low_memory=False
)

print("Dataset loaded successfully!")

df["accident_occurred"] = (
    df["accident_count"] > 0
).astype(int)

df["date"] = pd.to_datetime(df["date"])


# 2. Select features

numeric_features = [
    "hour", "minute", "day_of_week", "month", "day",
    "is_weekend", "is_peak_hour", "temperature",
    "rainfall", "visibility", "public_holiday",
    "event_flag", "roadwork_flag", "signal_timing",
    "road_capacity"
]

categorical_features = [
    "road_id",
    "weather_condition"
]

X = df[numeric_features + categorical_features]
y = df["accident_occurred"]


# 3. Chronological split

dates = sorted(df["date"].unique())

test_split_index = int(len(dates) * 0.80)
test_start_date = dates[test_split_index]

training_dates = dates[:test_split_index]

validation_split_index = int(len(training_dates) * 0.80)
validation_start_date = training_dates[validation_split_index]

train_mask = df["date"] < validation_start_date

validation_mask = (
    (df["date"] >= validation_start_date) &
    (df["date"] < test_start_date)
)

test_mask = df["date"] >= test_start_date

X_train = X.loc[train_mask]
X_validation = X.loc[validation_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_validation = y.loc[validation_mask]
y_test = y.loc[test_mask]


print("\n========== DATA SPLIT ==========")

print("Training records:", len(X_train))
print("Validation records:", len(X_validation))
print("Testing records:", len(X_test))

print("Training accidents:", int(y_train.sum()))
print("Validation accidents:", int(y_validation.sum()))
print("Testing accidents:", int(y_test.sum()))


# 4. Preprocessing

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


# 5. Random Forest with class weights

model = Pipeline(steps=[

    ("preprocessor", preprocessor),

    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))

])


# 6. Train model

print("\nTraining Class-Weighted Accident Model...")

model.fit(X_train, y_train)


# 7. Tune threshold using validation data

print("\nFinding best classification threshold...")

validation_prob = model.predict_proba(X_validation)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_validation,
    validation_prob
)

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

best_index = np.argmax(f1_scores)

best_threshold = thresholds[best_index]

print("Best threshold:", round(best_threshold, 4))
print("Validation F1-score:", round(f1_scores[best_index], 4))


# 8. Test predictions

print("\nEvaluating on untouched test data...")

y_prob = model.predict_proba(X_test)[:, 1]

y_pred = (y_prob >= best_threshold).astype(int)


# 9. Model evaluation

print("\n========== MODEL PERFORMANCE ==========")

print(
    f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%"
)

print(
    f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}"
)

print(
    f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}"
)

print(
    f"F1-score: {f1_score(y_test, y_pred, zero_division=0):.4f}"
)

print(
    f"PR-AUC: {average_precision_score(y_test, y_prob):.4f}"
)


# 10. Baseline comparison

baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

print("\nBaseline Accuracy:")

print(
    f"{accuracy_score(y_test, baseline_pred) * 100:.2f}%"
)


# 11. Classification report

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    target_names=["No Accident", "Accident"],
    zero_division=0
))


# 12. Confusion matrix

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))


# 13. Save experiment separately

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/accident_weighted_model.pkl"
)

joblib.dump(
    best_threshold,
    "models/accident_weighted_threshold.pkl"
)

print("\nClass-Weighted Accident Model saved successfully!")

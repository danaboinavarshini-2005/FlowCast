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
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "data/processed/future_accident_dataset.csv",
    low_memory=False
)

df["date"] = pd.to_datetime(df["date"])

print("Future accident dataset loaded!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. DEFINE TARGET AND FEATURES
# ==========================================

target = "future_accident_occurred"

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

categorical_features = [
    "road_id",
    "weather_condition"
]

features = numeric_features + categorical_features

X = df[features]
y = df[target]


# ==========================================
# 3. CHRONOLOGICAL DATA SPLIT
# ==========================================

unique_dates = sorted(df["date"].unique())

train_end = int(len(unique_dates) * 0.70)
validation_end = int(len(unique_dates) * 0.85)

train_dates = unique_dates[:train_end]
validation_dates = unique_dates[train_end:validation_end]
test_dates = unique_dates[validation_end:]

train_mask = df["date"].isin(train_dates)
validation_mask = df["date"].isin(validation_dates)
test_mask = df["date"].isin(test_dates)

X_train = X[train_mask]
y_train = y[train_mask]

X_val = X[validation_mask]
y_val = y[validation_mask]

X_test = X[test_mask]
y_test = y[test_mask]

print("\n========== DATA SPLIT ==========")

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

print("Training accidents:", y_train.sum())
print("Validation accidents:", y_val.sum())
print("Testing accidents:", y_test.sum())


# ==========================================
# 4. DATA PREPROCESSING
# ==========================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])


# ==========================================
# 5. TRAIN RANDOM FOREST
# ==========================================

model = Pipeline([
    ("preprocessor", preprocessor),

    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining Future Accident Prediction Model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================
# 6. FIND BEST THRESHOLD USING VALIDATION
# ==========================================

val_probabilities = model.predict_proba(X_val)[:, 1]

best_threshold = 0.5
best_f1 = 0

for threshold in np.arange(0.05, 0.96, 0.01):

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    score = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

print("\n========== THRESHOLD TUNING ==========")

print("Best threshold:", round(best_threshold, 4))
print("Validation F1:", round(best_f1, 4))


# ==========================================
# 7. FINAL TEST EVALUATION
# ==========================================

test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\n========== TEST RESULTS ==========")

print(
    "Accuracy:",
    round(accuracy_score(y_test, test_predictions), 4)
)

print(
    "Precision:",
    round(precision_score(y_test, test_predictions, zero_division=0), 4)
)

print(
    "Recall:",
    round(recall_score(y_test, test_predictions, zero_division=0), 4)
)

print(
    "F1 Score:",
    round(f1_score(y_test, test_predictions, zero_division=0), 4)
)

print(
    "PR-AUC:",
    round(average_precision_score(y_test, test_probabilities), 4)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_predictions))


# ==========================================
# 8. SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/future_accident_model.pkl"
)

joblib.dump(
    best_threshold,
    "models/future_accident_threshold.pkl"
)

print("\nModel saved successfully!")

print("models/future_accident_model.pkl")
print("models/future_accident_threshold.pkl")
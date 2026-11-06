
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "data/processed/future_accident_v4_dataset.csv",
    low_memory=False
)

df["date"] = pd.to_datetime(df["date"])

print("Future accident V4 dataset loaded!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. DEFINE FEATURES
# ==========================================

target = "future_accident_occurred"

numeric_features = [
    # Time features
    "hour",
    "minute",
    "day_of_week",
    "month",
    "day",
    "is_weekend",
    "is_peak_hour",

    # Weather features
    "temperature",
    "rainfall",
    "visibility",

    # Calendar and road features
    "public_holiday",
    "event_flag",
    "roadwork_flag",
    "signal_timing",
    "road_capacity",

    # Current traffic features
    "traffic_volume",
    "vehicle_count",
    "avg_speed",
    "occupancy",
    "travel_time",

    # Previous interval traffic features
    "prev_traffic_volume",
    "prev_vehicle_count",
    "prev_avg_speed",
    "prev_occupancy",
    "prev_travel_time"
]

categorical_features = [
    "road_id",
    "weather_condition",
    "congestion_level",
    "prev_congestion_level"
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
# 4. HANDLE CLASS IMBALANCE
# ==========================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nScale Pos Weight:", round(scale_pos_weight, 2))


# ==========================================
# 5. PREPROCESSING
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
# 6. BUILD XGBOOST MODEL
# ==========================================

xgb_classifier = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_lambda=5,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgb_classifier)
])


# ==========================================
# 7. TRAIN MODEL
# ==========================================

print("\nTraining Future Accident Model V4...")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 8. THRESHOLD TUNING
# ==========================================

val_probabilities = model.predict_proba(X_val)[:, 1]

best_threshold = 0.5
best_f1 = 0

for threshold in np.arange(0.01, 0.96, 0.01):

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    score = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

print("\n========== THRESHOLD TUNING ==========")

print("Best threshold:", round(best_threshold, 4))
print("Validation F1:", round(best_f1, 4))


# ==========================================
# 9. TEST EVALUATION
# ==========================================

test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\n========== V4 TEST RESULTS ==========")

print("Accuracy:",
      round(accuracy_score(y_test, test_predictions), 4))

print("Precision:",
      round(precision_score(y_test, test_predictions, zero_division=0), 4))

print("Recall:",
      round(recall_score(y_test, test_predictions, zero_division=0), 4))

print("F1 Score:",
      round(f1_score(y_test, test_predictions, zero_division=0), 4))

print("PR-AUC:",
      round(average_precision_score(y_test, test_probabilities), 4))

print("ROC-AUC:",
      round(roc_auc_score(y_test, test_probabilities), 4))

print("\nClassification Report:")

print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, test_predictions))


# ==========================================
# 10. FEATURE IMPORTANCE
# ==========================================

preprocessor_fitted = model.named_steps["preprocessor"]

feature_names = preprocessor_fitted.get_feature_names_out()

importances = model.named_steps[
    "classifier"
].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print("\n========== TOP 20 FEATURES ==========")

print(
    importance_df.head(20).to_string(index=False)
)


# ==========================================
# 11. SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

os.makedirs(
    "reports/accident_analysis",
    exist_ok=True
)

joblib.dump(
    model,
    "models/future_accident_v4_model.pkl"
)

joblib.dump(
    best_threshold,
    "models/future_accident_v4_threshold.pkl"
)

importance_df.to_csv(
    "reports/accident_analysis/v4_feature_importance.csv",
    index=False
)

print("\nV4 model saved successfully!")

print("models/future_accident_v4_model.pkl")
print("models/future_accident_v4_threshold.pkl")

print("\nFuture Accident Model V4 completed!")


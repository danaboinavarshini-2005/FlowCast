import pandas as pd
import numpy as np
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset

df = pd.read_csv(
"data/processed/flowcast_features.csv",
low_memory=False
)

print("Dataset loaded successfully!")

# Remove rows with missing target

df = df.dropna(subset=["traffic_volume"]).copy()

# Convert date

df["date"] = pd.to_datetime(df["date"])

# Define features

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

target = "traffic_volume"

X = df[numeric_features + categorical_features]
y = df[target]

# Chronological train-test split

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

# Random Forest model

model = Pipeline(steps=[
("preprocessor", preprocessor),
("regressor", RandomForestRegressor(
n_estimators=100,
max_depth=20,
min_samples_split=5,
random_state=42,
n_jobs=-1
))
])

# Train

print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

# Predict

y_pred = model.predict(X_test)

# Evaluate

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

# Save model

os.makedirs("models", exist_ok=True)

joblib.dump(
model,
"models/traffic_volume_model.pkl"
)

print("\nModel saved successfully!")
print("Location: models/traffic_volume_model.pkl")

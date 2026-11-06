import pandas as pd
import os

# Load master dataset
df = pd.read_csv("data/processed/flowcast_master.csv")

print("Master dataset loaded successfully!")

# Convert datetime
df["datetime"] = pd.to_datetime(df["datetime"])

# Extract date and time features
df["hour"] = df["datetime"].dt.hour
df["minute"] = df["datetime"].dt.minute
df["day_of_week"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day

# Weekend feature
df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# Peak-hour feature
df["is_peak_hour"] = (
    df["hour"].between(7, 10) |
    df["hour"].between(16, 19)
).astype(int)

# Sort records chronologically
df = df.sort_values(
    ["datetime", "road_id"]
).reset_index(drop=True)

# Save feature-engineered dataset
os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    "data/processed/flowcast_features.csv",
    index=False
)

# Display results
print("\n========== FEATURE ENGINEERING ==========")

print("Dataset shape:", df.shape)

print("\nNew features:")
print([
    "hour",
    "minute",
    "day_of_week",
    "month",
    "day",
    "is_weekend",
    "is_peak_hour"
])

print("\nFirst five records:")
print(df.head())

print("\nFeature-engineered dataset saved successfully!")
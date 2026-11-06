import pandas as pd
import os

# Load dataset
df = pd.read_csv(
    "data/processed/flowcast_features.csv",
    low_memory=False
)

print("Dataset loaded successfully!")

# Convert datetime
df["datetime"] = pd.to_datetime(df["datetime"])

# Sort records
df = df.sort_values(
    ["road_id", "datetime"]
).reset_index(drop=True)

# Create future timestamp
df["future_datetime"] = (
    df["datetime"] + pd.Timedelta(minutes=30)
)

# Prepare future accident lookup
future_data = df[
    ["road_id", "datetime", "accident_count"]
].copy()

future_data = future_data.rename(
    columns={
        "datetime": "future_datetime",
        "accident_count": "future_accident_count"
    }
)

# Match same road at exactly 30 minutes later
result = df.merge(
    future_data,
    on=["road_id", "future_datetime"],
    how="left"
)

# Check matching records
matched = result["future_accident_count"].notna()

print("\n========== FUTURE TARGET SUMMARY ==========")

print("Original records:", len(df))
print("Matched 30-minute records:", matched.sum())
print("Unmatched records:", (~matched).sum())

# Keep only records with a valid future interval
result = result[matched].copy()

# Create future accident target
result["future_accident_occurred"] = (
    result["future_accident_count"] > 0
).astype(int)

# Remove current accident information and temporary columns
result = result.drop(
    columns=[
        "accident_count",
        "future_accident_count",
        "future_datetime"
    ]
)

# Accident statistics
print("\n========== FUTURE ACCIDENT DISTRIBUTION ==========")

print(
    result["future_accident_occurred"].value_counts()
)

positive = result["future_accident_occurred"].sum()

print("\nFuture accident records:", positive)

print(
    "Future accident occurrence rate:",
    round(
        result["future_accident_occurred"].mean() * 100,
        3
    ),
    "%"
)

# Save dataset
os.makedirs("data/processed", exist_ok=True)

output_path = (
    "data/processed/future_accident_dataset.csv"
)

result.to_csv(output_path, index=False)

print("\nDataset saved successfully!")
print("Output:", output_path)
print("Final dataset shape:", result.shape)
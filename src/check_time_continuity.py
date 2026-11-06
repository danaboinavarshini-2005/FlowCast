
import pandas as pd

# Load dataset
df = pd.read_csv(
    "data/processed/flowcast_features.csv",
    low_memory=False
)

print("Dataset loaded successfully!")

# Convert existing datetime column
df["datetime"] = pd.to_datetime(df["datetime"])

# Sort by road and datetime
df = df.sort_values(
    ["road_id", "datetime"]
).reset_index(drop=True)

print("\n========== TIME CONTINUITY CHECK ==========")

# Check duplicate road-time records
duplicates = df.duplicated(
    subset=["road_id", "datetime"]
).sum()

print("Duplicate road-time records:", duplicates)

# Calculate time difference for each road
df["time_difference"] = (
    df.groupby("road_id")["datetime"].diff()
)

# Display time difference distribution
print("\nTime difference distribution:")

print(
    df["time_difference"]
    .value_counts()
    .head(10)
)

# Expected interval
expected_interval = pd.Timedelta(minutes=30)

# Find unexpected gaps
gaps = df[
    (df["time_difference"] != expected_interval)
    & df["time_difference"].notna()
]

print("\nUnexpected time gaps:", len(gaps))

# Display gap details
print("\nFirst 15 unexpected gaps:")

if len(gaps) > 0:
    print(
        gaps[
            ["road_id", "datetime", "time_difference"]
        ].head(15).to_string(index=False)
    )
else:
    print("No unexpected gaps found!")

# Summary
print("\n========== FINAL SUMMARY ==========")

print("Total records:", len(df))
print("Unique roads:", df["road_id"].nunique())
print("Duplicate records:", duplicates)
print("Unexpected gaps:", len(gaps))

print("\nTime continuity check completed!")


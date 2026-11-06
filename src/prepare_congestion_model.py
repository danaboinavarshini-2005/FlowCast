import pandas as pd

# Load dataset
df = pd.read_csv(
    "data/processed/flowcast_features.csv",
    low_memory=False
)

print("Dataset loaded successfully!")

# Check congestion distribution
print("\n========== CONGESTION DISTRIBUTION ==========")

print(df["congestion_level"].value_counts(dropna=False))

# Remove missing congestion labels
df = df.dropna(subset=["congestion_level"])

print("\n========== CLEANED CONGESTION DATA ==========")

print("Total records:", len(df))

print("\nClass distribution:")
print(df["congestion_level"].value_counts())

print("\nClass percentages:")
print(
    df["congestion_level"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nNumber of roads:", df["road_id"].nunique())

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())
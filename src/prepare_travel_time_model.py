import pandas as pd

# Load dataset

df = pd.read_csv(
"data/processed/flowcast_features.csv",
low_memory=False
)

print("Dataset loaded successfully!")

# Check target information

print("\n========== TRAVEL TIME ANALYSIS ==========")

print("\nTarget statistics:")
print(df["travel_time"].describe())

print("\nMissing values:")
print(df["travel_time"].isnull().sum())

print("\nZero or negative travel times:")
print((df["travel_time"] <= 0).sum())

print("\nNumber of roads:")
print(df["road_id"].nunique())

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())

print("\nFirst five travel-time records:")
print(df[["road_id", "date", "time_x", "travel_time"]].head())

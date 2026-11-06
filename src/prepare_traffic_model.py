import pandas as pd

# Load feature-engineered dataset
df = pd.read_csv("data/processed/flowcast_features.csv")

print("Dataset loaded successfully!")

# Remove records where target is missing
df = df.dropna(subset=["traffic_volume"])

# Select input features
features = [
    "road_id",
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

target = "traffic_volume"

# Create model dataset
model_data = df[features + [target]].copy()

# Check missing values
print("\n========== MODEL DATA CHECK ==========")

print("Dataset shape:", model_data.shape)

print("\nMissing values:")
print(model_data.isnull().sum())

print("\nTarget statistics:")
print(model_data[target].describe())

print("\nNumber of roads:")
print(model_data["road_id"].nunique())

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())
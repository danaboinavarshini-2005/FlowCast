import pandas as pd
import os

# Load cleaned datasets
traffic = pd.read_csv("data/processed/traffic_cleaned.csv")
weather = pd.read_csv("data/processed/weather_cleaned.csv")
calendar = pd.read_csv("data/processed/calendar_cleaned.csv")

print("Loaded cleaned datasets successfully.")

# Convert date and time into datetime values
traffic["datetime"] = pd.to_datetime(
    traffic["date"] + " " + traffic["time"]
)

weather["datetime"] = pd.to_datetime(
    weather["date"] + " " + weather["time"]
)

# Sort datasets
traffic = traffic.sort_values("datetime")
weather = weather.sort_values("datetime")

# Merge traffic with weather
merged = pd.merge_asof(
    traffic,
    weather,
    on="datetime",
    by=None,
    left_by="weather_station_id",
    right_by="station_id",
    direction="backward",
    tolerance=pd.Timedelta("1 hour")
)

# Remove unnecessary weather columns
merged = merged.drop(
    columns=["station_id", "date_y", "time_y"],
    errors="ignore"
)

# Rename traffic date column
merged = merged.rename(columns={"date_x": "date"})

# Merge with calendar
merged = pd.merge(
    merged,
    calendar,
    on="date",
    how="left"
)

# Sort final dataset
merged = merged.sort_values(
    ["road_id", "datetime"]
)

# Save master dataset
os.makedirs("data/processed", exist_ok=True)

merged.to_csv(
    "data/processed/flowcast_master.csv",
    index=False
)

# Display results
print("\n========== MASTER DATASET ==========")

print("Shape:", merged.shape)

print("\nColumns:")
print(merged.columns.tolist())

print("\nMissing values:")
print(merged.isnull().sum())

print("\nFirst five records:")
print(merged.head())

print("\nMaster dataset saved successfully!")
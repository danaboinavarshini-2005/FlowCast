import pandas as pd

df = pd.read_csv(
    "data/processed/future_accident_v4_dataset.csv",
    low_memory=False
)

print("\n========== DASHBOARD INPUT VALUES ==========")

print("\nAvailable Roads:")
print(sorted(df["road_id"].dropna().unique()))

print("\nWeather Conditions:")
print(sorted(df["weather_condition"].dropna().unique()))

print("\nCongestion Levels:")
print(sorted(df["congestion_level"].dropna().unique()))

print("\nPrevious Congestion Levels:")
print(sorted(df["prev_congestion_level"].dropna().unique()))

print("\nNumeric Feature Defaults:")
columns = [
    "temperature", "rainfall", "visibility",
    "signal_timing", "road_capacity",
    "traffic_volume", "vehicle_count",
    "avg_speed", "occupancy", "travel_time",
    "prev_traffic_volume", "prev_vehicle_count",
    "prev_avg_speed", "prev_occupancy",
    "prev_travel_time"
]

print(df[columns].median().round(2))
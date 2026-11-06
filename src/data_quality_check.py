import pandas as pd

# Load datasets
traffic = pd.read_csv("data/raw/traffic_sensor_log.csv")
weather = pd.read_csv("data/raw/weather_observations.csv")
calendar = pd.read_csv("data/raw/calendar_events.csv")


# 1. Traffic dataset checks
print("\n========== TRAFFIC DATASET ==========")

print("\nDate range:")
print(traffic["date"].min(), "to", traffic["date"].max())

print("\nTime values:")
print(traffic["time"].unique()[:15])

print("\nCongestion level counts:")
print(traffic["congestion_level"].value_counts(dropna=False))

print("\nNumerical column statistics:")
print(traffic[
    ["traffic_volume", "vehicle_count", "avg_speed",
     "occupancy", "travel_time", "accident_count", "road_capacity"]
].describe())

print("\nDuplicate samples:")
print(traffic[traffic.duplicated(keep=False)].head(10))

print("\nDuplicate timestamps by road:")
print(
    traffic.duplicated(
        subset=["road_id", "date", "time"]
    ).sum()
)


# 2. Weather dataset checks
print("\n========== WEATHER DATASET ==========")

print("\nDate range:")
print(weather["date"].min(), "to", weather["date"].max())

print("\nWeather condition counts:")
print(weather["weather_condition"].value_counts())

print("\nWeather numerical statistics:")
print(weather[
    ["temperature", "rainfall", "visibility"]
].describe())


# 3. Calendar dataset checks
print("\n========== CALENDAR DATASET ==========")

print("\nDate range:")
print(calendar["date"].min(), "to", calendar["date"].max())

print("\nPublic holidays:")
print(calendar["public_holiday"].value_counts())

print("\nEvents:")
print(calendar["event_flag"].value_counts())

print("\nRoadwork:")
print(calendar["roadwork_flag"].value_counts())


print("\n========== QUALITY CHECK COMPLETED ==========")

# Investigate duplicate traffic records

print("\n========== DUPLICATE INVESTIGATION ==========")

# Exact duplicate records
print("\nExact duplicate records:")
print(traffic.duplicated().sum())

# Duplicate road and timestamp combinations
duplicate_keys = traffic[
    traffic.duplicated(
        subset=["road_id", "date", "time"],
        keep=False
    )
].sort_values(["road_id", "date", "time"])

print("\nDuplicate road and timestamp records:")
print(duplicate_keys[
    ["road_id", "date", "time",
     "traffic_volume", "avg_speed",
     "occupancy", "congestion_level"]
].head(20))

# Check whether duplicate timestamps have different measurements
print("\nNumber of duplicate road-time combinations:")
print(
    duplicate_keys.groupby(
        ["road_id", "date", "time"]
    ).ngroups
)

# Investigate unusual values
print("\nNegative traffic volume records:")
print(traffic[traffic["traffic_volume"] < 0].head())

print("\nAverage speed above 120:")
print(traffic[traffic["avg_speed"] > 120].head())

print("\nOccupancy above 100:")
print(traffic[traffic["occupancy"] > 100].head())

print("\n========== INVESTIGATION COMPLETED ==========")
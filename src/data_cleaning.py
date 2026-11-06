import pandas as pd
import os

# File paths
RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"

os.makedirs(PROCESSED_PATH, exist_ok=True)

# Load original datasets
traffic = pd.read_csv(
    os.path.join(RAW_PATH, "traffic_sensor_log.csv")
)

weather = pd.read_csv(
    os.path.join(RAW_PATH, "weather_observations.csv")
)

calendar = pd.read_csv(
    os.path.join(RAW_PATH, "calendar_events.csv")
)

print("Original dataset sizes:")
print("Traffic:", traffic.shape)
print("Weather:", weather.shape)
print("Calendar:", calendar.shape)


# --------------------------------
# 1. CLEAN TRAFFIC DATA
# --------------------------------

print("\nCleaning traffic dataset...")

# Remove exact duplicate records
traffic = traffic.drop_duplicates()

# Convert negative traffic volume to missing
traffic.loc[
    traffic["traffic_volume"] < 0,
    "traffic_volume"
] = None

# Treat unusually high average speeds as invalid
traffic.loc[
    traffic["avg_speed"] > 120,
    "avg_speed"
] = None

# Standardize date format
traffic["date"] = pd.to_datetime(
    traffic["date"]
).dt.strftime("%Y-%m-%d")

# Save cleaned traffic data
traffic.to_csv(
    os.path.join(PROCESSED_PATH, "traffic_cleaned.csv"),
    index=False
)


# --------------------------------
# 2. CLEAN WEATHER DATA
# --------------------------------

print("Cleaning weather dataset...")

# Standardize weather labels
weather["weather_condition"] = (
    weather["weather_condition"]
    .str.strip()
    .str.lower()
)

weather["weather_condition"] = weather[
    "weather_condition"
].replace({
    "clear": "Clear",
    "cloudy": "Cloudy",
    "overcast": "Overcast",
    "rain": "Rain",
    "rainy": "Rain",
    "fog": "Fog",
    "foggy": "Fog"
})

# Convert weather dates to standard format
weather["date"] = pd.to_datetime(
    weather["date"],
    format="%d/%m/%Y"
).dt.strftime("%Y-%m-%d")

# Save cleaned weather data
weather.to_csv(
    os.path.join(PROCESSED_PATH, "weather_cleaned.csv"),
    index=False
)


# --------------------------------
# 3. CLEAN CALENDAR DATA
# --------------------------------

print("Cleaning calendar dataset...")

calendar["date"] = pd.to_datetime(
    calendar["date"]
).dt.strftime("%Y-%m-%d")

calendar.to_csv(
    os.path.join(PROCESSED_PATH, "calendar_cleaned.csv"),
    index=False
)


# --------------------------------
# 4. DISPLAY CLEANING RESULTS
# --------------------------------

print("\n========== CLEANING RESULTS ==========")

print("\nTraffic dataset:")
print("Shape:", traffic.shape)
print("Missing values:")
print(traffic.isnull().sum())

print("\nWeather dataset:")
print("Shape:", weather.shape)
print("Weather conditions:")
print(weather["weather_condition"].value_counts())
print("Missing values:")
print(weather.isnull().sum())

print("\nCalendar dataset:")
print("Shape:", calendar.shape)

print("\nAll cleaned datasets saved successfully!")

print("\nFiles created:")
print("data/processed/traffic_cleaned.csv")
print("data/processed/weather_cleaned.csv")
print("data/processed/calendar_cleaned.csv")
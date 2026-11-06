import pandas as pd
import os

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "data/processed/future_accident_dataset.csv",
    low_memory=False
)

df["datetime"] = pd.to_datetime(df["datetime"])

print("Dataset loaded!")
print("Original shape:", df.shape)


# ==========================================
# 2. PREPARE PREVIOUS INTERVAL DATA
# ==========================================

lag_columns = [
    "traffic_volume",
    "vehicle_count",
    "avg_speed",
    "occupancy",
    "travel_time",
    "congestion_level"
]

previous = df[
    ["road_id", "datetime"] + lag_columns
].copy()

# Previous record must be exactly 30 minutes earlier.
previous["datetime"] = (
    previous["datetime"] + pd.Timedelta(minutes=30)
)

# Rename previous interval columns.
previous = previous.rename(columns={
    "traffic_volume": "prev_traffic_volume",
    "vehicle_count": "prev_vehicle_count",
    "avg_speed": "prev_avg_speed",
    "occupancy": "prev_occupancy",
    "travel_time": "prev_travel_time",
    "congestion_level": "prev_congestion_level"
})


# ==========================================
# 3. MERGE PREVIOUS INTERVAL DATA
# ==========================================

df_v4 = df.merge(
    previous,
    on=["road_id", "datetime"],
    how="left",
    validate="one_to_one"
)

print("\nPrevious interval features added!")


# ==========================================
# 4. CHECK LAG FEATURE AVAILABILITY
# ==========================================

print("\n========== LAG FEATURE CHECK ==========")

for column in [
    "prev_traffic_volume",
    "prev_vehicle_count",
    "prev_avg_speed",
    "prev_occupancy",
    "prev_travel_time",
    "prev_congestion_level"
]:
    print(
        column,
        "missing:",
        df_v4[column].isna().sum()
    )


# ==========================================
# 5. SAVE DATASET
# ==========================================

os.makedirs("data/processed", exist_ok=True)

output_path = (
    "data/processed/future_accident_v4_dataset.csv"
)

df_v4.to_csv(output_path, index=False)

print("\nV4 dataset saved successfully!")
print("Output:", output_path)
print("Final shape:", df_v4.shape)
import pandas as pd
import numpy as np
import time
import argparse
from pathlib import Path
from datetime import datetime

# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "flowcast_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "simulated"
    / "sensor_readings.csv"
)

HISTORY_FILE = (
    PROJECT_ROOT / "data" / "simulated"
    / "sensor_history.csv"
)


# ==========================================
# GENERATE SENSOR READINGS
# ==========================================

def generate_sensor_readings():

    print("\nGenerating simulated sensor readings...")

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    df["datetime"] = pd.to_datetime(df["datetime"])

    df = df.sort_values("datetime")

    # Latest record for every road
    latest = df.groupby("road_id").tail(1).copy()

    sensor_columns = [
        "traffic_volume",
        "vehicle_count",
        "avg_speed",
        "occupancy",
        "travel_time"
    ]

    # Fill missing numeric values
    for col in sensor_columns:

        latest[col] = pd.to_numeric(
            latest[col], errors="coerce"
        )

        road_medians = df.groupby("road_id")[col].median()

        latest[col] = latest[col].fillna(
            latest["road_id"].map(road_medians)
        )

        latest[col] = latest[col].fillna(df[col].median())

    # Fill missing congestion labels
    latest["congestion_level"] = latest[
        "congestion_level"
    ].fillna(
        df["congestion_level"].mode().iloc[0]
    )

    # Generate sensor IDs
    latest["sensor_id"] = (
        "S" + latest["road_id"].str.extract(r"(\d+)$")[0]
    )

    # Add small random variations to sensor measurements
    for col in [
        "traffic_volume",
        "vehicle_count",
        "avg_speed",
        "occupancy"
    ]:

        variation = np.random.uniform(
            0.95, 1.05, len(latest)
        )

        latest[col] = latest[col] * variation

    latest["vehicle_count"] = (
        latest["vehicle_count"].round().astype(int)
    )

    latest["traffic_volume"] = (
        latest["traffic_volume"].round(2)
    )

    latest["avg_speed"] = (
        latest["avg_speed"].round(2)
    )

    latest["occupancy"] = (
        latest["occupancy"].clip(0, 100).round(2)
    )

    latest["travel_time"] = latest["travel_time"].round(2)

    # Timestamp
    current_timestamp = datetime.now()

    latest["timestamp"] = current_timestamp

    latest["data_source"] = "SIMULATED"

    # Select output columns
    output = latest[
        [
            "sensor_id",
            "road_id",
            "road_name",
            "traffic_volume",
            "vehicle_count",
            "avg_speed",
            "occupancy",
            "travel_time",
            "congestion_level",
            "timestamp",
            "data_source"
        ]
    ].copy()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save latest readings
    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # Save historical readings
    if HISTORY_FILE.exists():

        history = pd.read_csv(
            HISTORY_FILE,
            low_memory=False
        )

        # Add newly introduced columns to old history
        for col in output.columns:

            if col not in history.columns:
                history[col] = np.nan

        history = pd.concat(
            [history, output],
            ignore_index=True
        )

    else:

        history = output.copy()

    history.to_csv(
        HISTORY_FILE,
        index=False
    )

    print("\nSensor readings generated successfully!")

    print("Active sensors:", len(output))

    print("Timestamp:", current_timestamp)

    print("\nLatest readings:")
    print(output.head())

    print("\nHistorical records:", len(history))

    print("History saved to:", HISTORY_FILE)

    return output


# ==========================================
# MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Generate sensor readings continuously"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Interval between readings in seconds"
    )

    args = parser.parse_args()

    if args.continuous:

        print("Starting continuous sensor simulation...")

        print(
            "Update interval:",
            args.interval,
            "seconds"
        )

        print("Press CTRL+C to stop.")

        try:

            while True:

                generate_sensor_readings()

                time.sleep(args.interval)

        except KeyboardInterrupt:

            print("\nSensor simulation stopped.")

    else:

        generate_sensor_readings()
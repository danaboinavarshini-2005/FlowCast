
import pandas as pd
import os

# Project dataset paths
DATA_PATH = "data/raw"

# Load traffic dataset
traffic = pd.read_csv(
    os.path.join(DATA_PATH, "traffic_sensor_log.csv")
)

# Load weather dataset
weather = pd.read_csv(
    os.path.join(DATA_PATH, "weather_observations.csv")
)

# Load calendar dataset
calendar = pd.read_csv(
    os.path.join(DATA_PATH, "calendar_events.csv")
)


# Function to analyze datasets
def analyze_dataset(name, df):

    print("\n" + "=" * 60)
    print(f"{name} DATASET ANALYSIS")
    print("=" * 60)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nFirst Five Records:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Records:")
    print(df.duplicated().sum())

    print("\nData Types:")
    print(df.dtypes)


# Analyze all datasets
analyze_dataset("TRAFFIC", traffic)

analyze_dataset("WEATHER", weather)

analyze_dataset("CALENDAR", calendar)


print("\n======================================")
print("FLOWCAST DATA ANALYSIS COMPLETED")
print("======================================")


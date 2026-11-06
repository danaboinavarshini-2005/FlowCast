
import pandas as pd
import os

# Load dataset
df = pd.read_csv(
    "data/processed/flowcast_features.csv",
    low_memory=False
)

print("Dataset loaded successfully!")

# Create accident occurrence target
df["accident_occurred"] = (
    df["accident_count"] > 0
).astype(int)

# Create output directory
output_dir = "reports/accident_analysis"
os.makedirs(output_dir, exist_ok=True)

# Overall summary
print("\n========== OVERALL ACCIDENT SUMMARY ==========")

total_records = len(df)
accident_records = int(df["accident_occurred"].sum())
total_accidents = int(df["accident_count"].sum())

print("Total traffic records:", total_records)
print("Records with accidents:", accident_records)
print("Total recorded accidents:", total_accidents)

print(
    "Overall accident occurrence rate:",
    round(accident_records / total_records * 100, 3),
    "%"
)


# Function to calculate accident statistics
def analyze_category(data, column, title):

    result = (
        data.groupby(column, dropna=False)
        .agg(
            total_records=("accident_occurred", "size"),
            accident_records=("accident_occurred", "sum"),
            total_accidents=("accident_count", "sum")
        )
        .reset_index()
    )

    result["accident_rate_percent"] = (
        result["accident_records"]
        / result["total_records"] * 100
    )

    result = result.sort_values(
        "accident_rate_percent",
        ascending=False
    )

    result = result.round(3)

    print(f"\n========== {title} ==========")
    print(result.to_string(index=False))

    filename = column + "_accident_analysis.csv"

    result.to_csv(
        os.path.join(output_dir, filename),
        index=False
    )

    return result


# 1. Accident analysis by hour
hour_result = analyze_category(
    df,
    "hour",
    "ACCIDENT ANALYSIS BY HOUR"
)


# 2. Accident analysis by day
day_names = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}

df["day_name"] = df["day_of_week"].map(day_names)

day_result = analyze_category(
    df,
    "day_name",
    "ACCIDENT ANALYSIS BY DAY"
)


# 3. Accident analysis by road
road_result = analyze_category(
    df,
    "road_name",
    "ACCIDENT ANALYSIS BY ROAD"
)


# 4. Accident analysis by weather
df["weather_condition"] = (
    df["weather_condition"].fillna("Unknown")
)

weather_result = analyze_category(
    df,
    "weather_condition",
    "ACCIDENT ANALYSIS BY WEATHER"
)


# 5. Accident analysis by month
month_names = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May"
}

df["month_name"] = df["month"].map(month_names)

month_result = analyze_category(
    df,
    "month_name",
    "ACCIDENT ANALYSIS BY MONTH"
)


print("\n==============================================")
print("ACCIDENT RATE ANALYSIS COMPLETED SUCCESSFULLY!")
print("Results saved in:", output_dir)
print("==============================================")


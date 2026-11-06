import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SENSOR_FILE = PROJECT_ROOT / "data" / "simulated" / "sensor_readings.csv"


def generate_recommendations(df):

    recommendations = []

    for _, row in df.iterrows():

        actions = []
        priority = "Low"

        volume = row["traffic_volume"]
        speed = row["avg_speed"]
        occupancy = row["occupancy"]
        congestion = str(row["congestion_level"]).strip()

        # High traffic volume
        if volume >= 400:
            actions.append(
                "Prioritize this road for traffic management."
            )
            priority = "High"

        # High occupancy
        if occupancy >= 30:
            actions.append(
                "Consider reviewing traffic signal timing."
            )
            priority = "High"

        # Low vehicle speed
        if speed < 25:
            actions.append(
                "Investigate possible traffic congestion."
            )
            priority = "High"

        elif speed < 40:
            actions.append(
                "Monitor vehicle movement for developing congestion."
            )

            if priority != "High":
                priority = "Medium"

        # Congestion classification
        if congestion == "Severe":
            actions.append(
                "Take immediate traffic management attention."
            )
            priority = "High"

        elif congestion == "Heavy":
            actions.append(
                "Consider traffic diversion and signal adjustments."
            )
            priority = "High"

        elif congestion == "Moderate":
            actions.append(
                "Monitor traffic flow and optimize signal timing."
            )

            if priority == "Low":
                priority = "Medium"

        # Default recommendation
        if not actions:
            actions.append(
                "Traffic is flowing normally. Continue routine monitoring."
            )

        recommendations.append({

            "sensor_id": row["sensor_id"],
            "road_id": row["road_id"],
            "road_name": row["road_name"],
            "traffic_volume": volume,
            "avg_speed": speed,
            "occupancy": occupancy,
            "congestion_level": congestion,
            "priority": priority,
            "recommendation": " ".join(actions),
            "timestamp": row["timestamp"]

        })

    result = pd.DataFrame(recommendations)

    priority_order = {
        "High": 0,
        "Medium": 1,
        "Low": 2
    }

    result["priority_order"] = result["priority"].map(priority_order)

    result = result.sort_values(
        by=["priority_order", "traffic_volume"],
        ascending=[True, False]
    )

    return result.drop(columns=["priority_order"])


def main():

    if not SENSOR_FILE.exists():
        print("Sensor readings file not found!")
        return

    df = pd.read_csv(SENSOR_FILE)

    result = generate_recommendations(df)

    print("\n========== FLOWCAST TRAFFIC MANAGEMENT RECOMMENDATIONS ==========\n")

    for _, row in result.iterrows():

        print(f"Road: {row['road_name']} ({row['road_id']})")
        print(f"Priority: {row['priority']}")
        print(f"Congestion: {row['congestion_level']}")
        print(f"Recommendation: {row['recommendation']}")
        print("-" * 65)

    output_file = (
        PROJECT_ROOT / "reports" / "traffic_recommendations.csv"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(output_file, index=False)

    print("\nRecommendations saved successfully!")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    main()
import csv
import json
from pathlib import Path
from datetime import datetime


# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HISTORY_FILE = PROJECT_ROOT / "data" / "processed" / "prediction_history.csv"


# ==========================================
# SAVE PREDICTION
# ==========================================

def save_prediction(
    road_id,
    prediction_type,
    input_values,
    predicted_result
):

    # Create directory if it does not exist
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Prepare prediction record
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "road_id": road_id,
        "prediction_type": prediction_type,
        "input_values": json.dumps(input_values, default=str),
        "predicted_result": json.dumps(predicted_result, default=str)
    }

    # Check whether file already exists
    file_exists = HISTORY_FILE.exists()

    # Append prediction to CSV
    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as file:

        fieldnames = [
            "timestamp",
            "road_id",
            "prediction_type",
            "input_values",
            "predicted_result"
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(record)

    return True


# ==========================================
# LOAD PREDICTION HISTORY
# ==========================================

def load_prediction_history():

    import pandas as pd

    if not HISTORY_FILE.exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "road_id",
                "prediction_type",
                "input_values",
                "predicted_result"
            ]
        )

    return pd.read_csv(HISTORY_FILE)


# ==========================================
# CLEAR PREDICTION HISTORY
# ==========================================

def clear_prediction_history():

    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()

    return True
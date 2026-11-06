import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys

from pathlib import Path
from datetime import date, time


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SIMULATED_DIR = PROJECT_ROOT / "data" / "simulated"

HISTORY_FILE = SIMULATED_DIR / "sensor_history.csv"
SENSOR_FILE = SIMULATED_DIR / "sensor_readings.csv"
DATASET_FILE = DATA_DIR / "flowcast_features.csv"


# Prediction history module
from src.prediction_history import (
    save_prediction,
    load_prediction_history,
    clear_prediction_history
)


FALLBACK_COORDINATES = (16.3067, 80.4365)


# =========================================================
# 2. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="FlowCast | Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 3. CUSTOM CSS
# =========================================================

st.markdown("""
<style>
.stApp {
    background-color: #F4F7FB;
}
[data-testid="stSidebar"] {
    background-color: #10243A;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #10243A;
}
.subtitle {
    color: #64748B;
    font-size: 16px;
}
.section-title {
    color: #10243A;
    font-size: 23px;
    font-weight: 700;
}
div[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
.stButton > button {
    background-color: #1769AA;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 22px;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #0D4F85;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. LOAD TRAINED MODELS
# =========================================================

@st.cache_resource
def load_models():

    required_model_files = {
        "traffic": MODEL_DIR / "traffic_volume_model.pkl",
        "congestion": MODEL_DIR / "congestion_model.pkl",
        "travel": MODEL_DIR / "travel_time_model.pkl",
        "accident": MODEL_DIR / "future_accident_v4_model.pkl"
    }

    missing = [
        str(path) for path in required_model_files.values()
        if not path.exists()
    ]

    threshold_file = MODEL_DIR / "future_accident_v4_threshold.pkl"

    if missing or not threshold_file.exists():

        raise FileNotFoundError(
            "One or more trained model files are missing:\n"
            + "\n".join(
                missing + (
                    [str(threshold_file)]
                    if not threshold_file.exists()
                    else []
                )
            )
        )

    loaded = {
        name: joblib.load(path)
        for name, path in required_model_files.items()
    }

    threshold = float(joblib.load(threshold_file))

    return loaded, threshold


try:

    models, accident_threshold = load_models()

except Exception as exc:

    st.error(f"Unable to load trained models: {exc}")
    st.stop()


# =========================================================
# 5. LOAD HISTORICAL DATASET
# =========================================================

@st.cache_data
def load_dataset():

    if not DATASET_FILE.exists():
        return pd.DataFrame()

    data = pd.read_csv(
        DATASET_FILE,
        low_memory=False
    )

    if "datetime" in data.columns:

        data["datetime"] = pd.to_datetime(
            data["datetime"],
            errors="coerce"
        )

    return data


df = load_dataset()


# =========================================================
# 6. LOAD CURRENT SENSOR READINGS
# =========================================================

@st.cache_data(ttl=5)
def load_sensor_data():

    if not SENSOR_FILE.exists():
        return pd.DataFrame()

    data = pd.read_csv(
        SENSOR_FILE,
        low_memory=False
    )

    if "timestamp" in data.columns:

        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce"
        )

    required = {
        "road_id",
        "sensor_id",
        "traffic_volume",
        "vehicle_count",
        "avg_speed",
        "occupancy",
        "timestamp"
    }

    if not required.issubset(data.columns):
        return pd.DataFrame()

    data = data.dropna(
        subset=["road_id", "timestamp"]
    ).copy()

    for column in [
        "traffic_volume",
        "vehicle_count",
        "avg_speed",
        "occupancy"
    ]:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(
        subset=[
            "traffic_volume",
            "vehicle_count",
            "avg_speed",
            "occupancy"
        ]
    )

    data = (
        data.sort_values("timestamp")
        .drop_duplicates(
            subset=["road_id"],
            keep="last"
        )
        .sort_values("road_id")
        .reset_index(drop=True)
    )

    return data


sensor_df = load_sensor_data()


# =========================================================
# 7. LIVE WEATHER SERVICE
# =========================================================

@st.cache_data(ttl=600, show_spinner=False)
def fetch_live_weather(latitude, longitude):

    try:

        from src.weather_service import get_current_weather

        weather_data = get_current_weather(
            float(latitude),
            float(longitude)
        )

        if weather_data is None:
            return None

        return weather_data

    except Exception as exc:

        print("Dashboard Weather Error:", exc)

        return None


def get_road_coordinates(road_id):

    required = {
        "road_id",
        "latitude",
        "longitude"
    }

    if df.empty or not required.issubset(df.columns):

        return FALLBACK_COORDINATES

    rows = df[
        df["road_id"].astype(str) == str(road_id)
    ]

    if rows.empty:

        return FALLBACK_COORDINATES

    latitude = pd.to_numeric(
        rows.iloc[0]["latitude"],
        errors="coerce"
    )

    longitude = pd.to_numeric(
        rows.iloc[0]["longitude"],
        errors="coerce"
    )

    if pd.isna(latitude) or pd.isna(longitude):

        return FALLBACK_COORDINATES

    latitude = float(latitude)
    longitude = float(longitude)

    if not (-90 <= latitude <= 90):

        return FALLBACK_COORDINATES

    if not (-180 <= longitude <= 180):

        return FALLBACK_COORDINATES

    return latitude, longitude


def get_weather_for_road(road_id):

    latitude, longitude = get_road_coordinates(road_id)

    weather_data = fetch_live_weather(
        latitude,
        longitude
    )

    if weather_data is not None:

        return weather_data

    if (latitude, longitude) != FALLBACK_COORDINATES:

        weather_data = fetch_live_weather(
            FALLBACK_COORDINATES[0],
            FALLBACK_COORDINATES[1]
        )

    return weather_data


# =========================================================
# 8. SENSOR / TRAFFIC HELPER FUNCTIONS
# =========================================================

def classify_traffic(row):

    occupancy = float(row["occupancy"])
    speed = float(row["avg_speed"])

    if occupancy >= 30 or speed < 25:

        return "High Traffic"

    if occupancy >= 15 or speed < 40:

        return "Moderate Traffic"

    return "Low Traffic"


def get_latest_sensor(road_id):

    if sensor_df.empty:
        return None

    selected = sensor_df[
        sensor_df["road_id"].astype(str) == str(road_id)
    ]

    if selected.empty:
        return None

    return selected.sort_values(
        "timestamp"
    ).iloc[-1]


def format_number(value, decimals=2):

    try:

        if pd.isna(value):
            return "N/A"

        return f"{float(value):,.{decimals}f}"

    except (TypeError, ValueError):

        return "N/A"


if not sensor_df.empty:

    sensor_df["traffic_condition"] = sensor_df.apply(
        classify_traffic,
        axis=1
    )


# =========================================================
# 9. DATASET DATE RANGE
# =========================================================

if (
    not df.empty
    and "datetime" in df.columns
    and df["datetime"].notna().any()
):

    min_date = df["datetime"].min().date()
    max_date = df["datetime"].max().date()

else:

    min_date = date(2025, 1, 1)
    max_date = date(2025, 5, 31)


# =========================================================
# 10. HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🚦 FlowCast</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Traffic Analytics and Prediction System'
    '</div>',
    unsafe_allow_html=True
)

st.write("")
st.divider()


# =========================================================
# 11. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown("## 🚦 FlowCast")

st.sidebar.caption(
    "TRAFFIC INTELLIGENCE PLATFORM"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard Overview",
        "Live Sensor Monitoring",
        "Traffic Volume Prediction",
        "Congestion Prediction",
        "Travel Time Prediction",
        "Future Accident Prediction",
        "Traffic Management Recommendations",
        "Prediction History"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Model Information")

st.sidebar.write("📡 Simulated Traffic Sensors")
st.sidebar.write("🚗 Traffic Volume")
st.sidebar.write("🚦 Congestion")
st.sidebar.write("⏱️ Travel Time")
st.sidebar.write("⚠️ Future Accident V4")

st.sidebar.divider()

st.sidebar.caption(
    "Traffic Intelligence Platform"
)


# =========================================================
# 12. COMMON INPUT FUNCTION
# =========================================================

def get_common_inputs(default_road="NL-015"):

    st.markdown(
        '<div class="section-title">'
        'Road and Environmental Details'
        '</div>',
        unsafe_allow_html=True
    )

    road_options = [
        f"NL-{i:03d}"
        for i in range(1, 26)
    ]

    default_index = (
        road_options.index(default_road)
        if default_road in road_options else 0
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        road_id = st.selectbox(
            "Select Road",
            road_options,
            index=default_index,
            key=f"{page}_road"
        )

        selected_date = st.date_input(
            "Select Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key=f"{page}_date"
        )

    live_weather = get_weather_for_road(road_id)

    weather_options = [
        "Clear",
        "Cloudy",
        "Fog",
        "Overcast",
        "Rain"
    ]

    if live_weather:

        suggested_weather = live_weather.get(
            "weather_condition",
            "Cloudy"
        )

        if suggested_weather not in weather_options:
            suggested_weather = "Cloudy"

        suggested_temperature = float(
            live_weather.get("temperature", 18.0)
        )

        suggested_rainfall = float(
            live_weather.get("rainfall", 0.0)
        )

    else:

        suggested_weather = "Clear"
        suggested_temperature = 18.0
        suggested_rainfall = 0.0

    with col2:

        selected_time = st.time_input(
            "Select Time",
            value=time(8, 0),
            key=f"{page}_time"
        )

        weather = st.selectbox(
            "Weather Condition",
            weather_options,
            index=weather_options.index(suggested_weather),
            key=f"{page}_weather_{road_id}"
        )

    with col3:

        temperature = st.number_input(
            "Temperature (°C)",
            value=suggested_temperature,
            key=f"{page}_temperature_{road_id}"
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=max(0.0, suggested_rainfall),
            key=f"{page}_rainfall_{road_id}"
        )

    visibility = st.number_input(
        "Visibility (metres)",
        min_value=0.0,
        value=10000.0,
        key=f"{page}_visibility"
    )

    if live_weather:

        st.success(
            f"Live weather loaded for {road_id}. "
            f"Weather time: "
            f"{live_weather.get('weather_time', 'Latest available')} "
            "(Open-Meteo). You can adjust the values manually."
        )

        w1, w2, w3 = st.columns(3)

        w1.metric(
            "Humidity",
            f"{live_weather.get('humidity', 'N/A')}%"
        )

        w2.metric(
            "Wind Speed",
            f"{live_weather.get('wind_speed', 'N/A')} km/h"
        )

        w3.metric(
            "Current Rain",
            f"{live_weather.get('rain', 'N/A')} mm"
        )

    else:

        st.warning(
            "Live weather is unavailable. "
            "Default weather values are being used. "
            "You can enter weather conditions manually."
        )

    st.markdown("### Road Conditions")

    col4, col5, col6 = st.columns(3)

    with col4:

        signal_timing = st.number_input(
            "Signal Timing",
            min_value=0.0,
            value=44.0,
            key=f"{page}_signal"
        )

    with col5:

        road_capacity = st.number_input(
            "Road Capacity",
            min_value=0.0,
            value=1888.0,
            key=f"{page}_capacity"
        )

    with col6:

        public_holiday = st.selectbox(
            "Public Holiday",
            [0, 1],
            key=f"{page}_holiday"
        )

    col7, col8 = st.columns(2)

    with col7:

        event_flag = st.selectbox(
            "Special Event",
            [0, 1],
            key=f"{page}_event"
        )

    with col8:

        roadwork_flag = st.selectbox(
            "Roadwork",
            [0, 1],
            key=f"{page}_roadwork"
        )

    hour = selected_time.hour
    minute = selected_time.minute

    day_of_week = selected_date.weekday()
    month = selected_date.month
    day = selected_date.day

    is_weekend = int(day_of_week >= 5)

    is_peak_hour = int(
        (7 <= hour <= 10) or
        (16 <= hour <= 19)
    )

    return {
        "hour": hour,
        "minute": minute,
        "day_of_week": day_of_week,
        "month": month,
        "day": day,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
        "temperature": temperature,
        "rainfall": rainfall,
        "visibility": visibility,
        "public_holiday": public_holiday,
        "event_flag": event_flag,
        "roadwork_flag": roadwork_flag,
        "signal_timing": signal_timing,
        "road_capacity": road_capacity,
        "road_id": road_id,
        "weather_condition": weather
    }


# =========================================================
# 13. SENSOR METRICS
# =========================================================

def show_sensor_metrics(sensor):

    if sensor is None:

        st.warning(
            "No current sensor reading is available for this road."
        )

        return

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Traffic Volume",
        format_number(sensor["traffic_volume"])
    )

    c2.metric(
        "Vehicle Count",
        format_number(sensor["vehicle_count"], 0)
    )

    c3.metric(
        "Average Speed",
        format_number(sensor["avg_speed"])
    )

    c4.metric(
        "Occupancy",
        f"{format_number(sensor['occupancy'])}%"
    )

    st.caption(
        f"Sensor: {sensor['sensor_id']} | "
        f"Last updated: {sensor['timestamp']}"
    )


# =========================================================
# 14. DASHBOARD OVERVIEW
# =========================================================

if page == "Dashboard Overview":

    st.header("📊 Dashboard Overview")

    st.write(
        "Monitor traffic patterns, simulated sensors "
        "and AI-powered traffic predictions."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Prediction Models", "4")
    col2.metric("Roads Available", "25")
    col3.metric("Weather Types", "5")

    col4.metric(
        "Dataset Records",
        f"{len(df):,}" if not df.empty else "N/A"
    )

    st.divider()
    st.subheader("📡 Sensor Status")

    if not sensor_df.empty:

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "Active Sensors",
            sensor_df["sensor_id"].nunique()
        )

        s2.metric(
            "Total Vehicles",
            f"{sensor_df['vehicle_count'].sum():,.0f}"
        )

        s3.metric(
            "Average Speed",
            f"{sensor_df['avg_speed'].mean():.2f}"
        )

        s4.metric(
            "Average Occupancy",
            f"{sensor_df['occupancy'].mean():.2f}%"
        )

        st.subheader("🚦 Current Traffic Conditions")

        condition_counts = (
            sensor_df["traffic_condition"]
            .value_counts()
            .reindex(
                [
                    "Low Traffic",
                    "Moderate Traffic",
                    "High Traffic"
                ],
                fill_value=0
            )
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Low Traffic Roads",
            int(condition_counts["Low Traffic"])
        )

        c2.metric(
            "Moderate Traffic Roads",
            int(condition_counts["Moderate Traffic"])
        )

        c3.metric(
            "High Traffic Roads",
            int(condition_counts["High Traffic"])
        )

    else:

        st.warning(
            "Sensor readings are unavailable."
        )

    st.divider()
    st.subheader("Prediction Modules")

    c1, c2 = st.columns(2)

    with c1:

        st.info(
            "🚗 **Traffic Volume Prediction**\n\n"
            "Estimate expected traffic volume for a selected road and time."
        )

        st.info(
            "⏱️ **Travel Time Prediction**\n\n"
            "Estimate travel time using road and environmental conditions."
        )

        st.info(
            "📡 **Live Sensor Monitoring**\n\n"
            "Monitor simulated traffic readings across 25 roads."
        )

    with c2:

        st.info(
            "🚦 **Congestion Prediction**\n\n"
            "Classify traffic conditions into four congestion levels."
        )

        st.warning(
            "⚠️ **Future Accident Prediction**\n\n"
            "Estimate experimental accident probability for the next "
            "30-minute interval."
        )

    st.divider()
    st.subheader("🌦️ Current Weather")

    weather_road = st.selectbox(
        "Weather location (road)",
        [f"NL-{i:03d}" for i in range(1, 26)],
        index=14,
        key="overview_weather_road"
    )

    overview_weather = get_weather_for_road(
        weather_road
    )

    if overview_weather:

        wc1, wc2, wc3, wc4, wc5 = st.columns(5)

        wc1.metric(
            "Condition",
            overview_weather.get("weather_condition", "N/A")
        )

        wc2.metric(
            "Temperature",
            f"{overview_weather.get('temperature', 'N/A')} °C"
        )

        wc3.metric(
            "Humidity",
            f"{overview_weather.get('humidity', 'N/A')}%"
        )

        wc4.metric(
            "Rainfall",
            f"{overview_weather.get('rainfall', 'N/A')} mm"
        )

        wc5.metric(
            "Wind Speed",
            f"{overview_weather.get('wind_speed', 'N/A')} km/h"
        )

        st.caption(
            f"Source: Open-Meteo | Road: {weather_road} | "
            f"Weather time: "
            f"{overview_weather.get('weather_time', 'N/A')}"
        )

    else:

        st.warning(
            "Live weather could not be retrieved. "
            "Prediction pages retain manual weather inputs as a fallback."
        )

    if st.button(
        "🔄 Refresh Weather",
        key="overview_refresh_weather"
    ):

        fetch_live_weather.clear()
        st.rerun()

    st.divider()
    st.subheader("📈 Traffic Analytics")

    if not df.empty:

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            st.markdown("#### Traffic Volume by Hour")

            hourly = (
                df.groupby("hour")["traffic_volume"]
                .mean()
                .reset_index()
            )

            st.line_chart(
                hourly.set_index("hour")
            )

        with chart_col2:

            st.markdown("#### Congestion Distribution")

            congestion_counts = (
                df["congestion_level"].value_counts()
            )

            st.bar_chart(congestion_counts)

        st.divider()
        st.subheader("Dataset Information")

        st.write(
            f"**Dataset Period:** {min_date} to {max_date}"
        )

        st.write(
            f"**Total Records:** {len(df):,}"
        )

        st.write(
            "**Roads:** 25"
        )

    else:

        st.warning(
            "Traffic dataset could not be loaded."
        )


# =========================================================
# 15. LIVE SENSOR MONITORING
# =========================================================

elif page == "Live Sensor Monitoring":

    st.header("📡 Live Traffic Sensor Monitoring")

    st.warning(
        "SIMULATION MODE: These readings are generated from historical "
        "traffic records. They are not actual live sensor measurements."
    )

    if not sensor_df.empty:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Active Sensors",
            sensor_df["sensor_id"].nunique()
        )

        col2.metric(
            "Total Vehicles",
            f"{sensor_df['vehicle_count'].sum():,.0f}"
        )

        col3.metric(
            "Average Speed",
            f"{sensor_df['avg_speed'].mean():.2f}"
        )

        col4.metric(
            "Average Occupancy",
            f"{sensor_df['occupancy'].mean():.2f}%"
        )

        st.divider()
        st.subheader("🚦 Traffic Condition Summary")

        condition_counts = (
            sensor_df["traffic_condition"]
            .value_counts()
            .reindex(
                [
                    "Low Traffic",
                    "Moderate Traffic",
                    "High Traffic"
                ],
                fill_value=0
            )
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "🟢 Low Traffic",
            int(condition_counts["Low Traffic"])
        )

        c2.metric(
            "🟡 Moderate Traffic",
            int(condition_counts["Moderate Traffic"])
        )

        c3.metric(
            "🔴 High Traffic",
            int(condition_counts["High Traffic"])
        )

        st.bar_chart(condition_counts)

        st.divider()
        st.subheader("🚨 Traffic Alerts")

        high_traffic = sensor_df[
            sensor_df["traffic_condition"] == "High Traffic"
        ]

        if not high_traffic.empty:

            for _, row in high_traffic.iterrows():

                st.error(
                    f"High traffic detected on "
                    f"{row.get('road_name', row['road_id'])} "
                    f"({row['road_id']}). "
                    f"Occupancy: {row['occupancy']:.2f}% | "
                    f"Speed: {row['avg_speed']:.2f}"
                )

        else:

            st.success(
                "No high traffic alerts."
            )

        st.divider()
        st.subheader("Traffic Volume by Road")

        st.bar_chart(
            sensor_df.set_index("road_id")[["traffic_volume"]]
        )

        st.subheader("Vehicle Count by Road")

        st.bar_chart(
            sensor_df.set_index("road_id")[["vehicle_count"]]
        )

        st.subheader("Average Speed by Road")

        st.line_chart(
            sensor_df.set_index("road_id")[["avg_speed"]]
        )

        st.subheader("Road Occupancy by Road")

        st.bar_chart(
            sensor_df.set_index("road_id")[["occupancy"]]
        )

        st.divider()
        st.subheader("📋 Sensor Readings")

        display_columns = [
            column for column in [
                "sensor_id",
                "road_id",
                "road_name",
                "traffic_volume",
                "vehicle_count",
                "avg_speed",
                "occupancy",
                "traffic_condition",
                "timestamp"
            ]
            if column in sensor_df.columns
        ]

        st.dataframe(
            sensor_df[display_columns],
            width="stretch",
            hide_index=True
        )

        st.caption(
            f"Last generated: {sensor_df['timestamp'].max()}"
        )

        if st.button("🔄 Refresh Sensor Readings"):

            load_sensor_data.clear()
            st.rerun()

        try:

            from streamlit_autorefresh import st_autorefresh

            st_autorefresh(
                interval=30000,
                key="sensor_refresh"
            )

        except ImportError:

            st.info(
                "Install streamlit-autorefresh for automatic refresh."
            )

    else:

        st.error(
            "No sensor readings found."
        )

        st.code(
            "python src/simulate_traffic_sensors.py --continuous --interval 30"
        )


# =========================================================
# 16. TRAFFIC VOLUME PREDICTION
# =========================================================

elif page == "Traffic Volume Prediction":

    st.header("🚗 Traffic Volume Prediction")

    common = get_common_inputs()

    st.divider()
    st.subheader("📡 Latest Sensor Readings")

    sensor = get_latest_sensor(
        common["road_id"]
    )

    show_sensor_metrics(sensor)

    if st.button("Predict Traffic Volume"):

        try:

            features = pd.DataFrame(
                [common]
            ).drop(
                columns=["weather_condition"]
            )

            prediction = models["traffic"].predict(
                features
            )[0]

            # SAVE PREDICTION HISTORY
            save_prediction(
                road_id=common["road_id"],
                prediction_type="Traffic Volume",
                input_values=common,
                predicted_result=float(prediction)
            )

            st.success(
                "AI traffic volume prediction completed! "
                "Prediction saved to history."
            )

            st.metric(
                "Predicted Traffic Volume",
                f"{prediction:.2f}"
            )

        except Exception as exc:

            st.error(
                f"Traffic volume prediction failed: {exc}"
            )


# =========================================================
# 17. CONGESTION PREDICTION
# =========================================================

elif page == "Congestion Prediction":

    st.header("🚦 Congestion Level Prediction")

    common = get_common_inputs()

    st.divider()
    st.subheader("📡 Latest Sensor Readings")

    sensor = get_latest_sensor(
        common["road_id"]
    )

    show_sensor_metrics(sensor)

    if sensor is not None:

        condition = classify_traffic(sensor)

        if condition == "High Traffic":

            st.error(
                "🔴 Sensor Alert: High Traffic"
            )

        elif condition == "Moderate Traffic":

            st.warning(
                "🟡 Sensor Status: Moderate Traffic"
            )

        else:

            st.success(
                "🟢 Sensor Status: Low Traffic"
            )

        st.info(
            f"Sensor-based traffic condition: **{condition}**"
        )

    st.divider()

    if st.button("Predict Congestion"):

        try:

            features = pd.DataFrame(
                [common]
            ).drop(
                columns=["weather_condition"]
            )

            prediction = models["congestion"].predict(
                features
            )[0]

            # SAVE PREDICTION HISTORY
            save_prediction(
                road_id=common["road_id"],
                prediction_type="Congestion",
                input_values=common,
                predicted_result=str(prediction)
            )

            st.success(
                "AI congestion prediction completed! "
                "Prediction saved to history."
            )

            if prediction == "Free-flow":

                st.success(
                    f"🟢 {prediction}"
                )

            elif prediction == "Moderate":

                st.warning(
                    f"🟡 {prediction}"
                )

            else:

                st.error(
                    f"🔴 {prediction}"
                )

            st.caption(
                "AI model prediction and rule-based sensor condition "
                "are separate assessments."
            )

        except Exception as exc:

            st.error(
                f"Congestion prediction failed: {exc}"
            )


# =========================================================
# 18. TRAVEL TIME PREDICTION
# =========================================================

elif page == "Travel Time Prediction":

    st.header("⏱️ Travel Time Prediction")

    common = get_common_inputs()

    st.divider()
    st.subheader("📡 Latest Sensor Readings")

    sensor = get_latest_sensor(
        common["road_id"]
    )

    show_sensor_metrics(sensor)

    if sensor is not None:

        if sensor["avg_speed"] < 25:

            st.error(
                "🔴 Very low vehicle speed detected."
            )

        elif sensor["avg_speed"] < 40:

            st.warning(
                "🟡 Reduced vehicle speed detected."
            )

        else:

            st.success(
                "🟢 Vehicle speed is currently normal."
            )

    st.divider()

    if st.button("Predict Travel Time"):

        try:

            features = pd.DataFrame(
                [common]
            )

            prediction = models["travel"].predict(
                features
            )[0]

            # SAVE PREDICTION HISTORY
            save_prediction(
                road_id=common["road_id"],
                prediction_type="Travel Time",
                input_values=common,
                predicted_result=float(prediction)
            )

            st.success(
                "AI travel time prediction completed! "
                "Prediction saved to history."
            )

            st.metric(
                "Estimated Travel Time",
                f"{prediction:.2f} dataset units"
            )

            st.caption(
                "Travel time units follow the original dataset. "
                "Sensor readings are displayed separately."
            )

        except Exception as exc:

            st.error(
                f"Travel time prediction failed: {exc}"
            )


# =========================================================
# 19. FUTURE ACCIDENT PREDICTION
# =========================================================

elif page == "Future Accident Prediction":

    st.header("⚠️ Future Accident Risk Prediction")

    st.warning(
        "Experimental model: estimates accident probability for the next "
        "30-minute interval. It is not a reliable real-world safety warning."
    )

    common = get_common_inputs(
        default_road="NL-015"
    )

    road_id = common["road_id"]

    sensor = get_latest_sensor(
        road_id
    )

    st.divider()
    st.subheader("📡 Current Traffic Conditions")

    if sensor is not None:

        st.success(
            f"Current sensor connected: {sensor['sensor_id']}"
        )

        show_sensor_metrics(sensor)

        current_timestamp = pd.to_datetime(
            sensor["timestamp"]
        )

    else:

        st.warning(
            "No current sensor reading is available for this road. "
            "The prediction requires a current sensor timestamp."
        )

        current_timestamp = None

    st.subheader("Current Traffic Values")

    col1, col2, col3 = st.columns(3)

    with col1:

        traffic_volume = st.number_input(
            "Traffic Volume",
            min_value=0.0,
            value=(
                float(sensor["traffic_volume"])
                if sensor is not None else 361.5
            ),
            key="future_traffic_volume"
        )

        vehicle_count = st.number_input(
            "Vehicle Count",
            min_value=0.0,
            value=(
                float(sensor["vehicle_count"])
                if sensor is not None else 362.0
            ),
            key="future_vehicle_count"
        )

    with col2:

        avg_speed = st.number_input(
            "Average Speed",
            min_value=0.0,
            value=(
                float(sensor["avg_speed"])
                if sensor is not None else 44.9
            ),
            key="future_avg_speed"
        )

        occupancy = st.number_input(
            "Road Occupancy",
            min_value=0.0,
            value=(
                float(sensor["occupancy"])
                if sensor is not None else 27.0
            ),
            key="future_occupancy"
        )

    congestion_options = [
        "Free-flow",
        "Moderate",
        "Heavy",
        "Severe"
    ]

    with col3:

        sensor_travel_time = (
            sensor.get("travel_time")
            if sensor is not None else None
        )

        try:

            default_travel_time = (
                float(sensor_travel_time)
                if sensor_travel_time is not None
                and pd.notna(sensor_travel_time)
                else 3.12
            )

        except (TypeError, ValueError):

            default_travel_time = 3.12

        travel_time = st.number_input(
            "Current Travel Time",
            min_value=0.0,
            value=default_travel_time,
            key="future_travel_time"
        )

        sensor_congestion = (
            sensor.get("congestion_level")
            if sensor is not None else None
        )

        if sensor_congestion not in congestion_options:
            sensor_congestion = "Moderate"

        congestion = st.selectbox(
            "Current Congestion",
            congestion_options,
            index=congestion_options.index(sensor_congestion),
            key="future_congestion"
        )

    st.divider()
    st.subheader("🕒 Previous 30-Minute Conditions")

    previous_sensor = None
    history_debug = {}

    required_columns = [
        "traffic_volume",
        "vehicle_count",
        "avg_speed",
        "occupancy",
        "travel_time",
        "congestion_level"
    ]

    if (
        current_timestamp is not None
        and HISTORY_FILE.exists()
    ):

        try:

            history = pd.read_csv(
                HISTORY_FILE,
                low_memory=False
            )

            if (
                "timestamp" in history.columns
                and "road_id" in history.columns
            ):

                history["timestamp"] = pd.to_datetime(
                    history["timestamp"],
                    errors="coerce"
                )

                history = history.dropna(
                    subset=["timestamp"]
                )

                road_history = history[
                    history["road_id"].astype(str) == str(road_id)
                ].copy()

                missing_columns = [
                    column for column in required_columns
                    if column not in road_history.columns
                ]

                history_debug["total_history_rows"] = len(history)
                history_debug["road_history_rows"] = len(road_history)
                history_debug["missing_columns"] = missing_columns

                if not missing_columns:

                    for column in required_columns[:-1]:

                        road_history[column] = pd.to_numeric(
                            road_history[column],
                            errors="coerce"
                        )

                    road_history = road_history.dropna(
                        subset=required_columns + ["timestamp"]
                    )

                    target_timestamp = (
                        current_timestamp
                        - pd.Timedelta(minutes=30)
                    )

                    history_debug["current_timestamp"] = current_timestamp
                    history_debug["target_timestamp"] = target_timestamp

                    history_debug["complete_history_rows"] = len(
                        road_history
                    )

                    if not road_history.empty:

                        road_history["time_difference"] = (
                            road_history["timestamp"]
                            - target_timestamp
                        ).abs()

                        candidates = road_history[
                            road_history["time_difference"]
                            <= pd.Timedelta(minutes=2)
                        ].sort_values(
                            "time_difference"
                        )

                        if not candidates.empty:

                            previous_sensor = candidates.iloc[0]

                            history_debug["matched_difference"] = (
                                previous_sensor["time_difference"]
                            )

                        else:

                            closest = (
                                road_history
                                .sort_values("time_difference")
                                .head(3)
                            )

                            history_debug["closest_rows"] = closest[
                                ["timestamp", "time_difference"]
                            ].to_dict("records")

                else:

                    history_debug["error"] = (
                        "Required history columns are missing."
                    )

            else:

                history_debug["error"] = (
                    "History file must contain road_id and timestamp columns."
                )

        except Exception as exc:

            history_debug["error"] = str(exc)

    elif current_timestamp is None:

        history_debug["error"] = (
            "Current sensor timestamp is unavailable."
        )

    else:

        history_debug["error"] = (
            "Sensor history file does not exist."
        )

    if previous_sensor is not None:

        st.success(
            "Previous 30-minute sensor data retrieved automatically!"
        )

        st.caption(
            f"Matched reading: {previous_sensor['timestamp']} | "
            f"Difference from target: "
            f"{previous_sensor['time_difference']}"
        )

        p1, p2, p3 = st.columns(3)

        p1.metric(
            "Previous Traffic Volume",
            format_number(previous_sensor["traffic_volume"])
        )

        p1.metric(
            "Previous Vehicle Count",
            format_number(previous_sensor["vehicle_count"], 0)
        )

        p2.metric(
            "Previous Average Speed",
            format_number(previous_sensor["avg_speed"])
        )

        p2.metric(
            "Previous Occupancy",
            f"{format_number(previous_sensor['occupancy'])}%"
        )

        p3.metric(
            "Previous Travel Time",
            format_number(previous_sensor["travel_time"])
        )

        p3.metric(
            "Previous Congestion",
            str(previous_sensor["congestion_level"])
        )

    else:

        st.warning(
            "A valid previous reading could not be matched to the "
            "30-minute target. The prediction button will remain disabled "
            "until the required history record is available."
        )

        with st.expander(
            "Why is previous sensor data unavailable?"
        ):

            for key, value in history_debug.items():

                st.write(
                    f"**{key.replace('_', ' ').title()}:** {value}"
                )

            st.caption(
                "The selected road must have a complete historical sensor "
                "record within ±2 minutes of the current sensor timestamp "
                "minus 30 minutes."
            )

    st.divider()
    st.subheader("🤖 Future Accident Prediction")

    predict_disabled = previous_sensor is None

    if st.button(
        "Predict Future Accident Risk",
        disabled=predict_disabled,
        type="primary"
    ):

        try:

            accident_features = {
                **common,

                "traffic_volume": traffic_volume,
                "vehicle_count": vehicle_count,
                "avg_speed": avg_speed,
                "occupancy": occupancy,
                "travel_time": travel_time,
                "congestion_level": congestion,

                "prev_traffic_volume": float(
                    previous_sensor["traffic_volume"]
                ),

                "prev_vehicle_count": float(
                    previous_sensor["vehicle_count"]
                ),

                "prev_avg_speed": float(
                    previous_sensor["avg_speed"]
                ),

                "prev_occupancy": float(
                    previous_sensor["occupancy"]
                ),

                "prev_travel_time": float(
                    previous_sensor["travel_time"]
                ),

                "prev_congestion_level": str(
                    previous_sensor["congestion_level"]
                )
            }

            features = pd.DataFrame(
                [accident_features]
            )

            if hasattr(
                models["accident"],
                "feature_names_in_"
            ):

                expected = list(
                    models["accident"].feature_names_in_
                )

                if set(expected).issubset(features.columns):

                    features = features[expected]

                else:

                    missing_features = [
                        column for column in expected
                        if column not in features.columns
                    ]

                    raise ValueError(
                        "Missing accident model features: "
                        + ", ".join(missing_features)
                    )

            probabilities = (
                models["accident"].predict_proba(features)[0]
            )

            probability = float(
                probabilities[1]
            )

            prediction = int(
                probability >= accident_threshold
            )

            # SAVE PREDICTION HISTORY
            save_prediction(
                road_id=road_id,
                prediction_type="Future Accident Risk",
                input_values=accident_features,
                predicted_result={
                    "probability": probability,
                    "classification": (
                        "Elevated" if prediction == 1
                        else "Below Threshold"
                    ),
                    "threshold": accident_threshold
                }
            )

            result_col1, result_col2 = st.columns(2)

            result_col1.metric(
                "Estimated Accident Probability",
                f"{probability * 100:.2f}%"
            )

            result_col2.metric(
                "Classification",
                "Elevated" if prediction == 1 else "Below Threshold"
            )

            st.progress(
                float(np.clip(probability, 0.0, 1.0))
            )

            if prediction == 1:

                st.error(
                    "The model indicates elevated accident risk according "
                    "to its experimental threshold."
                )

            else:

                st.success(
                    "The model prediction is below its experimental "
                    "accident-risk threshold."
                )

            st.success(
                "Prediction saved to history."
            )

            st.caption(
                f"Experimental classification threshold: "
                f"{accident_threshold:.2f}. "
                "This output must not be used as a real-world safety alert."
            )

        except Exception as exc:

            st.error(
                f"Future accident prediction failed: {exc}"
            )


# =========================================================
# 20. TRAFFIC MANAGEMENT RECOMMENDATIONS
# =========================================================

elif page == "Traffic Management Recommendations":

    st.header("🚦 Traffic Management Recommendations")

    st.write(
        "Prioritize roads and suggest traffic-management actions "
        "using the latest simulated sensor readings."
    )

    st.warning(
        "SIMULATION MODE: Recommendations use simulated sensor data "
        "and initial rule-based thresholds. They do not control real "
        "traffic signals or replace decisions by traffic authorities."
    )

    if sensor_df.empty:

        st.error(
            "No current sensor readings are available."
        )

    elif "congestion_level" not in sensor_df.columns:

        st.error(
            "The sensor readings file does not contain the "
            "'congestion_level' column."
        )

    else:

        try:

            from src.traffic_recommendations import generate_recommendations

            recommendations = generate_recommendations(
                sensor_df.copy()
            )

            if recommendations.empty:

                st.info(
                    "No recommendations are available."
                )

            else:

                high_count = int(
                    (recommendations["priority"] == "High").sum()
                )

                medium_count = int(
                    (recommendations["priority"] == "Medium").sum()
                )

                low_count = int(
                    (recommendations["priority"] == "Low").sum()
                )

                st.subheader("Road Priority Summary")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Total Roads",
                    len(recommendations)
                )

                c2.metric(
                    "High Priority",
                    high_count
                )

                c3.metric(
                    "Medium Priority",
                    medium_count
                )

                c4.metric(
                    "Low Priority",
                    low_count
                )

                st.divider()
                st.subheader("🚨 High-Priority Roads")

                high_roads = recommendations[
                    recommendations["priority"] == "High"
                ]

                if high_roads.empty:

                    st.success(
                        "No high-priority roads detected."
                    )

                else:

                    for _, row in high_roads.iterrows():

                        st.error(
                            f"**{row['road_name']} ({row['road_id']})**\n\n"
                            f"Traffic volume: "
                            f"{format_number(row['traffic_volume'])} | "
                            f"Speed: {format_number(row['avg_speed'])} | "
                            f"Occupancy: "
                            f"{format_number(row['occupancy'])}%\n\n"
                            f"**Recommended action:** "
                            f"{row['recommendation']}"
                        )

                st.divider()

                st.subheader(
                    "📋 Prioritized Road Management Table"
                )

                display_columns = [
                    "road_id",
                    "road_name",
                    "traffic_volume",
                    "avg_speed",
                    "occupancy",
                    "congestion_level",
                    "priority",
                    "recommendation"
                ]

                display_columns = [
                    column for column in display_columns
                    if column in recommendations.columns
                ]

                st.dataframe(
                    recommendations[display_columns],
                    width="stretch",
                    hide_index=True
                )

                st.divider()
                st.subheader("📥 Export Recommendations")

                csv_data = recommendations.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Traffic Recommendations CSV",
                    data=csv_data,
                    file_name="flowcast_traffic_recommendations.csv",
                    mime="text/csv"
                )

                st.caption(
                    "Priority levels are assigned by the rule-based "
                    "thresholds in src/traffic_recommendations.py. "
                    "Review recommendations before taking real-world action."
                )

        except Exception as exc:

            st.error(
                "Unable to generate traffic management recommendations: "
                f"{exc}"
            )


# =========================================================
# 21. PREDICTION HISTORY
# =========================================================

elif page == "Prediction History":

    st.header("📜 Prediction History")

    st.write(
        "View, filter and download previously generated AI predictions."
    )

    history_df = load_prediction_history()

    if history_df.empty:

        st.info(
            "No prediction history available yet. "
            "Generate predictions to see them here."
        )

    else:

        # -------------------------------------------------
        # SUMMARY
        # -------------------------------------------------

        st.subheader("Prediction Summary")

        total_predictions = len(history_df)

        unique_roads = history_df["road_id"].nunique()

        prediction_types = history_df[
            "prediction_type"
        ].nunique()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Predictions",
            total_predictions
        )

        c2.metric(
            "Roads Analyzed",
            unique_roads
        )

        c3.metric(
            "Prediction Types",
            prediction_types
        )

        st.divider()

        # -------------------------------------------------
        # FILTERS
        # -------------------------------------------------

        st.subheader("Filter Predictions")

        col1, col2 = st.columns(2)

        with col1:

            roads = ["All"] + sorted(
                history_df["road_id"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_road = st.selectbox(
                "Filter by Road",
                roads,
                key="history_road_filter"
            )

        with col2:

            types = ["All"] + sorted(
                history_df["prediction_type"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_type = st.selectbox(
                "Filter by Prediction Type",
                types,
                key="history_type_filter"
            )

        filtered_history = history_df.copy()

        if selected_road != "All":

            filtered_history = filtered_history[
                filtered_history["road_id"].astype(str)
                == selected_road
            ]

        if selected_type != "All":

            filtered_history = filtered_history[
                filtered_history["prediction_type"]
                == selected_type
            ]

        st.divider()

        # -------------------------------------------------
        # HISTORY TABLE
        # -------------------------------------------------

        st.subheader("📋 Prediction Records")

        display_columns = [
            "timestamp",
            "road_id",
            "prediction_type",
            "predicted_result"
        ]

        available_columns = [
            column for column in display_columns
            if column in filtered_history.columns
        ]

        if filtered_history.empty:

            st.info(
                "No records match the selected filters."
            )

        else:

            st.dataframe(
                filtered_history[available_columns],
                width="stretch",
                hide_index=True
            )

            st.caption(
                f"Showing {len(filtered_history)} prediction records."
            )

        st.divider()

        # -------------------------------------------------
        # PREDICTION DISTRIBUTION
        # -------------------------------------------------

        st.subheader("📊 Prediction Type Distribution")

        if not filtered_history.empty:

            type_counts = filtered_history[
                "prediction_type"
            ].value_counts()

            st.bar_chart(type_counts)

        st.divider()

        # -------------------------------------------------
        # EXPORT HISTORY
        # -------------------------------------------------

        st.subheader("📥 Export Prediction History")

        csv_data = filtered_history.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Prediction History CSV",
            data=csv_data,
            file_name="flowcast_prediction_history.csv",
            mime="text/csv"
        )

        st.divider()

        # -------------------------------------------------
        # CLEAR HISTORY
        # -------------------------------------------------

        with st.expander("⚠️ Clear Prediction History"):

            st.warning(
                "This permanently deletes all saved prediction records."
            )

            confirm_clear = st.checkbox(
                "I understand that this will delete all prediction records.",
                key="confirm_clear_history"
            )

            if st.button(
                "Clear Prediction History",
                disabled=not confirm_clear,
                key="clear_history_button"
            ):

                clear_prediction_history()

                st.success(
                    "Prediction history cleared successfully."
                )

                st.rerun()


# =========================================================
# 22. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748B;">
    **FlowCast** | AI-Powered Traffic Analytics and Prediction System
    </div>
    """,
    unsafe_allow_html=True
)
# FlowCast – AI-Powered Smart Traffic Flow Prediction Platform

## 1. Project Overview

FlowCast is an AI-driven traffic analytics and prediction platform designed to help city traffic authorities understand traffic conditions and make proactive traffic-management decisions.

The system uses historical traffic sensor data, weather information, and calendar features to predict traffic volume, congestion levels, travel time, and future accident risk.

It also provides simulated sensor monitoring, traffic-management recommendations, and prediction history through an interactive Streamlit dashboard.

## 2. Key Features

- **Traffic Volume Prediction:** Predicts traffic volume using a trained machine-learning model.
- **Congestion Prediction:** Classifies traffic conditions into congestion categories.
- **Travel Time Prediction:** Estimates travel time using historical traffic patterns.
- **Future Accident Risk Prediction:** Estimates the probability of an accident in the next 30 minutes.
- **Live Sensor Monitoring:** Displays automatically updated simulated traffic sensor readings.
- **Weather Integration:** Retrieves current weather information using Open-Meteo.
- **Traffic Management Recommendations:** Prioritizes roads based on traffic conditions and suggests management actions.
- **Prediction History:** Stores predictions and supports filtering and CSV export.
- **Interactive Dashboard:** Provides separate pages for traffic analysis and prediction modules.

## 3. Technologies Used

**Programming Language:** Python

**Machine Learning:**
- Scikit-learn
- Random Forest
- XGBoost

**Data Processing:**
- Pandas
- NumPy

**Visualization:**
- Matplotlib
- Seaborn
- Plotly

**Dashboard:** Streamlit

**Other Libraries:** Joblib, Requests

**Weather API:** Open-Meteo

## 4. Project Structure

```text
FlowCast/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── simulated/
│
├── models/
│
├── notebooks/
│
├── report/
│
├── reports/
│
├── src/
│
├── tests/
│
└── requirements.txt
```

## 5. Machine Learning Models

FlowCast includes four prediction modules:

1. Traffic Volume Prediction – Random Forest Regressor.
2. Congestion Prediction – Random Forest Classifier.
3. Travel Time Prediction – Random Forest Regressor.
4. Future Accident Risk Prediction – Experimental XGBoost Classifier.

### Model Evaluation Results

| Prediction Model | Evaluation Metric | Result |
|---|---|---:|
| Traffic Volume | R² Score | 0.9540 |
| Congestion | Accuracy | 84.54% |
| Travel Time | R² Score | 0.7980 |
| Future Accident Risk | PR-AUC | 0.0150 |

**Important:** Accident prediction is experimental. Its low precision-recall performance means it should not be used for real-world safety alerts or emergency decisions.

Travel-time predictions are expressed in the dataset's units because the original unit has not been independently verified.

## 6. Installation and Execution

### Prerequisites

- Python 3.13 or a compatible Python version.
- pip package manager.

### Step 1: Clone the Repository

```bash
git clone https://github.com/danaboinavarshini-2005/FlowCast.git
cd FlowCast
```

Replace the placeholder with the actual repository URL.

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate the Environment

Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Dashboard

```bash
streamlit run dashboard/app.py
```

### Step 6: Run the Sensor Simulator

Open another terminal, activate the virtual environment, and run:

```bash
python src/simulate_traffic_sensors.py --continuous --interval 10
```

The simulator generates updated traffic readings for the dashboard.

## 7. Weather Integration

FlowCast uses the Open-Meteo API to retrieve current weather information.

Weather details include:
- Temperature
- Humidity
- Rainfall
- Wind speed
- Weather condition

Internet access is required for retrieving current weather information.

## 8. Important Limitations

- Sensor readings are simulated and do not come from physical traffic sensors.
- Traffic-management recommendations initially use rule-based thresholds.
- Accident-risk prediction is experimental and has limited predictive performance.
- Weather information is retrieved from an external API.
- Model predictions are estimates based on available historical data and should not replace decisions made by traffic authorities.

## 9. Future Enhancements

- Integration with real-time traffic sensors.
- Improved accident-risk prediction using larger and more reliable datasets.
- Integration with real-time road and traffic APIs.
- Advanced deep-learning models for traffic forecasting.
- Deployment on a cloud platform.
- Integration with intelligent traffic signal systems.

## 10. Project Conclusion

FlowCast demonstrates how machine learning, traffic sensor analytics, weather information, and interactive dashboards can be combined to support intelligent traffic monitoring and prediction.

The platform provides a foundation for proactive traffic management and can be further enhanced through real-world sensor integration and improved predictive models.
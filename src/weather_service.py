import requests


def get_current_weather(latitude, longitude):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "weather_code",
            "wind_speed_10m"
        ],
        "timezone": "auto"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        current = data["current"]

        weather_code = current["weather_code"]

        weather_mapping = {
            0: "Clear",
            1: "Clear",
            2: "Cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Fog",
            51: "Rain",
            53: "Rain",
            55: "Rain",
            61: "Rain",
            63: "Rain",
            65: "Rain",
            80: "Rain",
            81: "Rain",
            82: "Rain"
        }

        weather_condition = weather_mapping.get(
            weather_code,
            "Cloudy"
        )

        return {
            "temperature": current["temperature_2m"],
            "humidity": current["relative_humidity_2m"],
            "rainfall": current["precipitation"],
            "rain": current["rain"],
            "weather_code": weather_code,
            "weather_condition": weather_condition,
            "wind_speed": current["wind_speed_10m"],
            "weather_time": current["time"]
        }

    except Exception as e:

        print("Weather API Error:", e)

        return None
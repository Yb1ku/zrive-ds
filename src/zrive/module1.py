import requests
import pandas as pd
import matplotlib.pyplot as plt


COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}

VARIABLES = ["temperature_2m", "precipitation", "windspeed_10m"]


def connect_to_api(url, params):
    """Connects to the API and returns the JSON response, status code, and headers"""

    response = requests.get(url, params=params)
    status_code = response.status_code
    headers = response.headers

    match status_code:
        case code if 200 <= code < 300:
            return response.json(), status_code, headers
        case code if 400 <= code < 500:
            raise Exception(f"Client error: {status_code}")
        case code if 500 <= code < 600:
            raise Exception(f"Server error: {status_code}")
        case _:
            raise Exception(f"Unexpected status code: {status_code}")


def get_data_meteo_api(url, city):
    """Fetches historical weather data for a given city from the Open-Meteo API"""

    lat, lon = COORDINATES[city]["latitude"], COORDINATES[city]["longitude"]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2010-01-01",
        "end_date": "2020-01-01",
        "hourly": VARIABLES,
    }
    data, _, _ = connect_to_api(url, params)
    metrics = data.get("hourly", {})
    return metrics


def prepare_data_to_plot(data, groupby="ME"):
    """Formats the data for plotting"""

    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df = df.resample(groupby).mean()
    df.index = df.index.strftime("%Y-%m")
    return df


def plot_data(data, city):
    """Plots the already formatted data"""

    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))

    ax1.plot(data.index, data["temperature_2m"], label="Temperature (°C)", color="red")
    ax2.plot(
        data.index, data["windspeed_10m"], label="Wind Speed (km/h)", color="green"
    )
    ax3.bar(
        data.index,
        data["precipitation"],
        label="Precipitation (mm)",
        color="blue",
        alpha=0.4,
    )

    ax1.set_xlabel("Date")
    ax1.set_xticks(data.index[::12])
    ax1.set_ylabel("Temperature (°C)", color="red")
    ax2.set_ylabel("Wind Speed (km/h)", color="green")
    ax3.set_ylabel("Precipitation (mm)", color="blue")

    plt.grid()
    plt.title(f"Historical Weather Data for {city}")
    plt.savefig("src/zrive/outputs/plot.png")


def main(city="Madrid"):
    url = "https://archive-api.open-meteo.com/v1/archive"

    data = get_data_meteo_api(url, city)
    data = prepare_data_to_plot(data)
    plot_data(data, city)


if __name__ == "__main__":
    main("Madrid")

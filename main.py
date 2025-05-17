from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import logging

import requests
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

import numpy as np

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)

# Load trained model
model = None
try:
    model = joblib.load("model.pkl")
    print("✅ Model Loaded Successfully!")
except FileNotFoundError:
    print("❌ model.pkl not found! Train and save the model first.")

# Weather Input schema
class WeatherInfo(BaseModel):
    wind_speed: float
    wind_dir: float
    wave_height: float
    wave_dir: float
    wave_period: float

class VesselSpec(BaseModel):
    vessel_speed: float
    vessel_length: float
    beam: float
    draft: float
    heading: float

#CorOrdinates
class Coordinate(BaseModel):
    latitude: float
    longitude: float

#API Request Input
class APIRequestPayload(BaseModel):
    vessel_spec: VesselSpec
    current_location: Coordinate
    destination_location: Coordinate

class WeatherFactorDetails:
    wind_speed: float
    wind_dir: float
    wave_height: float
    wave_dir: float
    wave_period: float
    latitude: float
    longitude: float
    weather_factor: float


# API Token and Headers
def get_intermediate_ports(current: Coordinate, destination: Coordinate):
    # url = "http://k8s-developer-08ea8755f7-901870132.ap-south-1.elb.amazonaws.com:8088/api/v2/distance"
    # token = "your_api_token_here"
    # entity_id = "your_entity_id_here"
    # user_id = "your_user_id_here"
    #
    # payload = {
    #     "vessel_location": {"latitude": current.latitude, "longitude": current.longitude},
    #     "port_location":  {"latitude": current.latitude, "longitude": current.longitude}
    # }
    # headers = {
    #     "Authorization": f"Bearer {token}",
    #     "Content-Type": "application/json",
    #     "entity_id": entity_id,
    #     "user_id": user_id
    # }
    # try:
    #     response = requests.post(url, headers=headers, json=payload, timeout=10)
    #     response.raise_for_status()
    #     data = response.json()
    #     coordinates = data.get("data", {}).get("points", {}).get("coordinates", [])
    #     if not coordinates:
    #         print("No waypoints found in response.")
    coordinates = [
        [12.646619, 101.147913], [12.590852, 101.144846], [12.395662, 101.21913],
        [10.246048, 103.5], [8.169472, 104.908817], [8.302803, 105.272939],
        [8.556436, 105.581669], [9.3564, 106.6436], [10.690019, 108.405398],
        [11.747736, 109.5], [23.56968, 118.550441], [23.586335, 118.56918],
        [23.900686, 118.922865], [24.215036, 119.27655], [24.529386, 119.630236],
        [24.843737, 119.983921], [25.158087, 120.337606], [25.51688, 120.576684],
        [25.903444, 120.834268], [26.290008, 121.091852], [26.676572, 121.349436],
        [26.852312, 121.463589], [27.242473, 121.71702], [27.632634, 121.970452],
        [27.883961, 122.111679], [28.287972, 122.338702], [28.691983, 122.565725],
        [29.095994, 122.792749], [29.12171, 122.802922], [29.548086, 122.971603],
        [29.974463, 123.140283], [30.40084, 123.308964], [30.621919, 123.402193],
        [30.757103, 123.422336], [31.016513, 123.395885], [32.970047, 126.471861],
        [34.645368, 129.021767], [45.763482, 141.842929], [45.865152, 143.612884],
        [49.930445, 155.693328], [50.794248, 157], [62.070107, 180],
        [64.158657, -180], [66.109342,-172.177131], [69.718604, -169.403781],
        [69.718604, -180], [77.763817, 105.614691]]
    waypoints = []
    for coord in coordinates:
        logging.info("Coordinates", coord)
        if len(coord) == 2:  # Ensure correct structure
            coordinate = Coordinate(longitude=coord[1], latitude=coord[0])
            waypoints.append(coordinate)
            logging.info("Coordinate", coordinate)
        else:
            print(f"⚠️ Skipping malformed coordinate: {coord}")
    return waypoints
    # except requests.exceptions.RequestException as e:
    #     print(f"Request failed: {e}")
    #     return []

def fetch_weather_data(current_location: Coordinate):
    api_key="9715deab1181ffbaecbd94c17df6788c"

    # Fetch wave data
    wave_url = "https://marine-api.open-meteo.com/v1/marine"
    marine_params = {
        "latitude": current_location.latitude,
        "longitude": current_location.longitude,
        "daily": ["wave_height_max", "wave_direction_dominant", "wave_period_max"],
        "forecast_days": 9
    }
    marine_response = requests.get(wave_url, params=marine_params)

    if marine_response.status_code != 200:
        raise Exception(f"Failed to fetch wave data: {marine_response.status_code} - {marine_response.text}")

    marine_data = marine_response.json().get("daily", {})
    num_days = len(marine_data.get("wave_height_max", []))

    if num_days != 9:
        raise ValueError(f"Expected 16 days of wave data, but received {num_days}")

    weather_data = []
    for i in range(num_days):
        wd =  WeatherInfo(wave_period = float(marine_data.get("wave_period_max", [0.0]*num_days)[i] or 1.0),
                          wave_dir = float(marine_data.get("wave_direction_dominant", [0.0]*num_days)[i] or 1.0),
                        wave_height = float(marine_data.get("wave_height_max", [0.0]*num_days)[i] or 1.0),
                          wind_dir = 1.0, wind_speed = 1)
        weather_data.append(wd)

    # Fetch wind data
    wind_url = f"https://api.openweathermap.org/data/3.0/onecall?lat={current_location.latitude}&lon={current_location.longitude}&exclude=minutely,hourly,alerts&units=metric&appid={api_key}"
    wind_response = requests.get(wind_url)

    if wind_response.status_code != 200:
        raise Exception(f"Failed to fetch wind data: {wind_response.status_code} - {wind_response.text}")

    wind_data = wind_response.json()
    for i, day in enumerate(wind_data.get('daily', [])):
        if i < num_days:
            wind_speed_ms= day.get("wind_speed", 1.0)
            wind_speed_knots=1.94384*wind_speed_ms
            weather_data[i].wind_speed=wind_speed_knots
            weather_data[i].wind_dir = day.get("wind_deg", 1)

    return weather_data

def relative_angle(weather_dir, vessel_heading):
    return abs(weather_dir - vessel_heading) % 360

# Calculate Weather Factor
def calculate_weather_factor(weather_info: WeatherInfo, vessel_spec: VesselSpec):


    wind_angle = relative_angle(weather_info.wind_dir, vessel_spec.heading)
    wave_angle = relative_angle(weather_info.wave_dir, vessel_spec.heading)

    wind_resistance_coeff = 0.0005 * vessel_spec.vessel_length / vessel_spec.beam
    wave_resistance_coeff = 0.002 * vessel_spec.draft

    def wind_effect_factor(wind_angle):
        if wind_angle < 90:
            return 1.0  # Headwind
        elif 90 <= wind_angle <= 150:
            return 0.5  # Crosswind
        else:
            return -0.2  # Tailwind

    wind_effect = wind_effect_factor(wind_angle)
    wind_resistance = wind_resistance_coeff * weather_info.wind_speed**2 * wind_effect
    wave_resistance = wave_resistance_coeff * weather_info.wave_height**2 * (1 + 0.1 * weather_info.wave_period) * math.cos(math.radians(wave_angle))

    weather_factor = (wind_resistance + wave_resistance)
    return round(weather_factor*10, 2)

#scaling weather factor
def scale_weather_factor(value, min_factor, max_factor, min_range=1, max_range=10):
    if max_factor == min_factor:
        return min_range  # Prevent division by zero by assigning the minimum scale value
    return min_range + (max_range - min_range) * (value - min_factor) / (max_factor - min_factor)

#Train ML Model
def train_and_predict_model(df_weather):

    if "scaled_weather_factor" in df_weather.columns:
        X = df_weather.drop(columns=["scaled_weather_factor"])  # Features
        y = df_weather["scaled_weather_factor"]  # Target

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize & Train the XGBRegressor model
        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        print("✅ Model trained successfully!")
    else:
        raise ValueError("❌ 'final_weather_factor' column not found in df_weather.")

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"\nModel Mean Squared Error: {mse:.2f}")
    print("\nPredicted Weather Factor: ",predictions[0])
    future_predictions=model.predict(X.sample(n=3,random_state=42)) #n is no. of days
    if isinstance(future_predictions, np.ndarray):
        future_predictions = future_predictions.tolist()
    else:
        future_predictions = [float(future_predictions)]
    joblib.dump(model, "model.pkl")
    return model,future_predictions

def process_weather_factor(input_model: APIRequestPayload):
    logging.info("process_weather_factor called")
    # Get intermediate ports
    intermediate_ports_df = get_intermediate_ports(input_model.current_location, input_model.destination_location)
    print("Intermediate Points:")
    print("\n", intermediate_ports_df)
    weather_data_list = []

    w_data = fetch_weather_data(input_model.current_location)

    for weather_entry in w_data:
        weather_info = weather_entry.model_copy()

        weather_factor_details = WeatherFactorDetails()
        weather_factor_details.latitude = input_model.current_location.latitude
        weather_factor_details.longitude = input_model.current_location.longitude
        weather_factor_details.weather_factor = calculate_weather_factor(weather_info, input_model.vessel_spec)

        # if isinstance(input_model.current_location, Coordinate):  # Check if it's a Coordinate object
        #     weather_entry_copy["LATITUDE"] = input_model.current_location.latitude
        #     weather_entry_copy["LONGITUDE"] = input_model.current_location.longitude
        # elif isinstance(input_model.current_location, (tuple, list)) and len(input_model.current_location) == 2:
        #     weather_entry_copy["LATITUDE"], weather_entry_copy["LONGITUDE"] = input_model.current_location
        # else:
        #     raise ValueError("Invalid current_location format. Expected a tuple (LAT, LON).")

        weather_data_list.append(weather_factor_details)

    # Fetch weather data for intermediate ports & calculate weather factor
    for row in intermediate_ports_df:
        weather_data = WeatherInfo
        is_error = bool
        try:
            weather_data = fetch_weather_data(row)
            is_error = False
        except Exception as e:
            print("Exception while fetch weather data")
            is_error = True

        if not is_error:
            for wd in weather_data:
                wd_copy = WeatherFactorDetails()
                wd_copy.longitude = row.longitude
                wd_copy.latitude = row.latitude
                wd_copy.wind_dir = wd.wind_dir
                wd_copy.wind_speed = wd.wind_speed
                wd_copy.wave_period = wd.wave_period
                wd_copy.wave_height = wd.wave_height
                wd_copy.wave_dir = wd.wave_dir
                wd_copy.weather_factor = calculate_weather_factor(wd, input_model.vessel_spec)
                weather_data_list.append(wd_copy)

    # Fetch weather data for destination & calculate weather factor
    weather_data = fetch_weather_data(input_model.destination_location)
    for wd in weather_data:
        wd_copy = WeatherFactorDetails()
        wd_copy.longitude = input_model.destination_location.longitude
        wd_copy.latitude = input_model.destination_location.latitude
        wd_copy.wind_dir = wd.wind_dir
        wd_copy.wind_speed = wd.wind_speed
        wd_copy.wave_period = wd.wave_period
        wd_copy.wave_height = wd.wave_height
        wd_copy.wave_dir = wd.wave_dir
        wd_copy.weather_factor = calculate_weather_factor(wd, input_model.vessel_spec)
        weather_data_list.append(wd_copy)

    result = None
    # Convert all collected data into DataFrame (Handle empty list case)
    if not weather_data_list:
        print("No weather data available.")
    else:
        df_weather = pd.DataFrame([vars(wd) for wd in weather_data_list])

        df_weather = df_weather.select_dtypes(include=["number"])

        # Ensure 'weather_factor' column exists
        if "weather_factor" in df_weather.columns and not df_weather["weather_factor"].isnull().all():
            # Compute min and max weather factor for normalization
            min_factor = df_weather["weather_factor"].min()
            max_factor = df_weather["weather_factor"].max()

            # Handle case where min_factor == max_factor (avoid division by zero)
            if min_factor == max_factor:
                df_weather["scaled_weather_factor"] = 1.0  # Assign default scaled value
            else:
                df_weather["scaled_weather_factor"] = df_weather["weather_factor"].apply(
                    lambda x: scale_weather_factor(x, min_factor, max_factor)
                )

            # Compute final weather factor as the mean
            final_weather_factor = df_weather["scaled_weather_factor"].mean()
            df_weather["final_weather_factor"] = final_weather_factor

            print("\nFinal Weather Data for All Points:\n", df_weather)

            # Calling ML model
            model, future_factors = train_and_predict_model(df_weather)

            # Compute and return the mean of `future_factors`
            if isinstance(future_factors, (list, np.ndarray)) and len(future_factors) > 0:
                future_mean = sum(future_factors) / len(future_factors)
            else:
                future_mean = None  # Handle empty list case

            print("\nPredicted Weather Factors for Next 3 Days:", future_factors)
            print("Mean Future Weather Factor:", future_mean)

            # Return the mean future weather factor
            result = {
                "final_weather_factor": final_weather_factor,
                "mean_future_factor": future_mean
            }
        else:
            print("Weather_factor column is missing or all values are NaN.")
            result = None

    # Return the result (dictionary containing mean values)
    return result


@app.post("/predict/")
def predict_weather_factor(input_data: APIRequestPayload):
    if model is None:
        return {"error": "ML model not loaded. Please train and save the model first."}
    return process_weather_factor(input_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

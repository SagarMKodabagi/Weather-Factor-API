# Weather-Factor-API
The Weather Factor API predicts the impact of weather on maritime routes using machine learning. It analyzes conditions like wind and waves along a vessel's path and returns a weather correction factor to support voyage planning and charter rate estimation.

# Features
Predicts weather impact using wind, waves, current, and other marine conditions.

Built with FastAPI for fast and reliable performance.

Integrated with an XGBoost regression model trained on historical marine data.

Accepts port locations, vessel specs, and weather data as input.

# Tech Stack
Python

FastAPI

XGBoost

Pandas, NumPy

Uvicorn (for local dev server)

# Usage
Clone the repository

# bash
Copy code
git clone https://github.com/your-username/weather-factor-api.git
cd weather-factor-api
Install dependencies

# bash
Copy code
pip install -r requirements.txt
Run the API

# bash
Copy code
uvicorn main:app --reload
Access docs at:

# bash
Copy code
http://localhost:8000/docs

# Sample Request
json
Copy code
POST /predict-weather-factor
{
  "start_port": "Rotterdam",
  "end_port": "Singapore",
  "vessel_dwt": 50000,
  "departure_date": "2025-06-01"
}

# Output
json
Copy code
{
  "weather_factor": 1.12,
  "message": "Weather factor successfully predicted"
}

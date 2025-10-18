"""
F1 Race Predictor from Historical Dataset (with Automated Weather)

This script loads the pre-prepared 'f1_features_dataset.csv', trains a model,
fetches a live weather forecast, and predicts the outcome of a new race based
on user-provided qualifying results.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import fastf1
import requests
from datetime import datetime
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")


OPENWEATHER_API_KEY = "2b4a875e9486ebdf58990405dda6e721" 

# 2. Set the year and event name for your prediction
PREDICTION_YEAR = 2025
PREDICTION_EVENT_NAME = "United States Grand Prix"

# 3. Enter the qualifying results for the race you want to predict
QUALIFYING_RESULTS = {
    'Driver': ['VER', 'NOR', 'PIA', 'HUL', 'RUS', 'ALO', 'SAI', 'HAM', 'ALB', 'LEC', 
               'ANT', 'HAD', 'GAS', 'STR', 'LAW', 'BEA', 'COL', 'TSU', 'OCO', 'BOR'],
    'TeamName': ['Red Bull Racing', 'McLaren', 'McLaren', 'Kick Sauber', 'Mercedes', 'Aston Martin',
                 'Williams', 'Ferrari', 'Williams', 'Ferrari', 'Mercedes', 'Racing Bulls', 'Alpine',
                 'Aston Martin', 'Racing Bulls', 'Haas F1 Team', 'Alpine', 'Red Bull Racing',
                 'Haas F1 Team', 'Kick Sauber'],
    'QualiRank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
}

# QUALIFYING_RESULTS = {
#     'Driver': ['RUS', 'VER', 'PIA', 'ANT', 'NOR', 'HAM', 'LEC', 'HAD', 'BEA', 'ALO',
#                'HUL', 'LAW', 'TSU', 'BOR', 'STR', 'COL', 'OCO', 'GAS', 'ALB', 'SAI'],
#     'TeamName': ['Mercedes', 'Red Bull Racing', 'McLaren', 'Mercedes', 'McLaren', 
#                  'Ferrari', 'Ferrari', 'Racing Bulls', 'Haas F1 Team', 'Aston Martin',
#                  'Kick Sauber', 'Racing Bulls', 'Red Bull Racing', 'Kick Sauber', 
#                  'Aston Martin', 'Alpine', 'Haas F1 Team', 'Alpine', 'Williams', 'Williams'],
#     'QualiRank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# }

CIRCUIT_COORDINATES = {
    'Sakhir': (26.0325, 50.5106), 'Jeddah': (21.6319, 39.1044), 'Melbourne': (-37.8497, 144.9683),
    'Suzuka': (34.8431, 136.5411), 'Shanghai': (31.3389, 121.2203), 'Miami': (25.9581, -80.2389),
    'Imola': (44.3439, 11.7167), 'Monte-Carlo': (43.7347, 7.4206), 'Montréal': (45.5000, -73.5228),
    'Barcelona': (41.5700, 2.2611), 'Spielberg': (47.2197, 14.7647), 'Silverstone': (52.0786, -1.0169),
    'Budapest': (47.5789, 19.2486), 'Spa-Francorchamps': (50.4372, 5.9714), 'Zandvoort': (52.3888, 4.5409),
    'Monza': (45.6156, 9.2811), 'Baku': (40.3725, 49.8533), 'Singapore': (1.2914, 103.8644),
    'Austin': (30.1328, -97.6411), 'Mexico City': (19.4042, -99.0907), 'São Paulo': (-23.7036, -46.6997),
    'Las Vegas': (36.1147, -115.1728), 'Lusail': (25.4900, 51.4542), 'Yas Island': (24.4672, 54.6031)
}

def get_weather_forecast(api_key: str, year: int, event_name: str) -> dict:
    """Fetches weather forecast for a given F1 event."""
    print(f"\nFetching event details for {year} {event_name}...")
    try:
        event = fastf1.get_event(year, event_name)
        circuit_location = event['Location']
        race_date = event.get_race().date

        if circuit_location not in CIRCUIT_COORDINATES:
            print(f"-> Error: Coordinates for '{circuit_location}' not found.")
            return None
        
        lat, lon = CIRCUIT_COORDINATES[circuit_location]
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url)
        response.raise_for_status()
        forecast_data = response.json()

        race_datetime = datetime.combine(race_date.date(), datetime.min.time()).replace(hour=14)
        closest_forecast = min(forecast_data['list'], key=lambda x: abs(datetime.fromtimestamp(x['dt']) - race_datetime))
        
        weather = {
            'AirTemp': closest_forecast['main']['temp'],
            'TrackTemp': closest_forecast['main']['temp'] + 7, # Approximation
            'Humidity': closest_forecast['main']['humidity'],
            'WindSpeed': closest_forecast['wind']['speed']
        }
        print(f"-> Weather forecast for race day: {weather['AirTemp']:.1f}°C, Humidity {weather['Humidity']}%")
        return weather
    except Exception as e:
        print(f"-> Could not get weather forecast: {e}")
        return None

if __name__ == "__main__":
    DATASET_FILE = 'f1_features_dataset.csv'
    try:
        df = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        print(f"Error: Dataset file '{DATASET_FILE}' not found.")
        print("Please run 'prepare_f1_data.py' first to generate the data.")
        exit()

    X = df.drop(columns=['RacePosition', 'Year', 'EventName', 'FullName'])
    y = df['RacePosition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1)
    
    # --- THIS IS THE FIX ---
    # The problematic arguments have been removed from the .fit() method
    # to ensure compatibility with older versions of the XGBoost library.
    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)], 
              verbose=False)

    print("\n--- Model Self-Evaluation ---")
    predictions_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions_test)
    print(f"Model's Mean Absolute Error on test data: {mae:.4f}")
    print(f"This means the model is, on average, off by ~{mae:.2f} positions.")

    weather_forecast = get_weather_forecast(OPENWEATHER_API_KEY, PREDICTION_YEAR, PREDICTION_EVENT_NAME)
    if weather_forecast is None:
        print("-> Using default weather values as fallback.")
        weather_forecast = {'AirTemp': 25.0, 'TrackTemp': 32.0, 'Humidity': 60.0, 'WindSpeed': 2.0}

    new_race_df = pd.DataFrame(QUALIFYING_RESULTS)
    for key, value in weather_forecast.items():
        new_race_df[key] = value

    print("\nPredicting outcome for the new race...")
    new_race_encoded = pd.get_dummies(new_race_df, columns=['TeamName'])
    
    new_race_aligned = new_race_encoded.reindex(columns=X_train.columns, fill_value=0)
    
    predictions = model.predict(new_race_aligned)

    results_df = new_race_df[['Driver', 'TeamName', 'QualiRank']].copy()
    results_df['PredictedValue'] = predictions
    results_df['PredictedPosition'] = results_df['PredictedValue'].rank().astype(int)

    print("\n🏁 Predicted Race Results 🏁")
    print("-----------------------------------")
    print(results_df.sort_values('PredictedPosition').drop(columns=['PredictedValue']).to_string(index=False))
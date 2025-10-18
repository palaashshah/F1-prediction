import fastf1
import pandas as pd
from datetime import datetime
import os

def create_f1_dataset(years, output_csv_path):
    """
    Fetches detailed data for all F1 sessions (Race, Qualifying, Practices, Sprints)
    and their corresponding weather data for the specified years. Compiles it into a single CSV file.

    Args:
        years (list): A list of integers representing the seasons to fetch.
        output_csv_path (str): The file path to save the final CSV.
    """
    # --- Setup ---
    cache_path = 'fastf1_cache'
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    
    try:
        fastf1.Cache.enable_cache(cache_path)
        print(f"FastF1 cache enabled at: {os.path.abspath(cache_path)}")
    except Exception as e:
        print(f"Error enabling cache: {e}. Please ensure the directory is writable.")

    all_session_results = []
    
    # Get today's date (UTC) to filter out future races
    today = pd.to_datetime(datetime.utcnow().date())

    for year in years:
        print(f"\n{'='*40}")
        print(f"🏎️  Fetching schedule for {year} season...")
        print(f"{'='*40}")

        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"Could not fetch schedule for {year}. Reason: {e}")
            continue

        for index, event in schedule.iterrows():
            event_date = event['EventDate'].date()
            if pd.to_datetime(event_date) > today:
                print(f"\nSkipping future event: {event['EventName']} {year}")
                continue

            print(f"\nProcessing Event: {event['EventName']} {year}")

            session_names = ['Race', 'Qualifying', 'Sprint', 'Practice 1', 'Practice 2', 'Practice 3']
            
            for session_name in session_names:
                try:
                    session = fastf1.get_session(year, event['RoundNumber'], session_name)
                    
                    # MODIFICATION: Now loading weather data by setting weather=True
                    session.load(telemetry=False, weather=True, messages=False)

                    if session.results is None or session.results.empty:
                        print(f"  - No results for '{session_name}'. Skipping.")
                        continue
                    
                    print(f"  - Successfully loaded results for '{session_name}'.")

                    session_df = session.results.copy()
                    
                    # --- NEW: Add Weather Data ---
                    # Calculate average weather conditions for the session and add them to the results
                    if session.weather_data is not None and not session.weather_data.empty:
                        weather_df = session.weather_data
                        # Convert relevant columns to numeric, coercing errors
                        weather_cols = ['AirTemp', 'TrackTemp', 'Humidity', 'WindSpeed']
                        for col in weather_cols:
                            weather_df[col] = pd.to_numeric(weather_df[col], errors='coerce')
                        
                        # Calculate and add mean values
                        session_df['AvgAirTemp'] = weather_df['AirTemp'].mean()
                        session_df['AvgTrackTemp'] = weather_df['TrackTemp'].mean()
                        session_df['AvgHumidity'] = weather_df['Humidity'].mean()
                        session_df['AvgWindSpeed'] = weather_df['WindSpeed'].mean()
                        print("    - Added average weather data.")
                    else:
                        # If no weather data is available, fill with None/NaN
                        session_df['AvgAirTemp'] = None
                        session_df['AvgTrackTemp'] = None
                        session_df['AvgHumidity'] = None
                        session_df['AvgWindSpeed'] = None
                        print("    - No weather data available for this session.")
                    
                    session_df['Year'] = year
                    session_df['EventName'] = event['EventName']
                    session_df['SessionName'] = session_name
                    
                    all_session_results.append(session_df)

                except Exception as e:
                    if "session type not found" in str(e).lower():
                         print(f"  - Session '{session_name}' does not exist for this event. Skipping.")
                    else:
                        print(f"  - Could not process '{session_name}'. Reason: {e}")

    if not all_session_results:
        print("\nNo data was processed. The output file will not be created.")
        return

    print("\nCompiling all data into a single DataFrame...")
    final_df = pd.concat(all_session_results, ignore_index=True)

    # Reorder columns for better readability
    id_cols = ['Year', 'EventName', 'SessionName', 'FullName', 'TeamName', 'Position']
    weather_cols = ['AvgAirTemp', 'AvgTrackTemp', 'AvgHumidity', 'AvgWindSpeed']
    data_cols = [col for col in final_df.columns if col not in id_cols + weather_cols]
    final_df = final_df[id_cols + weather_cols + data_cols]
    
    final_df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Success! Data including weather saved to '{output_csv_path}'")


if __name__ == "__main__":
    target_years = [2025]
    output_filename = "f1_all_sessions_data_2025.csv"
    
    create_f1_dataset(target_years, output_filename)


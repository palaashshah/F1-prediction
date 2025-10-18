"""
F1 Data Preparation Script (Simplified Logic)

This script builds a clean, simple, and robust dataset for F1 prediction.
It ignores all unreliable practice and sprint data, focusing only on the most
predictive features: Qualifying results and Race weather.
"""
import pandas as pd
import glob

def prepare_simplified_f1_data(filepath_pattern, output_path):
    """
    Loads raw data and builds a simplified, clean feature-engineered DataFrame.
    """
    # --- Step 1: Load and filter data ---
    print("Step 1: Loading and filtering raw data...")
    csv_files = glob.glob(filepath_pattern)
    if not csv_files:
        raise FileNotFoundError(f"No files found for pattern: '{filepath_pattern}'.")
    
    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    
    # Keep ONLY Race and Qualifying sessions
    sessions_to_keep = ['Race', 'Qualifying']
    df = df[df['SessionName'].isin(sessions_to_keep)]
    print(f"-> Loaded and filtered down to {len(df)} Race/Qualifying records.")

    # Get a unique list of race events to iterate over
    race_events = df[df['SessionName'] == 'Race'][['Year', 'EventName']].drop_duplicates().sort_values(by=['Year', 'EventName'])
    
    final_data_rows = []

    print("\nStep 2: Building simplified dataset for each race event...")
    # --- Step 2: Iterate through each event ---
    for _, row in race_events.iterrows():
        year, event_name = row['Year'], row['EventName']
        
        event_df = df[(df['Year'] == year) & (df['EventName'] == event_name)]
        
        race_session = event_df[event_df['SessionName'] == 'Race']
        quali_session = event_df[event_df['SessionName'] == 'Qualifying']

        drivers_in_event = quali_session['FullName'].unique()

        # --- Step 3: Build a row for each driver ---
        for driver in drivers_in_event:
            driver_row = {'Year': year, 'EventName': event_name, 'FullName': driver}

            team_info = quali_session[quali_session['FullName'] == driver]
            driver_row['TeamName'] = team_info['TeamName'].iloc[0] if not team_info.empty else 'Unknown'

            race_info = race_session[race_session['FullName'] == driver]
            driver_row['RacePosition'] = race_info['Position'].iloc[0] if not race_info.empty else None

            quali_info = quali_session[quali_session['FullName'] == driver]
            driver_row['QualiRank'] = quali_info['Position'].iloc[0] if not quali_info.empty else 20
            
            driver_row['AirTemp'] = race_session['AvgAirTemp'].mean()
            driver_row['TrackTemp'] = race_session['AvgTrackTemp'].mean()
            driver_row['Humidity'] = race_session['AvgHumidity'].mean()
            driver_row['WindSpeed'] = race_session['AvgWindSpeed'].mean()

            final_data_rows.append(driver_row)

    # --- Step 4: Create and clean final DataFrame ---
    print("\nStep 3: Assembling final DataFrame...")
    final_df = pd.DataFrame(final_data_rows)
    final_df.dropna(subset=['RacePosition'], inplace=True)
    final_df.fillna(20, inplace=True)
    final_df = pd.get_dummies(final_df, columns=['TeamName'], prefix='Team', drop_first=True)
    
    final_df.to_csv(output_path, index=False)
    

if __name__ == "__main__":
    try:
        prepare_simplified_f1_data(
            filepath_pattern='f1_all_sessions_data_20*.csv',
            output_path='f1_features_dataset.csv'
        )
        print("\n" + "="*50)
        print("✅ SUCCESS! Simplified dataset created.")
        print(f"   Saved to 'f1_features_dataset.csv'")
        print("="*50)
    except Exception as e:
        print(f"\nAn error occurred: {e}")


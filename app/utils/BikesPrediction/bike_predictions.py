import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
import requests
from math import radians, sin, cos, sqrt, atan2
from flask import Flask, jsonify
from flask_cors import CORS
import os
import joblib
from pathlib import Path

# Configuration
app = Flask(__name__)
CORS(app)
cwd = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(cwd, "saved_models")
FORCE_RETRAIN = False  # Change to True to force retrain models
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

EVENTS_API_URL = "https://failteireland.azure-api.net/opendata-api/v2/events"
FORECAST_DAYS = 7
EVENT_RADIUS_KM = 1.0
EVENT_IMPACT_FACTOR = 0.8

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * (2 * atan2(sqrt(a), sqrt(1-a)))

def fetch_events_bikes():
    events = []
    try:
        response = requests.get(EVENTS_API_URL, timeout=10)
        if response.ok:
            for event in response.json().get('value', []):
                try:
                    start = pd.to_datetime(event['startDate'], utc=True).astimezone('Europe/Dublin').date()
                    end = pd.to_datetime(event['endDate'], utc=True).astimezone('Europe/Dublin').date()
                    events.append({
                        'dates': pd.date_range(start, end).date.tolist(),
                        'lat': event['location']['geo']['latitude'],
                        'lon': event['location']['geo']['longitude']
                    })
                except KeyError:
                    continue
    except Exception as e:
        print(f"Event fetch error: {e}")
    return events

def load_and_preprocess_data(cwd):
    file_paths = [f"{cwd}/dublin-bikes_station_status_{month}2024.csv" 
                  for month in ['05', '06', '07', '08', '09']]
    
    for f in file_paths:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Data file not found: {f}")

    df = pd.concat([pd.read_csv(f) for f in file_paths], ignore_index=True)
    df["last_reported"] = pd.to_datetime(df["last_reported"])
    
    stations = df[['station_id', 'name', 'lat', 'lon']].drop_duplicates('station_id')
    
    daily = df.groupby(['station_id', pd.Grouper(key='last_reported', freq='D')]).agg(
        bikes=('num_bikes_available', 'mean'),
        stands=('num_docks_available', 'mean')
    ).reset_index()
    
    daily['weekday'] = daily['last_reported'].dt.weekday
    daily['weekend'] = daily['weekday'] >= 5
    
    return daily, stations

def generate_predictions(daily_data, stations, events):
    current_date = datetime.now().date()
    future_dates = [current_date + timedelta(days=i) for i in range(1, FORECAST_DAYS+1)]
    predictions = []
    mae_values = []

    for station_id in daily_data['station_id'].unique():
        model_path = os.path.join(MODEL_DIR, f"station_{station_id}.joblib")
        station_data = daily_data[daily_data['station_id'] == station_id].set_index('last_reported')
        lat, lon = stations[stations['station_id'] == station_id][['lat', 'lon']].values[0]
        
        # Model loading/training
        try:
            if not FORCE_RETRAIN and os.path.exists(model_path):
                model_fit = joblib.load(model_path)
                print(f"Loaded existing model for station {station_id}")
            else:
                model = SARIMAX(station_data['bikes'], order=(1,1,1), seasonal_order=(1,1,1,7))
                model_fit = model.fit(disp=False)
                joblib.dump(model_fit, model_path)
                print(f"Trained and saved model for station {station_id}")
            
            forecast = model_fit.forecast(steps=FORECAST_DAYS)
        except Exception as e:
            print(f"Model error for station {station_id}: {e}")
            forecast = pd.Series([np.nan]*FORECAST_DAYS)
            mae_values.append(np.nan)
            continue

        # Event impact calculation
        affected_dates = set()
        for event in events:
            if calculate_distance(lat, lon, event['lat'], event['lon']) <= EVENT_RADIUS_KM:
                affected_dates.update(set(future_dates) & set(event['dates']))
        
        # Apply adjustments
        adjusted = [v * (EVENT_IMPACT_FACTOR if d in affected_dates else 1) 
                   for v, d in zip(forecast.values, future_dates)]
        
        predictions.append(pd.DataFrame({
            'station_id': station_id,
            'date': future_dates,
            'bikes': adjusted,
            'stands': station_data['stands'].iloc[-1]
        }))
        
        mae_values.append(mean_absolute_error(station_data['bikes'], model_fit.fittedvalues))

    return pd.concat(predictions), np.nanmean(mae_values)

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    try:
        daily_data, stations = load_and_preprocess_data(cwd)
        events = fetch_events_bikes()
        predictions_df, mae = generate_predictions(daily_data, stations, events)
        results = predictions_df.merge(stations, on='station_id')
        
        output = []
        for station_id, group in results.groupby('station_id'):
            output.append({
                'station_id': int(station_id),
                'station_name': group['name'].iloc[0],
                'latitude': float(group['lat'].iloc[0]),
                'longitude': float(group['lon'].iloc[0]),
                'predictions': [{
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'bikes': round(row['bikes'], 1),
                    'stands': round(row['stands'], 1)
                } for _, row in group.iterrows()]
            })
        
        return jsonify({
            'status': 'success',
            'data': output,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'mae': round(mae, 2),
                'stations_count': len(output),
                'forecast_days': FORECAST_DAYS
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
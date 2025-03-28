# from flask import Flask, render_template_string, jsonify, request

from openrouteservice import Client

from datetime import datetime, timedelta
from functools import lru_cache


from flask import Blueprint, request, jsonify
 # Assuming get_route is imported here
import requests
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
from math import radians, sin, cos, sqrt, asin
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Configuration (should match your main app)
cwd = os.path.dirname(os.path.abspath(__file__))
# print("cccccccccccccccccwwwwwwwwwwwwwwwwwddddddddddddd", cwd)
# MODEL_DIR = os.path.join(cwd, "/BikesPrediction/saved_models")
MODEL_DIR = (f"{cwd}/BikesPrediction/saved_models")
print("MODEL_DIR", MODEL_DIR)
FORECAST_DAYS = 7
EVENT_RADIUS_KM = 1.0
EVENT_IMPACT_FACTOR = 0.8
DUBLIN_BIKES_API_URL = "https://api.jcdecaux.com/vls/v1/stations?contract=dublin&apiKey=5c8736d088fe1fc1388a8982d30072aa64aaf032"
REAL_TIME_API_URL = "https://api.nationaltransport.ie/gtfsr/v2/Vehicles?format=json"
EVENTS_API_URL = "https://failteireland.azure-api.net/opendata-api/v2/events"
SUBSCRIPTION_KEY = "988e6458483340cd8599cacbbe75acb3"
MONITORS_API_URL = "https://data.smartdublin.ie/sonitus-api/api/monitors"
DATA_API_URL = "https://data.smartdublin.ie/sonitus-api/api/data"
API_USERNAME = "dublincityapi"
API_PASSWORD = "Xpa5vAQ9ki"
ORS_API_KEY = "5b3ce3597851110001cf62485c2afa0d8bdf46a08bec95a1d1e35e69"
ors_client = Client(key=ORS_API_KEY)
headers = {"x-api-key": SUBSCRIPTION_KEY}
BIKE_NOTIFICATION_THRESHOLDS = {
    'percentage': 0.3,  # 30% difference
    'absolute': 5,      # Minimum 5 bikes difference
    'min_capacity': 10  # Only consider stations with at least 10 bike capacity
}

@lru_cache(maxsize=128)
def get_air_pollution_data():
    """Get current air quality index (AQI) data for Dublin based on PM1 readings."""
    try:
        monitors = fetch_monitors()
        if not monitors:
            return {"status": "error", "message": "No air quality monitors found"}
        
        # Get PM1 data from all monitors
        monitor_data = fetch_all_monitor_data(monitors)
        if not monitor_data:
            return {"status": "error", "message": "No PM1 data available"}
        
        # Calculate AQI for each reading
        aqi_points = []
        for lat, lon, pm1 in monitor_data:
            aqi = calculate_aqi(pm1)
            aqi_points.append({
                "latitude": lat,
                "longitude": lon,
                "aqi": round(aqi, 1),
                "pm1": pm1,
                "health_impact": "Good" if aqi <= 50 else 
                               "Moderate" if aqi <= 100 else
                               "Unhealthy for Sensitive Groups" if aqi <= 150 else
                               "Unhealthy"
            })
        
        return {
            "status": "success",
            "data": aqi_points,
            "generated_at": datetime.now().isoformat(),
            "metadata": {
                "monitors_count": len(aqi_points),
                "average_aqi": round(sum(p["aqi"] for p in aqi_points)/len(aqi_points), 1)
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to get air pollution data: {str(e)}"}

# NOISE POLLUTION FUNCTIONS
def fetch_noise_monitor_data(monitor_id):
    """Fetch noise level data for a specific monitor."""
    try:
        params = {
            "username": API_USERNAME,
            "password": API_PASSWORD,
            "monitor": monitor_id,
            "start": int(datetime.now().timestamp()) - 3600,  # Last hour
            "end": int(datetime.now().timestamp())
        }
        response = requests.post(DATA_API_URL, data=params)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                for entry in data:
                    if "leq" in entry:  # Equivalent sound level
                        return float(entry["leq"])
        return None
    except Exception as e:
        print(f"Error fetching noise data for monitor {monitor_id}: {str(e)}")
        return None

def fetch_all_noise_monitor_data(monitors):
    """Fetch noise data from all monitors in parallel."""
    monitor_data = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_monitor = {
            executor.submit(fetch_noise_monitor_data, monitor['serial_number']): monitor 
            for monitor in monitors
        }
        for future in as_completed(future_to_monitor):
            monitor = future_to_monitor[future]
            try:
                leq = future.result()
                if leq is not None:
                    lat = float(monitor.get("latitude"))
                    lon = float(monitor.get("longitude"))
                    monitor_data.append((lat, lon, leq))
            except Exception as exc:
                print(f"Noise monitor {monitor['serial_number']} error: {exc}")
    return monitor_data

@lru_cache(maxsize=128)
def get_noise_pollution_data():
    """Get current noise pollution levels across Dublin."""
    try:
        monitors = fetch_monitors()
        if not monitors:
            return {"status": "error", "message": "No noise monitors found"}
        
        noise_data = fetch_all_noise_monitor_data(monitors)
        if not noise_data:
            return {"status": "error", "message": "No noise data available"}
        
        return {
            "status": "success",
            "data": [{
                "latitude": lat,
                "longitude": lon,
                "noise_level": round(leq, 1),
                "description": "Quiet" if leq <= 50 else 
                             "Moderate" if leq <= 65 else 
                             "Loud" if leq <= 80 else 
                             "Very Loud"
            } for lat, lon, leq in noise_data],
            "generated_at": datetime.now().isoformat(),
            "metadata": {
                "monitors_count": len(noise_data),
                "average_db": round(sum(d[2] for d in noise_data)/len(noise_data), 1)
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to get noise pollution data: {str(e)}"}

def get_route(start, end, avoid_areas=None, profile="driving-car"):
    try:
        route_params = {
            "coordinates": [start, end],
            "profile": profile,
            "format": "geojson"
        }
        if avoid_areas:
            route_params["options"] = {"avoid_polygons": avoid_areas}
        route = ors_client.directions(**route_params)
        return [(point[1], point[0]) for point in route['features'][0]['geometry']['coordinates']]
    except Exception as e:
        print(f"Error fetching route: {e}")
        return []

# Helper Functions
@lru_cache(maxsize=128)
def fetch_bus_locations():
    """Fetch bus locations for heatmap."""
    try:
        response = requests.get(REAL_TIME_API_URL, headers=headers)
        response.raise_for_status()
        data = response.json()
        bus_locations = []
        for entity in data.get("entity", []):
            vehicle = entity.get("vehicle", {})
            position = vehicle.get("position", {})
            lat = position.get("latitude")
            lon = position.get("longitude")
            if lat is not None and lon is not None:
                bus_locations.append([lat, lon, 1])  # Intensity set to 1
        return bus_locations
    except Exception as e:
        return {"status": "error", "message": f"Failed to fetch bus locations: {str(e)}"}

def get_normal_route(start_lat, start_lon, end_lat, end_lon):
    """Get a normal route between two points."""
    try:
        start_coord = (start_lon, start_lat)
        end_coord = (end_lon, end_lat)
        route = get_route(start_coord, end_coord)
        return {"route": route}
    except Exception as e:
        return {"status": "error", "message": f"Failed to get normal route: {str(e)}"}

def fetch_realtime_data():
    """Fetch real-time bus data."""
    try:
        response = requests.get(REAL_TIME_API_URL, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None

def create_bus_congestion_avoidance_areas(bus_locations, threshold=3, grid_size=0.0005):
    """Create polygons to avoid bus congestion."""
    from collections import defaultdict
    if not bus_locations:
        return None
    grid_counts = defaultdict(int)
    for lat, lon, _ in bus_locations:
        grid_x = round(lon / grid_size) * grid_size
        grid_y = round(lat / grid_size) * grid_size
        grid_counts[(grid_x, grid_y)] += 1
    avoidance_polygons = []
    for (grid_x, grid_y), count in grid_counts.items():
        if count >= threshold:
            radius = 0.0006
            polygon = {
                "type": "Polygon",
                "coordinates": [[
                    [grid_x - radius, grid_y - radius],
                    [grid_x + radius, grid_y - radius],
                    [grid_x + radius, grid_y + radius],
                    [grid_x - radius, grid_y + radius],
                    [grid_x - radius, grid_y - radius]
                ]]
            }
            avoidance_polygons.append(polygon)
    return {"type": "MultiPolygon", "coordinates": [p["coordinates"] for p in avoidance_polygons]} if avoidance_polygons else None

def get_sustainable_route(start_lat, start_lon, end_lat, end_lon):
    """Get a sustainable route avoiding bus congestion."""
    try:
        data = fetch_realtime_data()
        bus_locations = []
        if data:
            for entity in data["entity"]:
                vehicle_data = entity.get("vehicle", {})
                if "position" in vehicle_data:
                    lat = vehicle_data["position"]["latitude"]
                    lon = vehicle_data["position"]["longitude"]
                    bus_locations.append([lat, lon, 1])
        avoid_bus_polygons = create_bus_congestion_avoidance_areas(bus_locations)
        start_coord = (start_lon, start_lat)
        end_coord = (end_lon, end_lat)
        route = get_route(start_coord, end_coord, avoid_bus_polygons)
        return {"route": route}
    except Exception as e:
        return {"status": "error", "message": f"Failed to get sustainable route: {str(e)}"}

def fetch_monitors():
    params = {"username": API_USERNAME, "password": API_PASSWORD}
    response = requests.post(MONITORS_API_URL, data=params)
    return response.json() if response.status_code == 200 else None

# Function to fetch real-time data for a specific monitor
def fetch_monitor_data(monitor_id):
    params = {
        "username": API_USERNAME,
        "password": API_PASSWORD,
        "monitor": monitor_id,
        "start": 1738503900,
        "end": 1738590300
    }
    response = requests.post(DATA_API_URL, data=params)
    return response.json() if response.status_code == 200 else None

# Parallel fetch monitor data using ThreadPoolExecutor
def fetch_all_monitor_data(monitors):
    monitor_data = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_monitor = {executor.submit(fetch_monitor_data, monitor['serial_number']): monitor for monitor in monitors}
        for future in as_completed(future_to_monitor):
            monitor = future_to_monitor[future]
            try:
                data = future.result()
            except Exception as exc:
                print(f"Monitor {monitor['serial_number']} generated an exception: {exc}")
                data = None
            if data and isinstance(data, list):
                for d in data:
                    if isinstance(d, dict) and "pm1" in d:
                        lat = float(monitor.get("latitude"))
                        lon = float(monitor.get("longitude"))
                        pm1_value = float(d["pm1"])
                        monitor_data.append((lat, lon, pm1_value))
                        break
    return monitor_data

def get_clean_route_data(start_lat, start_lon, end_lat, end_lon):
    """Get a route with the least air pollution."""
    try:
        monitors = fetch_monitors()
        monitor_data = fetch_all_monitor_data(monitors) if monitors else []
        start_coord = (start_lon, start_lat)
        end_coord = (end_lon, end_lat)
        route = get_clean_route(start_coord, end_coord, monitor_data)
        return {"route": route}
    except Exception as e:
        return {"status": "error", "message": f"Failed to get clean route: {str(e)}"}

@lru_cache(maxsize=128)
def fetch_events():
    """Fetch event data."""
    try:
        response = requests.get(EVENTS_API_URL, headers=headers)
        response.raise_for_status()
        events_data = response.json()
        return events_data.get("value", [])
    except Exception as e:
        return {"status": "error", "message": f"Failed to fetch events: {str(e)}"}

def load_and_preprocess_data(cwd):
    """Load and preprocess bike data."""
    print("CWD: ", cwd)
    file_paths = [f"{cwd}/BikesPrediction/dublin-bikes_station_status_{month}2024.csv" 
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

def fetch_events_bikes():
    """Fetch events affecting bike availability."""
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

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * (2 * atan2(sqrt(a), sqrt(1-a)))

def generate_predictions(daily_data, stations, events):
    """Generate bike availability predictions."""
    current_date = datetime.now().date()
    future_dates = [current_date + timedelta(days=i) for i in range(0, FORECAST_DAYS)]
    predictions = []
    mae_values = []
    for station_id in daily_data['station_id'].unique():
        model_path = os.path.join(MODEL_DIR, f"station_{station_id}.joblib")
        station_data = daily_data[daily_data['station_id'] == station_id].set_index('last_reported')
        lat, lon = stations[stations['station_id'] == station_id][['lat', 'lon']].values[0]
        try:
            if os.path.exists(model_path):
                model_fit = joblib.load(model_path)
            else:
                model = SARIMAX(station_data['bikes'], order=(1,1,1), seasonal_order=(1,1,1,7))
                model_fit = model.fit(disp=False)
                joblib.dump(model_fit, model_path)
            forecast = model_fit.forecast(steps=FORECAST_DAYS)
        except Exception as e:
            print(f"Model error for station {station_id}: {e}")
            forecast = pd.Series([np.nan]*FORECAST_DAYS)
            mae_values.append(np.nan)
            continue
        affected_dates = set()
        for event in events:
            if calculate_distance(lat, lon, event['lat'], event['lon']) <= EVENT_RADIUS_KM:
                affected_dates.update(set(future_dates) & set(event['dates']))
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

@lru_cache(maxsize=128)
def get_predictions_data():
    """Get bike availability predictions."""
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
        return {
            'status': 'success',
            'data': output,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'mae': round(mae, 2),
                'stations_count': len(output),
                'forecast_days': FORECAST_DAYS
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to generate predictions: {str(e)}"}

@lru_cache(maxsize=128)
def get_bike_notifications_data():
    """Get notifications for bike availability discrepancies."""
    try:
        bike_response = requests.get(DUBLIN_BIKES_API_URL, timeout=10)
        bike_response.raise_for_status()
        realtime_stations = bike_response.json()
        daily_data, stations = load_and_preprocess_data(cwd)
        events = fetch_events_bikes()
        predictions_df, _ = generate_predictions(daily_data, stations, events)
        results = predictions_df.merge(stations, on='station_id')
        today = datetime.now().date()
        notifications = []
        for station in realtime_stations:
            try:
                station_id = station['number']
                current_bikes = station['available_bikes']
                total_capacity = station['bike_stands']
                if total_capacity < BIKE_NOTIFICATION_THRESHOLDS['min_capacity'] or station['status'] != 'OPEN':
                    continue
                prediction = results[
                    (results['station_id'] == station_id) & 
                    (results['date'] == today)
                ]
                if not prediction.empty:
                    predicted = prediction['bikes'].values[0]
                    absolute_diff = abs(predicted - current_bikes)
                    percentage_diff = absolute_diff / predicted if predicted > 0 else 0
                    if (percentage_diff >= BIKE_NOTIFICATION_THRESHOLDS['percentage'] and 
                        absolute_diff >= BIKE_NOTIFICATION_THRESHOLDS['absolute']):
                        notifications.append({
                            'station_id': station_id,
                            'station_name': station['name'],
                            'position': station['position'],
                            'current_bikes': current_bikes,
                            'predicted_bikes': round(predicted, 1),
                            'percentage_diff': round(percentage_diff * 100, 1),
                            'absolute_diff': round(absolute_diff, 1),
                            'total_capacity': total_capacity,
                            'status': station['status'],
                            'last_updated': datetime.fromtimestamp(station['last_update']/1000).isoformat()
                        })
            except KeyError as e:
                print(f"Missing key in station data: {e}")
                continue
        return {
            'status': 'success',
            'notifications': sorted(notifications, key=lambda x: x['percentage_diff'], reverse=True),
            'thresholds': BIKE_NOTIFICATION_THRESHOLDS,
            'generated_at': datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to get bike notifications: {str(e)}"}

def calculate_aqi(pm1):
    if pm1 <= 12:
        return (50 / 12) * pm1
    elif pm1 <= 35.4:
        return 50 + (49 / 23.4) * (pm1 - 12)
    elif pm1 <= 55.4:
        return 100 + (49 / 20) * (pm1 - 35.4)
    elif pm1 <= 150.4:
        return 150 + (99 / 95) * (pm1 - 55.4)
    else:
        return 200 + (299 / 149.6) * (pm1 - 150.4)

def generate_points_in_bounding_box(start, end, num_points=100):
    min_lat = min(start[1], end[1])
    max_lat = max(start[1], end[1])
    min_lon = min(start[0], end[0])
    max_lon = max(start[0], end[0])

    points = []
    for _ in range(num_points):
        lat = np.random.uniform(min_lat, max_lat)
        lon = np.random.uniform(min_lon, max_lon)
        points.append((lat, lon))
    return points

# Function to calculate AQI for multiple points
def calculate_aqi_for_points(monitor_data, points):
    aqi_values = []
    for lat, lon in points:
        total_aqi = 0
        count = 0
        for m_lat, m_lon, pm1 in monitor_data:
            distance = math.sqrt((lat - m_lat) ** 2 + (lon - m_lon) ** 2)
            if distance < 0.01:  # Consider monitors within 0.01 degrees (~1 km)
                aqi = calculate_aqi(pm1)
                total_aqi += aqi
                count += 1
        if count > 0:
            aqi_values.append((lat, lon, total_aqi / count))
        else:
            aqi_values.append((lat, lon, 0))  # Default AQI if no monitors nearby
    return aqi_values

# Assuming get_clean_route is defined elsewhere; if not, here's a placeholder
def get_clean_route(start, end, monitor_data):
    """Placeholder for get_clean_route (should be defined in app.utils.gridmap)."""
    points = generate_points_in_bounding_box(start, end, num_points=100)
    aqi_values = calculate_aqi_for_points(monitor_data, points)

    # Find the point with the least AQI
    min_aqi_point = min(aqi_values, key=lambda x: x[2])
    clean_route = get_route(start, (min_aqi_point[1], min_aqi_point[0]), profile="driving-car")
    clean_route += get_route((min_aqi_point[1], min_aqi_point[0]), end, profile="driving-car")
    return clean_route

# Add these near other model paths at the top
PEDESTRIAN_DATA_PATH = (f"{cwd}/PedestrianPrediction/pedestrian-counts-1-jan-9-march-2025.xlsx")
LOCATION_COORDS_PATH = (f"{cwd}/PedestrianPrediction/dublin-city-centre-footfall-counter-locations-18072023.csv")
PEDESTRIAN_MODEL_PATH = (f"{cwd}/PedestrianPrediction/pedestrian_count_model.pkl")

# Initialize pedestrian components after bike model initialization
# ------------------------------------------------------------------
# Load or train pedestrian model
pedestrian_model = None
location_coords = None

def initialize_pedestrian_components():
    global pedestrian_model, location_coords
   
    # Load coordinates data
    if os.path.exists(LOCATION_COORDS_PATH):
        location_coords = pd.read_csv(LOCATION_COORDS_PATH)
        location_coords = location_coords[['Eco-Visio Oupput', 'Latitude', 'Longitude']]\
            .rename(columns={'Eco-Visio Oupput': 'Location'})\
            .dropna(subset=['Latitude', 'Longitude'])
   
    # Load or train model
    if os.path.exists(PEDESTRIAN_MODEL_PATH):
        pedestrian_model = joblib.load(PEDESTRIAN_MODEL_PATH)
    else:
        df = pd.read_excel(PEDESTRIAN_DATA_PATH)
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.rename(columns={df.columns[0]: "Datetime"}, inplace=True)
       
        # Feature engineering
        df["Hour"] = df["Datetime"].dt.hour
        df["Day"] = df["Datetime"].dt.day
        df["Month"] = df["Datetime"].dt.month
        df["Weekday"] = df["Datetime"].dt.weekday
       
        # Reshape data
        df_long = df.melt(
            id_vars=["Datetime", "Hour", "Day", "Month", "Weekday"],
            var_name="Location",
            value_name="Pedestrian_Count"
        ).dropna()

        # Prepare features
        X = pd.get_dummies(df_long[["Hour", "Day", "Month", "Weekday", "Location"]],
                          columns=["Location"], drop_first=True)
        y = df_long["Pedestrian_Count"]
       
        # Train and save model
        pedestrian_model = RandomForestRegressor(n_estimators=100, random_state=42)
        pedestrian_model.fit(X, y)
        joblib.dump(pedestrian_model, PEDESTRIAN_MODEL_PATH)

# Initialize when app starts
initialize_pedestrian_components()

# Add the new API endpoint
#@app.route('/api/pedestrian_predictions', methods=['GET'])

@lru_cache(maxsize=128)
def get_pedestrian_predictions():
    if not pedestrian_model or location_coords is None:
        return {
            "status": "error",
            "message": "Pedestrian prediction system not initialized"
        }, 500

    try:
        current_datetime = datetime.now()
        valid_locations = location_coords['Location'].unique()
        
        # Create prediction dataframe
        current_data = pd.DataFrame({
            "Datetime": [current_datetime] * len(valid_locations),
            "Location": valid_locations,
            "Hour": current_datetime.hour,
            "Day": current_datetime.day,
            "Month": current_datetime.month,
            "Weekday": current_datetime.weekday()
        })

        # Prepare features
        X_current = pd.get_dummies(current_data[["Hour", "Day", "Month", "Weekday", "Location"]],
                                 columns=["Location"], drop_first=True)
        
        # Align features with model
        missing_cols = set(pedestrian_model.feature_names_in_) - set(X_current.columns)
        for col in missing_cols:
            X_current[col] = 0
        X_current = X_current[pedestrian_model.feature_names_in_]

        # Make predictions and convert types
        predictions = pedestrian_model.predict(X_current)
        current_data["Predicted_Count"] = predictions.astype(int)

        # Merge with coordinates
        # merged_data = current_data.merge(
        #     location_coords[['Location', 'Latitude', 'Longitude']].astype(float),
        #     on='Location',
        #     how='inner'
        # )
        merged_data = current_data.merge(location_coords, on='Location', how='inner')

        # Convert all data to native Python types
        output_data = []
        for _, row in merged_data.iterrows():
            output_data.append({
                "location": str(row['Location']),
                "latitude": float(row['Latitude']),
                "longitude": float(row['Longitude']),
                "predicted_count": int(row['Predicted_Count']),
                "hour": int(row['Hour']),
                "datetime": row['Datetime'].isoformat()
            })

        # Calculate mean as native float
        mean_pred = float(merged_data['Predicted_Count'].mean())

        return {
            'status': 'success',
            'data': output_data,
            'generated_at': datetime.now().isoformat(),
            'location_count': len(output_data),
            'model_type': 'RandomForest',
            'mean_prediction': round(mean_pred, 1)
        }, 200

    except Exception as e:
        return {
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }, 500
   
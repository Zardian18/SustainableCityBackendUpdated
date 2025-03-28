from flask import Flask, render_template_string
import requests
import folium
from folium.plugins import HeatMap
import numpy as np
import json
from shapely.geometry import shape, Point
from scipy.interpolate import Rbf
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)

# Sonitus API credentials and endpoints
API_USERNAME = "dublincityapi"
API_PASSWORD = "Xpa5vAQ9ki"
MONITORS_API_URL = "https://data.smartdublin.ie/sonitus-api/api/monitors"
DATA_API_URL = "https://data.smartdublin.ie/sonitus-api/api/data"

# Function to load Dublin boundary polygon from a GeoJSON file
def load_dublin_polygon_from_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    # Assumes the first feature is Dublin's boundary
    return shape(data['features'][0]['geometry'])

# Generate random points within a given polygon
def generate_random_points_in_polygon(polygon, num_points):
    np.random.seed(42)  # Fixed seed for reproducibility
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    while len(points) < num_points:
        random_point = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(random_point):
            points.append(random_point)
    return points

# Function to fetch all available monitors
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
                        no2_value = float(d["pm1"])
                        monitor_data.append((lat, lon, no2_value))
                        break
    print(monitor_data)
    return monitor_data

# Generate heatmap with RBF interpolation limited to Dublin land area
def generate_heatmap(monitors):
    map_center = [53.349805, -6.26031]
    m = folium.Map(location=map_center, zoom_start=12)

    # Fetch monitor data in parallel
    monitor_data = fetch_all_monitor_data(monitors)
    
    if len(monitor_data) < 3:
        return m._repr_html_()  # Not enough data to interpolate

    latitudes, longitudes, no2_values = zip(*monitor_data)
    rbf_interpolator = Rbf(latitudes, longitudes, no2_values, function='linear')

    # Load the Dublin boundary polygon from file
    dublin_polygon = load_dublin_polygon_from_file(r'app\utils\dublin_boundary.geojson')

    # Generate random points within Dublin boundary
    num_samples = 1000
    random_points = generate_random_points_in_polygon(dublin_polygon, num_samples)
    lat_samples = [pt.y for pt in random_points]  # Shapely Points: x=lon, y=lat
    lon_samples = [pt.x for pt in random_points]

    # Interpolate NO2 values at these points
    interpolated_no2 = rbf_interpolator(lat_samples, lon_samples)

    # Prepare heatmap data
    heatmap_data = [[lat, lon, max(0, val)] for lat, lon, val in zip(lat_samples, lon_samples, interpolated_no2)]
    HeatMap(heatmap_data, radius=25, max_zoom=12, blur=20).add_to(m)

    return m._repr_html_()

@app.route("/", methods=["GET"])
def index():
    monitors = fetch_monitors()
    heatmap_html = generate_heatmap(monitors) if monitors else "Error fetching monitors"
    
    return render_template_string("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Dublin Air Quality Heatmap</title>
        </head>
        <body>
            <h1>Dublin Air Quality Heatmap</h1>
            <div id="map">{{ heatmap_html|safe }}</div>
        </body>
        </html>
    """, heatmap_html=heatmap_html)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Blueprint, jsonify, request
from app.utils.gridmap import fetch_realtime_data, get_route
from app.services.prediction_service import make_prediction
import json
import requests
import folium
import time
import numpy as np
from flask import Flask, render_template_string, jsonify
from folium.plugins import HeatMap
from openrouteservice import Client

# Create a Blueprint for prediction routes
prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/prediction', methods=['GET'])
def predict():
    input_data = request.args.getlist('input', type=float)
    if not input_data:
        input_data = [0.6, 0.7, 0.4, 0.3, 0.7, 0.9, 0.2]  # Default input if none is provided

    if len(input_data) != 7:
        return jsonify({"error": "Expected 7 input values."}), 400

    prediction_result = make_prediction(input_data)
    return jsonify({"prediction": prediction_result}), 200


@prediction_bp.route('/compare', methods=['GET'])
def compare_routes():
    """Displays both normal and sustainable routes along with bus heatmap."""
    data = fetch_realtime_data()
    bus_locations = []
    start_coord = (-6.26031, 53.349805)
    end_coord = (-6.24889, 53.33306)
    
    if data:
        for entity in data["entity"]:
            vehicle_data = entity.get("vehicle", {})
            if "position" in vehicle_data:
                lat = vehicle_data["position"]["latitude"]
                lon = vehicle_data["position"]["longitude"]
                bus_locations.append([lat, lon, 1])
    
    map_center = [(start_coord[1] + end_coord[1]) / 2, (start_coord[0] + end_coord[0]) / 2]
    compare_map = folium.Map(location=map_center, zoom_start=14)
    
    if bus_locations:
        HeatMap(bus_locations, min_opacity=0.3, radius=15, blur=10, max_zoom=1).add_to(compare_map)
    
    folium.Marker((start_coord[1], start_coord[0]), popup="Start", icon=folium.Icon(color='green')).add_to(compare_map)
    folium.Marker((end_coord[1], end_coord[0]), popup="End", icon=folium.Icon(color='red')).add_to(compare_map)
    
    # Normal Route
    normal_route_coords = get_route(start_coord, end_coord)
    if normal_route_coords:
        folium.PolyLine(normal_route_coords, color="red", weight=5, tooltip="Normal Route").add_to(compare_map)
    
    # Sustainable Route
    sustainable_route_coords = get_route(start_coord, end_coord, avoid_areas)
    if sustainable_route_coords:
        folium.PolyLine(sustainable_route_coords, color="blue", weight=5, tooltip="Sustainable Route").add_to(compare_map)
    
    return render_template_string("""
        <html>
        <head><title>Route Comparison</title></head>
        <body>
            <h2>Comparison of Normal vs Sustainable Route</h2>
            <iframe srcdoc='{{ map_html }}' width="100%" height="600px"></iframe>
        </body>
        </html>
    """, map_html=compare_map._repr_html_())

# Avoidance area setup
avoid_point = (53.334021, -6.245198)
radius = 0.0045
avoid_areas = {
    "type": "Polygon",
    "coordinates": [[
        [avoid_point[1] - radius, avoid_point[0] - radius],
        [avoid_point[1] + radius, avoid_point[0] - radius],
        [avoid_point[1] + radius, avoid_point[0] + radius],
        [avoid_point[1] - radius, avoid_point[0] + radius],
        [avoid_point[1] - radius, avoid_point[0] - radius]   
    ]]
}
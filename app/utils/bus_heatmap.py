from flask import Flask, render_template_string, jsonify, request  # Import 'request' from Flask
import requests  # Import requests separately
from flask_cors import CORS
import folium
import openrouteservice
import numpy as np
from folium.plugins import HeatMap

# API Keys
ORS_API_KEY = "5b3ce3597851110001cf62485c2afa0d8bdf46a08bec95a1d1e35e69"
DUBLIN_BUS_API_KEY = "988e6458483340cd8599cacbbe75acb3"  # Dublin Bus API key
DUBLIN_BIKES_API_URL = "https://api.jcdecaux.com/vls/v1/stations?contract=dublin&apiKey=5c8736d088fe1fc1388a8982d30072aa64aaf032"
REAL_TIME_API_URL = "https://api.nationaltransport.ie/gtfsr/v2/Vehicles?format=json"
EVENTS_API_URL = "https://failteireland.azure-api.net/opendata-api/v2/events"

# Flask app
app = Flask(__name__)
CORS(app)
ors_client = openrouteservice.Client(key=ORS_API_KEY)

def fetch_api_data(url):
    try:
        headers = {"x-api-key": DUBLIN_BUS_API_KEY} if "nationaltransport.ie" in url else {}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def get_here_route(start, end, transport_mode):
    """Fetch route using HERE API"""
    try:
        base_url = "https://intermodal.router.hereapi.com/v8/routes"
        params = {
            "origin": start,
            "destination": end,
            "vehicle[modes]": transport_mode,
            "vehicle[enable]": "routeHead",
            "transit[enable]": "routeTail",
            "apiKey": "hlXCzjlcPk332nvf2NkwRo--8WxK8J6Qp_cLVIcnW1E"
        }
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

    
def calculate_segment_bbox(coord1, coord2):
    """Calculate bounding box for a route segment"""
    min_lat = min(coord1[0], coord2[0])
    max_lat = max(coord1[0], coord2[0])
    min_lon = min(coord1[1], coord2[1])
    max_lon = max(coord1[1], coord2[1])
    return [min_lon, min_lat, max_lon, max_lat]

def detect_congestion(segment_bbox, buses, events, transport_mode):
    """Check congestion in a segment using simple heuristics"""
    BUS_THRESHOLDS = {'car': 5, 'bike': 2, 'walking': 1}
    EVENT_IMPACT = {'high': 3, 'medium': 2, 'low': 1}
    
    # Count buses in bounding box
    bus_count = sum(1 for bus in buses.get('buses', []) 
                   if (segment_bbox[0] <= bus['longitude'] <= segment_bbox[2]) and 
                      (segment_bbox[1] <= bus['latitude'] <= segment_bbox[3]))
    
    # Check events in area
    event_score = sum(EVENT_IMPACT.get(e.get('impact', 'low'), 0) 
                 for e in events.get('events', [])
                 if (segment_bbox[0] <= e['lon'] <= segment_bbox[2]) and 
                    (segment_bbox[1] <= e['lat'] <= segment_bbox[3]))
    
    # Simple congestion logic
    if transport_mode == 'car':
        return bus_count > BUS_THRESHOLDS['car'] or event_score > 3
    elif transport_mode == 'bike':
        return bus_count > BUS_THRESHOLDS['bike'] or event_score > 2
    else:
        return bus_count > BUS_THRESHOLDS['walking'] or event_score > 1

def get_route_internal(start, end):
    """Fetch the base route from OpenRouteService"""
    try:
        # Parse coordinates and ensure they are floats
        start_lat, start_lon = map(float, start.split(','))
        end_lat, end_lon = map(float, end.split(','))
    except ValueError:
        return {"error": "Invalid coordinate format. Use 'lat,lon'."}
    
    try:
        # ORS expects coordinates in [lon, lat] order
        coords = [[start_lon, start_lat], [end_lon, end_lat]]
        
        # Request the route from ORS
        route = ors_client.directions(
            coordinates=coords,
            profile='driving-car',
            format='geojson'
        )
        
        # Extract route geometry and convert to [lat, lon] for frontend
        features = route.get('features', [])
        if not features:
            return {"error": "No route found between the points"}
        
        geometry = features[0]['geometry']['coordinates']
        coordinates = [[point[1], point[0]] for point in geometry]  # Convert to lat,lon
        
        # Extract summary and instructions
        summary = features[0]['properties'].get('summary', {})
        segments = features[0]['properties'].get('segments', [{}])
        instructions = []
        for segment in segments:
            for step in segment.get('steps', []):
                instructions.append({
                    "instruction": step.get('instruction', ''),
                    "distance": step.get('distance', 0),
                    "duration": step.get('duration', 0)
                })
        
        return {
            "route": {
                "coordinates": coordinates,
                "distance": summary.get('distance', 0),  # Meters
                "duration": summary.get('duration', 0),  # Seconds
                "instructions": instructions
            }
        }
    except openrouteservice.exceptions.ApiError as e:
        return {"error": f"ORS API Error: {str(e)}"}
    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}

def calculate_segment_bbox(coord1, coord2):
    """Calculate bounding box for a route segment"""
    min_lat = min(coord1[0], coord2[0])
    max_lat = max(coord1[0], coord2[0])
    min_lon = min(coord1[1], coord2[1])
    max_lon = max(coord1[1], coord2[1])
    return [min_lon, min_lat, max_lon, max_lat]

def calculate_detour(coord1, coord2):
    """Calculate detour point using simple offset"""
    return [
        (coord1[0] + coord2[0])/2 + 0.0005,  # Small latitude offset
        (coord1[1] + coord2[1])/2 - 0.0005   # Small longitude offset
    ]



def get_route_with_waypoints(start, end, waypoints):
    """Get route with additional waypoints"""
    try:
        coords = parse_coordinates(start) + waypoints + parse_coordinates(end)
        route = ors_client.directions(
            coordinates=coords,
            profile='driving-car',
            format='geojson',
            options={'avoid_features': ['ferries']}
        )
        return format_route_response(route)
    except Exception as e:
        return {"error": str(e)}

def parse_coordinates(coord_str):
    """Parse 'lat,lon' string into [lon, lat] list"""
    lat, lon = map(float, coord_str.split(','))
    return [[lon, lat]]

def format_route_response(route):
    """Format ORS response for API output"""
    features = route.get('features', [])
    if not features:
        return {"error": "No route found"}
    
    geometry = features[0]['geometry']['coordinates']
    coordinates = [[point[1], point[0]] for point in geometry]  # Convert to lat,lon
    
    summary = features[0]['properties'].get('summary', {})
    return {
        "route": {
            "coordinates": coordinates,
            "distance": summary.get('distance', 0),
            "duration": summary.get('duration', 0)
        }
    }



@app.route('/heatmap', methods=['GET'])
def get_data():
    bike_data = fetch_api_data(DUBLIN_BIKES_API_URL)
    bus_data = fetch_api_data(REAL_TIME_API_URL)
    events_data = fetch_api_data(EVENTS_API_URL)

    return jsonify({
        "bikes": bike_data,
        "buses": bus_data,
        "events": events_data
    })

# @app.route('/route', methods=['GET'])
# def get_route():
#     # Get start and end coordinates from query parameters
#     start = request.args.get('start')
#     end = request.args.get('end')
    
#     if not start or not end:
#         return jsonify({"error": "Missing start or end parameters"}), 400
    
#     try:
#         # Parse coordinates and ensure they are floats
#         start_lat, start_lon = map(float, start.split(','))
#         end_lat, end_lon = map(float, end.split(','))
#     except ValueError:
#         return jsonify({"error": "Invalid coordinate format. Use 'lat,lon'."}), 400
    
#     try:
#         # ORS expects coordinates in [lon, lat] order
#         coords = [[start_lon, start_lat], [end_lon, end_lat]]
        
#         # Request the route from ORS
#         route = ors_client.directions(
#             coordinates=coords,
#             profile='driving-car',
#             format='geojson'
#         )
        
#         # Extract route geometry and convert to [lat, lon] for frontend
#         features = route.get('features', [])
#         if not features:
#             return jsonify({"error": "No route found between the points"}), 404
        
#         geometry = features[0]['geometry']['coordinates']
#         coordinates = [[point[1], point[0]] for point in geometry]  # Convert to lat,lon
        
#         # Extract summary and instructions
#         summary = features[0]['properties'].get('summary', {})
#         segments = features[0]['properties'].get('segments', [{}])
#         instructions = []
#         for segment in segments:
#             for step in segment.get('steps', []):
#                 instructions.append({
#                     "instruction": step.get('instruction', ''),
#                     "distance": step.get('distance', 0),
#                     "duration": step.get('duration', 0)
#                 })
        
#         return jsonify({
#             "route": {
#                 "coordinates": coordinates,
#                 "distance": summary.get('distance', 0),  # Meters
#                 "duration": summary.get('duration', 0),  # Seconds
#                 "instructions": instructions
#             }
#         })
#     except openrouteservice.exceptions.ApiError as e:
#         return jsonify({"error": f"ORS API Error: {str(e)}"}), 500
#     except Exception as e:
#         return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# @app.route('/optimized_route', methods=['GET'])
# def get_optimized_route():
#     start = request.args.get('start')
#     end = request.args.get('end')
#     transport_mode = request.args.get('mode', 'car')
    
#     if not start or not end:
#         return jsonify({"error": "Missing start or end parameters"}), 400
    
#     route_data = get_here_route(start, end, transport_mode)
    
#     if 'error' in route_data:
#         return jsonify(route_data), 500
    
#     return jsonify(route_data)

# @app.route('/compare')
# def compare_routes():
#     # Test coordinates
#     start = "53.34366705,-6.254436"
#     end = "53.312211176459776,-6.249847412109376"
    
#     # Create base map
#     start_lat, start_lon = map(float, start.split(','))
#     m = folium.Map(location=[start_lat, start_lon], zoom_start=14)

#     # Add markers
#     folium.Marker([start_lat, start_lon], tooltip='Start', icon=folium.Icon(color='green')).add_to(m)
#     end_lat, end_lon = map(float, end.split(','))
#     folium.Marker([end_lat, end_lon], tooltip='End', icon=folium.Icon(color='red')).add_to(m)

#     # Get regular route
#     with app.test_client() as client:
#         response = client.get(f'/route?start={start}&end={end}')
#         regular_route = response.json.get('route', {}).get('coordinates', [])

#     # Get optimized route coordinates from complex JSON
#     optimized_coords = []
#     with app.test_client() as client:
#         response = client.get(f'/optimized_route?start={start}&end={end}')
#         optimized_data = response.json
        
#         try:
#             # Extract coordinates from all sections
#             for section in optimized_data['routes'][0]['sections']:
#                 # Add departure location
#                 dep = section['departure']['place']['location']
#                 optimized_coords.append([dep['lat'], dep['lng']])
                
#                 # Add arrival location
#                 arr = section['arrival']['place']['location']
#                 optimized_coords.append([arr['lat'], arr['lng']])
                
#             # Remove duplicates while preserving order
#             # seen = set()
#             # optimized_coords = [x for x in optimized_coords if tuple(x) not in seen and not seen.add(tuple(x))]
#         except (KeyError, IndexError) as e:
#             print(f"Error parsing optimized route: {e}")

#     # Add routes to map
#     if regular_route:
#         folium.PolyLine(
#             locations=regular_route,
#             color='blue',
#             weight=5,
#             opacity=0.7,
#             tooltip='Regular Route (ORS)'
#         ).add_to(m)

#     if optimized_coords:
#         folium.PolyLine(
#             locations=optimized_coords,
#             color='red', 
#             weight=5,
#             opacity=0.7,
#             tooltip='Optimized Route (HERE)',
#             dash_array='5,5'
#         ).add_to(m)

#     # Add layer control
#     folium.LayerControl().add_to(m)

#     return m._repr_html_()
# # import polyline  # Ensure this is installed: pip install polyline

    
if __name__ == '__main__':
    app.run(debug=True)
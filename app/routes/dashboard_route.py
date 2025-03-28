from flask import Blueprint, request, jsonify

from app.utils.gridmap import fetch_bus_locations, fetch_events, get_bike_notifications_data, get_clean_route_data, get_normal_route, get_predictions_data, get_sustainable_route, get_air_pollution_data, get_noise_pollution_data, get_pedestrian_predictions

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/', methods=['GET'])
def dashboard():
    """
    Aggregate responses from various API routes into a single dashboard response.
    Routes requiring coordinates are included only if parameters are provided.
    """
    # Extract optional coordinates from query parameters
    start_lat = request.args.get("start_lat", type=float)
    start_lon = request.args.get("start_lon", type=float)
    end_lat = request.args.get("end_lat", type=float)
    end_lon = request.args.get("end_lon", type=float)
    has_coordinates = all([start_lat is not None, start_lon is not None, 
                          end_lat is not None, end_lon is not None])

    # Initialize dashboard response
    dashboard_data = {
        "bus_heatmap": fetch_bus_locations(),
        "events": fetch_events(),
        "predictions": get_predictions_data(),
        "bike_notifications": get_bike_notifications_data(),
        "air_pollution": get_air_pollution_data(),
        "pedestrian": get_pedestrian_predictions(),
        "normal_route": None,
        "sustainable_route": None,
        "clean_route": None
    }

    # Include route data if coordinates are provided
    if has_coordinates:
        dashboard_data["normal_route"] = get_normal_route(start_lat, start_lon, end_lat, end_lon)
        dashboard_data["sustainable_route"] = get_sustainable_route(start_lat, start_lon, end_lat, end_lon)
        dashboard_data["clean_route"] = get_clean_route_data(start_lat, start_lon, end_lat, end_lon)

    return jsonify(dashboard_data)

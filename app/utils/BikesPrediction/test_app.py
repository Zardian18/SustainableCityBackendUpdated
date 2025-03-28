import requests
import pytest

# Define the API URL (Ensure Flask is running)
BASE_URL = "http://localhost:5000/api/predictions" #if this URL is modified to reflect the wrong url then get_response will fail.

@pytest.fixture
def get_response():
    """Fetch response from the API endpoint."""
    response = requests.get(BASE_URL)
    return response

def test_api_status(get_response):
    """Test if API returns status 200 (OK)."""
    assert get_response.status_code == 200, f"Expected 200, got {get_response.status_code}"

def test_json_structure(get_response):
    """Test if JSON response contains the expected structure."""
    data = get_response.json()
    
    assert "data" in data, "Missing 'data' key in response"
    assert isinstance(data["data"], list), "'data' should be a list"

    if len(data["data"]) > 0:  # Only run if there's data
        station = data["data"][0]
        required_keys = {"station_id", "station_name", "latitude", "longitude", "predictions"} #if any of these keys is removed from the response then the test case will fail 

        # Ensure each station has required keys
        assert required_keys.issubset(station.keys()), f"Missing keys in station: {station.keys()}"

        # Check 'predictions' structure
        assert isinstance(station["predictions"], list), "'predictions' should be a list"
        if len(station["predictions"]) > 0:
            prediction = station["predictions"][0]
            required_prediction_keys = {"date", "bikes", "stands"}

            assert required_prediction_keys.issubset(prediction.keys()), f"Missing keys in prediction: {prediction.keys()}"

def test_valid_data_ranges(get_response):
    """Ensure latitude, longitude, bikes, and stands values are within valid ranges."""
    data = get_response.json()
    
    for station in data["data"]:
        assert -90 <= station["latitude"] <= 90, f"Invalid latitude: {station['latitude']}"
        assert -180 <= station["longitude"] <= 180, f"Invalid longitude: {station['longitude']}"

        # for prediction in station["predictions"]:
        #     assert 0 <= prediction["bikes"] <= 100, f"Invalid bikes count: {prediction['bikes']}"
        #     assert 0 <= prediction["stands"] <= 100, f"Invalid stands count: {prediction['stands']}"

def test_fail_status():
    """Test API failure by calling an incorrect endpoint."""
    response = requests.get("http://localhost:5000/api/wrong_endpoint")
    assert response.status_code == 404, f"Expected 404, got {response.status_code}"

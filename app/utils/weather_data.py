import requests
import xml.etree.ElementTree as ET

'''
Call fetch_forecast from your code to get the data for the particular latitude 
and longitude in JSON format
'''

def get_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print('Error:', response.status_code)
        return None

def parse_weather_data(data):
    root = ET.fromstring(data)
    data_dict = {}
    for product in root.findall('./product'):
        for time in product:
            key = (time.attrib["from"], time.attrib["to"])
            for location in time:
                param = {}
                for parameter in location:
                    value = None
                    attributes = list(parameter.attrib.keys())
                    expected_field = ["value", "mps", "percent", "deg", "number"]
                    for field in expected_field:
                        if field in attributes:
                            value = field
                            break
                    tag = parameter.tag
                    if(tag == "symbol"):
                        tag = parameter.attrib["id"]
                    if value:
                        param[tag] = parameter.attrib[value]
                data_dict[key] = param
    print(data_dict)
    return data_dict

def fetch_forecast(lat, long):
    url = f"http://openaccess.pf.api.met.ie/metno-wdb2ts/locationforecast?lat={lat};long={lon}"
    data = get_data(url)
    parse_weather_data(data)

#54.7210798611, -8.7237392806

if __name__=='__main__':
    lat = input("Enter the latitude: ")
    lon = input("Enter the longitude: ")
    url = f"http://openaccess.pf.api.met.ie/metno-wdb2ts/locationforecast?lat={lat};long={lon}"
    data = get_data(url)
    parse_weather_data(data)
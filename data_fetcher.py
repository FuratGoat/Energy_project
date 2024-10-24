import requests
import json

def fetch_data(api_key, url):
    """Fetch data from API and save to JSON"""
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        with open('energy_data.json', 'w') as f:
            json.dump(data, f, indent=4)
        return data
    return None

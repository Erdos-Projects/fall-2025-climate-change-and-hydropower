import json
import urllib.request
import time
import math
import pandas as pd

params = {}
def add_params(fname):
    with open(fname) as f:
        contents = f.read()
        obj = json.loads(contents)
        for feature in obj['features']:
            params[int(feature['id'])] = feature['properties']['parameter_description']

add_params('params.json') # Downloaded these files earlier
add_params('params_2.json')

def get_json(url):
    time.sleep(.2) # Their API wants no more than 5 requests per second
    f = urllib.request.urlopen(url)
    contents = f.read()
    return json.loads(contents)

def find_closest_feature(lat, long, features):
    min_dist = math.inf
    min_ind = -1
    for index, feature in enumerate(features):
        long_f, lat_f = feature['geometry']['coordinates']
        dist = (lat - lat_f) ** 2 + (long - long_f) ** 2
        if dist < min_dist:
            min_dist = dist
            min_ind = index
    return features[min_ind]

API_BASE = 'https://api.waterdata.usgs.gov/ogcapi/v0/collections/'
# Code should be a 5 digit string. ie 10 is water temperature. It needs to be passed as '00010'
def code_search(lat, long, code): # Searches for nearest station offering the parameter
    for eps_pow in range(-2, 1):
        eps = 10 ** eps_pow
        lat_min, lat_max = lat - eps, lat + eps
        long_min, long_max = long - eps, long + eps
        bbox = ','.join(map(str, [long_min, lat_min, long_max, lat_max]))
        response = get_json(API_BASE+f'latest-continuous/items?f=json&lang=en-US&bbox={bbox}&limit=1000&properties=id,monitoring_location_id,parameter_code&parameter_code={code}')
        if response['numberReturned'] > 0: # Station found in bounding box
            feature = find_closest_feature(lat, long, response['features'])
            return feature['properties']['monitoring_location_id'], feature['geometry']['coordinates']
    return None # Nothing found nearby

def station_search(lat, long): # Searches for closest station given the latitude and longitude
    for eps_pow in range(-2, 1):
        eps = 10 ** eps_pow
        lat_min, lat_max = lat - eps, lat + eps
        long_min, long_max = long - eps, long + eps
        bbox = ','.join(map(str, [long_min, lat_min, long_max, lat_max]))
        response = get_json(API_BASE+f'monitoring-locations/items?f=json&lang=en-US&bbox={bbox}&limit=1000&properties=id')
        if response['numberReturned'] > 0:
            feature = find_closest_feature(lat, long, response['features'])
            return feature['id'], feature['geometry']['coordinates']

def search_stations(lat, long, code = None):
    if code is not None:
        return code_search(lat, long, code)
    else:
        return station_search(lat, long)

def get_recent_values(station, code):
    print(API_BASE+f'daily/items?f=json&lang=en-US&limit=100&skipGeometry=true&offset=0&monitoring_location_id={station}&parameter_code={code}&time=P1M')
    response = get_json(API_BASE+f'daily/items?f=json&lang=en-US&limit=100&skipGeometry=true&offset=0&monitoring_location_id={station}&parameter_code={code}&time=P1M')
    L = []
    for feature in response['features']:
        props = feature['properties']
        L.append({
            'measurement': props['value'],
            'units': props['unit_of_measure'],
            'datetime': props['time'],
            'approval': props['approval_status'],
            'statistic_id': props['statistic_id'] # Controls average, min, max, etc.
        })
    return pd.DataFrame(L)

# Example finding nearest station to Eugene Oregon with cubic discharge
# Running this counts against the rate limit
def test():
    print('Searching for the following parameter:\t', params[60])
    station, coords = code_search(44.0521, -123.0897, '00060')
    print('Found station:\t' station)
    data = get_recent_values(station, '00060')
    print(data)
    return data

import pandas as pd
import shapely
import numpy as np
import geopandas as gpd
import requests
import itertools
from math import radians, cos, sin, asin, sqrt

import matplotlib
import matplotlib.pyplot as plt
import time
import random

print("MapBox Access: ")
access = input()

# Generate isochrone using Mapbox API
def isochrone(lon,lat,access=access):
    r = requests.get(f"https://api.mapbox.com/isochrone/v1/mapbox/walking/{lon},{lat}?contours_minutes=30,60&contours_colors=6706ce,04e813&polygons=true&access_token={access}")
    response = r.json()
    try:
        iso30 = shapely.geometry.Polygon(response['features'][1]['geometry']['coordinates'][0])
        iso60 = shapely.geometry.Polygon(response['features'][0]['geometry']['coordinates'][0])
    except:
        print(response)
        iso30, iso60 = None, None
    time.sleep(0.02)
    return iso30,iso60

def haversine(pt1, pt2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    if 'tuple' in str(type(pt1)).lower():
        lon1, lat1 = pt1
        lon2, lat2 = pt2
    else:
        lon1, lat1 = pt1.x, pt1.y
        lon2, lat2 = pt2.x, pt2.y

    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def driving_time(sources,destinations,access=access,naive=False):
    """
    Input:
    - sources: List of shapely coordinates
    - destinations: List of shapely coordinates
    
    Output:
    - result: n_sources x n_destinations matrix pertaining to drive times in minutes
    """
    # Declare speed for conversion
    SPEED = 60
    # Set up result matrix
    result = np.zeros((len(sources),len(destinations))) # Generate driving time matrix
    
    if naive:
        for i in range(len(sources)):
            for j in range(len(destinations)):
                # Compute distance in km, divide by speed (km/h), convert to minutes
                result[i][j] = (haversine((sources[i].x, sources[i].y),
                                          (destinations[j].x, destinations[j].y))/SPEED)*60
        return result
    
    # Format to string for MapBox
    sources = [f"{pt.x},{pt.y}" for pt in sources] # Reformat sources
    destinations = [f"{pt.x},{pt.y}" for pt in destinations] # Reformat destinations    
    for i in range(len(sources)):
        for j in range(len(destinations)//20+1):
            # Iterate source by source, take 20 destinations per query
            source = sources[i] 
            destination = destinations[j*20:(j*20)+20]
            
            # Formatting for API
            num_dests = len(destination)
            destination_ids = ";".join(list(map(str, list(range(1,num_dests+1)))))
            destination = ";".join(destination)
            
            # API Call
            r = requests.get(f"https://api.mapbox.com/directions-matrix/v1/mapbox/driving/{source};{destination}?sources=0&destinations={destination_ids}&access_token={access}")
            r = r.json()
            
            # Update result matrix
            result[i][j*20:(j*20)+num_dests] = np.array(r['durations'])
            
            # Sleep, max of 60 calls per minute
            time.sleep(1)
    
    # Output results in minutes
    result = result/60
    
    return result


import pandas as pd
import shapely
import numpy as np
import geopandas as gpd
import requests
import itertools

import matplotlib
import matplotlib.pyplot as plt

import random
from helper_functions.mapbox_helper import * 

print("Google Access Key:")
google_access = input()

def geocode(address, prov="", country="Philippines", access=google_access):
    x = requests.get(f"https://maps.googleapis.com/maps/api/geocode/json?address={address},{prov},{country}&key={access}")
    x = x.json()
    coords = x['results'][0]['geometry']['location']
    prec = x['results'][0]['geometry']['location_type']
    lat, lon = coords['lat'], coords['lng']
    return lon, lat, prec

def nearest_road(coords, google_access=google_access):
    result = []
    for i in range((len(coords)//100)+1):
        # Take 100 coordinates at a time
        lst = coords[i*100:(i+1)*100] if i!=(len(coords)//100) else coords[i*100:] 
        lst = [(pt.x,pt.y) for pt in lst] 
        
        # Turn into string, query it, then obtain points
        query = "|".join([f"{pt[1]},{pt[0]}" for pt in lst])
        x = requests.get(f"https://roads.googleapis.com/v1/nearestRoads?points={query}&key={google_access}").json()
        
        # Format results into dictionary by ID
        road_lst = [(None,None)]*len(lst)
        if 'snappedPoints' in x.keys():
            x = x['snappedPoints']
            for d in x:
                road_lst[d['originalIndex']] = (d['location']['longitude'],d['location']['latitude'])
        
        # Use haversine formula to find distance between points
        distances = [haversine(pt1, pt2) if (pt2[0] and pt2[1]) else None for (pt1, pt2) in zip(lst, road_lst)]
        result.extend(distances)
    return result

def subset_hrsl(hrsl, gdf):
    """
    Input:
    - hrsl: HRSL dataframe
    - gdf: Geodataframe pertaining to area to subset the HRSL to
    
    Output:
    - subset_hrsl: The subsetted HRSL dataframe
    """
    
    polygon = gdf.iloc[0]['geometry']
    if 'MultiPolygon' in str(type(polygon)):
        max_y = max([max(p.exterior.coords.xy[1]) for p in polygon])
        min_y = min([min(p.exterior.coords.xy[1]) for p in polygon])
        max_x = max([max(p.exterior.coords.xy[0]) for p in polygon])
        min_x = min([min(p.exterior.coords.xy[0]) for p in polygon])
    else:
        max_y = max(polygon.exterior.coords.xy[1])
        min_y = min(polygon.exterior.coords.xy[1])
        max_x = max(polygon.exterior.coords.xy[0])
        min_x = min(polygon.exterior.coords.xy[0])
        
    # Get all the HRSL populations within Antipolo
    subset_hrsl = hrsl.loc[(hrsl.latitude>min_y)&(hrsl.latitude<max_y)&(hrsl.longitude>min_x)&(hrsl.longitude<max_x)]
    temp = gpd.sjoin(subset_hrsl, gdf, how='left', op='within')
    subset_hrsl = temp.loc[temp.ISO.notnull()].reset_index(drop=True)
    subset_hrsl = subset_hrsl.drop('index_right', axis=1)
    return subset_hrsl

def generate_isochrones(gdf, col='geometry'):
    """
    Input: 
    - gdf: Dataframe with a column named `col` corresponding to coordinates of points
    
    Output: 
    - gdf30: Geodataframe with 30 minute isochrones as its `geometry` column
    - gdf60: Geodataframe with 60 minute isochrones as its `geometry` column
    """
    isochrones = [isochrone(p.x,p.y) for p in gdf[col]]
    gdf['i30'] = [item[0] for item in isochrones]
    gdf['i60'] = [item[1] for item in isochrones]

    gdf30 = gpd.GeoDataFrame(gdf, geometry=gdf['i30'])
    gdf60 = gpd.GeoDataFrame(gdf, geometry=gdf['i60'])
    
    return gdf30, gdf60
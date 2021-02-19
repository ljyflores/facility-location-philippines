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

def bounding_box(polygon):
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
    return max_y, min_y, max_x, min_x

def generate_candidates(gdf, sample_size=None, spacing=1):
    """
    Input: 
    Geodataframe, geometry corresponds to LGU polygon
    
    Note:
    Expect only be 1 row corresponding to desired LGU
    
    Output: 
    Geodataframe, geometry corresponds to coordinates of candidate sites
    """
    
    # Generate top, bottom, left, and rightmost points
    polygon = gdf.iloc[0]['geometry']
    max_y, min_y, max_x, min_x = bounding_box(polygon)
    
    # Generate sites
    sites = []
    for x in np.linspace(min_x,max_x,round((max_x-min_x)/(0.01*spacing))):
        for y in np.linspace(min_y,max_y,round((max_y-min_y)/(0.01*spacing))):
            p = shapely.geometry.Point(x,y)
            if any(p.within(geom) for geom in gdf['geometry']):
                sites.append(p)
    
    if sample_size:
        sites = random.sample(sites, sample_size)
        
    all_sites = gpd.GeoDataFrame({'geometry':sites}, geometry=sites, crs='EPSG:4326').reset_index()
    return all_sites

def add_record(d, lst):
    """
    Function that checks if a record is already present in a dictionary
    Keeps track of a dictionary d
    Returns True if the record is NEW (i.e. not yet in the dictionary)
    Returns False otherwise
    """
    if lst[0] not in d:
        d[lst[0]] = {}
        if len(lst)==1:
            return True
    if len(lst)==1:
        return False
    else:
        return add_record(d[lst[0]], lst[1:])

def sample_sets(lst, set_size, n_results, redundant_dict):
    """
    Inputs:
    - lst: List of HRSL ids
    - set_size: Size of each candidate set
    - n_results: Number of candidate sets to generate
    - redundant_dict: Dictionary containing site ids as keys, 
    and list of redundant sites as a list (value)
    
    Output:
    Dataframe with Set IDs (set_id) and sets (site_set)
    """
    print("Generating candidates")
    if n_results=='all':
        result = list(itertools.combinations(lst, set_size))
    else:
        count = 0
        result = []
        records = {}
        while count < n_results:
            test = sorted(random.sample(lst, set_size))
            if add_record(records, test):
                if check_valid_candidate(test, redundant_dict):
                    result.append(test)
                    count += 1
    print("Generating dataframe")
    result = pd.DataFrame([(i,item) for (i,item) in enumerate(result)], 
                          columns=['set_id','site_set'])
    return result

def generate_redundant_sites(lst, threshold=2):
    # Generate distance matrix
    site_site_mat = np.zeros((len(lst),len(lst)))
    for i in range(len(lst)):
        site_site_mat[i] = list(map(lambda p: haversine(lst[i], p), lst))
        
    # Identify pairs whose distance is below the threshold
    xs, ys = (site_site_mat < threshold).nonzero()
    d = {}
    for (x,y) in zip(xs,ys):
        if x!=y:
            if x in d:
                d[x].append(y)
            else:
                d[x] = [y]
    return d

def check_valid_candidate(candidate, d):
    for i in range(len(candidate)):
        if candidate[i] not in d:
            continue
        if any([item in d[candidate[i]] for item in candidate[i+1:]]):
            return False
    return True
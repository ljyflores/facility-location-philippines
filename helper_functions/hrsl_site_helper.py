import pandas as pd
import shapely
import numpy as np
import geopandas as gpd
import requests
import itertools

import matplotlib
import matplotlib.pyplot as plt

import random

def make_site_hrsl_dict(sites, hrsl, pop_col='population_2020'):
    """
    Inputs:
    - sites: Geodataframe containing candidate sites, geometry
             corresponds to 30 minute isochrone polygon around the site
    - hrsl: Geodataframe containing HRSL sites, geometry corresponds
            to coordinates of HRSL population
            
    Output:
    Dictionary with site index as key, and list containg tuples 
    (<HRSL index>, <population>) as value
    """
    # Merge the chosen HRSL points with the populations
    sites.crs, hrsl.crs = 'EPSG:4326', 'EPSG:4326'
    sites_w_pop = gpd.sjoin(sites, hrsl, how='left', op='contains')
    
    # Collect all the HRSL indices per site
    keys = sites_w_pop['index']
    values = [(a,b) for (a,b) in zip(sites_w_pop['index_right'], 
                                     sites_w_pop[pop_col])]
    d = {}
    for (key,value) in zip(keys,values):
        if key in d:
            d[key].append(value)
        else:
            d[key] = [value]
    return d

def make_site_set_hrsl_dict(site_hrsl_dict, 
                             site_sets,
                             id_col = 'set_id',
                             site_col = 'site_set'):
    """
    Inputs:
    - site_hrsl_dict: Dictionary with site index as key,
                      and list of tuples (hrsl_idx, population) 
                      as value
    - site_sets: Dataframe with column for set_ids <id_col>
                 and column with list of sets <site_col>
    - id_col: Column name of site_sets for set IDs
    - site_col: Column name of site_sets for sites
    
    Output:
    Dictionary with set ID as key, and a dictionary as the value
    Dictionary contains (1) site_ids: corresponding site IDs,
                        (2) population: total population,
                        (3) hrsl_ids: corresponding HRSL IDs
    """
    site_set_hrsl_dict = {}
    for i, row in site_sets.iterrows():
        compiled_HRSL = [site_hrsl_dict[k] for k in row[site_col]]
        compiled_HRSL = set(list(itertools.chain.from_iterable(compiled_HRSL)))
        compiled_idx = [item[0] for item in compiled_HRSL]
        compiled_pop = [item[1] for item in compiled_HRSL]
        site_set_hrsl_dict[row[id_col]] = {'site_ids': row[site_col],
                                           'population': compiled_pop,
                                           'hrsl_ids': compiled_idx}
    return site_set_hrsl_dict

def add_coords(site_set_hrsl, site_coords):
    """
    Inputs:
    - site_set_hrsl: Dictionary with site_set ID as key, and dictionary containing
                     list of site_ids ('site_ids') as value
    - site_coords: Dictionary with site ID as key, and coordinates as value
    """
    for key in site_set_hrsl.keys():
        site_set_hrsl[key]['coords'] = [site_coords[idx] for idx in site_set_hrsl[key]['site_ids']]
    return site_set_hrsl


import pandas as pd
import shapely
import numpy as np
import geopandas as gpd
import requests
import itertools
import functools
import matplotlib
import matplotlib.pyplot as plt
from helper_functions.hrsl_site_helper import *

import random

# Formulas from Jia et al (2019)
def expected_demand(population, dist, beds, u, a, s):
    return u*(population**a)*(beds**s)*dist

def dist_decay(time, b, t):
    return 1/(1+((time/t)**b))

def attractiveness_score(beds, dist, s=0.40):
    """
    Input: 
    - beds: Number of beds at facility
    - dist: Travel time in minutes from facility to population center
    
    Calculates attractiveness score of a hospital from a population center as:
    beds^constant x distance
    """
    return (beds**s)*dist
    
def expected_demand_norm(population, dist, norm, beds, u=0.20, a=0.66):
    """
    Input: 
    - population: Population at population center
    - dist: Traven time in minutes from facility to population center
    - norm: Total attractiveness score of other hospitals
    - beds: Number of beds at new facility
    - u: Constant, default=0.20
    - a: Constant, default=0.66
    
    Calculates expected number of visitors to a facility from a population center as
    u*(population**a)*attractiveness_score(beds, dist)/(attractiveness_score(beds, dist)+norm)
    """
    return u*(population**a)*attractiveness_score(beds, dist)/(attractiveness_score(beds, dist)+norm)

# Mapreduce functions
def reducer(p, c):
    if p[1] > c[1]:
        return p
    return c

def mapreduce(site_set_hrsl, mapper):
    print("Map Step")
    #step 1
    mapped = map(mapper, iter(site_set_hrsl.values()))
    mapped = zip(iter(site_set_hrsl.values()), mapped)
    print("Reduce Step")
    #step 2:
    reduced = functools.reduce(reducer, mapped)
    return reduced

# Metric calculation functions
def calculate_no_competition(demand_matrix, k, site_coords):
    scores = demand_matrix.sum(axis=0)
    result = {0: {'site_ids':sorted(range(len(scores)), key=lambda i: scores[i])[-k:]}}
    result = add_coords(result, site_coords)
    result = (result[0], sum(sorted(scores)[-k:]))
    return result

def compute_metric_population_single(d):
    """
    Input:
    - site_set_hrsl: Dictionary of site sets
    
    Output:
    - site_set_hrsl: Dictionary with 'metric' key appended 
    """
    return np.nansum(d['population'])

def compute_metric_dist_decay_single(dct, hrsl_pop_c, exp_demand):
    """
    Input:
    - site_set_hrsl: Dictionary of site sets
    - hrsl_pop: List of HRSL populations, ordered by HRSL_id
    - exp_demand: n_HRSL_points x n_facilities matrix of expected demand values
    
    Output:
    - site_set_hrsl: Dictionary with 'metric' key appended 
    """    
    total_demand = 0
    hrsl_pop = np.array(hrsl_pop_c.copy())
    for site_id in dct['site_ids']:
        actual_demand = np.minimum(exp_demand[:,site_id], hrsl_pop)
        hrsl_pop = hrsl_pop - actual_demand
        total_demand += np.nansum(actual_demand)
    return total_demand

def compute_exp_demand(hrsl_pop_column, time_matrix_c, beds, u, a, s, b, t):
    demand = time_matrix_c.copy()
    compute_exp_demand = lambda col: [expected_demand(pop, 
                                                      dist_decay(d, b, t), 
                                                      beds, 
                                                      u, 
                                                      a, 
                                                      s) for (pop, d) in zip(hrsl_pop_column, 
                                                                             col)]
    exp_demand = np.apply_along_axis(compute_exp_demand, 0, demand)
    return exp_demand

def compute_hospital_attractiveness(hosp_matrix_c, bed_capacity):
    """
    Input:
    - hosp_matrix_c: n_HRSL_points x n_hospitals matrix of drive times
    - bed_capacity: List of bed capacities of n_hospitals in the same order as hosp_matrix_c
    
    Output:
    - hosp_matrix[0]: List of total hospital_attractiveness values for n_sites
    """
    # Turn matrices into dataframes
    hosp_matrix = pd.DataFrame(hosp_matrix_c.copy()) if not 'DataFrame' in str(type(hosp_matrix_c)) else hosp_matrix_c.copy()

    # Compute "attractiveness score" from each population center to each existing hospital
    for i, row in hosp_matrix.iterrows():
        hosp_matrix.loc[i] = [attractiveness_score(beds, dist_decay(d)) for (beds,d) in zip(bed_capacity, row)]

    # Get the current total attractiveness score at each HRSL point
    hosp_matrix = hosp_matrix.sum(axis=1).reset_index()
    
    return hosp_matrix[0]

def compute_site_attractiveness(time_matrix_c, hospital_attractiveness, hrsl_pop):
    """
    Input:
    - time_matrix_c: n_HRSL_points x n_sites matrix of drive times
    - hospital_attractiveness: List of total hospital_attractiveness values for n_sites
    - hrsl_pop: List of populations for n_HRSL_points
    
    Output:
    - site_attractiveness: Dictionary of n_sites values where the key is the site ID,
    and the attractiveness value is the value
    """
    # Turn the site set-HRSL matrix into a dataframe
    time_matrix = pd.DataFrame(time_matrix_c.copy()) if not 'DataFrame' in str(type(time_matrix_c)) else time_matrix_c.copy()
    
    # Add the attractiveness score column to the site set-HRSL matrix
    time_matrix['norm'] = hospital_attractiveness
    
    # Calculate score per site set-HRSL pair, 
    # incorporating population, distance decay, 
    # current attractiveness score from other hospitals, and beds
    for i, row in time_matrix.iterrows():
        if i%1000==0:
            print(i)
        for col in time_matrix.columns[:-1]:
            time_matrix.loc[i, col] = expected_demand_norm(hrsl_pop[i], 
                                                           dist_decay(row[col]),
                                                           row['norm'],
                                                           100)
    # Remove the norm before summing
    time_matrix = time_matrix.drop('norm', axis=1)

    # Sum total expected visitors per site
    site_attractiveness = time_matrix.T.sum(axis=1).reset_index()
    
    # Turn into dictionary
    site_attractiveness = {key:value for (key,value) in zip(site_attractiveness['index'],
                                                            site_attractiveness[0])}
    return site_attractiveness

def compute_metric_competition(site_set_hrsl, site_attractiveness):
    """
    Input:
    - site_set_hrsl: Dictionary of site sets
    - site_attractiveness: List of attractiveness values for n_sites
    
    Output:
    - site_set_hrsl: Dictionary with 'metric' key appended 
    """
    for key in site_set_hrsl.keys():
        site_set_hrsl[key]['metric'] = np.nansum([site_attractiveness[idx] for idx in site_set_hrsl[key]['site_ids']])
    return site_set_hrsl

def compute_metric_competition_single(d, site_attractiveness):
    """
    Input:
    - site_set_hrsl: Dictionary of site sets
    - site_attractiveness: List of attractiveness values for n_sites
    
    Output:
    - Competition metric score
    """
    return np.nansum([site_attractiveness[idx] for idx in d['site_ids']])

# def pick_highest_metric(site_set_hrsl, metric='metric'):
#     """
#     Input:
#     - site_set_hrsl: Dictionary of site sets
    
#     Output:
#     - Dictionary for the site_set with the highest `metric` value
#     """
#     max_idx = np.argmax([val[metric] for val in site_set_hrsl.values()])
#     return site_set_hrsl[list(site_set_hrsl.keys())[max_idx]]


# def compute_metric_population(site_set_hrsl):
#     """
#     Input:
#     - site_set_hrsl: Dictionary of site sets
    
#     Output:
#     - site_set_hrsl: Dictionary with 'metric' key appended 
#     """
#     for key in site_set_hrsl.keys():
#         site_set_hrsl[key]['metric'] = np.nansum(site_set_hrsl[key]['population'])
#     return site_set_hrsl

# def compute_metric_dist_decay(site_set_hrsl, hrsl_pop_c, time_matrix,\
#                              beds=1, u=0.20, a=0.66, s=0.40, b=2.14, t=6.29):
#     """
#     Input:
#     - site_set_hrsl: Dictionary of site sets
#     - hrsl_pop_c: List of HRSL populations, ordered by HRSL_id
#     - time_matrix: n_HRSL_points x n_facilities matrix of drive times
    
#     Output:
#     - site_set_hrsl: Dictionary with 'metric' key appended 
#     """
#     for site_set in site_set_hrsl.keys():
#         hrsl_pop = hrsl_pop_c.copy()
#         site_ids = site_set_hrsl[site_set]['site_ids']
#         total_demand = 0
#         for site_id in site_ids:
#             exp_demand = [expected_demand(pop, dist_decay(d, b, t), beds, u, a, s) for (pop, d) in zip(hrsl_pop, time_matrix[:,site_id])]
#             actual_demand = [min(expected, actual) for (expected,actual) in zip(exp_demand, hrsl_pop)]
#             hrsl_pop = hrsl_pop - actual_demand
#             total_demand += np.nansum(actual_demand)
#         site_set_hrsl[site_set]['metric'] = total_demand
#     return site_set_hrsl
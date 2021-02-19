import pandas as pd
import shapely
import numpy as np
import itertools

import random

def compute_expected_demand(hrsl, hosp, union, pop_col='population_2020'):
    
    # Subset the dataframe to HRSL populations (1) with people and (2) within 30 minutes of a facility
    temp_hrsl = hrsl.loc[hrsl[pop_col]>0]
    covered_idx = [i for i,geom in enumerate(temp_hrsl['geometry']) if geom.within(union)]
    temp_hrsl = temp_hrsl.loc[covered_idx]
    
    # Iterate randomly through the rows and reduce capacity at each iteration
    for i, row in temp_hrsl.loc[random.sample(list(temp_hrsl.index), temp_hrsl.shape[0])].iterrows():
    
        # Get all the hospitals which are within 30 mins of the HRSL point
        temp = hosp.loc[hosp.geometry.apply(lambda x: x.contains(row['geometry']))]

        # If the HRSL's population can be completely serviced, set to 0
        if sum(temp['capacity']) >= row[pop_col]:
            hosp.loc[temp.index,'capacity'] = temp['capacity']-(row[pop_col]*temp['capacity']/sum(temp['capacity']))
            hrsl.loc[i,pop_col] = 0
        # If not, subtract only the serviceable population
        else:
            hrsl.loc[i,pop_col] -= sum(temp['capacity'])
            hosp.loc[temp.index,'capacity'] = 0
    return hrsl
import pandas as pd
import shapely
import numpy as np
import geopandas as gpd
import requests
import itertools
from shapely.ops import cascaded_union
import matplotlib
import matplotlib.pyplot as plt
import random
from helper_functions.hrsl_site_helper import *

def plot_result(result, mun, background=False, hrsl=None, hrsl_col=None, hosp=None, height=5, width=7, norm=False):
    """
    Inputs:
    - result: Dictionary containing key called 'coords' for points to plot, and 'metric' to show
    - mun: Geodataframe representing the entire province/municipality being studied
    - background: Boolean for whether the hrsl/hosp values will be added, default to False
    - hrsl: HRSL geodataframe
    - hrsl_col: Column to plot in `hrsl`
    - hosp: Hospital geodataframe
    - height: Height per image, Default to 5
    - width: Width per image, Default to 7
    - norm: Specifies upper limit for HRSL legend, Default to False
    """
    
    plot = gpd.GeoDataFrame(result[0]['coords'], geometry=result[0]['coords'], crs='EPSG:4236')
    norm = matplotlib.colors.Normalize(0,norm,clip=True) if norm else matplotlib.colors.Normalize(0,max(hrsl[hrsl_col]),clip=True)
    ax = mun.plot(color='white', edgecolor='black', linewidth=0.25, figsize=(width,height))
    if background:
        hrsl.plot(ax=ax, column=hrsl_col, cmap='Blues', markersize=0.25, legend=True, norm=norm)
        hosp.plot(ax=ax, color='orange', marker='P', markersize=50)
    plot.plot(ax=ax, color='red', markersize=50)
    ax.axis('off')
    
def plot_multiple(results, shape, mun, background=False, hrsl=None, hrsl_col=None, hosp=None, height=8, width=6):
    """
    Inputs:
    - results: List of dictionaries containing key called 'coords' for points to plot, and 'metric' to show
    - shape: Tuple of (width, height) where width is number of plots on x axis, height on y axis
    - mun: Geodataframe representing the entire province/municipality being studied
    - background: Boolean for whether the hrsl/hosp values will be added, default to False
    - hrsl: HRSL geodataframe OR list of HRSL geodataframes if different ones are to be used per plot
    - hrsl_col: Column to plot in `hrsl`
    - hosp: Hospital geodataframe
    - height: Height per image, Default to 8
    - width: Width per image, Default to 6
    """
    fig, axs = plt.subplots(shape[0], shape[1], figsize=(shape[0]*width, shape[1]*height))
    fig.subplots_adjust(hspace = .2, wspace=.001)
    if 'GeoDataFrame' in str(type(hrsl)):
        hrsl = [hrsl]*(shape[0]*shape[1])
    for i in range(len(results)):
        result = results[i]
        plot = gpd.GeoDataFrame(result[0]['coords'], geometry=result[0]['coords'], crs='EPSG:4236')
        if shape[0]==1:
            mun.plot(ax=axs[i], color='white', edgecolor='black', linewidth=0.25)
            if background:
                hrsl[i].plot(ax=axs[i], column=hrsl_col, cmap='Blues', markersize=0.25, legend=True)
                hosp.plot(ax=axs[i], color='orange', marker='P', markersize=50)
            plot.plot(ax=axs[i], color='red')
        else:
            mun.plot(ax=axs[i//shape[1]][i%shape[1]], color='white', edgecolor='black', linewidth=0.25)
            if background:
                hrsl[i].plot(ax=axs[i//shape[1]][i%shape[1]], column=hrsl_col, cmap='Blues', markersize=0.25, legend=True)
                hosp.plot(ax=axs[i//shape[1]][i%shape[1]], color='orange', marker='P', markersize=50)
            plot.plot(ax=axs[i//shape[1]][i%shape[1]], color='red')
    plt.show()
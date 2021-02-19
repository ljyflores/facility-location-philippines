import os
import numpy as np
import pandas as pd
import cv2 
import tifffile as tiff
import matplotlib.pyplot as plt
import shapely
import geopandas as gpd
from helper_functions.mapbox_helper import haversine
import pickle

colors = {
    'red': [np.array([0,0,200]), np.array([25,25,255])],
    'violet': [np.array([50,80,30]), np.array([255,105,255])],
    'yellow': [np.array([0,200,200]), np.array([50,255,255])],
    'pink': [np.array([150,150,180]), np.array([211,177,241])]
}
hazard_legend = {
    'Earthquake Induced Landslide': ['red'],
    'Ground Rupture': ['red'],
    'Ground Shaking': ['red'],
    'Storm Surge': ['red'],
    'Flood': ['red'],
    'Liquefaction': ['red']
}

def pad_folder_images(root):
    # Read in the images
    image_names = [f"{root}/{item}" for item in os.listdir(root) if '.png' in item]
    images = [cv2.imread(image) for image in image_names]
    
    # Determine max size and size to be padded
    max_size = np.max([image.shape for image in images], axis=0)
    pad_size = [max_size-np.array(image.shape) for image in images]
    
    # Pad and output the images
    output_img = []
    for i in range(len(images)):
        new_img = cv2.copyMakeBorder(images[i], 
                                     top = 0,
                                     bottom = pad_size[i][0], 
                                     left = 0,
                                     right = pad_size[i][1], 
                                     borderType = cv2.BORDER_CONSTANT, 
                                     value = 0)
        output_img.append(new_img)
    return image_names, output_img

def mask_hazard(img, color_filters, colors=colors):
    mask_lst = [cv2.inRange(img, colors[color][0],colors[color][1]) for color in color_filters]
    mask = sum(mask_lst)
    return mask

def produce_masks(filenames, images, hazard_legend=hazard_legend):
    mask_lst = []
    for (name, img) in zip(filenames, images):
        for hazard in hazard_legend.keys():
            if hazard in name:
                mask_lst.append(mask_hazard(img, hazard_legend[hazard]))
                break
    return mask_lst

def find_hazard_coords(mask_lst, top, bottom, left, right, indiv=False):
    mask = sum(mask_lst)
    lon = np.linspace(left, right, mask.shape[1])
    lat = np.linspace(bottom, top, mask.shape[0])
    mask = mask.flatten()
    lon_tiled = np.tile(lon, len(lat))
    lat_tiled = np.tile(np.array([lat]).T, len(lon)).flatten()
    if indiv:
        result = []
        for m in mask_lst:
            df = pd.DataFrame({'longitude':lon_tiled,
                               'latitude':lat_tiled,
                               'mask':m.flatten()})
            df = df.loc[df['mask']>250].reset_index(drop=True)
            result.append(df)
        return result
    else:
        df = pd.DataFrame({'longitude':lon_tiled,
                           'latitude':lat_tiled,
                           'mask':mask})
        df = df.loc[df['mask']>250].reset_index(drop=True)
        return df

def produce_hazard_coords(folder, top, bottom, left, right):
    filenames, images = pad_folder_images(folder)
    mask_lst = produce_masks(filenames, images)
    hazard_coords = find_hazard_coords(mask_lst, 
                                   top, 
                                   bottom,
                                   left,
                                   right)
    return hazard_coords

def produce_hazard_coords_by_hazard(folder, top, bottom, left, right):
    filenames, images = pad_folder_images(folder)
    mask_lst = produce_masks(filenames, images)
    hazard_coords = find_hazard_coords(mask_lst, 
                                   top, 
                                   bottom,
                                   left,
                                   right,
                                   indiv=True)
    return hazard_coords, filenames

def subset_hazards(df_hazards, mun):
    """
    Inputs:
    - df_hazards: Dataframe containing 'longitude' and 'latitude' column corresponding to hazard sites
    - mun: Geodataframe corresponding to province/municipality 
    
    Outputs:
    - df_hazards: Subsetted geodataframe from df_hazards, geometry corresponds to coordinates of hazard sites
    """
    # Subset hazardous sites to relevant region
    df_hazards['geometry'] = [shapely.geometry.Point(x,y) for (x,y) in zip(df_hazards['longitude'],df_hazards['latitude'])]
    df_hazards = gpd.GeoDataFrame(df_hazards, geometry=df_hazards['geometry'], crs='EPSG:4326')
    df_hazards = gpd.sjoin(mun, df_hazards, how='left', op='contains')
    
    # Keep only longitude and latitude columns
    df_hazards = df_hazards[['longitude','latitude']].reset_index(drop=True)
    df_hazards['geometry'] = [shapely.geometry.Point(x,y) for (x,y) in zip(df_hazards['longitude'],df_hazards['latitude'])]
    df_hazards = gpd.GeoDataFrame(df_hazards, geometry=df_hazards['geometry'], crs='EPSG:4326')
    
    return df_hazards

def select_good_sites(all_sites, df_hazards, km=1):
    KM_TO_DEGREES = 0.009009
    
    # Turn coordinates into matrices
    site_coords = np.array([[pt.x,pt.y] for pt in all_sites['geometry']])
    hazard_coords = np.array([[x,y] for (x,y) in zip(df_hazards['longitude'],df_hazards['latitude'])])

    # Identify all coordinates at least 1km from a hazard point
    good_sites = []
    for i, site in enumerate(site_coords):
        min_manhattan = np.min(np.sum(np.abs(hazard_coords-site), axis=1))
        if min_manhattan >= (KM_TO_DEGREES*km):
            good_sites.append(i)

    # Keep only the good sites
    all_sites = all_sites.loc[all_sites['index'].isin(good_sites)]
    
    return all_sites

# Hazard Mapping Functions
def open_pkl(file):
    with (open(file,"rb")) as openfile:
        result = pickle.load(openfile)
    return result
def to_gdf(df):
    df['geometry'] = [shapely.geometry.Point(x,y) for (x,y) in zip(df['longitude'],df['latitude'])]
    df = gpd.GeoDataFrame(df, geometry=df['geometry'], crs='EPSG:4326')
    return df

# Identify distance to hazard per site
def return_closest_hazards(coord_lst, hazard_coords, filenames):
    result = []
    for i, hazard_df in enumerate(hazard_coords):
        min_dist = []
        for site in coord_lst:
            min_dist.append(min(hazard_df['geometry'].apply(lambda x: haversine(site, x))))
        result.append((filenames[i], min_dist))
    return result

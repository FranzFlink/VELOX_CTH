###just some functions that are needed to load simulations/ velox high resoulation timeseries


import os
import glob
import numpy as np
import re
import netCDF4 as nc
import datetime 
from scipy.interpolate import interp1d

def sortby(x):
    return int(re.findall( '\d+', x)[-1])

def load_ts_high_res_unmasked(flight_path):
    path = os.path.join('/projekt_agmwend/data/EUREC4A/06_Flights',flight_path ,'VELOX/VELOX_327kveL/10x10_Central_Pixel/')
    filename    =  flight_path[7:]+'_SOD_Center_10x10_Pixel_Envelope_Diff_Minus_Inhomo.txt'
    ts_central = np.loadtxt(path+filename, skiprows = 1).T

    return ts_central

def load_sims(flight_path, v=1):
    
    sim_dir = os.path.join('/projekt_agmwend/data/EUREC4A/06_Flights', flight_path, 'VELOX/VELOX_327kveL/Simulation_Cloud_Free_Different_Zout')
    sim_path = sorted(glob.glob(sim_dir + '/*'), key = sortby)
    sims = [] 
    sim_path =  [item for item in sim_path if 'REPTRAN-MEDIUM' in item]
    if len(sim_path) == 0:
        sim_path =  [item for item in sim_path if 'filter_function' in item]
    for item in sim_path:
        sims.append(np.loadtxt(item,skiprows=36,usecols=(0,3)))    
    sims = np.array(sims)
    
    return sims
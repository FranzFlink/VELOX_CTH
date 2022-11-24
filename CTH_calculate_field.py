### filename : cth_2d_calc.py
### author   : Joshua Müller  
### For a detailed explanation of this file, have a look at the interactive Notebook "name.ipynb" or the 
### corresponding pdf-version "name.pdf"

import numpy as np 
import datetime 
import os 
import glob
import xarray as xr
import eurec4a
from intake import open_catalog
from scipy.stats import linregress
from numba import njit
from tqdm import tqdm
from CTH_helper import *


from time import time as pytime

f = 'Flight_20200205a'

fnum = str(f[7:-1])

Y, M, D = int(fnum[0:4]), int(fnum[4:6]), int(fnum[6:])
f_format = f'HALO-{fnum[6:]}{fnum[4:6]}'

path = '/projekt_agmwend/data/EUREC4A/06_Flights/'+f+'/VELOX/VELOX_327kveL/'
tb_name = 'EUREC4A_HALO_VELOX_BT_Filter_01_'+str(f[7:-1])+'_v0.4.nc'
cm_name = 'EUREC4A_HALO_VELOX_cloudmask_'+str(f[7:-1])+'_v4.1.nc'
params_name = glob.glob(os.path.join('/projekt_agmwend/data/EUREC4A/06_Flights',f,'BAHAMAS')+'/*.nc')[0]
wl_name = glob.glob(os.path.join('/projekt_agmwend/data/EUREC4A/06_Flights', f, 'WALES','*V2.1.nc'))[0]

#name of the output file to be generated
cth_name = 'EUREC4A_HALO_VELOX_CTH_'+f+'_'

### xrcm : Cloud Mask 
### xrtb : Brightness Temperature
### xrwl : WALES LIDAR 
### xrparamas : Flight parameters (altitude, pitch and roll)
### xrcoff : lapse_rate and offset, generated from CTH_calculate_coffs.py

xrcm  = xr.open_dataset(path+cm_name)
xrtb = xr.open_dataset(path+tb_name)
xrwl = xr.open_dataset(wl_name)
xrparams = xr.open_dataset(params_name)
xrcoff = xr.open_dataset(f'/home/jomueller/{fnum}_coffs.nc')

all_flight_segments = eurec4a.get_flight_segments()
circles = [item for item in all_flight_segments['HALO'][f'HALO-{str(f[11:-1])}']['segments'] if 'circle' in item['name'] and len(item['dropsondes']['GOOD']) > 5 ]

### Loading the simulated Brighness temperatures, these were simulated for different flying altitudes (height) to calculate the
### absorbtion due to the atmosphere
sims = load_sims(f)

sim_datetime = [datetime.datetime(Y,M,D) + datetime.timedelta(seconds=sims[0,i,0]) for i in range(len(sims[0,:,0]))]
sim_np64 = np.array(sim_datetime, dtype=np.datetime64)
    
if sims.shape[0] == 12:
    height = np.array([250, 500, 1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])
else:
    height = np.array([250, 500, 750, 1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])

xr_sim = xr.Dataset(
    data_vars=dict(
        dT=(["height", "time"], sims[:,:,1]),

    ),
    coords=dict(
        time=sim_np64,
        height=height,
    ),
    attrs=dict(description=f"VELOX{fnum} simulated brightness temperature for different airplane altitudes"))


sim_interp = xr_sim.interp(height=np.arange(1,15001),
    method="linear",
    kwargs={"fill_value": "extrapolate"},).interp_like(xrtb)

xrHALO = xr.Dataset(
    data_vars=dict(
        lat=(["time"], xrparams['IRS_LAT'].values),
        lon=(["time"], xrparams['IRS_LON'].values),
        alt=(["time"], xrparams['IRS_ALT'].values),
        roll=(["time"], xrparams['IRS_PHI'].values),
        pitch=(["time"], xrparams['IRS_THE'].values),
        hdg=(["time"], xrparams['IRS_HDG'].values),
    ),
    coords=dict(

        time=xrparams['TIME'].values,
    ),
)


### As the vza given is the same for every timestep, here is the "corrected vza" including pitch and roll angles


def compute_vza(roll, pitch):

    alpha, beta, pitch, roll = np.radians(35), np.radians(28), np.radians(pitch), np.radians(roll)
    alpha_rolled = np.abs(np.linspace(-alpha/2 + roll, alpha/2 + roll, 640)) 
    beta_pitched = np.abs(np.linspace(-beta/2 + pitch, beta/2 + pitch, 512))
    X, Y = np.meshgrid(beta_pitched, alpha_rolled, copy=False)
    
    return np.degrees(np.arctan(((np.tan(X))**2 + (np.tan(Y))**2)**0.5))

### calculate the distance between the camera and any pixel including pitch and roll 

def dist3D(vza, height):
    vza = np.radians(vza)
    return (height**2 * (np.tan(vza)**2 + 1))**0.5


def precalc_percentiles(cth):

    percentiles = np.array([cth.quantile(q) for q in np.linspace(0,1,102)])
    percentiles_range = [np.where((cth >= percentiles[i-1]) & (cth <= percentiles[i+1]))[0] for i in range(1,101)]

    return np.array([cth[wales_index].mean() for wales_index in percentiles_range])


percentiles_mean = precalc_percentiles(xrwl['cloud_top'])

### guessing the cth with a statistical approach - if the brightness temperature is between the i-th and i+1th  
### percentile compared to the nadir brightness temperature, then so should the cth be in the i-th to i+1th percentile 
### in the corresponding cth distribution measured by wales lidar. 

@njit

def guess_cth(bt_array, cm_array ,bt_center, percentiles_mean=percentiles_mean[::-1]):

    cth_array = np.zeros(bt_array.shape) * np.nan

    for i in range(bt_array.shape[0]):
        for j in range(bt_array.shape[1]):
            if cm_array[i,j] > 0:
                q = int(np.count_nonzero(bt_array[i,j] > bt_center) / len(bt_center) * 100)
                cth_array[i,j] = percentiles_mean[q]

    return cth_array




params_interp = xrHALO.interp_like(xrtb)
coff_interp   = xrcoff.interp_like(xrtb)

#filled_offset = coff_interp.offset.interpolate_na(dim='time', method='linear').coarsen(time=10, boundary='trim').mean().interp_like(xrtb)
#filled_lapse_rate = coff_interp.lapse_rate.interpolate_na(dim='time', method='linear').coarsen(time=10, boundary='trim').mean().interp_like(xrtb)

averaged_offset = coff_interp.offset.coarsen(time=60, boundary='trim').mean().interp_like(xrtb.time)
averaged_lapse_rate = coff_interp.lapse_rate.coarsen(time=60, boundary='trim').mean().interp_like(xrtb.time)

cth_array = np.zeros(xrtb['BT_2D'].shape)
N = xrtb.time.shape[0]
CM_center = xrcm.cloud_mask.isel(y=slice(251,261), x=slice(315, 325)).mean(dim = {'x', 'y'})

# tqdm for showing progressbar 

print('...')

cth_array = np.zeros(xrtb['BT_2D'].shape) 
bt_center = xrtb['BT_Center'][CM_center > 0].values
cms = xrcm['cloud_mask'].transpose('x', 'y', 'time').values
alt, roll, pitch = params_interp['alt'].values, params_interp['roll'].values, params_interp['pitch'].values
BT_sims = sim_interp['dT'].values
BT_timesteps = xrtb['BT_2D'].values
lapse_rates = averaged_lapse_rate.interpolate_na(dim="time", method="linear").values
offsets = averaged_offset.interpolate_na(dim="time", method="linear").values

ranges = [np.arange(np.where(xrtb.time==xrtb.time.sel(time=circles[i]['start'], method='nearest'))[0], np.where(xrtb.time==xrtb.time.sel(time=circles[i]['end'], method='nearest'))[0]) for i in range(len(circles))]
c = 0

for circle in ranges:
    c += 1
    print(f'processing circle {c} out of {len(circles)}')
    circle_name = f"C{c}.nc"
    j = 0
    cth_array = np.zeros((len(circle), 640, 512))
    CF = np.zeros(len(circle))

    for i in tqdm(circle):

        if (xrcm.CF_max.isel(time=i) > 0.0003):

            # lapse_rate = averaged_lapse_rate.isel(time=i).values
            # offset = averaged_offset.isel(time=i).values
            # alt, roll, pitch = params_interp['alt'].isel(time=i), params_interp['roll'].isel(time=i), params_interp['pitch'].isel(time=i)
            # distance = np.array(dist3D(compute_vza(pitch, roll), alt.values), dtype=np.int64)
            # index = np.where((xrcm['cloud_mask'].transpose('x', 'y', 'time')[:,:,i] > 0))
            # BT_sim = sim_interp['dT'].isel(time=i)
            # BT_timestep = xrtb['BT_2D'].isel(time=i)
            # cth_guess = np.array(guess_cth(BT_timestep.values, xrcm['cloud_mask'].transpose('x', 'y', 'time')[:,:,i].values ,xrtb['BT_Center'][CM_center > 0].values), dtype=np.int32)
            # BT_cloudy = BT_timestep.values[index]
            # # xarray indexing is weird ... and slow -> change to numpy indexing 
            # atmos_absorbtion_cloud_top = BT_sim.sel(height=cth_guess[index]).values
            # atmos_absorbtion_airplane  = BT_sim.sel(height=distance[index]).values
            # cth_array[i][index] =  lapse_rate * ( (atmos_absorbtion_cloud_top - atmos_absorbtion_airplane) + BT_cloudy ) + offset

            # Changing to numpy as it's much faster than xarray 
    
            distance = np.array(dist3D(compute_vza(pitch[i], roll[i]), alt[i]), dtype=np.int64)
            index = np.where((cms[:,:,i] > 0))
            lapse_rate, offset = lapse_rates[i], offsets[i]

            cth_guess = np.array(guess_cth(BT_timesteps[i,:,:], cms[:,:,i], bt_center), dtype=np.int32)

            BT_cloudy = BT_timesteps[i][index]

            atmos_absorbtion_cloud_top = BT_sims[cth_guess[index], i]
            atmos_absorbtion_airplane  = BT_sims[distance[index], i]

            cth_array[j][index] =  lapse_rate * ( (atmos_absorbtion_cloud_top - atmos_absorbtion_airplane) + BT_cloudy ) + offset
            
            CF[j] = np.count_nonzero(cth_array[j]) / (512*640)
        else: 
            CF[j] = 0
        j += 1
        
### Making a pretty output_dataset with some attributes and included vza, vaa for each circle

    output_dataset = xr.Dataset(
        data_vars = dict(
            #vza = xrtb['vza'],
            #vaa = xrtb['vza'],
            #cloud_mask = xrcm['cloud_mask']
            CTH = (['time','x','y'], np.array(cth_array, dtype=np.int32)),
            CF_max = (['time'], CF)
        ),
        coords = dict(
            time = xrtb['time'][circle]
        ),
        attrs = dict(
            title = 'Two-dimensional cloud-top height with 1 Hz temporal resolution derived from VELOX brightness temperature during the EUREC4A field campaign.',
            version = 'Version v0.31 from 2022-02-06',
            comment_1 = 'cloud-height is derived from combination of 7.70 -12.00 micrometer VELOX brightness temperature with dropsondes and cross-calibrated with WALES cloud-top height. Applied cloud_mask is provided with treshold cloud_mask == 2.' ,
            variable = 'CTH',
            author = 'Michael Schäfer, André Ehrlich, Anna Luebke, Jakob Thoböll, Kevin Wolf, Joshua Müller, Manfred Wendisch',
            history = '2022-06-21 : updated dataset attributes, 2022-08-17 : improved cross calibration with wales lidar, 2022-09-30 : segmented files into circles',
            created_on = '2022-02-06'
        )
        )
    print(f'Saving CTHs to {cth_name}{circle_name}')
    output_dataset.to_netcdf(cth_name+circle_name)
    output_dataset.close()



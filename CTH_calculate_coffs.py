import numpy as np 
import datetime 
import os 
import glob
import xarray as xr
import eurec4a
from intake import open_catalog
from CTH_helper import *
from scipy.stats import linregress


### This script is used to calculate the nadir-CTH height with VELOX 10.8µm brightness temperature and Level 3 Joanne Dropsonde
### data during the EUREC4A. For the correction of the atmospheric absorbtion ... is used. A simple linear fit is used to determine
### the lapse rate and offset to compute the CTH : 
###     CTH = lapse rate * corrected brightness temperature + offset
### After the computation of the CTH, VELOX is compared with WALES Lidar to correct the offest:
###     corrected offset = offset - (CTH_VELOX - CTH_WALES)
### The output of the script is the nadir CTH timeseries, lapse rate and corrected offset

print('Start')
f = 'Flight_20200209a'
fnum = str(f[7:-1])

Y, M, D = int(fnum[0:4]), int(fnum[4:6]), int(fnum[6:])


f_format = f'HALO-{fnum[4:6]}{fnum[6:]}'

velox_path = '/projekt_agmwend/data/EUREC4A/06_Flights/'+f+'/VELOX/VELOX_327kveL/'
dropsonde_path = '/home/jomueller/EUREC4A_JOANNE_Dropsonde-RD41_Level_3_v2.0.0.nc'
wales_path = glob.glob(os.path.join('/projekt_agmwend/data/EUREC4A/06_Flights', f, 'WALES','*V2.1.nc'))[0]
params_path = glob.glob(os.path.join('/projekt_agmwend/data/EUREC4A/06_Flights',f,'BAHAMAS')+'/*.nc')[0]



ts_hr_unmasked = load_ts_high_res_unmasked(f)
dt_datetime_unmasked = [datetime.datetime(Y,M,D) + datetime.timedelta(seconds=ts_hr_unmasked[0,i]) for i in range(len(ts_hr_unmasked[0,:]))]
dt_np64_unmasked = np.array(dt_datetime_unmasked, dtype=np.datetime64)

xr_ts_hr_unmasked = xr.Dataset(
    data_vars=dict(
        BT=(["time"], ts_hr_unmasked[1,:]),
        dT=(["time"], ts_hr_unmasked[2,:]),

    ),
    coords=dict(
        time=dt_np64_unmasked,
    ),
    attrs=dict(description=f"VELOX{fnum} 10x10 NADIR PIXEL MEAN BT"))

xrwl = xr.open_dataset(wales_path)
xrds = xr.open_dataset(dropsonde_path)
xrparams = xr.open_dataset(params_path)

#import the circle segments and select the circles with more than 5 dropsondes for the corresponding flight

all_flight_segments = eurec4a.get_flight_segments()
circles = [item for item in all_flight_segments['HALO'][f_format]['segments'] if 'circle' in item['name'] and len(item['dropsondes']['GOOD']) > 5 ]
select_flight = np.array([f_format in word.item() for word in xrds['sonde_id']])

ds_sel = xrds.sel(sonde_id = select_flight)
ds_sel['ta'] = ds_sel['ta'] - 273.15

sims = load_sims(f)

print(sims)

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
    attrs=dict(description=f"VELOX{fnum} 10x10 NADIR PIXEL MEAN BT"))

sim_interp = xr_sim.interp(height=np.arange(1,13001),
    method="linear",
    kwargs={"fill_value": "extrapolate"},)



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

def dist3D(pitch, roll, height):
    # 3D Pythogras to calculate distance between airplane and ground 
    pitch, roll = np.radians(pitch), np.radians(roll)
    return np.sqrt(height**2 * (np.tan(pitch)**2 + np.tan(roll)**2 + 1) )

alt = dist3D(xrHALO['pitch'], xrHALO['roll'], xrHALO['alt'])


sim_temps = []
xrwl_interp   = xrwl['cloud_top'].interp_like(xr_sim)
xrHALO_interp = alt.interp_like(xr_sim)
f_alt_mean = xrHALO_interp.mean()

for time in xr_sim.time:
    f_alt = xrHALO_interp.sel(time=time)
    if np.isnan(xrwl_interp.sel(time=time)):
        cth = xrwl_interp.sel(time=slice(time-np.timedelta64(5, 'm'), time+np.timedelta64(5, 'm'))).mean()
        if np.isnan(cth):
            cth = 1
    else:
        cth = xrwl_interp.sel(time=time)
    if np.isnan(xrHALO_interp.sel(time=time)):
        f_alt = f_alt_mean

    
    T_correct = sim_interp.sel(time=time, height=int(cth)).dT.values - sim_interp.sel(time=time,height=int(f_alt)).dT.values

    sim_temps.append(T_correct)

print('...')

xr_sim = xr_sim.assign(dict(T = (['time'], sim_temps)))

sim_hr_interp= xr_sim['T'].interp_like(xr_ts_hr_unmasked)

BT_masked = xr_ts_hr_unmasked.BT.where((xr_ts_hr_unmasked.dT - xr_ts_hr_unmasked.BT) > 0.5) - 273.15

xr_ts_hr_corr = xr.Dataset(
    data_vars=dict(
        BT=(["time"], BT_masked.values + sim_hr_interp.values ),
    ),
    coords=dict(

        time=dt_np64_unmasked,
    ),

    attrs=dict(description=f"VELOX{fnum} CORRECTED TS"))

slopes, intercepts = [], []

for y in ds_sel['ta']:
    mask = ~np.isnan(y) & ~np.isnan(y.alt)
    slope, intercept, r, p, se = linregress(y[mask][:250], y.alt[mask][:250])
    slopes.append(slope)
    intercepts.append(intercept)

def height(temperature, lapse_rate, offset):
    return lapse_rate * temperature + offset

print('...')

xr_coff = xr.Dataset(
    data_vars=dict(
        lapse_rate = (['time'], slopes),
        offset = (['time'], intercepts)
    ),  
    coords=dict(

        time= ds_sel['launch_time'].values,
    ),

    attrs=dict(description="cut"))


xr_coff_interp = xr_coff.interp_like(xr_ts_hr_unmasked)

H = height(xr_ts_hr_corr['BT'],xr_coff_interp['lapse_rate'], xr_coff_interp['offset'])

delta = H - xrwl['cloud_top'].interp_like(H)
off_coarsed = (xr_coff_interp['offset'] - delta).coarsen(time=600, boundary='trim').mean().interp_like(xr_ts_hr_unmasked)
H = height(xr_ts_hr_corr['BT'] ,xr_coff_interp['lapse_rate'], off_coarsed)

print('...')


output_dataset = xr.Dataset(
    data_vars = dict(
        CTH = (['time'], H.values),
        lapse_rate = (['time'], xr_coff_corr_interp['lapse_rate'].values),
        offset  = (['time'], xr_coff_interp['offset'].values)
    ),
    coords = dict(
        time = H.time
    )
)

print(output_dataset['lapse_rate'].mean(), output_dataset['lapse_rate'].std())

output_dataset.to_netcdf(f'/projekt_agmwend/data/EUREC4A/11_VELOX-Tools/VELOX_CTH/coffs/{fnum}_coffs.nc')


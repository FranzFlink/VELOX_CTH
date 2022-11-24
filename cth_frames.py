import numpy as np 
import datetime 
import os 
import matplotlib.pyplot as plt
import glob
import time
from scipy.optimize import curve_fit
import seaborn as sns
import xarray as xr
import netCDF4 as nc
from tqdm import tqdm
import eurec4a
from intake import open_catalog
from scipy.stats import  linregress
import pandas as pd

print('start')
print('...')

def pixel_to_meter(pitch, roll, height, alpha=35.5, beta=28.7):
    pitch = np.radians(pitch)
    roll = np.radians(roll)
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    xlen = (np.tan(alpha/2 + roll) + np.tan(alpha/2 - roll)) * height
    ylen = (np.tan(beta/2 + pitch) + np.tan(beta/2 - pitch)) * height
    return xlen, ylen

all_flight_segments = eurec4a.get_flight_segments()

#Select flight and circle

circle = 6
f = 'Flight_20200131a'


fnum = str(f[7:-1])
Y, M, D = int(fnum[0:4]), int(fnum[4:6]), int(fnum[6:])
f_format = f'HALO-{fnum[6:]}{fnum[4:6]}'


os.chdir('/projekt_agmwend/data/EUREC4A/11_VELOX-Tools/VELOX_CTH/quicklooks/')
if not os.path.isdir(f'{fnum}'):
    os.mkdir(f'{fnum}')
os.chdir(f'/projekt_agmwend/data/EUREC4A/11_VELOX-Tools/VELOX_CTH/quicklooks/{fnum}')
if not os.path.isdir(f'C{circle}'):
    os.mkdir(f'C{circle}')
os.chdir(f'/projekt_agmwend/data/EUREC4A/11_VELOX-Tools/VELOX_CTH/quicklooks/{fnum}/C{circle}')

print('processed files are saved into :\n'+os.getcwd())

fnum = str(f[7:-1])
Y, M, D = int(fnum[0:4]), int(fnum[4:6]), int(fnum[6:])
f_format = f'HALO-{fnum[4:6]}{fnum[6:]}'

circles = [item for item in all_flight_segments['HALO'][f'{f_format}']['segments'] if 'circle' in item['name'] and len(item['dropsondes']['GOOD']) > 5 ]


path = '/projekt_agmwend/data/EUREC4A/06_Flights/'+f+'/VELOX/VELOX_327kveL/'
cth_name = 'EUREC4A_HALO_VELOX_CTH_'+fnum+'_v0.3.nc'
tb_name = 'EUREC4A_HALO_VELOX_BT_Filter_01_'+fnum+'_v0.4.nc'
cm_name = 'EUREC4A_HALO_VELOX_cloudmask_'+fnum+'_v4.1.nc'
wl_name = '/projekt_agmwend/data/EUREC4A/06_Flights'+ f+'WALES/EUREC4A_HALO_WALES_cloudtop_'+fnum+'_V2.1.nc'
params_name = glob.glob(os.path.join('/projekt_agmwend/data/EUREC4A/06_Flights',f,'BAHAMAS')+'/*.nc')[0]
cth_path = f'/projekt_agmwend/data/EUREC4A/06_Flights/{f}/VELOX/VELOX_327kveL/CTH/EUREC4A_HALO_VELOX_CTH_Flight_{fnum}a_C{circle}.nc'
xrcth = xr.open_dataset(cth_path)
xrcm = xr.open_dataset(path + cm_name)


xrtb = xr.open_dataset(path+tb_name)
xrparams = xr.open_dataset(params_name)
wl_path = glob.glob(os.path.join('/projekt_agmwend/data/EUREC4A/06_Flights', f, 'WALES','*V2.1.nc'))
xrwl = xr.open_dataset(wl_path[0])

xrds = xr.open_dataset('/home/jomueller/EUREC4A_JOANNE_Dropsonde-RD41_Level_3_v2.0.0.nc')

xrHALO = xr.Dataset(
    data_vars=dict(
        lat=(["time"], xrparams['IRS_LAT'].values),
        lon=(["time"], xrparams['IRS_LON'].values),
        alt=(["time"], xrparams['IRS_ALT'].values),
        roll=(["time"], xrparams['IRS_PHI'].values),
        pitch=(["time"], xrparams['IRS_THE'].values),
        hdg=(["time"], xrparams['IRS_HDG'].values),
        gs=(["time"], xrparams['IRS_GS'].values),

    ),
    coords=dict(

        time=xrparams['TIME'].values,
    ),
)


print('...')

tt = xrtb.time.values
tb = xrtb.BT_2D.values
cth = xrcth.CTH.values
ranges = [np.arange(np.where(xrtb.time==xrtb.time.sel(time=circles[i]['start'], method='nearest'))[0], np.where(xrtb.time==xrtb.time.sel(time=circles[i]['end'], method='nearest'))[0]) for i in range(len(circles))]
rg = ranges[circle-1]
p = 0
print('...')


pys = np.round(pixel_to_meter(xrHALO['pitch'], xrHALO['roll'], xrHALO['alt'])[1] / 512)
gs  = np.round(xrHALO['gs'])
dys = np.array(np.round(gs / pys, 0), dtype='int32')
time = [[tt[i] for j in range(dys[i])] for i in rg]
lt = [item for sublist in time for item in sublist]

import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

arrs = [tb[i,:,256: 256 + dys[i]] for i in rg]
cths = [cth[i,:,256: 256 + dys[rg[i]]] for i in range(len(rg))]
#mask = [cm[400: 400 + dys[i], :, i] for i in rg]
im = np.concatenate(arrs, axis=1)
im2 = np.concatenate(cths, axis=1, dtype=np.float32)

im2[im2 == 0] = im2[im2 ==0] * np.nan


pltcth = xrcth.CTH.isel(x=slice(315,325),y=slice(251,261)).mean(dim={'x', 'y'})
v_s, v_e = np.arange(0, 50000, 200), np.arange(10000, 60000, 200)

interp_time = [pd.date_range(time[i][0], time[i+1][0], periods=len(time[i])).values for i in range(len(time) -1)]
it = [item for sublist in interp_time for item in sublist]

cth_stripe = xr.Dataset(
    data_vars=dict(CTH = (['swath', 'time'], im2[:,:len(it)])),
    coords=dict(time = it, swath = np.arange(0, 640),
    ),
)


cth_min = np.nanquantile(im2, 0.001)
cth_max = np.nanquantile(im2, 0.999)

tb_min = np.min(im)
tb_max = np.max(im)



for index in tqdm(range(250)):

    view_start = v_s[index]
    view_stop = v_e[index]

    y_lims = [0, 640]
    x_lims = mdates.date2num([lt[view_start], lt[view_stop]])

    fig, (ax1,ax2,ax3) = plt.subplots(3, figsize=(35,10), sharex=True)



    fig1 = ax1.contourf(im[:,view_start:view_stop], levels=np.linspace(tb_min, tb_max, 12),extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
            cmap='rainbow')


            
    fig2 = ax2.contourf(im2[:,view_start:view_stop], levels=np.linspace(cth_min, cth_max, 25), extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]])

    ax1.xaxis_date()
    #pltcth.sel(time=slice(lt[view_start], lt[view_stop])).plot(linestyle='--', marker='.', markersize=10, c='b', ax=ax3, label='VELOX 2D')

    #H_coarsed.sel(time=slice(lt[view_start], lt[view_stop])).plot(linestyle='', marker='.', markersize=5, c='r', ax=ax3, label='VELOX nadir')
    xrwl.cloud_top.sel(time=slice(lt[view_start], lt[view_stop])).plot(linestyle='--', marker='.', markersize=10, c='g', ax=ax3, label='WALES')
    cth_stripe.CTH.sel(time=slice(lt[view_start], lt[view_stop])).isel(swath=slice(315, 325)).mean(dim={'swath'}).rolling(time=10, center=True).mean().plot(linestyle='', marker='.', markersize=5, c='b', label='VELOX rolling mean')

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='0.5%', pad=0.05)
    ax1.tick_params(axis='both', which='minor', labelsize=8)
    cb = fig.colorbar(fig1, cax=cax, orientation='vertical', label='BT [°C]')
    cb.ax.tick_params(labelsize=8)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes('right', size='0.5%', pad=0.05)
    ax2.tick_params(axis='both', which='minor', labelsize=8)
    cb2 = fig.colorbar(fig2, cax=cax2, orientation='vertical', label='CTH [m]')

    cb2.ax.tick_params(labelsize=8)
    divider = make_axes_locatable(ax3)
    cax2 = divider.append_axes('right', size='0.5%', pad=0.05)
    cax2.set_visible(False)
    ax1.axhline(320, c='k', linewidth=5, alpha=.3)
    ax2.axhline(320, c='r', linewidth=5, alpha=.3)
    ax1.set_ylabel('swath arcoss\ntrack [km]')
    ax2.set_ylabel('swath across\ntrack [km]')
    ax3.legend(loc='upper left')
    ax2.set_yticks([0, 220, 320, 420, 640])
    ax2.set_yticklabels([-3.2, -1, 0, 1, 3.2])
    ax1.set_yticks([0, 220, 320, 420, 640])
    ax1.set_yticklabels([-3.2, -1, 0, 1, 3.2])
    from matplotlib.dates import DateFormatter
    date_form = DateFormatter("%H:%M")
    ax1.xaxis.set_major_formatter(date_form)
    ax3.set_xlabel(f'time @ {f}')
    ax3.xaxis.set_minor_locator(mdates.SecondLocator(bysecond=np.arange(0, 60, 5)))
    ax1.grid(which='minor')
    ax2.grid(which='minor')
    ax3.grid(which='minor')


    ax1.grid(which='major', lw=1, c='k')
    ax2.grid(which='major', lw=1, c='k')
    ax3.grid(which='major', lw=1, c='k')

    ax1.grid(which='both',visible=None, axis='y')
    ax2.grid(which='both',visible=None, axis='y')
    ax3.grid(which='both',visible=None, axis='y')
    ax3.set_ylim(0, 3000)

    # for ax in (ax1, ax2):
    #         ax4 = ax.twiny()
    #         ax4.set_xticks(ax.get_xticks())
    #         #ax4.set_xbound(ax.get_xbound())

    ax22 = ax2.secondary_xaxis('top')
    ax22.set_xlabel('along track [km]')
    ax22.set_xticks(ax2.get_xticks())
    ax22.set_xticklabels([12 * i for i in range(len(ax2.get_xticks()))])

    plt.tight_layout(pad=0.5)
    plt.savefig(f'{p}.png')
    p += 1
    plt.close()



import numpy as np 
import datetime 
import os 
import matplotlib.pyplot as plt
import glob
import sys
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
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
from matplotlib.offsetbox import AnchoredText
import warnings
warnings.filterwarnings('ignore')

bash_input = sys.argv[1]

fligth_names = f'Flight_{bash_input}a'

print(fligth_names)

plt.rcParams.update({'font.size': 12,
                     'axes.titlesize': 20,
                     'axes.labelsize': 18,
                     'axes.labelpad': 14,
                     'lines.linewidth': 1,
                     'lines.markersize': 10,
                     'xtick.labelsize' : 18,
                     'ytick.labelsize' : 18,
                     'xtick.top' : True,
                     'xtick.direction' : 'in',
                     'ytick.right' : True,
                     'ytick.direction' : 'in',
                     'axes.grid' : True, 
                     'figure.autolayout' : True})

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

def make_frames(c, f):

    #circle = 3
    #f = 'Flight_20200202a'
    circle = c

    fnum = str(f[7:-1])
    #Y, M, D = int(fnum[0:4]), int(fnum[4:6]), int(fnum[6:])
    f_format = f'HALO-{fnum[6:]}{fnum[4:6]}'

    print(f)
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
    cth_path = f'/projekt_agmwend/data/EUREC4A/06_Flights/{f}/VELOX/VELOX_327kveL/VELOX_CTH/EUREC4A_HALO_VELOX_CTH_Flight_{fnum}a_C{circle}.nc'
    xrcth = xr.open_dataset(cth_path)
    xrcm = xr.open_dataset(path + cm_name)
    coff_path = f'/projekt_agmwend/data/EUREC4A/11_VELOX-Tools/VELOX_CTH/coffs_v2/{fnum}_coffs.nc'



    xrtb = xr.open_dataset(path+tb_name)
    xrparams = xr.open_dataset(params_name)
    wl_path = glob.glob(os.path.join('/projekt_agmwend/data/EUREC4A/06_Flights', f, 'WALES','*V2.1.nc'))
    xrwl = xr.open_dataset(wl_path[0])

    xrds = xr.open_dataset('/home/jomueller/EUREC4A_JOANNE_Dropsonde-RD41_Level_3_v2.0.0.nc')
    coffs = xr.open_dataset(coff_path)


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

    im2[im2 <= 0] = im2[im2 <= 0] * np.nan

    pltcth = xrcth.CTH.isel(x=slice(315,325),y=slice(251,261)).mean(dim={'x', 'y'})
    v_s, v_e = np.arange(0, 50000, 200), np.arange(10000, 60000, 200)

    interp_time = [pd.date_range(time[i][0], time[i+1][0], periods=len(time[i])).values for i in range(len(time) -1)]
    it = [item for sublist in interp_time for item in sublist]

    cth_stripe = xr.Dataset(
        data_vars=dict(CTH = (['swath', 'time'], im2[:,:len(it)])),
        coords=dict(time = it, swath = np.arange(0, 640),
        ),
    )

    #correction shouldn't be needed anymore snice it's done in CTH_calc_fields.py

    #dif =  coffs.CTH.interp_like(cth_stripe) - cth_stripe.CTH.isel(swath=slice(315, 325)).mean(dim={'swath'}).rolling(time=10, center=True).mean()
    #ddif = np.outer(np.ones(640), dif.values)

    #im2 = im2 + np.nanmean(ddif)
    # im2 = im2[:,:len(it)] + ddif

    cth_min = 500
    cth_max = np.round(np.nanquantile(im2, 0.999), -2) + 100

    tb_min = int(np.nanquantile(im, 0.001))
    tb_max = int(np.nanquantile(im, 0.999))+1

    print(cth_min, cth_max, tb_min, tb_max)

    for index in tqdm(range(250)):
        view_start = v_s[index]
        view_stop = v_e[index]

        extent = [xrHALO.lon.min()-1, xrHALO.lon.max()+1, xrHALO.lat.min()-1, xrHALO.lat.max()+1]
        fig = plt.figure(figsize=(35, 15), tight_layout=True)
        spec = fig.add_gridspec(4, 4, width_ratios=[1, 1, 1, .5])
        ax0 = fig.add_subplot(spec[0, 3], projection=crs.PlateCarree())
        axt = fig.add_subplot(spec[0, :3])
        axt.set_axis_off()
        axt.text(0.5, 0.5, f"{f_format} : Circle {circle}", fontsize=32)

        #end of current viewing slice is depicted position

        lats = xrHALO.lat.sel(time=slice(lt[0], lt[view_stop])).values 
        lons = xrHALO.lon.sel(time=slice(lt[0], lt[view_stop])).values

        ax0.scatter(lons, lats, transform=crs.PlateCarree(), s=1, zorder=3, linestyle='--', color='darkblue')
        ax0.scatter(lons[-10:], lats[-10:], transform=crs.PlateCarree(), s=10, color='red')
        ax0.set_extent(extent)
        ax0.scatter(-59.543, 13.193, c='k', label='Barbados')
        ax0.legend(loc='upper right')

        gl = ax0.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)


        land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                            edgecolor='black', linewidth=3,
                                            facecolor='k')

        ocean_10m = cfeature.NaturalEarthFeature('physical', 'ocean', '10m',
                                            edgecolor='face',
                                            facecolor=cfeature.COLORS['water'])

        ax0.add_feature(land_10m)
        ax0.add_feature(ocean_10m)

        lon = abs(lons[-1])

        text = AnchoredText(f'HALO @ {lats[-1]:.2f}°N {lon:.2f}°W',
                        loc=4, prop={'size': 12}, frameon=True)
        ax0.add_artist(text)

        gl.xlabel_style = {'size': 18}
        gl.ylabel_style = {'size': 18}

        gl.ylabels_right = False
        gl.xlabels_bottom = False
        #gl.xlabels_top = True

        ax1 = fig.add_subplot(spec[1, :])
        ax2 = fig.add_subplot(spec[2, :], sharex=ax1)
        ax3 = fig.add_subplot(spec[3, :], sharex=ax1)


        y_lims = [0, 640]
        x_lims = mdates.date2num([lt[view_start], lt[view_stop]])

        fig1 = ax1.contourf(im[:,view_start:view_stop], levels=np.linspace(tb_min, tb_max, 25),extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
                cmap='rainbow')


                
        fig2 = ax2.contourf(im2[:,view_start:view_stop], levels=np.linspace(cth_min, cth_max, 25), extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]])

        ax1.xaxis_date()
        #pltcth.sel(time=slice(lt[view_start], lt[view_stop])).plot(linestyle='--', marker='.', markersize=10, c='b', ax=ax3, label='VELOX 2D')

        #H_coarsed.sel(time=slice(lt[view_start], lt[view_stop])).plot(linestyle='', marker='.', markersize=5, c='r', ax=ax3, label='VELOX nadir')
        xrwl.cloud_top.sel(time=slice(lt[view_start], lt[view_stop])).plot(linestyle='--', marker='.', markersize=10, c='g', ax=ax3, label='WALES')
        #cth_stripe.CTH.sel(time=slice(lt[view_start], lt[view_stop])).isel(swath=slice(315, 325)).mean(dim={'swath'}).rolling(time=10, center=True).mean().plot(linestyle='', marker='.', markersize=5, c='b', label='VELOX rolling mean')

        coffs.CTH.sel(time=slice(lt[view_start], lt[view_stop])).plot(linestyle='--', marker='.', markersize=10, c='darkblue', ax=ax3, label='VELOX')

        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='0.5%', pad=0.05)
        ax1.tick_params(axis='both', which='minor', labelsize=18)
        cb = fig.colorbar(fig1, cax=cax, orientation='vertical', label='BT [°C]')
        cb.ax.tick_params(labelsize=18)
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes('right', size='0.5%', pad=0.05)
        ax2.tick_params(axis='both', which='minor', labelsize=18)
        cb2 = fig.colorbar(fig2, cax=cax2, orientation='vertical', label='CTH [m]')
        cb2.ax.tick_params(labelsize=18)
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
        ax3.set_xlabel(f'time')
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
        ax3.set_ylim(0, cth_max)

        # for ax in (ax1, ax2):
        #         ax4 = ax.twiny)
        #         ax4.set_xticks(ax.get_xticks())
        #         #ax4.set_xbound(ax.get_xbound())

        ax22 = ax2.secondary_xaxis('top')
        ax22.set_xlabel('along track [km]')
        ax22.set_xticks(ax2.get_xticks())
        ax22.set_xticklabels([12 * i for i in range(len(ax2.get_xticks()))])
    
        plt.suptitle("  ")
        #plt.tight_layout(pad=0.5)
        plt.savefig(f'{p}.png')
        p += 1
        plt.close()

#fligth_names = ['Flight_20200205a']

#input the flightname as bash argument in the form YYYYMMDD


for circle in np.arange(1, 7):
    make_frames(circle, fligth_names)


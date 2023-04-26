import xarray as xr
import os
from time import time

os.chdir('/projekt_agmwend/data/EUREC4A/06_Flights/')

for path in os.listdir(os.getcwd()):
    if os.path.isfile(f'/projekt_agmwend/data/EUREC4A/06_Flights/{path}/VELOX/VELOX_327kveL/EUREC4A_HALO_VELOX_BT_Filter_01_{path[-9:-1]}_v0.4.nc'):

        xrts = xr.open_dataset(f'/projekt_agmwend/data/EUREC4A/06_Flights/{path}/VELOX/VELOX_327kveL/EUREC4A_HALO_VELOX_BT_Filter_01_{path[-9:-1]}_v0.4.nc')
        #xrts = xr.open_dataset(base+f'{path}/VELOX/VELOX_327kveL/EUREC4A_HALO_VELOX_BT_Filter_01_{path[-9:-1]}_v0.4.nc')
        s = time()
        xrts_nadir = xrts.BT_2D.isel(x=slice(315, 325), y=slice(251,261)).mean(dim={'x', 'y'})
        xrts_nadir.to_netcdf(f'/projekt_agmwend/data/EUREC4A/06_Flights/{path}/VELOX/VELOX_327kveL/10x10_Central_Pixel/EUREC4A_{path}_ROI_Center_10x10_Pixel_Filter1.nc', mode='w')
        d = time() -s
        
        print('Saved file as 10x10_Central_Pixel/EUREC4A_{path}_ROI_Center_10x10_Pixel_Filter1.nc')
        print(f'in {d:02f}sec')
    else:
        print(f'{path} has no VELOX data')
 
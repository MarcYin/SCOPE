import rioxarray
import xarray as xr
import numpy as np
import os
import pandas as pd
import datetime

from pathlib import Path

start_year = 2018
end_year = 2025
fname = f'/work/scratch-pw3/marc/CPEO/ERA5/ERA5_LAND_HOURLY_{start_year}-{end_year}_scope.nc'
# load weather data
ds = xr.open_dataset(fname)
# ds = ds.transpose('time', 'lat', 'lon')
ds.rio.write_crs('EPSG:4326', inplace=True)
ds.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)



output_dir = '/gws/nopw/j04/nceo_isp/CPEO/US_ground_SIF/subsets' 
troposif_dir =  os.path.join(output_dir, 'US_ground_SIF_best_daily.nc')
troposif_ds = xr.open_dataset(troposif_dir)
troposif_ds.rio.write_crs('EPSG:4326', inplace=True)
troposif_ds.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)



def read_s2_bio(site, year):
    S2_data_folder = Path(f"/gws/nopw/j04/nceo_isp/CPEO/S2_data/{site}_{year}/")

    s2_sur_ref_ds = xr.open_dataset(S2_data_folder / f"s2a_sur_ref.nc")
    # Create a filename for storing NPZ outputs
    npz_filename = S2_data_folder / f"{site}_{year}.npz"

    f = np.load(npz_filename, allow_pickle=True)
    doys = f.f.doys
    post_bio_tensor = f.f.post_bio_tensor * 1.
    mask = f.f.mask
    # reshape the post_bio_tensor to match the shape of mask
    post_bio_tensor = post_bio_tensor.reshape(mask.shape[0], mask.shape[1], 7, len(doys))
    geotransform = f.f.geotransform
    crs = f.f.crs
    x_coords = geotransform[0] + np.arange(mask.shape[1]) * geotransform[1]
    y_coords = geotransform[3] + np.arange(mask.shape[0]) * geotransform[5]

    # Index	Parameter	Scale
    # 0	N	1/100.
    # 1	cab	1/100.
    # 2	cm	1/10000.
    # 3	cw	1/10000.
    # 4	lai	1/100.
    # 5	ala	1/100.
    # 6	cbrown	1/1000.

    bio_names = ['N', 'cab', 'cm', 'cw', 'lai', 'ala', 'cbrown']
    scales = [1/100., 1/100., 1/10000., 1/10000., 1/100., 1/100., 1/1000.]

    post_bio_time = [datetime.datetime.strptime(f'{year}{str(int(doy))}', '%Y%j') for doy in doys]
    # create xr.DataArray for the post_bio_tensor
    post_bio_da = xr.DataArray(post_bio_tensor, dims=['y', 'x', 'band', 'time'], coords=[y_coords, x_coords, bio_names, post_bio_time])
    
        # Get time coordinate
    time = post_bio_da.coords['time'].values

    # Find unique times and their first index
    _, index = np.unique(time, return_index=True)

    # Select only the unique times
    post_bio_da = post_bio_da.isel(time=index)
    # rescale the post_bio_da using the scales
    for i, scale in enumerate(scales):
        post_bio_da[:, :, i, :] = post_bio_da[:, :, i, :] * scale

    # georeference the post_bio_da
    post_bio_da = post_bio_da.rio.set_spatial_dims(x_dim='x', y_dim='y')
    post_bio_da = post_bio_da.rio.write_crs(s2_sur_ref_ds.rio.crs)

    post_bio_scale_params = f.f.dat.reshape(mask.shape[0], mask.shape[1], 15)
    bio_names = ['N', 'cab', 'cm', 'cw', 'lai', 'ala', 'cbrown', 'n0', 'm0', 'n1', 'm1', 'BSMBrightness', 'BSMlat', 'BSMlon', 'SMC']
    post_bio_scale_da = xr.DataArray(post_bio_scale_params, dims=['y', 'x', 'band'], coords=[y_coords, x_coords, bio_names])
    post_bio_scale_da = post_bio_scale_da.rio.set_spatial_dims(x_dim='x', y_dim='y')
    post_bio_scale_da = post_bio_scale_da.rio.write_crs(s2_sur_ref_ds.rio.crs)


    return post_bio_da, post_bio_scale_da

df = pd.read_csv('/home/users/marcyin/CPEO/US_ground_SIF/Crops_SIF_VegIndices_IL_NE_2136/data/site_locations.csv')
for index, row in df.iterrows():
    site = row['site']
    for year in eval(row['years']):
        if year > 2017:
            print(f"Processing {site} for year {year}")
            post_bio_da, post_bio_scale_da = read_s2_bio(site, year)
            
            # import rasterio
            # coordinates = row.longitude, row.latitude
            # # reproject to post_bio_da crs
            # coordinates = rasterio.warp.transform('EPSG:4326', post_bio_da.rio.crs, [coordinates[0]], [coordinates[1]])
            # # buffer 200 meters
            # buffer = 200
            # x_min = coordinates[0][0] - buffer
            # x_max = coordinates[0][0] + buffer
            # y_min = coordinates[1][0] - buffer
            # y_max = coordinates[1][0] + buffer
            
            
            # # clip post_bio_da
            # post_bio_da = post_bio_da.rio.clip_box(x_min, y_min, x_max, y_max)
            # post_bio_scale_da = post_bio_scale_da.rio.clip_box(x_min, y_min, x_max, y_max)


            bounds = post_bio_da.rio.transform_bounds('EPSG:4326')
            
            x_slice = slice(bounds[0], bounds[2])
            y_slice = slice(bounds[3], bounds[1])
            time_slice = slice(post_bio_da.time[0], post_bio_da.time[-1])
            # time_mask = (ds.time >= np.datetime64(datetime.datetime(year, 6, 1))) & (ds.time <= np.datetime64(datetime.datetime(year, 11, 30)))

            weather_year_ds = ds.sel(time=time_slice)
            weather_year_ds.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=True)
            weather_year_ds = weather_year_ds.rio.clip_box(*bounds, allow_one_dimensional_raster=True, auto_expand=True)

            troposif_year_ds = troposif_ds.sel(time=time_slice).rio.clip_box(*bounds, allow_one_dimensional_raster=True)
            time_grid = troposif_year_ds.delta_time.mean(dim=['y', 'x'], skipna=True)
            troposif_year_ds = troposif_year_ds.isel(time=~pd.isnull(time_grid), drop=True)
            time_grid = time_grid[~pd.isnull(time_grid)]
            troposif_year_ds['time'] = time_grid
            # drop the delta_time variable
            troposif_year_ds = troposif_year_ds.drop_vars('delta_time')

            weather_year_ds = weather_year_ds.interp(time=time_grid.values, method="linear")
            post_bio_da = post_bio_da.interp(time=time_grid.values, method="linear")            
            weather_year_ds = weather_year_ds.rio.reproject_match(post_bio_da, resampling=0)
            troposif_year_ds = troposif_year_ds.rio.reproject_match(post_bio_da, resampling=0)

            data_vars = xr.Dataset()
            data_vars['Rin'] = weather_year_ds['Rin']
            data_vars['Rli'] = weather_year_ds['Rli']
            data_vars['Ta'] = weather_year_ds['Ta']
            data_vars['ea'] = weather_year_ds['ea']
            data_vars['p'] = weather_year_ds['p']
            data_vars['u'] = weather_year_ds['u']
            # data_vars['u'] = weather_year_ds['u']
            # data_vars['u'] = weather_year_ds['u']

            data_vars['LAI'] = post_bio_da.sel(band='lai')
            data_vars['N'] = post_bio_da.sel(band='N')
            data_vars['Cab'] = post_bio_da.sel(band='cab')
            data_vars['CCa'] = post_bio_da.sel(band='cab') * 0.25
            data_vars['Cdm'] = post_bio_da.sel(band='cm')
            data_vars['Cw'] = post_bio_da.sel(band='cw')
            data_vars['ala'] = post_bio_da.sel(band='ala')
            data_vars['Cs'] = post_bio_da.sel(band='cbrown')

            soil_params =  post_bio_scale_da.sel(band=['BSMBrightness', 'BSMlat', 'BSMlon', 'SMC'])
            # add a new dimension to soil_params
            soil_params = soil_params.expand_dims(dim='time')
            soil_params['time'] = time_grid[:1]
            soil_params = soil_params.sel(time=time_grid, method='nearest')
            soil_params['time'] = time_grid.values

            data_vars['BSMBrightness'] = soil_params.sel(band='BSMBrightness')
            data_vars['BSMlat'] = soil_params.sel(band='BSMlat')
            data_vars['BSMlon'] = soil_params.sel(band='BSMlon')
            data_vars['SMC'] = soil_params.sel(band='SMC')

            data_vars['tts'] = troposif_year_ds['solar_zenith_angle']
            data_vars['tto'] = troposif_year_ds['viewing_zenith_angle']
            data_vars['psi'] = (troposif_year_ds['viewing_azimuth_angle'] - troposif_year_ds['solar_azimuth_angle']) % 360

            # only keep time, x, y and spatial_ref in the coordinates
            coord_names = data_vars._coord_names
            to_remove = [name for name in coord_names if name not in ['time', 'x', 'y', 'spatial_ref']]
            data_vars = data_vars.drop_vars(to_remove)
            data_vars = data_vars.transpose('x', 'y', 'time')


            # SCOPE configuration attributes
            scope_config = {
                'lite':1,
                'calc_fluor': 1,
                'calc_planck': 0,
                'calc_xanthophyllabs': 0,
                'soilspectrum': 1,
                'Fluorescence_model': 0,
                'apply_T_corr': 1,
                'verify': 0,
                'mSCOPE': 0,
                'calc_directional': 0,
                'calc_vert_profiles': 0,
                'soil_heat_method': 2,
                'calc_rss_rbs': 1,
                'MoninObukhov': 1,
                'save_spectral': 0
            }
        
            # # SCOPE configuration attributes
            # self.scope_config = {
            #     'calc_fluor': 1,
            #     'calc_planck': 0,
            #     'calc_xanthophyllabs': 0,
            #     'save_spectral': 0,
            #     'lite': 1,
            #     'soilspectrum': 1,
            #     'MoninObukhov': 1,
            #     'soil_heat_method': 2,
            # }

            data_vars.attrs.update(scope_config)

            # File references (will be set relative to scope_matlab_dir if provided)
            file_refs = {
                'optipar_file': 'input/fluspect_parameters/Optipar2017_ProspectD.mat',
                'soil_file': 'input/soil_spectra/soilnew.txt',
                'atmos_file': 'input/radiationdata/FLEX-S3_std.atm',
            }
            scope_path = Path('/home/users/marcyin/SCOPE/SCOPE-2.1')
            for ref_name, ref_path in file_refs.items():
                full_path = scope_path / ref_path
                data_vars.attrs[ref_name] = full_path.as_posix()
            
            data_vars.to_netcdf(f'/home/users/marcyin/SCOPE/experiments/{site}_{year}_input.nc')
            


# import xarray as xr
# import pylab as plt
# from pathlib import Path
# inout_files = Path('/home/users/marcyin/SCOPE/experiments/').glob('*input.nc')
# for input_file in inout_files:
#     ds = xr.open_dataset(input_file)
#     print(input_file)
#     ds.LAI[:, :, ::15].plot.imshow(col='time', col_wrap=4)
#     plt.show()

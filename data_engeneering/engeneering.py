import xarray as xr
import numpy as np

def engineer_physical_features(ds_surf, ds_pres):
    """
    Computes 42 physical variables from raw ERA5 tensors.
    Assumes ds_surf and ds_pres have been aligned by 'valid_time', 'latitude', and 'longitude'.
    """
    # Initialize a new Dataset to hold only the engineered variables
    ds_eng = xr.Dataset(coords=ds_surf.coords)
    
    # Mathematical Constants
    EPSILON = 1e-6  # Prevents division-by-zero errors in denominators
    L_V = 2.5e6     # Latent heat of vaporization (J/kg)
    C_P = 1004.0    # Specific heat of air at constant pressure (J/(kg K))
    
    # 1-3. Wind Speeds (Magnitudes)
    ds_eng['v10_mag'] = np.sqrt(ds_surf['u10']**2 + ds_surf['v10']**2)
    ds_eng['v850_mag'] = np.sqrt(ds_pres['u'].sel(pressure_level=850)**2 + ds_pres['v'].sel(pressure_level=850)**2)
    ds_eng['v250_mag'] = np.sqrt(ds_pres['u'].sel(pressure_level=250)**2 + ds_pres['v'].sel(pressure_level=250)**2)
    
    # 4-6. Relative Vorticity (Curl of the wind vector)
    for lvl in [850, 500, 250]:
        u = ds_pres['u'].sel(pressure_level=lvl)
        v = ds_pres['v'].sel(pressure_level=lvl)
        dv_dx = v.differentiate('longitude')
        du_dy = u.differentiate('latitude')
        ds_eng[f'vort_{lvl}'] = dv_dx - du_dy
        
    # 7-10. Vertical Wind Shear
    ds_eng['shear_u_deep'] = ds_pres['u'].sel(pressure_level=250) - ds_pres['u'].sel(pressure_level=850)
    ds_eng['shear_v_deep'] = ds_pres['v'].sel(pressure_level=250) - ds_pres['v'].sel(pressure_level=850)
    ds_eng['shear_u_low'] = ds_pres['u'].sel(pressure_level=850) - ds_surf['u10']
    ds_eng['shear_v_low'] = ds_pres['v'].sel(pressure_level=850) - ds_surf['v10']

    # 11-13. Lapse Rates (Negative vertical temperature gradient)
    ds_eng['lapse_low'] = -(ds_pres['t'].sel(pressure_level=850) - ds_pres['t'].sel(pressure_level=1000)) / \
                           (ds_pres['z'].sel(pressure_level=850) - ds_pres['z'].sel(pressure_level=1000) + EPSILON)
    ds_eng['lapse_mid'] = -(ds_pres['t'].sel(pressure_level=500) - ds_pres['t'].sel(pressure_level=850)) / \
                           (ds_pres['z'].sel(pressure_level=500) - ds_pres['z'].sel(pressure_level=850) + EPSILON)
    ds_eng['lapse_high'] = -(ds_pres['t'].sel(pressure_level=250) - ds_pres['t'].sel(pressure_level=500)) / \
                            (ds_pres['z'].sel(pressure_level=250) - ds_pres['z'].sel(pressure_level=500) + EPSILON)
    
    # 14-15. Geopotential Thickness
    ds_eng['thick_1000_500'] = ds_pres['z'].sel(pressure_level=500) - ds_pres['z'].sel(pressure_level=1000)
    ds_eng['thick_850_500'] = ds_pres['z'].sel(pressure_level=500) - ds_pres['z'].sel(pressure_level=850)
    
    # 16-18. Thermodynamic Indexes
    ds_eng['inv_strength'] = ds_pres['t'].sel(pressure_level=850) - ds_surf['t2m']
    ds_eng['theta_e_850'] = ds_pres['t'].sel(pressure_level=850) + (L_V / C_P) * ds_pres['q'].sel(pressure_level=850)
    
    shear_500_10_sq = (ds_pres['u'].sel(pressure_level=500) - ds_surf['u10'])**2 + \
                      (ds_pres['v'].sel(pressure_level=500) - ds_surf['v10'])**2
    ds_eng['brn_proxy'] = ds_surf['cape'] / (0.5 * shear_500_10_sq + EPSILON)

    # 19-21. Topographical Forcings
    ds_eng['slope_x'] = ds_surf['z'].differentiate('longitude')
    ds_eng['slope_y'] = ds_surf['z'].differentiate('latitude')
    ds_eng['orographic_lift'] = (ds_surf['u10'] * ds_eng['slope_x']) + (ds_surf['v10'] * ds_eng['slope_y'])
    
    # 22-25. Advection and Pressure Gradients
    t_850 = ds_pres['t'].sel(pressure_level=850)
    q_850 = ds_pres['q'].sel(pressure_level=850)
    ds_eng['adv_t_850'] = -(ds_pres['u'].sel(pressure_level=850) * t_850.differentiate('longitude') + 
                            ds_pres['v'].sel(pressure_level=850) * t_850.differentiate('latitude'))
    ds_eng['adv_q_850'] = -(ds_pres['u'].sel(pressure_level=850) * q_850.differentiate('longitude') + 
                            ds_pres['v'].sel(pressure_level=850) * q_850.differentiate('latitude'))
    
    ds_eng['grad_sp_x'] = ds_surf['sp'].differentiate('longitude')
    ds_eng['grad_sp_y'] = ds_surf['sp'].differentiate('latitude')

    # 26-28. Vertical Integrations (Sum across 1000, 850, 500, 250)
    ds_eng['col_q'] = ds_pres['q'].sum(dim='pressure_level')
    ds_eng['col_clwc'] = ds_pres['clwc'].sum(dim='pressure_level')
    ds_eng['col_ciwc'] = ds_pres['ciwc'].sum(dim='pressure_level')
    
    # 29-33. Flux and Radiation
    flux_q_x = ds_pres['u'].sel(pressure_level=850) * q_850
    flux_q_y = ds_pres['v'].sel(pressure_level=850) * q_850
    ds_eng['flux_q_x'] = flux_q_x
    ds_eng['flux_q_y'] = flux_q_y
    ds_eng['mfc_850'] = -(flux_q_x.differentiate('longitude') + flux_q_y.differentiate('latitude'))
    
    ds_eng['rad_balance'] = ds_surf['ssr'] + ds_surf['ttr']
    ds_eng['grad_cc_x'] = ds_pres['cc'].sel(pressure_level=500).differentiate('longitude')

    # 34-37. Cyclical time extraction mapped to continuous vectors
    times = ds_surf['valid_time'].dt
    hours = times.hour
    days = times.dayofyear
    
    ds_eng['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
    ds_eng['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
    ds_eng['day_sin'] = np.sin(2 * np.pi * days / 365.25)
    ds_eng['day_cos'] = np.cos(2 * np.pi * days / 365.25)

    # 38-42. Operational Diagnostics
    ds_eng['k_index_proxy'] = t_850 - ds_pres['t'].sel(pressure_level=500) + (q_850 * 1000)
    ds_eng['w_diff'] = ds_pres['w'].sel(pressure_level=500) - ds_pres['w'].sel(pressure_level=850)
    
    d_1000 = ds_pres['d'].sel(pressure_level=1000)
    ds_eng['d_anom_1000'] = d_1000 - d_1000.mean(dim=['valid_time', 'latitude', 'longitude'])
    
    u_500 = ds_pres['u'].sel(pressure_level=500)
    v_500 = ds_pres['v'].sel(pressure_level=500)
    ds_eng['def_stretch'] = u_500.differentiate('longitude') - v_500.differentiate('latitude')
    ds_eng['def_shear'] = v_500.differentiate('longitude') + u_500.differentiate('latitude')
    
    return ds_eng

# Example Execution:
# ds_engineered = engineer_physical_features(ds_surface, ds_pressure)
# print(list(ds_engineered.data_vars.keys()))
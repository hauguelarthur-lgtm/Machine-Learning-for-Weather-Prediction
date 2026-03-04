import xarray as xr
import numpy as np

def engineer_physical_features(ds_surf, ds_pres):
    """
    Computes 42 physical variables from raw ERA5 tensors.
    Assumes ds_surf and ds_pres have been mathematically aligned.
    """
    ds_eng = xr.Dataset(coords=ds_surf.coords)
    
    # ---------------------------------------------------------
    # MATHEMATICAL CONSTANTS
    # ---------------------------------------------------------
    EPSILON = 1e-6      # Prevents division-by-zero singularities
    L_V = 2.5e6         # Latent heat of vaporization (J/kg)
    C_P = 1004.0        # Specific heat of air at constant pressure (J/(kg K))
    G_0 = 9.80665       # Standard gravitational acceleration (m/s^2)
    R_D = 287.058       # Specific gas constant for dry air (J/(kg K))
    P_0 = 1000.0        # Standard reference pressure (hPa)
    R_EARTH = 6371000.0 # Mean radius of the Earth (m)

    # ---------------------------------------------------------
    # SPHERICAL AND TOPOGRAPHICAL BOUNDARY MATRICES
    # ---------------------------------------------------------
    lat_rad = np.deg2rad(ds_surf['latitude'])
    
    # Differential step scalars mapping Euclidean coordinates to the S^2 manifold
    dx_scalar = 1.0 / (R_EARTH * np.cos(lat_rad) * np.pi / 180.0)
    dy_scalar = 1.0 / (R_EARTH * np.pi / 180.0)

    # Convert surface geopotential to strict geometric elevation
    geometric_elevation = ds_surf['z'] / G_0
    
    # Extract geopotential heights for the free atmosphere
    z_1000 = ds_pres['z'].sel(pressure_level=1000) / G_0
    z_850  = ds_pres['z'].sel(pressure_level=850) / G_0
    z_500  = ds_pres['z'].sel(pressure_level=500) / G_0
    z_250  = ds_pres['z'].sel(pressure_level=250) / G_0

    # ---------------------------------------------------------
    # 1-3. KINETIC VELOCITY (Magnitudes)
    # ---------------------------------------------------------
    ds_eng['v10_mag'] = np.sqrt(ds_surf['u10']**2 + ds_surf['v10']**2)
    ds_eng['v850_mag'] = np.sqrt(ds_pres['u'].sel(pressure_level=850)**2 + ds_pres['v'].sel(pressure_level=850)**2)
    ds_eng['v250_mag'] = np.sqrt(ds_pres['u'].sel(pressure_level=250)**2 + ds_pres['v'].sel(pressure_level=250)**2)
    
    # ---------------------------------------------------------
    # 4-6. RELATIVE VORTICITY (S^2 Curl)
    # ---------------------------------------------------------
    for lvl in [850, 500, 250]:
        u = ds_pres['u'].sel(pressure_level=lvl)
        v = ds_pres['v'].sel(pressure_level=lvl)

        dv_dx = v.differentiate('longitude') * dx_scalar
        du_dy = u.differentiate('latitude') * dy_scalar
        
        # Spherical curvature correction term induced by meridian convergence
        curvature = (u / R_EARTH) * np.tan(lat_rad)
        
        ds_eng[f'vort_{lvl}'] = dv_dx - du_dy + curvature
        
    # ---------------------------------------------------------
    # 7-10. VERTICAL WIND SHEAR
    # ---------------------------------------------------------
    ds_eng['shear_u_deep'] = ds_pres['u'].sel(pressure_level=250) - ds_pres['u'].sel(pressure_level=850)
    ds_eng['shear_v_deep'] = ds_pres['v'].sel(pressure_level=250) - ds_pres['v'].sel(pressure_level=850)
    ds_eng['shear_u_low'] = ds_pres['u'].sel(pressure_level=850) - ds_surf['u10']
    ds_eng['shear_v_low'] = ds_pres['v'].sel(pressure_level=850) - ds_surf['v10']

    # ---------------------------------------------------------
    # 11-13. ENVIRONMENTAL LAPSE RATES
    # ---------------------------------------------------------
    # Low-level lapse rate strictly anchored to geometric topology, resolving subterranean clamping
    dz_low = z_850 - geometric_elevation
    # Boolean mask: False (0) where 850hPa is subterranean
    valid_mask = (dz_low > 0).astype(np.float32) 
    
    ds_eng['lapse_low'] = (-(ds_pres['t'].sel(pressure_level=850) - ds_surf['t2m']) / 
                           (dz_low.clip(min=EPSILON))) * valid_mask
    
    ds_eng['lapse_mid'] = -(ds_pres['t'].sel(pressure_level=500) - ds_pres['t'].sel(pressure_level=850)) / \
                           (z_500 - z_850 + EPSILON)
    
    ds_eng['lapse_high'] = -(ds_pres['t'].sel(pressure_level=250) - ds_pres['t'].sel(pressure_level=500)) / \
                            (z_250 - z_500 + EPSILON)
    
    # ---------------------------------------------------------
    # 14-15. GEOPOTENTIAL THICKNESS (Physical Meters)
    # ---------------------------------------------------------
    ds_eng['thick_1000_500'] = z_500 - z_1000
    ds_eng['thick_850_500']  = z_500 - z_850
    
    # ---------------------------------------------------------
    # 16-18. THERMODYNAMIC CAPE PROXIES
    # ---------------------------------------------------------
    ds_eng['inv_strength'] = ds_pres['t'].sel(pressure_level=850) - ds_surf['t2m']

    t_850 = ds_pres['t'].sel(pressure_level=850)
    q_850 = ds_pres['q'].sel(pressure_level=850)
    
    # Exact derivation of adiabatic Potential Temperature (theta)
    theta_850 = t_850 * (P_0 / 850.0) ** (R_D / C_P)
    
    # Equivalent Potential Temperature (theta_e) resolving fluid expansion
    ds_eng['theta_e_850'] = theta_850 * np.exp((L_V * q_850) / (C_P * t_850))
    
    shear_500_10_sq = (ds_pres['u'].sel(pressure_level=500) - ds_surf['u10'])**2 + \
                      (ds_pres['v'].sel(pressure_level=500) - ds_surf['v10'])**2
    ds_eng['brn_proxy'] = ds_surf['cape'] / (0.5 * shear_500_10_sq + EPSILON)

    # ---------------------------------------------------------
    # 19-21. TOPOGRAPHICAL FORCINGS
    # ---------------------------------------------------------
    ds_eng['slope_x'] = geometric_elevation.differentiate('longitude') * dx_scalar
    ds_eng['slope_y'] = geometric_elevation.differentiate('latitude') * dy_scalar
    ds_eng['orographic_lift'] = (ds_surf['u10'] * ds_eng['slope_x']) + (ds_surf['v10'] * ds_eng['slope_y'])
    
    # ---------------------------------------------------------
    # 22-25. FLUID ADVECTION AND KINEMATIC GRADIENTS
    # ---------------------------------------------------------
    ds_eng['adv_t_850'] = -(ds_pres['u'].sel(pressure_level=850) * (t_850.differentiate('longitude') * dx_scalar) + 
                            ds_pres['v'].sel(pressure_level=850) * (t_850.differentiate('latitude') * dy_scalar))
    
    ds_eng['adv_q_850'] = -(ds_pres['u'].sel(pressure_level=850) * (q_850.differentiate('longitude') * dx_scalar) + 
                            ds_pres['v'].sel(pressure_level=850) * (q_850.differentiate('latitude') * dy_scalar))
    
    ds_eng['grad_sp_x'] = ds_surf['sp'].differentiate('longitude') * dx_scalar
    ds_eng['grad_sp_y'] = ds_surf['sp'].differentiate('latitude') * dy_scalar

    # ---------------------------------------------------------
    # 26-28. MASS-WEIGHTED VERTICAL INTEGRATION (Hydrostatic)
    # ---------------------------------------------------------
    # Constant scalar isolating standard gravity and converting hPa coordinates to Pascal
    mass_scalar = 100.0 / G_0 
    
    ds_eng['col_q'] = np.abs(ds_pres['q'].integrate(coord='pressure_level') * mass_scalar)
    ds_eng['col_clwc'] = np.abs(ds_pres['clwc'].integrate(coord='pressure_level') * mass_scalar)
    ds_eng['col_ciwc'] = np.abs(ds_pres['ciwc'].integrate(coord='pressure_level') * mass_scalar)
    
    # ---------------------------------------------------------
    # 29-33. FLUX AND RADIATION DIVERGENCE
    # ---------------------------------------------------------
    flux_q_x = ds_pres['u'].sel(pressure_level=850) * q_850
    flux_q_y = ds_pres['v'].sel(pressure_level=850) * q_850
    ds_eng['flux_q_x'] = flux_q_x
    ds_eng['flux_q_y'] = flux_q_y
    
    ds_eng['mfc_850'] = -((flux_q_x.differentiate('longitude') * dx_scalar) + 
                          (flux_q_y.differentiate('latitude') * dy_scalar))
    
    ds_eng['rad_balance'] = ds_surf['ssr'] + ds_surf['ttr']
    ds_eng['grad_cc_x'] = ds_pres['cc'].sel(pressure_level=500).differentiate('longitude') * dx_scalar

    # ---------------------------------------------------------
    # 34-37. CYCLICAL EMBEDDINGS
    # ---------------------------------------------------------
    times = ds_surf['valid_time'].dt
    hours = times.hour
    days = times.dayofyear
    
    ds_eng['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
    ds_eng['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
    ds_eng['day_sin'] = np.sin(2 * np.pi * days / 365.25)
    ds_eng['day_cos'] = np.cos(2 * np.pi * days / 365.25)

    # ---------------------------------------------------------
    # 38-42. MACROSCOPIC DIAGNOSTICS
    # ---------------------------------------------------------
    ds_eng['k_index_proxy'] = t_850 - ds_pres['t'].sel(pressure_level=500) + (q_850 * 1000)
    ds_eng['w_diff'] = ds_pres['w'].sel(pressure_level=500) - ds_pres['w'].sel(pressure_level=850)
    
    d_1000 = ds_pres['d'].sel(pressure_level=1000)
    ds_eng['d_anom_1000'] = d_1000 - d_1000.mean(dim=['valid_time', 'latitude', 'longitude'])
    
    u_500 = ds_pres['u'].sel(pressure_level=500)
    v_500 = ds_pres['v'].sel(pressure_level=500)
    ds_eng['def_stretch'] = (u_500.differentiate('longitude') * dx_scalar) - (v_500.differentiate('latitude') * dy_scalar)
    ds_eng['def_shear'] = (v_500.differentiate('longitude') * dx_scalar) + (u_500.differentiate('latitude') * dy_scalar)
    
    return ds_eng
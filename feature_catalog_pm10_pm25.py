#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature catalog for the PM10+PM2.5 low-visibility S2 datasets.

The order mirrors s2_data.py for the main PM10+PM2.5 month-tail dataset:
dynamic window, five continuous static fields, one vegetation category, then
feature-engineering columns. The VERA-inspired feature names are included so
the same catalog can describe datasets built by s2_data_aerosol_vera.py.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


BASE_DYNAMIC_FEATURES: List[Dict[str, str]] = [
    {
        "feature": "RH2M",
        "block": "dynamic_12h",
        "based_on": "2 m relative humidity at each hour in the 12 h window",
        "calculation": "Direct Tianji field after variable renaming.",
        "scientific_meaning": "Near-surface saturation controls fog and mist formation and aerosol hygroscopic growth.",
    },
    {
        "feature": "T2M",
        "block": "dynamic_12h",
        "based_on": "2 m air temperature at each hour",
        "calculation": "Direct Tianji field after variable renaming.",
        "scientific_meaning": "Sets saturation vapor pressure, cooling tendency and fog persistence conditions.",
    },
    {
        "feature": "PRECIP",
        "block": "dynamic_12h",
        "based_on": "Surface precipitation at each hour",
        "calculation": "Direct Tianji field; log1p-transformed in model preprocessing.",
        "scientific_meaning": "Represents hydrometeor extinction and wet-scavenging or wet-weather low visibility.",
    },
    {
        "feature": "MSLP",
        "block": "dynamic_12h",
        "based_on": "Mean sea-level pressure at each hour",
        "calculation": "Direct Tianji field after variable renaming.",
        "scientific_meaning": "Synoptic pressure pattern proxy for stable high pressure, fronts and advection regime.",
    },
    {
        "feature": "SW_RAD",
        "block": "dynamic_12h",
        "based_on": "Downward shortwave radiation at the surface",
        "calculation": "Direct Tianji field; log1p-transformed in model preprocessing.",
        "scientific_meaning": "Daytime heating and mixing proxy; low values at night favor radiative cooling.",
    },
    {
        "feature": "U10",
        "block": "dynamic_12h",
        "based_on": "10 m zonal wind",
        "calculation": "Direct Tianji field after variable renaming.",
        "scientific_meaning": "Low-level advection and shear component affecting ventilation and fog displacement.",
    },
    {
        "feature": "WSPD10",
        "block": "dynamic_12h",
        "based_on": "10 m wind speed",
        "calculation": "Direct Tianji or derived wind-speed field.",
        "scientific_meaning": "Weak wind supports stagnation; stronger wind ventilates fog and aerosols.",
    },
    {
        "feature": "V10",
        "block": "dynamic_12h",
        "based_on": "10 m meridional wind",
        "calculation": "Direct Tianji field after variable renaming.",
        "scientific_meaning": "Low-level advection and shear component affecting moisture and aerosol transport.",
    },
    {
        "feature": "WDIR10",
        "block": "dynamic_12h",
        "based_on": "10 m wind direction",
        "calculation": "Direct Tianji or derived wind-direction field.",
        "scientific_meaning": "Flow-regime indicator for terrain-channeling and moist-air source direction.",
    },
    {
        "feature": "CAPE",
        "block": "dynamic_12h",
        "based_on": "Convective available potential energy",
        "calculation": "Direct Tianji field; log1p-transformed in model preprocessing.",
        "scientific_meaning": "Convective instability proxy; helps separate warm convective low visibility from radiative fog.",
    },
    {
        "feature": "LCC",
        "block": "dynamic_12h",
        "based_on": "Low cloud cover",
        "calculation": "Direct Tianji field after variable renaming.",
        "scientific_meaning": "Cloud and near-surface liquid-water proxy linked to radiation and visibility loss.",
    },
    {
        "feature": "T_925",
        "block": "dynamic_12h",
        "based_on": "925 hPa temperature",
        "calculation": "Direct Tianji field after variable renaming.",
        "scientific_meaning": "Lower-tropospheric thermal structure and inversion context.",
    },
    {
        "feature": "RH_925",
        "block": "dynamic_12h",
        "based_on": "925 hPa relative humidity",
        "calculation": "Direct Tianji field after variable renaming.",
        "scientific_meaning": "Vertical humidity structure; distinguishes shallow fog from deeper moist layers.",
    },
    {
        "feature": "U_925",
        "block": "dynamic_12h",
        "based_on": "925 hPa zonal wind",
        "calculation": "Direct Tianji field after variable renaming.",
        "scientific_meaning": "Low-level shear and advection above the surface.",
    },
    {
        "feature": "WSPD925",
        "block": "dynamic_12h",
        "based_on": "925 hPa wind speed",
        "calculation": "Direct Tianji or derived wind-speed field.",
        "scientific_meaning": "Ventilation and shear proxy above the surface layer.",
    },
    {
        "feature": "V_925",
        "block": "dynamic_12h",
        "based_on": "925 hPa meridional wind",
        "calculation": "Direct Tianji field after variable renaming.",
        "scientific_meaning": "Low-level shear and moisture transport above the surface.",
    },
    {
        "feature": "DP_1000",
        "block": "dynamic_12h",
        "based_on": "1000 hPa dew-point temperature",
        "calculation": "Direct Tianji field after variable renaming.",
        "scientific_meaning": "Near-surface moisture content and saturation proximity.",
    },
    {
        "feature": "DP_925",
        "block": "dynamic_12h",
        "based_on": "925 hPa dew-point temperature",
        "calculation": "Direct Tianji field after variable renaming.",
        "scientific_meaning": "Lower-tropospheric moisture reservoir and vertical moisture gradient.",
    },
    {
        "feature": "Q_1000",
        "block": "dynamic_12h",
        "based_on": "1000 hPa specific humidity",
        "calculation": "Direct Tianji field after variable renaming.",
        "scientific_meaning": "Absolute near-surface moisture supply for condensation and haze growth.",
    },
    {
        "feature": "Q_925",
        "block": "dynamic_12h",
        "based_on": "925 hPa specific humidity",
        "calculation": "Direct Tianji field after variable renaming.",
        "scientific_meaning": "Vertical moisture stratification and lower-tropospheric moisture support.",
    },
    {
        "feature": "W_925",
        "block": "dynamic_12h",
        "based_on": "925 hPa vertical velocity",
        "calculation": "Direct Tianji omega field after variable renaming.",
        "scientific_meaning": "Vertical motion proxy affecting cloud, mixing and moisture convergence.",
    },
    {
        "feature": "W_1000",
        "block": "dynamic_12h",
        "based_on": "1000 hPa vertical velocity",
        "calculation": "Direct Tianji omega field after variable renaming.",
        "scientific_meaning": "Near-surface vertical motion proxy for mixing and convergence.",
    },
    {
        "feature": "DPD",
        "block": "dynamic_12h",
        "based_on": "T2M and D2M",
        "calculation": "DPD = T2M - D2M; D2M is computed from T2M and RH2M if absent.",
        "scientific_meaning": "Dew-point depression; small values indicate near saturation.",
    },
    {
        "feature": "INVERSION",
        "block": "dynamic_12h",
        "based_on": "T_925 and T2M",
        "calculation": "INVERSION = T_925 - T2M.",
        "scientific_meaning": "Low-level stability proxy; positive values suppress vertical mixing.",
    },
]

DYNAMIC_APPEND_FEATURES: List[Dict[str, str]] = [
    {
        "feature": "zenith",
        "block": "dynamic_12h",
        "based_on": "station latitude, longitude and UTC valid time",
        "calculation": "pvlib apparent solar zenith angle; zero-filled if solar calculation fails.",
        "scientific_meaning": "Solar geometry proxy for night, radiative cooling and daytime mixing.",
    },
    {
        "feature": "PM10_ugm3",
        "block": "dynamic_12h",
        "based_on": "station PM10 time series matched to Tianji time and station",
        "calculation": "Nearest time/station match; max with zero; kg m-3 to ug m-3 by multiplying 1e12.",
        "scientific_meaning": "Coarse and total aerosol loading contributing to extinction.",
    },
    {
        "feature": "PM25_ugm3",
        "block": "dynamic_12h",
        "based_on": "station PM2.5 time series matched to Tianji time and station",
        "calculation": "Nearest time/station match; max with zero; kg m-3 to ug m-3 by multiplying 1e12.",
        "scientific_meaning": "Fine aerosol loading; important for hygroscopic growth and scattering.",
    },
]

STATIC_FEATURES: List[Dict[str, str]] = [
    {
        "feature": "lat_norm",
        "block": "static",
        "based_on": "station latitude",
        "calculation": "lat / 90.",
        "scientific_meaning": "Climatological latitude gradient in radiation, temperature and fog regime.",
    },
    {
        "feature": "lon_norm",
        "block": "static",
        "based_on": "station longitude",
        "calculation": "lon / 180.",
        "scientific_meaning": "Regional circulation and climatological location proxy.",
    },
    {
        "feature": "orography",
        "block": "static",
        "based_on": "nearest terrain height h",
        "calculation": "h at nearest grid point.",
        "scientific_meaning": "Altitude affects temperature, pressure, cloud base and station representativeness.",
    },
    {
        "feature": "orography_anom",
        "block": "static",
        "based_on": "nearest terrain height and local terrain window",
        "calculation": "h - mean(h in a radius-2 grid window).",
        "scientific_meaning": "Local valley/ridge exposure proxy for pooling and drainage flows.",
    },
    {
        "feature": "orography_std",
        "block": "static",
        "based_on": "local terrain window",
        "calculation": "std(h in a radius-2 grid window).",
        "scientific_meaning": "Terrain complexity proxy for sub-grid representativeness and local circulations.",
    },
    {
        "feature": "veg_type",
        "block": "static_category",
        "based_on": "nearest vegetation class htcc",
        "calculation": "Mapped to an integer category and passed through an embedding layer.",
        "scientific_meaning": "Land-cover proxy for surface moisture, roughness and radiation response.",
    },
]

BASE_FOG_FEATURES: List[Dict[str, str]] = [
    {
        "feature": "sat_dpd",
        "block": "feature_engineering",
        "based_on": "last-hour RH2M and DPD",
        "calculation": "clip(RH2M/100,0,1) * 1/(1+exp(DPD/2)).",
        "scientific_meaning": "Near-saturation proxy combining high humidity and small dew-point depression.",
    },
    {
        "feature": "wind_favourability",
        "block": "feature_engineering",
        "based_on": "last-hour WSPD10",
        "calculation": "Gaussian preference centered at 3.5 m s-1 with sigma 2.5.",
        "scientific_meaning": "Moderate weak wind can sustain fog without strong ventilation.",
    },
    {
        "feature": "stability_ri",
        "block": "feature_engineering",
        "based_on": "last-hour INVERSION and WSPD10",
        "calculation": "tanh((INVERSION/(WSPD10^2+0.1))/2).",
        "scientific_meaning": "Richardson-like stability proxy; stable weak-wind layers trap moisture and aerosols.",
    },
    {
        "feature": "night_clear_radiation",
        "block": "feature_engineering",
        "based_on": "last-hour zenith, LCC and SW_RAD",
        "calculation": "1(zenith>90) * clip(1-LCC/0.3,0,1) * (1-clip(SW_RAD/800,0,1)).",
        "scientific_meaning": "Nighttime clear-sky radiative-cooling potential.",
    },
    {
        "feature": "rh_surface_minus_925",
        "block": "feature_engineering",
        "based_on": "last-hour RH2M and RH_925",
        "calculation": "tanh((RH2M - RH_925)/50).",
        "scientific_meaning": "Vertical humidity contrast; highlights shallow surface moisture pooling.",
    },
    {
        "feature": "fog_potential",
        "block": "feature_engineering",
        "based_on": "RH2M, wind_favourability, stability_ri and night_clear_radiation",
        "calculation": "0.4*RH_norm + 0.25*wind_fav + 0.2*positive stability + 0.15*night cooling.",
        "scientific_meaning": "Composite heuristic for radiative fog favorability.",
    },
    {
        "feature": "rh2m_delta_3h",
        "block": "feature_engineering",
        "based_on": "RH2M sequence",
        "calculation": "RH2M(t) - RH2M(t-3 h).",
        "scientific_meaning": "Recent humidification tendency.",
    },
    {
        "feature": "rh2m_delta_6h",
        "block": "feature_engineering",
        "based_on": "RH2M sequence",
        "calculation": "RH2M(t) - RH2M(t-6 h).",
        "scientific_meaning": "Medium-term humidification tendency.",
    },
    {
        "feature": "rh2m_std_12h",
        "block": "feature_engineering",
        "based_on": "RH2M sequence",
        "calculation": "standard deviation over the 12 h window.",
        "scientific_meaning": "Humidity variability and persistence proxy.",
    },
    {
        "feature": "rh2m_range_12h",
        "block": "feature_engineering",
        "based_on": "RH2M sequence",
        "calculation": "max(RH2M)-min(RH2M) over the 12 h window.",
        "scientific_meaning": "Amplitude of humidity evolution.",
    },
    {
        "feature": "t2m_delta_3h",
        "block": "feature_engineering",
        "based_on": "T2M sequence",
        "calculation": "T2M(t) - T2M(t-3 h).",
        "scientific_meaning": "Recent cooling or warming tendency.",
    },
    {
        "feature": "t2m_delta_6h",
        "block": "feature_engineering",
        "based_on": "T2M sequence",
        "calculation": "T2M(t) - T2M(t-6 h).",
        "scientific_meaning": "Medium-term cooling or warming tendency.",
    },
    {
        "feature": "t2m_std_12h",
        "block": "feature_engineering",
        "based_on": "T2M sequence",
        "calculation": "standard deviation over the 12 h window.",
        "scientific_meaning": "Thermal variability and stability of the near-surface layer.",
    },
    {
        "feature": "t2m_range_12h",
        "block": "feature_engineering",
        "based_on": "T2M sequence",
        "calculation": "max(T2M)-min(T2M) over the 12 h window.",
        "scientific_meaning": "Diurnal cooling/heating amplitude.",
    },
    {
        "feature": "wspd10_delta_3h",
        "block": "feature_engineering",
        "based_on": "WSPD10 sequence",
        "calculation": "WSPD10(t) - WSPD10(t-3 h).",
        "scientific_meaning": "Recent ventilation change.",
    },
    {
        "feature": "wspd10_delta_6h",
        "block": "feature_engineering",
        "based_on": "WSPD10 sequence",
        "calculation": "WSPD10(t) - WSPD10(t-6 h).",
        "scientific_meaning": "Medium-term wind-speed tendency.",
    },
    {
        "feature": "wspd10_std_12h",
        "block": "feature_engineering",
        "based_on": "WSPD10 sequence",
        "calculation": "standard deviation over the 12 h window.",
        "scientific_meaning": "Wind variability and turbulence/ventilation regime proxy.",
    },
    {
        "feature": "wspd10_range_12h",
        "block": "feature_engineering",
        "based_on": "WSPD10 sequence",
        "calculation": "max(WSPD10)-min(WSPD10) over the 12 h window.",
        "scientific_meaning": "Amplitude of ventilation changes.",
    },
    {
        "feature": "rh2m_accel",
        "block": "feature_engineering",
        "based_on": "RH2M sequence",
        "calculation": "(RH2M(t)-RH2M(t-3 h)) - (RH2M(t-3 h)-RH2M(t-6 h)).",
        "scientific_meaning": "Acceleration of humidification or drying.",
    },
    {
        "feature": "humid_cold_proxy",
        "block": "feature_engineering",
        "based_on": "last-hour RH2M and T2M",
        "calculation": "RH2M * exp(-T2M_C/10).",
        "scientific_meaning": "Cold moist near-surface condition proxy.",
    },
    {
        "feature": "night_low_cloud_proxy",
        "block": "feature_engineering",
        "based_on": "last-hour zenith and LCC",
        "calculation": "1(zenith>90) * (1-LCC).",
        "scientific_meaning": "As coded, nighttime low-cloud/clear-sky contrast proxy for radiative cooling context.",
    },
    {
        "feature": "cold_humid_weak_wind_flag",
        "block": "feature_engineering",
        "based_on": "last-hour RH2M, T2M and WSPD10",
        "calculation": "1(RH2M>90 and T2M_C<10 and WSPD10<4).",
        "scientific_meaning": "Rule-based cold, humid, weak-wind fog-favoring state.",
    },
    {
        "feature": "rh_low_cloud_ratio",
        "block": "feature_engineering",
        "based_on": "last-hour RH2M and LCC",
        "calculation": "RH2M / (100*LCC + 1).",
        "scientific_meaning": "High surface humidity relative to modeled low-cloud cover; possible shallow fog signal.",
    },
    {
        "feature": "rh_squared",
        "block": "feature_engineering",
        "based_on": "last-hour RH2M",
        "calculation": "(RH2M/100)^2.",
        "scientific_meaning": "Nonlinear amplification of near-saturation humidity.",
    },
    {
        "feature": "low_level_shear",
        "block": "feature_engineering",
        "based_on": "last-hour U10,V10,U_925,V_925",
        "calculation": "tanh(sqrt((U_925-U10)^2+(V_925-V10)^2)/8).",
        "scientific_meaning": "Low-level shear affecting mixing, turbulence and fog erosion.",
    },
    {
        "feature": "wind_direction_turning",
        "block": "feature_engineering",
        "based_on": "last-hour U10,V10,U_925,V_925",
        "calculation": "0.5*(1-cos(theta_925-theta_10)).",
        "scientific_meaning": "Directional shear/turning proxy for stratification and advection changes.",
    },
    {
        "feature": "convective_wet_proxy",
        "block": "feature_engineering",
        "based_on": "last-hour CAPE and PRECIP",
        "calculation": "product of sigmoid(log1p(CAPE)-log(200)) and sigmoid(log1p(PRECIP)-log(0.1)).",
        "scientific_meaning": "Convective wet-weather visibility-loss proxy distinct from radiative fog.",
    },
    {
        "feature": "daytime_mixing_proxy",
        "block": "feature_engineering",
        "based_on": "last-hour SW_RAD, WSPD10 and INVERSION",
        "calculation": "sigmoid(SW_RAD-150) * sigmoid(WSPD10-4) * sigmoid(-INVERSION+0.5).",
        "scientific_meaning": "Daytime turbulent mixing and fog-dissipation potential.",
    },
    {
        "feature": "ventilation_proxy",
        "block": "feature_engineering",
        "based_on": "last-hour WSPD10 and low_level_shear",
        "calculation": "tanh(WSPD10*(1+shear_mag)/12).",
        "scientific_meaning": "Ventilation strength for dispersing droplets and aerosols.",
    },
    {
        "feature": "moisture_stratification",
        "block": "feature_engineering",
        "based_on": "last-hour Q_1000 and Q_925",
        "calculation": "tanh((Q_1000-Q_925)*1500).",
        "scientific_meaning": "Low-level moisture pooling and vertical moisture gradient.",
    },
    {
        "feature": "omega_contrast",
        "block": "feature_engineering",
        "based_on": "last-hour W_925 and W_1000",
        "calculation": "tanh((W_925-W_1000)/0.25).",
        "scientific_meaning": "Vertical-motion contrast linked to mixing, lifting and cloud formation.",
    },
    {
        "feature": "warm_instability_proxy",
        "block": "feature_engineering",
        "based_on": "last-hour INVERSION and T2M",
        "calculation": "tanh((-INVERSION + max(T2M_C-18,0)*0.25)/3).",
        "scientific_meaning": "Warm-season instability/mixing proxy for non-radiative low visibility regimes.",
    },
]

VERA_FEATURES: List[Dict[str, str]] = [
    {
        "feature": "vera_pm25_fraction",
        "block": "feature_engineering_vera_optional",
        "based_on": "last-hour PM2.5 and PM10",
        "calculation": "PM2.5 / (PM10 + eps), clipped to [0,1.5].",
        "scientific_meaning": "Fine-particle fraction affecting scattering efficiency.",
    },
    {
        "feature": "vera_coarse_pm_log",
        "block": "feature_engineering_vera_optional",
        "based_on": "last-hour PM10 and PM2.5",
        "calculation": "log1p(max(PM10-PM2.5,0)).",
        "scientific_meaning": "Coarse aerosol mass proxy for extinction.",
    },
    {
        "feature": "vera_growth_fine_log",
        "block": "feature_engineering_vera_optional",
        "based_on": "last-hour RH2M",
        "calculation": "log1p(clip((1-RH)^-0.85,1,10)).",
        "scientific_meaning": "Hygroscopic growth factor for fine aerosol under high RH.",
    },
    {
        "feature": "vera_growth_coarse_log",
        "block": "feature_engineering_vera_optional",
        "based_on": "last-hour RH2M",
        "calculation": "log1p(clip((1-RH)^-0.45,1,6)).",
        "scientific_meaning": "Hygroscopic growth factor for coarse aerosol.",
    },
    {
        "feature": "vera_hydrated_pm25",
        "block": "feature_engineering_vera_optional",
        "based_on": "PM2.5 and fine-aerosol RH growth",
        "calculation": "log1p(PM2.5) * vera_growth_fine_log.",
        "scientific_meaning": "Hydrated fine-aerosol scattering proxy.",
    },
    {
        "feature": "vera_hydrated_coarse",
        "block": "feature_engineering_vera_optional",
        "based_on": "coarse PM and coarse-aerosol RH growth",
        "calculation": "log1p(max(PM10-PM2.5,0)) * vera_growth_coarse_log.",
        "scientific_meaning": "Hydrated coarse-aerosol scattering proxy.",
    },
    {
        "feature": "vera_aerosol_ext_proxy",
        "block": "feature_engineering_vera_optional",
        "based_on": "hydrated PM2.5 and hydrated coarse PM",
        "calculation": "0.80*hydrated_pm25 + 0.25*hydrated_coarse.",
        "scientific_meaning": "Aerosol extinction proxy inspired by explicit scattering intermediates.",
    },
    {
        "feature": "vera_near_saturation_activation",
        "block": "feature_engineering_vera_optional",
        "based_on": "RH2M and DPD",
        "calculation": "sigmoid((RH2M-95)/2.5) * sigmoid(-DPD/1.5).",
        "scientific_meaning": "Activation of droplet/liquid-water formation near saturation.",
    },
    {
        "feature": "vera_liquid_water_proxy",
        "block": "feature_engineering_vera_optional",
        "based_on": "near-saturation activation, LCC and INVERSION",
        "calculation": "near_sat*(0.5+0.5*LCC)*(1+0.25*stability).",
        "scientific_meaning": "Cloud/fog liquid-water extinction proxy.",
    },
    {
        "feature": "vera_precip_ext_proxy",
        "block": "feature_engineering_vera_optional",
        "based_on": "PRECIP and near-saturation activation",
        "calculation": "log1p(PRECIP)*(0.5+0.5*near_sat).",
        "scientific_meaning": "Precipitation-particle extinction proxy.",
    },
    {
        "feature": "vera_total_ext_proxy",
        "block": "feature_engineering_vera_optional",
        "based_on": "aerosol, liquid-water and precipitation proxies",
        "calculation": "log1p(aerosol_ext + 2*liquid_water + 0.5*precip_ext).",
        "scientific_meaning": "Total extinction-like intermediate before visibility conversion.",
    },
    {
        "feature": "vera_inverse_visibility_proxy",
        "block": "feature_engineering_vera_optional",
        "based_on": "total_ext proxy",
        "calculation": "1/(0.10 + total_ext), clipped to [0,10].",
        "scientific_meaning": "Visibility-like transform of extinction proxy; monotonic but not a full diagnosis.",
    },
    {
        "feature": "vera_rh_error_sensitivity",
        "block": "feature_engineering_vera_optional",
        "based_on": "RH2M and fine-aerosol growth",
        "calculation": "log1p(clip((RH/(1-RH))*g_fine,0,80)).",
        "scientific_meaning": "Sensitivity of aerosol extinction to humidity/input error near saturation.",
    },
    {
        "feature": "vera_ext_std_12h",
        "block": "feature_engineering_vera_optional",
        "based_on": "12 h aerosol-extinction proxy sequence",
        "calculation": "standard deviation of aerosol_ext over 12 h.",
        "scientific_meaning": "Temporal variability of aerosol scattering potential.",
    },
    {
        "feature": "vera_ext_trend_3h",
        "block": "feature_engineering_vera_optional",
        "based_on": "12 h aerosol-extinction proxy sequence",
        "calculation": "aerosol_ext(t) - aerosol_ext(t-3 h).",
        "scientific_meaning": "Recent increase or decrease in aerosol extinction potential.",
    },
    {
        "feature": "vera_ext_peak_12h",
        "block": "feature_engineering_vera_optional",
        "based_on": "12 h aerosol-extinction proxy, Q_1000-Q_925 and WSPD10",
        "calculation": "max(aerosol_ext_12h) + moisture-pooling/weak-wind adjustment.",
        "scientific_meaning": "Peak aerosol extinction under moisture pooling and weak ventilation.",
    },
]

TIME_FEATURES: List[Dict[str, str]] = [
    {
        "feature": "month_sin",
        "block": "feature_engineering",
        "based_on": "UTC sample month",
        "calculation": "sin(2*pi*month/12).",
        "scientific_meaning": "Seasonal cycle encoded without a discontinuity at year boundary.",
    },
    {
        "feature": "month_cos",
        "block": "feature_engineering",
        "based_on": "UTC sample month",
        "calculation": "cos(2*pi*month/12).",
        "scientific_meaning": "Seasonal cycle companion coordinate.",
    },
    {
        "feature": "hour_sin",
        "block": "feature_engineering",
        "based_on": "UTC sample hour",
        "calculation": "sin(2*pi*hour/24).",
        "scientific_meaning": "Diurnal phase encoded without a discontinuity at midnight.",
    },
    {
        "feature": "hour_cos",
        "block": "feature_engineering",
        "based_on": "UTC sample hour",
        "calculation": "cos(2*pi*hour/24).",
        "scientific_meaning": "Diurnal phase companion coordinate.",
    },
]


def dynamic_features_for_count(dyn_vars_count: int) -> List[Dict[str, str]]:
    features = BASE_DYNAMIC_FEATURES.copy()
    if dyn_vars_count >= 25:
        features.append(DYNAMIC_APPEND_FEATURES[0])
    if dyn_vars_count >= 26:
        features.append(DYNAMIC_APPEND_FEATURES[1])
    if dyn_vars_count >= 27:
        features.append(DYNAMIC_APPEND_FEATURES[2])
    return features[:dyn_vars_count]


def fe_features_for_dim(extra_feat_dim: int) -> List[Dict[str, str]]:
    main = BASE_FOG_FEATURES + TIME_FEATURES
    vera = BASE_FOG_FEATURES + VERA_FEATURES + TIME_FEATURES
    if extra_feat_dim <= len(main):
        return main[:extra_feat_dim]
    if extra_feat_dim <= len(vera):
        return vera[:extra_feat_dim]
    out = vera.copy()
    for i in range(len(out), extra_feat_dim):
        out.append(
            {
                "feature": f"extra_feature_{i:02d}",
                "block": "feature_engineering_unknown",
                "based_on": "not described in local feature catalog",
                "calculation": "Inspect the data-build script that produced this dataset.",
                "scientific_meaning": "Unknown until verified against the source data-build script.",
            }
        )
    return out


def catalog_rows(dyn_vars_count: int = 27, extra_feat_dim: int = 36) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    position = 0
    for item in dynamic_features_for_count(dyn_vars_count):
        row = dict(item)
        row.update(
            {
                "position": str(position),
                "span": "12 hourly values",
                "used_in_main_pm10_pm25": "yes",
            }
        )
        rows.append(row)
        position += 1
    for item in STATIC_FEATURES:
        row = dict(item)
        row.update(
            {
                "position": str(position),
                "span": "station static value",
                "used_in_main_pm10_pm25": "yes",
            }
        )
        rows.append(row)
        position += 1
    for item in fe_features_for_dim(extra_feat_dim):
        row = dict(item)
        row.update(
            {
                "position": str(position),
                "span": "single engineered value per sample",
                "used_in_main_pm10_pm25": "yes" if item["block"] != "feature_engineering_vera_optional" else "optional_vera_dataset",
            }
        )
        rows.append(row)
        position += 1
    return rows


def permutation_groups(window_size: int, dyn_vars_count: int, extra_feat_dim: int) -> List[Dict[str, object]]:
    groups: List[Dict[str, object]] = []
    dyn_features = dynamic_features_for_count(dyn_vars_count)
    for i, item in enumerate(dyn_features):
        cols = [t * dyn_vars_count + i for t in range(window_size)]
        groups.append(
            {
                "feature": item["feature"],
                "block": item["block"],
                "columns": cols,
                "n_columns": len(cols),
            }
        )

    split_dyn = window_size * dyn_vars_count
    for j, item in enumerate(STATIC_FEATURES):
        groups.append(
            {
                "feature": item["feature"],
                "block": item["block"],
                "columns": [split_dyn + j],
                "n_columns": 1,
            }
        )

    split_extra = split_dyn + 6
    for j, item in enumerate(fe_features_for_dim(extra_feat_dim)):
        groups.append(
            {
                "feature": item["feature"],
                "block": item["block"],
                "columns": [split_extra + j],
                "n_columns": 1,
            }
        )
    return groups


def write_catalog(rows: Sequence[Dict[str, str]], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "position",
        "block",
        "feature",
        "span",
        "based_on",
        "calculation",
        "scientific_meaning",
        "used_in_main_pm10_pm25",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| # | Block | Feature | Based on / calculation | Scientific meaning |\n")
        f.write("|---:|---|---|---|---|\n")
        for row in rows:
            calc = f"{row.get('based_on','')}; {row.get('calculation','')}"
            f.write(
                "| {position} | {block} | `{feature}` | {calc} | {meaning} |\n".format(
                    position=row.get("position", ""),
                    block=row.get("block", ""),
                    feature=row.get("feature", ""),
                    calc=calc.replace("|", "/"),
                    meaning=row.get("scientific_meaning", "").replace("|", "/"),
                )
            )


__all__ = [
    "BASE_DYNAMIC_FEATURES",
    "DYNAMIC_APPEND_FEATURES",
    "STATIC_FEATURES",
    "BASE_FOG_FEATURES",
    "VERA_FEATURES",
    "TIME_FEATURES",
    "catalog_rows",
    "dynamic_features_for_count",
    "fe_features_for_dim",
    "permutation_groups",
    "write_catalog",
]

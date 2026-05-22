| # | Block | Feature | Based on / calculation | Scientific meaning |
|---:|---|---|---|---|
| 0 | dynamic_12h | `RH2M` | 2 m relative humidity at each hour in the 12 h window; Direct Tianji field after variable renaming. | Near-surface saturation controls fog and mist formation and aerosol hygroscopic growth. |
| 1 | dynamic_12h | `T2M` | 2 m air temperature at each hour; Direct Tianji field after variable renaming. | Sets saturation vapor pressure, cooling tendency and fog persistence conditions. |
| 2 | dynamic_12h | `PRECIP` | Surface precipitation at each hour; Direct Tianji field; log1p-transformed in model preprocessing. | Represents hydrometeor extinction and wet-scavenging or wet-weather low visibility. |
| 3 | dynamic_12h | `MSLP` | Mean sea-level pressure at each hour; Direct Tianji field after variable renaming. | Synoptic pressure pattern proxy for stable high pressure, fronts and advection regime. |
| 4 | dynamic_12h | `SW_RAD` | Downward shortwave radiation at the surface; Direct Tianji field; log1p-transformed in model preprocessing. | Daytime heating and mixing proxy; low values at night favor radiative cooling. |
| 5 | dynamic_12h | `U10` | 10 m zonal wind; Direct Tianji field after variable renaming. | Low-level advection and shear component affecting ventilation and fog displacement. |
| 6 | dynamic_12h | `WSPD10` | 10 m wind speed; Direct Tianji or derived wind-speed field. | Weak wind supports stagnation; stronger wind ventilates fog and aerosols. |
| 7 | dynamic_12h | `V10` | 10 m meridional wind; Direct Tianji field after variable renaming. | Low-level advection and shear component affecting moisture and aerosol transport. |
| 8 | dynamic_12h | `WDIR10` | 10 m wind direction; Direct Tianji or derived wind-direction field. | Flow-regime indicator for terrain-channeling and moist-air source direction. |
| 9 | dynamic_12h | `CAPE` | Convective available potential energy; Direct Tianji field; log1p-transformed in model preprocessing. | Convective instability proxy; helps separate warm convective low visibility from radiative fog. |
| 10 | dynamic_12h | `LCC` | Low cloud cover; Direct Tianji field after variable renaming. | Cloud and near-surface liquid-water proxy linked to radiation and visibility loss. |
| 11 | dynamic_12h | `T_925` | 925 hPa temperature; Direct Tianji field after variable renaming. | Lower-tropospheric thermal structure and inversion context. |
| 12 | dynamic_12h | `RH_925` | 925 hPa relative humidity; Direct Tianji field after variable renaming. | Vertical humidity structure; distinguishes shallow fog from deeper moist layers. |
| 13 | dynamic_12h | `U_925` | 925 hPa zonal wind; Direct Tianji field after variable renaming. | Low-level shear and advection above the surface. |
| 14 | dynamic_12h | `WSPD925` | 925 hPa wind speed; Direct Tianji or derived wind-speed field. | Ventilation and shear proxy above the surface layer. |
| 15 | dynamic_12h | `V_925` | 925 hPa meridional wind; Direct Tianji field after variable renaming. | Low-level shear and moisture transport above the surface. |
| 16 | dynamic_12h | `DP_1000` | 1000 hPa dew-point temperature; Direct Tianji field after variable renaming. | Near-surface moisture content and saturation proximity. |
| 17 | dynamic_12h | `DP_925` | 925 hPa dew-point temperature; Direct Tianji field after variable renaming. | Lower-tropospheric moisture reservoir and vertical moisture gradient. |
| 18 | dynamic_12h | `Q_1000` | 1000 hPa specific humidity; Direct Tianji field after variable renaming. | Absolute near-surface moisture supply for condensation and haze growth. |
| 19 | dynamic_12h | `Q_925` | 925 hPa specific humidity; Direct Tianji field after variable renaming. | Vertical moisture stratification and lower-tropospheric moisture support. |
| 20 | dynamic_12h | `W_925` | 925 hPa vertical velocity; Direct Tianji omega field after variable renaming. | Vertical motion proxy affecting cloud, mixing and moisture convergence. |
| 21 | dynamic_12h | `W_1000` | 1000 hPa vertical velocity; Direct Tianji omega field after variable renaming. | Near-surface vertical motion proxy for mixing and convergence. |
| 22 | dynamic_12h | `DPD` | T2M and D2M; DPD = T2M - D2M; D2M is computed from T2M and RH2M if absent. | Dew-point depression; small values indicate near saturation. |
| 23 | dynamic_12h | `INVERSION` | T_925 and T2M; INVERSION = T_925 - T2M. | Low-level stability proxy; positive values suppress vertical mixing. |
| 24 | dynamic_12h | `zenith` | station latitude, longitude and UTC valid time; pvlib apparent solar zenith angle; zero-filled if solar calculation fails. | Solar geometry proxy for night, radiative cooling and daytime mixing. |
| 25 | dynamic_12h | `PM10_ugm3` | station PM10 time series matched to Tianji time and station; Nearest time/station match; max with zero; kg m-3 to ug m-3 by multiplying 1e12. | Coarse and total aerosol loading contributing to extinction. |
| 26 | dynamic_12h | `PM25_ugm3` | station PM2.5 time series matched to Tianji time and station; Nearest time/station match; max with zero; kg m-3 to ug m-3 by multiplying 1e12. | Fine aerosol loading; important for hygroscopic growth and scattering. |
| 27 | static | `lat_norm` | station latitude; lat / 90. | Climatological latitude gradient in radiation, temperature and fog regime. |
| 28 | static | `lon_norm` | station longitude; lon / 180. | Regional circulation and climatological location proxy. |
| 29 | static | `orography` | nearest terrain height h; h at nearest grid point. | Altitude affects temperature, pressure, cloud base and station representativeness. |
| 30 | static | `orography_anom` | nearest terrain height and local terrain window; h - mean(h in a radius-2 grid window). | Local valley/ridge exposure proxy for pooling and drainage flows. |
| 31 | static | `orography_std` | local terrain window; std(h in a radius-2 grid window). | Terrain complexity proxy for sub-grid representativeness and local circulations. |
| 32 | static_category | `veg_type` | nearest vegetation class htcc; Mapped to an integer category and passed through an embedding layer. | Land-cover proxy for surface moisture, roughness and radiation response. |
| 33 | feature_engineering | `sat_dpd` | last-hour RH2M and DPD; clip(RH2M/100,0,1) * 1/(1+exp(DPD/2)). | Near-saturation proxy combining high humidity and small dew-point depression. |
| 34 | feature_engineering | `wind_favourability` | last-hour WSPD10; Gaussian preference centered at 3.5 m s-1 with sigma 2.5. | Moderate weak wind can sustain fog without strong ventilation. |
| 35 | feature_engineering | `stability_ri` | last-hour INVERSION and WSPD10; tanh((INVERSION/(WSPD10^2+0.1))/2). | Richardson-like stability proxy; stable weak-wind layers trap moisture and aerosols. |
| 36 | feature_engineering | `night_clear_radiation` | last-hour zenith, LCC and SW_RAD; 1(zenith>90) * clip(1-LCC/0.3,0,1) * (1-clip(SW_RAD/800,0,1)). | Nighttime clear-sky radiative-cooling potential. |
| 37 | feature_engineering | `rh_surface_minus_925` | last-hour RH2M and RH_925; tanh((RH2M - RH_925)/50). | Vertical humidity contrast; highlights shallow surface moisture pooling. |
| 38 | feature_engineering | `fog_potential` | RH2M, wind_favourability, stability_ri and night_clear_radiation; 0.4*RH_norm + 0.25*wind_fav + 0.2*positive stability + 0.15*night cooling. | Composite heuristic for radiative fog favorability. |
| 39 | feature_engineering | `rh2m_delta_3h` | RH2M sequence; RH2M(t) - RH2M(t-3 h). | Recent humidification tendency. |
| 40 | feature_engineering | `rh2m_delta_6h` | RH2M sequence; RH2M(t) - RH2M(t-6 h). | Medium-term humidification tendency. |
| 41 | feature_engineering | `rh2m_std_12h` | RH2M sequence; standard deviation over the 12 h window. | Humidity variability and persistence proxy. |
| 42 | feature_engineering | `rh2m_range_12h` | RH2M sequence; max(RH2M)-min(RH2M) over the 12 h window. | Amplitude of humidity evolution. |
| 43 | feature_engineering | `t2m_delta_3h` | T2M sequence; T2M(t) - T2M(t-3 h). | Recent cooling or warming tendency. |
| 44 | feature_engineering | `t2m_delta_6h` | T2M sequence; T2M(t) - T2M(t-6 h). | Medium-term cooling or warming tendency. |
| 45 | feature_engineering | `t2m_std_12h` | T2M sequence; standard deviation over the 12 h window. | Thermal variability and stability of the near-surface layer. |
| 46 | feature_engineering | `t2m_range_12h` | T2M sequence; max(T2M)-min(T2M) over the 12 h window. | Diurnal cooling/heating amplitude. |
| 47 | feature_engineering | `wspd10_delta_3h` | WSPD10 sequence; WSPD10(t) - WSPD10(t-3 h). | Recent ventilation change. |
| 48 | feature_engineering | `wspd10_delta_6h` | WSPD10 sequence; WSPD10(t) - WSPD10(t-6 h). | Medium-term wind-speed tendency. |
| 49 | feature_engineering | `wspd10_std_12h` | WSPD10 sequence; standard deviation over the 12 h window. | Wind variability and turbulence/ventilation regime proxy. |
| 50 | feature_engineering | `wspd10_range_12h` | WSPD10 sequence; max(WSPD10)-min(WSPD10) over the 12 h window. | Amplitude of ventilation changes. |
| 51 | feature_engineering | `rh2m_accel` | RH2M sequence; (RH2M(t)-RH2M(t-3 h)) - (RH2M(t-3 h)-RH2M(t-6 h)). | Acceleration of humidification or drying. |
| 52 | feature_engineering | `humid_cold_proxy` | last-hour RH2M and T2M; RH2M * exp(-T2M_C/10). | Cold moist near-surface condition proxy. |
| 53 | feature_engineering | `night_low_cloud_proxy` | last-hour zenith and LCC; 1(zenith>90) * (1-LCC). | As coded, nighttime low-cloud/clear-sky contrast proxy for radiative cooling context. |
| 54 | feature_engineering | `cold_humid_weak_wind_flag` | last-hour RH2M, T2M and WSPD10; 1(RH2M>90 and T2M_C<10 and WSPD10<4). | Rule-based cold, humid, weak-wind fog-favoring state. |
| 55 | feature_engineering | `rh_low_cloud_ratio` | last-hour RH2M and LCC; RH2M / (100*LCC + 1). | High surface humidity relative to modeled low-cloud cover; possible shallow fog signal. |
| 56 | feature_engineering | `rh_squared` | last-hour RH2M; (RH2M/100)^2. | Nonlinear amplification of near-saturation humidity. |
| 57 | feature_engineering | `low_level_shear` | last-hour U10,V10,U_925,V_925; tanh(sqrt((U_925-U10)^2+(V_925-V10)^2)/8). | Low-level shear affecting mixing, turbulence and fog erosion. |
| 58 | feature_engineering | `wind_direction_turning` | last-hour U10,V10,U_925,V_925; 0.5*(1-cos(theta_925-theta_10)). | Directional shear/turning proxy for stratification and advection changes. |
| 59 | feature_engineering | `convective_wet_proxy` | last-hour CAPE and PRECIP; product of sigmoid(log1p(CAPE)-log(200)) and sigmoid(log1p(PRECIP)-log(0.1)). | Convective wet-weather visibility-loss proxy distinct from radiative fog. |
| 60 | feature_engineering | `daytime_mixing_proxy` | last-hour SW_RAD, WSPD10 and INVERSION; sigmoid(SW_RAD-150) * sigmoid(WSPD10-4) * sigmoid(-INVERSION+0.5). | Daytime turbulent mixing and fog-dissipation potential. |
| 61 | feature_engineering | `ventilation_proxy` | last-hour WSPD10 and low_level_shear; tanh(WSPD10*(1+shear_mag)/12). | Ventilation strength for dispersing droplets and aerosols. |
| 62 | feature_engineering | `moisture_stratification` | last-hour Q_1000 and Q_925; tanh((Q_1000-Q_925)*1500). | Low-level moisture pooling and vertical moisture gradient. |
| 63 | feature_engineering | `omega_contrast` | last-hour W_925 and W_1000; tanh((W_925-W_1000)/0.25). | Vertical-motion contrast linked to mixing, lifting and cloud formation. |
| 64 | feature_engineering | `warm_instability_proxy` | last-hour INVERSION and T2M; tanh((-INVERSION + max(T2M_C-18,0)*0.25)/3). | Warm-season instability/mixing proxy for non-radiative low visibility regimes. |
| 65 | feature_engineering | `month_sin` | UTC sample month; sin(2*pi*month/12). | Seasonal cycle encoded without a discontinuity at year boundary. |
| 66 | feature_engineering | `month_cos` | UTC sample month; cos(2*pi*month/12). | Seasonal cycle companion coordinate. |
| 67 | feature_engineering | `hour_sin` | UTC sample hour; sin(2*pi*hour/24). | Diurnal phase encoded without a discontinuity at midnight. |
| 68 | feature_engineering | `hour_cos` | UTC sample hour; cos(2*pi*hour/24). | Diurnal phase companion coordinate. |

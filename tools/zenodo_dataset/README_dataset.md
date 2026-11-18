# HAZBRA – User guide

**HAZBRA: a climate hazard dataset for Brazil with observed and projected indicators at grid and municipal scale**
Version 1.0 · Licence CC BY 4.0 · Code: https://github.com/Luisgcattelan/HAZBRA (doi:10.5281/zenodo.24319623)

HAZBRA provides 16 indicators of extreme rainfall, drought, heat and compound hazards for Brazil:

* **Observed** (historical): climatologies, trends and GEV return levels from three gridded datasets –
  CHIRPS, MERGE and BR-DWGD (labelled `XAVIER` in the files);
* **Projected**: changes at global warming levels (GWLs) of +1.5, +2 and +3 °C from 19 bias-corrected CMIP6 models
  of CLIMBRA – climatologies, return levels, ensemble statistics, model agreement, return-period shifts,
  delta-change projections and annual time series;
* on **regular grids** (NetCDF4) and for the **5,570 municipalities** of Brazil (Parquet tables, IBGE 2024 codes).

The source datasets (CHIRPS, MERGE, BR-DWGD, CLIMBRA) are not redistributed.

---

## 1. Quick start

1. Download the zips you need (see §2) and unpack them **in the same folder** – together they rebuild the tree
   `netcdf/` and `municipalities/`. `MD5SUMS.txt` lets you check the downloads (`md5sum -c MD5SUMS.txt`).
2. Gridded data – e.g. projected change of the 25-year Rx5day at +2 °C (ensemble median, %):

   ```python
   import xarray as xr
   ds = xr.open_dataset("netcdf/CLIMBRA/Rainfall/Rx5day/ensemble/ensemble_anomaly_return_period.nc")
   da = ds["Rx5day_anom_perc"].sel(ensemble="median", GWL="2", return_period=25)
   ```

3. Municipal data – the same quantity per municipality:

   ```python
   import pandas as pd, geopandas as gpd
   df = pd.read_parquet("municipalities/CLIMBRA/Rainfall/Rx5day/ensemble_anomaly_return_period.parquet")
   sub = df[(df.variable == "Rx5day_anom_perc") & (df.ensemble == "median") & (df.GWL == "2") & (df.return_period == 25)]
   geo = gpd.read_file("municipalities/geometry/municipalities_2024.gpkg", layer="municipalities")
   gdf = geo.merge(sub[["CD_MUN", "mean", "method"]], on="CD_MUN")
   ```

   (`GWL` and `ensemble` are text columns, `return_period` is an integer.)

## 2. Files in this record

| File | Contents |
|---|---|
| `README.md` | this user guide |
| `README_municipalities.md` | detailed description of the municipal tables (columns, aggregation method, checks) |
| `netcdf_CLIMBRA_<Hazard>.zip` | CLIMBRA products per model and ensemble (`netcdf/CLIMBRA/<Hazard>/...`) |
| `netcdf_Observational_<Hazard>.zip` | observational products (`netcdf/Observational/<Hazard>/...`) |
| `municipalities_CLIMBRA_<Hazard>.zip` | municipal tables of the CLIMBRA ensemble products |
| `municipalities_Observations_<Hazard>.zip` | municipal tables of the observational products |
| `municipalities_geometry.zip` | municipality polygons used (+ 25 km buffers), state outlines, removed island/lagoon parts |
| `municipalities_weights.zip` | grid-cell × municipality area weights (only needed to reproduce the aggregation) |
| `netcdf_outliers.zip` | quality-control flags of every modified return-level cell + summary tables (§7) |
| `MD5SUMS.txt` | checksums of the zips |

`<Hazard>` is `Rainfall`, `Drought`, `Heat` or `Compound`.

## 3. Indicators

| Hazard | Indicator (file name) | Definition | Unit | Observations | GEV¹ |
|---|---|---|---|---|---|
| Rainfall | `Rx1day` | annual maximum 1-day precipitation | mm | CHIRPS, MERGE, XAVIER | yes |
| Rainfall | `Rx5day` | annual maximum consecutive 5-day precipitation | mm | CHIRPS, MERGE, XAVIER | yes |
| Rainfall | `CWD` | maximum number of consecutive days with P ≥ 1 mm | days | CHIRPS, MERGE, XAVIER | yes |
| Rainfall | `prcptot` | annual total precipitation on days with P ≥ 1 mm | mm | CHIRPS, MERGE, XAVIER | – |
| Rainfall | `pr_season` | seasonal mean precipitation (DJF, MAM, JJA, SON), as daily mean × 30 | mm/month | CHIRPS, MERGE, XAVIER | – |
| Drought | `CDD` | maximum number of consecutive days with P < 1 mm | days | CHIRPS, MERGE, XAVIER | yes |
| Drought | `CDD_season` | CDD within each season | days | CHIRPS, MERGE, XAVIER | yes² |
| Drought | `SPI3` | seasonal minimum of the 3-month Standardized Precipitation Index | – | CHIRPS, MERGE, XAVIER | yes² (min) |
| Drought | `SPI6` | annual minimum of the 6-month SPI | – | CHIRPS, MERGE, XAVIER | yes (min) |
| Heat | `TXX` | annual maximum of daily maximum temperature | °C | XAVIER | yes |
| Heat | `TNN` | annual minimum of daily minimum temperature | °C | XAVIER | yes (min) |
| Heat | `Frost` | longest annual spell of days with TN < 6 °C (Brazilian frost-risk threshold) | days | XAVIER | – |
| Heat | `WSDI` | warm spell duration index: days in spells of ≥ 6 days with TX > calendar-day 90th percentile of 1981–2010 | days | XAVIER | – |
| Compound | `HI` | NOAA Heat Index from TX and daily relative humidity: `HI_mean`, `HI_max` (°C) and days per class (below) | °C, days | XAVIER | yes |
| Compound | `FWI` | Canadian Fire Weather Index (TX, RH, wind, P): `FWI_mean`, `FWI_max` and days per class (below) | –, days | XAVIER | yes |
| Compound | `Dry_Hot` | days with TX above its calendar-day 75th percentile and P below the 25th percentile of wet-day P (1981–2010, 5-day window) | days | XAVIER | – |

¹ GEV = return levels (5, 10, 25, 50, 100 years) and GEV parameters; *(min)* = fitted to minima, levels from the lower tail.
Return-period shifts (§5) are provided for every GEV indicator except the seasonal ones (² `CDD_season`, `SPI3`).

**Class counts.** HI (°C): `days_caution` 27–32, `days_extreme_caution` 32–41, `days_danger` 41–54,
`days_extreme_danger` ≥ 54, `days_hot_total` ≥ 27. FWI: `days_low` 0–5, `days_moderate` 5–12, `days_high` 12–22,
`days_very_high` 22–50, `days_extreme` ≥ 50, `days_danger_total` ≥ 5 (lower bound included, upper bound excluded).

HI and FWI use the 10 CLIMBRA models that provide relative humidity and wind (ACCESS-ESM1-5, CMCC-ESM2, EC-EARTH3,
INM-CM4_8, INM-CM5, IPSL-CM6A-LR, MIROC6, MPI-ESM1-2, MRI-ESM2, NorESM2-MM); all other indicators use the 19 models.

## 4. Periods and grids

| | Grid | Series | Reference (climatology, GEV) | Trend |
|---|---|---|---|---|
| CHIRPS v2.0 | 0.05°, 780 × 800 | 1981–2024 | 1981–2010 | 1981–2023 |
| MERGE (INPE) | 0.1°, 800 × 600 (South America) | 2001–2024 | 2001–2020 | 2001–2023 |
| BR-DWGD (`XAVIER`) | 0.1°, 393 × 391 | 1981–2023 | 1981–2010 | 1981–2023 |
| CLIMBRA (19 models) | 0.25°, 162 × 168 | 1980–2100 (SSP2-4.5, SSP5-8.5) | `present` = 1981–2010; GWL windows (below) | – |

* **Coordinates**: `lat`, `lon` in degrees (WGS84), cell centres. Cells outside each dataset's land domain are NaN.
* **GWL windows**: for each model, the 30-year window centred on the year the model reaches +1.5, +2 or +3 °C relative
  to 1850–1900 in SSP5-8.5 (Hauser et al. 2022 timings, 20-year window extended by 5 years on each side). The
  `GWL` coordinate is `present`, `1.5`, `2`, `3` (text). Indicators are computed on the full daily series first and
  then cut to each window.
* **CLIMBRA calendar**: days that do not exist in a model's calendar are skipped. The year **2014 is missing** in all
  models (the historical files end in 2013 and the scenarios start in 2015), so windows containing 2014 have 29 years.
* **Seasonal indicators** (`pr_season`, `CDD_season`, `SPI3`) have a `season` dimension (DJF, MAM, JJA, SON).
* **SPI** uses a gamma distribution fitted to the whole series of each source (observational record; 1980–2100 per model).
* **Model names** in the files are the CLIMBRA short names: ACCESS-CM2, ACCESS-ESM1-5, CMCC-ESM2, EC-EARTH3 (EC-Earth3),
  GFDL-CM4, GFDL-ESM4, HadGEM3-GC31-LL, INM-CM4_8 (INM-CM4-8), INM-CM5 (INM-CM5-0), IPSL-CM6A-LR, KACE (KACE-1-0-G),
  KIOST (KIOST-ESM), MIROC6, MPI-ESM1-2 (MPI-ESM1-2-HR), MRI-ESM2 (MRI-ESM2-0), NESM3, NorESM2-MM, TaiESM1, UKESM1-0-LL.

## 5. Gridded products (`netcdf/`)

### 5.1 Observations – `netcdf/Observational/<Hazard>/<Indicator>/<DATASET>_<kind>.nc`

| kind | Contents | Dimensions |
|---|---|---|
| `climatology` | mean of the annual (or seasonal) values over the reference period | lat, lon [, season] |
| `timeseries` | annual (or seasonal) series | time, lat, lon [, season] |
| `timeseries_monthly` | monthly SPI-3 series (SPI3 only) | time, lat, lon |
| `return_period` | GEV return levels | return_period (5…100), lat, lon [, season] |
| `params` | GEV parameters `c`, `loc`, `scale` | dparams, lat, lon [, season] |
| `trend` | `p_values`, `slopes` (Sen, units per year), `intercepts` – Hamed & Rao (1998) modified Mann–Kendall test; significant if p < 0.05 | lat, lon |

Multi-variable indicators (HI, FWI) hold one variable per statistic (`HI_mean`, `HI_max`, `days_*`); their trend files
use `<var>_p_values`, `<var>_slopes`, `<var>_intercepts`.

### 5.2 CLIMBRA per model – `netcdf/CLIMBRA/<Hazard>/<Indicator>/individual_models/<MODEL>_<kind>.nc`

| kind | Contents | Dimensions |
|---|---|---|
| `climatology` | mean over each window | GWL (present, 1.5, 2, 3), lat, lon |
| `return_period` | GEV return levels per window | GWL, return_period, lat, lon |
| `params` | GEV parameters per window (`c`, `loc`, `scale`) | GWL, dparams, lat, lon |
| `timeseries` | annual series 1980–2100 (not for SPI3/SPI6) | scenario (ssp245, ssp585), time, lat, lon |
| `rp_shift` | `<var>_hist_return_level` (present-day level of the T-year event), `<var>_new_return_period` (years, 1–1000), `<var>_probability_ratio` (= T / new return period; > 1 more frequent) | GWL (1.5, 2, 3), event_rp, lat, lon |

### 5.3 CLIMBRA ensemble – `netcdf/CLIMBRA/<Hazard>/<Indicator>/ensemble/`

The `ensemble` dimension holds statistics **across models**: `mean`, `median`, `p10`, `p90` (computed per grid cell).

| File | Contents |
|---|---|
| `ensemble_climatology.nc`, `ensemble_return_period.nc`, `ensemble_timeseries.nc` | ensemble statistics of the per-model products |
| `ensemble_anomaly_climatology.nc`, `ensemble_anomaly_return_period.nc` | change relative to `present`, per model then across models: `<var>_anom` (absolute), `<var>_anom_perc` (%), `<var>_model_agreement` |
| `ensemble_rp_shift.nc` | ensemble (p10, median, p90) of the per-model return-period shift, plus `<var>_frac_more_frequent`, `<var>_rp_model_agreement`, `<var>_n_models`, `<var>_frac_capped` |
| `ensemble_delta_<DATASET>_climatology.nc`, `ensemble_delta_<DATASET>_return_period.nc` | **delta-change projections**: observed baseline of `<DATASET>` (interpolated bilinearly to the 0.25° grid) + ensemble statistic of the model change; `GWL = present` is the observed baseline itself. Day counts are bounded by 0 and 365 (92 for CDD_season), amounts by 0 |

**Model agreement** (`<var>_model_agreement`): number of models with a positive change, kept only where at least 66 %
of the models agree on the sign of the change; NaN means low agreement.
`<var>_anom_perc` is NaN where the present-day value is 0.

## 6. Municipal products (`municipalities/`)

Every gridded observational file (`climatology`, `return_period`, `params`, `trend`, `timeseries`) and every CLIMBRA
**ensemble** file is aggregated to the 5,570 municipalities of the IBGE 2024 mesh (lagoons and oceanic islands removed):

* `municipalities/Observations/<Hazard>/<Indicator>/<DATASET>_<kind>.parquet`
* `municipalities/CLIMBRA/<Hazard>/<Indicator>/<ensemble file>.parquet`

Long format: one row per municipality × variable × combination of the NetCDF dimensions. Key columns: `CD_MUN`
(IBGE code, text), `NM_MUN`, `NM_UF`, `variable`, the dimension columns (`ensemble`, `GWL`, `return_period`, …),
`mean` (area-weighted), `std`, `min`, `max`, `method` (`polygon`, `buffer25km` or `nearest`), and for agreement / trend
variables `agree_area_frac`, `robust` (≥ 50 % of the area with ≥ 66 % model agreement), `sig_area_frac`, `mean_sig`.
Small municipalities with less than one valid grid cell (or < 50 % of their area covered) use a 25 km buffer, then the
nearest valid cell within 50 km. See `README_municipalities.md` for all columns and checks.

## 7. Quality control of return levels (`netcdf/outliers/`)

Return-level fields were screened for unstable GEV fits: cells deviating from the 5 × 5 local median by more than six
standard deviations, and values beyond physical limits, were removed and filled by nearest-neighbour interpolation;
day counts above 365 (92 for CDD_season) were capped instead. For each source:

* `netcdf/outliers/<CLIMBRA|Observational>/<Hazard>/<Indicator>/<SOURCE>_outliers.nc`: `<var>_flag`
  (0 kept, 1 spatial outlier, 2 beyond the limit – capped for day counts, removed otherwise) and `<var>_raw`
  (original value where flagged);
* `outliers_by_indicator.csv`, `outliers_summary.csv`, `outliers_detail.csv`, `outliers_summary.md`: shares of
  modified cells; `outliers_log.csv`: full log of the filter runs.

## 8. Citation

Please cite the data paper (in preparation) and this record:
Cattelan, L. G., Amorim, P. B. de, Rodrigues, D. T., Buller, A., Vinagreiro, C. E., Moraes, R. J. N. P., Dalla Riva, E.,
& Mattos, C. HAZBRA: a climate hazard dataset for Brazil with observed and projected indicators at grid and municipal
scale (Version 1.0). Zenodo. https://doi.org/10.5281/zenodo.24320753

and the source datasets: CHIRPS (Funk et al. 2015), MERGE (Rozante et al. 2010, 2020), BR-DWGD (Xavier et al. 2022),
CLIMBRA (Ballarin et al. 2023).

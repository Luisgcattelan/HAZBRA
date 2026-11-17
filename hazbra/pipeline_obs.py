"""Observational driver: daily CHIRPS / MERGE / BR-DWGD (XAVIER) -> annual indicators -> products.

Per dataset and indicator it writes (hazard/indicator/):
    <DATASET>_timeseries.nc     annual (or seasonal-annual) series
    <DATASET>_climatology.nc    mean over the climatological period (XAVIER, CHIRPS 1981-2010; MERGE 2001-2020)
    <DATASET>_return_period.nc  GEV return levels 5-100 yr         (only indicators with `gev`)
    <DATASET>_params.nc         GEV parameters                     (only indicators with `gev`)
    <DATASET>_trend.nc          Mann-Kendall p-value + Sen's slope / intercept over the full record
                                (XAVIER 1981-2023, CHIRPS 1981-2023, MERGE 2001-2023)
    <DATASET>_timeseries_monthly.nc  monthly SPI3 only
Outlier filter of the return levels: flagged pixels in <outliers>/Observational/<Hazard>/<Indicator>/<DATASET>_outliers.nc,
rows appended to <outliers>/outliers_log.csv.

All indicator-specific science lives in `indicators.py`.
"""
from __future__ import annotations

import time
from pathlib import Path

import xarray as xr

from .indicators import OBS_PERIODS, OBS_TO_STD, Ctx, Indicator
from .io import open_obs, save_nc
from .outliers import append_log, drop_outliers, outlier_paths, save_flags
from .stats import compute_trend, gev_params, gev_return_levels, mann_kendall, mann_kendall_multi


LAT_CHUNK = 10        # dask chunk (rows) inside a latitude band


def out_dir(cfg, ind: Indicator) -> Path:
    return Path(cfg.paths.netcdf_obs) / ind.hazard / ind.name


def expected_files(ind: Indicator, dataset: str, out: Path) -> list[Path]:
    kinds = ["timeseries", "climatology", "trend"] + (["return_period", "params"] if ind.gev else [])
    return [out / f"{dataset}_{k}.nc" for k in kinds]


def _name(obj, name: str):
    return obj.rename(name) if isinstance(obj, xr.DataArray) else obj


def process_dataset(cfg, ind: Indicator, dataset: str, overwrite: bool = False) -> bool:
    out = out_dir(cfg, ind)
    out.mkdir(parents=True, exist_ok=True)
    if not overwrite and all(p.exists() for p in expected_files(ind, dataset, out)):
        print(f"=== {ind.name} / {dataset}: outputs exist, skipping (use --overwrite)")
        return False

    t0 = time.time()
    print(f"\n=== {ind.name} / {dataset}")
    per = OBS_PERIODS[dataset]
    first_year = str(int(per["series"][0]) - 1) if ind.seasonal else per["series"][0]   # DJF needs the previous Dec
    series_slice = slice(f"{first_year}-01-01", f"{per['series'][1]}-12-31")
    clim_slice = slice(f"{per['clim'][0]}-01-01", f"{per['clim'][1]}-12-31")
    trend_slice = slice(f"{per['trend'][0]}-01-01", f"{per['trend'][1]}-12-31")

    # ---- daily data -> annual indicator, in latitude bands (every indicator is computed cell by cell, so
    #      bands give identical results while bounding memory; each band is in memory and split in small
    #      lat chunks so dask computes it in parallel, with the whole time axis in one chunk)
    raw = open_obs(cfg, dataset, ind.obs_vars, series_slice, reindex_full=dataset in ind.obs_reindex_daily)
    daily = {OBS_TO_STD[k]: v for k, v in raw.items()}
    del raw
    ctx = Ctx(calib=slice("1981", "2010"), mask=False, product="OBS")
    nlat = next(iter(daily.values())).sizes["lat"]
    rows = int(getattr(cfg.calc, "obs_tile_rows", nlat))
    annual_parts, monthly_parts = [], []
    for i0 in range(0, nlat, rows):
        tb = time.time()
        band = {k: v.isel(lat=slice(i0, i0 + rows)).load().chunk({"time": -1, "lat": LAT_CHUNK})
                for k, v in daily.items()}
        if ind.prepare_obs:
            band = ind.prepare_obs(band)
        a = ind.annual(band, ctx)
        if ind.reduce_window:                   # SPI: the indicator is monthly, annual = minimum
            m = a.load()
            monthly_parts.append(m)
            a = ind.reduce_window(m)
        annual_parts.append(a.load())
        del band
        print(f"  lat rows {i0}-{min(i0 + rows, nlat)} of {nlat}: {(time.time() - tb) / 60:.1f} min")
    annual = xr.concat(annual_parts, dim="lat") if len(annual_parts) > 1 else annual_parts[0]
    monthly = (xr.concat(monthly_parts, dim="lat") if len(monthly_parts) > 1 else monthly_parts[0]) \
        if monthly_parts else None
    del annual_parts, monthly_parts

    # ---- products (GEV fit and trend in parallel over lat chunks)
    in_clim = annual.sel(time=clim_slice)
    clim = in_clim.mean("time").load()

    rp = params = None
    stats, flags = [], {}
    if ind.gev:
        in_clim_c = in_clim.chunk({"time": -1, "lat": LAT_CHUNK})
        rp = _name(gev_return_levels(in_clim_c, cfg.calc.return_periods, ind.gev), ind.name).load()
        rp = drop_outliers(rp, product="OBS", stats=stats, flags=flags,
                           context={"product": "OBS", "indicator": ind.name, "source": dataset})
        params = _name(gev_params(in_clim_c), ind.name).load()

    if ind.seasonal:                             # drop the helper year before the record starts
        annual = annual.sel(time=slice(per["series"][0], "2024"))
    in_trend = annual.sel(time=trend_slice).chunk({"time": -1, "lat": LAT_CHUNK})
    trend = compute_trend(mann_kendall_multi(in_trend) if ind.multi_variable else mann_kendall(in_trend))

    # ---- write
    save_nc(_name(annual, ind.name), out / f"{dataset}_timeseries.nc", ind.name)
    save_nc(_name(clim, ind.name), out / f"{dataset}_climatology.nc", ind.name)
    save_nc(trend, out / f"{dataset}_trend.nc")
    if rp is not None:
        save_nc(rp, out / f"{dataset}_return_period.nc", ind.name)
        save_nc(params, out / f"{dataset}_params.nc", ind.name)
        nc_path, log_path = outlier_paths(cfg, "OBS", ind.hazard, ind.name, dataset)
        save_flags(flags, nc_path)
        append_log(stats, log_path)
    if ind.obs_monthly_series and monthly is not None:
        save_nc(_name(monthly.drop_vars("prob_of_zero", errors="ignore"), ind.name),
                out / f"{dataset}_timeseries_monthly.nc", ind.name)

    print(f"  done in {(time.time() - t0) / 60:.1f} min")
    return True


def run(cfg, ind: Indicator, datasets=None, overwrite: bool = False) -> None:
    if datasets is None:
        datasets = ind.obs_datasets
        if cfg.test.enabled:
            datasets = [d for d in datasets if d in cfg.test.datasets]
    for d in datasets:
        process_dataset(cfg, ind, d, overwrite=overwrite or cfg.calc.overwrite)

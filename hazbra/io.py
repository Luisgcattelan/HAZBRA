"""Reading raw inputs and writing NetCDF products."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# float32 + zlib (+ byte shuffle) for every data variable we write
ENC32 = {"dtype": "float32", "zlib": True, "complevel": 4, "shuffle": True}

# attrs.units that xclim needs; applied to every variable present (CLIMBRA names and BR-DWGD names)
UNITS = {
    "tasmax": "degC", "tasmin": "degC", "tas": "degC", "hur": "%", "pr": "mm/d", "sfcWind": "m/s",
    "Tmax": "degC", "Tmin": "degC", "RH": "%", "u2": "m/s",
}


# --------------------------------------------------------------------------- writing
def save_nc(ds: xr.Dataset | xr.DataArray, path: str | os.PathLike, name: str | None = None) -> None:
    """Write float32 + zlib-compressed NetCDF. A DataArray needs `name` (or its own name)."""
    if isinstance(ds, xr.DataArray):
        name = name or ds.name
        if name is None:
            raise ValueError("A DataArray needs a name to be saved.")
        ds = ds.to_dataset(name=name)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, encoding={v: ENC32 for v in ds.data_vars})


def fix_units(ds: xr.Dataset) -> xr.Dataset:
    for v in ds.data_vars:
        if v in UNITS:
            ds[v].attrs["units"] = UNITS[v]
    return ds


# --------------------------------------------------------------------------- time axis helpers
def to_datetime64_time(obj):
    """Replace a cftime / non-ns time axis by datetime64[ns] via its ISO date string.

    Works for 'noleap' and standard calendars (a 360-day calendar would fail on 30 February).
    """
    if obj.time.values.dtype != "datetime64[ns]":
        dates = pd.to_datetime([pd.Timestamp(t.strftime("%Y-%m-%d")) for t in obj.time.values])
    else:
        dates = pd.to_datetime(obj.time.values)
    obj["time"] = dates
    return obj


def reindex_full_daily(obj):
    """Datetime axis + a complete daily axis (missing days -> NaN)."""
    obj = to_datetime64_time(obj)
    all_days = pd.date_range(start=obj.time.values.min(), end=obj.time.values.max(), freq="D")
    return obj.reindex(time=all_days)


def to_noleap_daily(da: xr.DataArray) -> xr.DataArray:
    """Convert to a noleap calendar and reindex to a continuous daily axis (gaps -> NaN)."""
    try:
        if hasattr(da, "convert_calendar"):
            da = da.convert_calendar("noleap", align_on="year")
        else:  # older xarray
            from xarray.coding.calendar_ops import convert_calendar as xr_convert_calendar
            da = xr_convert_calendar(da, "noleap", align_on="year")
    except Exception:
        # fallback: just drop 29 Feb
        da = da.sel(time=~((da.time.dt.month == 2) & (da.time.dt.day == 29)))

    cal = getattr(da.time.dt, "calendar", "proleptic_gregorian")
    t0, t1 = da.time.values[0], da.time.values[-1]
    if np.issubdtype(da.time.dtype, np.datetime64):
        full = xr.date_range(start=np.datetime64(t0), end=np.datetime64(t1), freq="D")
    else:
        full = xr.cftime_range(start=t0, end=t1, freq="D", calendar=cal)
    return da.reindex(time=full)


def seasonal_to_annual_time(da: xr.DataArray) -> xr.DataArray:
    """After groupby(['time.year', 'time.season']): rename `year` -> `time` (= 1 January of that year)."""
    da = da.rename({"year": "time"})
    new_time = pd.to_datetime(da.time.values.astype(str) + "-01-01")
    return da.assign_coords(time=new_time)


# --------------------------------------------------------------------------- CLIMBRA inputs
def climbra_files(raw_climbra: Path, model: str, varname: str, scenario: str = "ssp585") -> list[str]:
    """Files `<raw>/<var>/<MODEL>-<var>-{hist,<scenario>}.nc` of one model."""
    folder = Path(raw_climbra) / varname
    if not folder.is_dir():
        return []
    files = sorted(f for f in os.listdir(folder) if f.startswith(model) and ("hist" in f or scenario in f))
    return [str(folder / f) for f in files]


def open_climbra(cfg, model: str, varnames, scenario: str = "ssp585") -> xr.Dataset | None:
    """Lazy dataset with all `varnames` of one model/scenario (historical + scenario), or None if a
    variable is missing for this model (e.g. only 10 of the 19 models have hur / sfcWind / rss)."""
    parts = []
    for v in varnames:
        files = climbra_files(cfg.paths.raw_climbra, model, v, scenario)
        if not files:
            return None
        parts.append(xr.open_mfdataset(files, combine="by_coords", chunks="auto", use_cftime=True))
    ds = xr.merge(parts)
    from .config import subset_box
    return subset_box(ds, cfg)


# --------------------------------------------------------------------------- observational inputs
def open_obs(cfg, dataset: str, varnames, period: slice | None = None, load: bool = False,
             reindex_full: bool = False) -> dict[str, xr.DataArray]:
    """Daily observational variables as {name: DataArray} (BR-DWGD names for XAVIER).

    `load=True` reads them into memory (needed by run-length / multi-variable indicators); otherwise
    they stay lazy (dask) and only the annual result is loaded.
    `reindex_full=True` completes the daily axis (missing days -> NaN) BEFORE selecting `period`
    (needed for rolling-window indicators on MERGE, which has missing days).
    """
    from .config import subset_box
    out = {}
    for v in varnames:
        if dataset == "XAVIER":
            folder = Path(cfg.paths.raw_xavier) / v
            files = sorted(str(folder / f) for f in os.listdir(folder) if f.endswith(".nc"))
            ds = xr.open_mfdataset(files, combine="by_coords", chunks="auto")
            ds = ds.rename({"latitude": "lat", "longitude": "lon"})
        elif dataset in ("CHIRPS", "MERGE"):
            if v != "pr":
                raise ValueError(f"{dataset} only provides precipitation (requested {v}).")
            path = cfg.paths.chirps_daily if dataset == "CHIRPS" else cfg.paths.merge_daily
            ds = xr.open_dataset(path, chunks="auto")
        else:
            raise ValueError(f"Unknown observational dataset {dataset}")
        ds = subset_box(ds, cfg)
        if reindex_full:
            ds = reindex_full_daily(ds)
        if period is not None:
            ds = ds.sel(time=period)
        ds = fix_units(ds)
        out[v] = ds[v].load() if load else ds[v]
    return out

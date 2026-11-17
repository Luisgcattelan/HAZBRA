"""Outlier filtering of return-level fields (GEV fits can explode in a few grid cells).

For every 2-D field (lat, lon):
  1. spatial outliers: |value - 5x5 median| > 6 * std(|value - median|)          -> removed
  2. absolute outliers: value beyond a per-variable physical threshold            -> removed
     (day counts `days_*`, CDD, CWD and CDD_season are CAPPED at their threshold – 365 days, 92 for
     CDD_season – instead of being removed)
  3. removed cells are filled by spatial interpolation (rioxarray `interpolate_na`);
     cells that were NaN in the input (ocean / outside the domain) stay NaN.
"""
from __future__ import annotations

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from scipy.ndimage import median_filter

# Absolute thresholds on return levels (positive: values >= thr are removed; negative: values <= thr).
THRESHOLDS = {
    "FWI_max": 100, "FWI_mean": 100,
    "CDD": 365, "CDD_season": 92, "CWD": 365,
    "days_low": 365, "days_moderate": 365, "days_high": 365, "days_very_high": 365,
    "days_extreme": 365, "days_danger_total": 365,
    "days_caution": 365, "days_extreme_caution": 365, "days_danger": 365,
    "days_extreme_danger": 365, "days_hot_total": 365,
    "HI_mean": 120, "HI_max": 180,
    "TXX": 70, "TNN": -30,
    "Rx1day": 3000, "Rx5day": 5000,
    "frost": 360,
    "SPI3": -15, "SPI6": -15,
}
# CLIMBRA Rx1day uses a tighter bound than the observational one (original scripts: 2500 vs 3000).
THRESHOLD_OVERRIDES = {("CLIMBRA", "Rx1day"): 2500}


CAPPED_VARS = {"CDD", "CWD", "CDD_season"}     # spell lengths, bounded by the year / season length


def is_capped(var: str) -> bool:
    """Day counts and spell lengths are bounded by the period length: cap at the threshold
    (365 days; 92 = longest season for CDD_season), do not interpolate."""
    return var.startswith("days_") or var in CAPPED_VARS


def threshold_for(var: str, product: str = "OBS") -> float | None:
    return THRESHOLD_OVERRIDES.get((product, var), THRESHOLDS.get(var))


FLAG_MEANINGS = "kept spatial_outlier beyond_threshold"   # 0 / 1 / 2 in <var>_flag


def drop_outliers_2d(da: xr.DataArray, threshold: float, spatial_filter: bool = True,
                     verbose: bool = False, counts: dict | None = None, cap: bool = False) -> xr.DataArray:
    """Filter one (lat, lon) field. A leading extra dimension (e.g. `season`) is passed through the
    same code path (the median filter then also acts across it).
    `cap=True`: values beyond `threshold` are set to `threshold` (only spatial outliers are interpolated).

    If `counts` is a dict, it receives: n_valid (finite input cells), n_spatial (spatial outliers),
    n_threshold (beyond the absolute threshold, after the spatial step), n_replaced (cells whose value
    changed), max_abs_change and `flag` (int8 array: 0 kept, 1 spatial outlier, 2 beyond the threshold).
    """
    out = da
    outliers = np.zeros(da.shape, dtype=bool)
    if spatial_filter:
        local_median = median_filter(da.values, size=5)
        diff = np.abs(da.values - local_median)
        spatial_threshold = 6 * np.nanstd(diff)
        outliers = diff > spatial_threshold
        if verbose:
            print(f"  spatial threshold {spatial_threshold:.2f}: {int(outliers.sum())} outliers")
        out = da.where(~outliers, np.nan)

    if cap:
        beyond = out > threshold if threshold >= 0 else out < threshold
        clipped = out.where(~beyond, threshold).where(np.isfinite(out), -9999)
    elif threshold >= 0:
        beyond = out >= threshold
        clipped = out.where(out < threshold, -9999)
    else:
        beyond = out <= threshold
        clipped = out.where(out > threshold, -9999)

    clipped = clipped.rio.write_nodata(-9999, inplace=False)
    clipped = clipped.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    clipped = clipped.rio.write_crs("EPSG:4326", inplace=False)
    filled = clipped.rio.interpolate_na()
    result = filled.where(~np.isnan(da), np.nan)          # restore the original NaNs

    if counts is not None:
        beyond = np.asarray(beyond)
        valid = np.isfinite(da.values)
        change = np.abs(result.values - da.values)
        changed = valid & ~(change <= 1e-6 * np.maximum(np.abs(da.values), 1.0))   # NaN change counts as changed
        counts.update(
            n_valid=int(valid.sum()),
            n_spatial=int(outliers.sum()),
            n_threshold=int(beyond.sum()),
            n_replaced=int(changed.sum()),
            max_abs_change=float(np.nanmax(np.where(changed, change, np.nan))) if changed.any() else 0.0,
            flag=np.where(beyond, 2, np.where(outliers, 1, 0)).astype("int8"),
        )
    return result


def drop_outliers(ds: xr.Dataset | xr.DataArray, product: str = "OBS", spatial_filter: bool = True,
                  verbose: bool = False, stats: list | None = None, context: dict | None = None,
                  flags: dict | None = None) -> xr.Dataset:
    """Apply the filter to every variable that has a threshold, looping over `GWL` (CLIMBRA only)
    and `return_period`.  Variables with 'days' in the name are filtered twice.

    stats : list -> one row (dict) per variable / GWL / return period / pass is appended, with the keys of
            `context` (e.g. indicator, source) and the counts / proportions of the filter.
    flags : dict -> receives `<var>_flag` (int8, 0 kept / 1 spatial outlier / 2 beyond threshold) and
            `<var>_raw` (unfiltered value where the flag is > 0, NaN elsewhere) for every filtered variable.
    """
    ds = ds.to_dataset() if not isinstance(ds, xr.Dataset) else ds.copy()
    context = context or {}
    for var in list(ds.data_vars):
        thr = threshold_for(var, product)
        if thr is None:
            continue
        da = ds[var]
        gwl_slices = range(da.sizes["GWL"]) if "GWL" in da.dims else [None]
        per_gwl, flag_gwl = [], []
        for ig in gwl_slices:
            da_g = da if ig is None else da.isel(GWL=ig)
            per_rp, flag_rp = [], []
            for irp in range(da_g.sizes["return_period"]):
                field = da_g.isel(return_period=irp)
                flag = np.zeros(field.shape, dtype="int8")
                for pass_no in (1, 2) if "days" in var else (1,):
                    counts = {}
                    field = drop_outliers_2d(field, thr, spatial_filter, verbose, counts, cap=is_capped(var))
                    flag = np.maximum(flag, counts.pop("flag"))
                    if stats is not None:
                        n = max(counts["n_valid"], 1)
                        stats.append({**context, "variable": var, "threshold": thr,
                                      "GWL": "obs" if ig is None else str(da["GWL"].values[ig]),
                                      "return_period": int(da_g["return_period"].values[irp]),
                                      "pass": pass_no, "spatial_filter": spatial_filter, "capped": is_capped(var),
                                      **counts,
                                      "prop_spatial": counts["n_spatial"] / n,
                                      "prop_threshold": counts["n_threshold"] / n,
                                      "prop_replaced": counts["n_replaced"] / n})
                per_rp.append(field)
                flag_rp.append(field.copy(data=flag))
            per_gwl.append(xr.concat(per_rp, dim="return_period"))
            flag_gwl.append(xr.concat(flag_rp, dim="return_period"))
        filtered = per_gwl[0] if ig is None else xr.concat(per_gwl, dim="GWL")
        if flags is not None:
            fl = (flag_gwl[0] if ig is None else xr.concat(flag_gwl, dim="GWL")).astype("int8")
            fl = fl.transpose(*da.dims)
            fl.attrs = {"long_name": f"outlier flag of {var}", "flag_values": [0, 1, 2],
                        "flag_meanings": FLAG_MEANINGS, "threshold": thr,
                        "threshold_action": "capped at threshold" if is_capped(var) else "removed and interpolated"}
            flags[f"{var}_flag"] = fl
            flags[f"{var}_raw"] = da.where(fl > 0).assign_attrs(
                long_name=f"unfiltered {var} where flagged (NaN elsewhere)")
        ds[var] = filtered
    return ds


# --------------------------------------------------------------------------- outputs
def outlier_paths(cfg, product: str, hazard: str, indicator: str, source: str):
    """(<outliers>/<CLIMBRA|Observational>/<Hazard>/<Indicator>/<source>_outliers.nc, consolidated log CSV)."""
    from pathlib import Path
    root = Path(cfg.paths.outliers)
    sub = "CLIMBRA" if product == "CLIMBRA" else "Observational"
    return root / sub / hazard / indicator / f"{source}_outliers.nc", root / "outliers_log.csv"


def save_flags(flags: dict, path) -> None:
    """Flag + raw-value NetCDF of one model / dataset (int8 flags, float32 raw values, zlib)."""
    from pathlib import Path
    if not flags:
        return
    out = xr.Dataset(flags)
    out.attrs.update(description="Return-level cells changed by the HAZBRA outlier filter",
                     flag_meanings="0 kept, 1 spatial outlier (|x - 5x5 median| > 6 std), 2 beyond the absolute "
                                   "threshold (capped for day counts, removed + interpolated otherwise)")
    enc = {v: ({"dtype": "int8", "zlib": True, "complevel": 4} if v.endswith("_flag")
               else {"dtype": "float32", "zlib": True, "complevel": 4}) for v in out.data_vars}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    out.to_netcdf(tmp, encoding=enc)
    tmp.replace(path)


def append_log(stats: list, path) -> None:
    """Append the rows of one run to the consolidated CSV shared by all indicators and runs.

    Each row carries `run_id` (timestamp + SLURM job id) and `run_time`, so re-runs can be told apart; the
    summary keeps the latest run per product / indicator / source.  The rows are written in a single
    append (with an advisory lock where the file system supports it) so parallel jobs do not interleave.
    """
    import csv
    import datetime
    import io
    import os
    from pathlib import Path
    if not stats:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    run_id = f"{now}_{os.environ.get('SLURM_JOB_ID', 'local')}"
    cols = ["run_id", "run_time", "product", "indicator", "source", "variable", "GWL", "return_period", "pass",
            "threshold", "capped", "spatial_filter", "n_valid", "n_spatial", "n_threshold", "n_replaced",
            "prop_spatial", "prop_threshold", "prop_replaced", "max_abs_change"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    for r in stats:
        w.writerow({"run_id": run_id, "run_time": now, **r})
    with open(path, "a", newline="") as fh:
        try:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX)
        except Exception:
            pass
        if fh.tell() == 0:
            fh.write(",".join(cols) + "\n")
        fh.write(buf.getvalue())
        fh.flush()

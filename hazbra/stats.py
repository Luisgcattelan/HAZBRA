"""Extreme-value (GEV) and trend statistics applied to annual indicator series."""
from __future__ import annotations

import numpy as np
import xarray as xr
from xclim.indices.stats import fit, frequency_analysis


# --------------------------------------------------------------------------- GEV
def gev_return_levels(annual, return_periods, mode: str = "max"):
    """T-year return levels from a GEV fitted to the annual series (xclim `frequency_analysis`).

    mode="max": upper tail (levels exceeded on average once every T years);
    mode="min": lower tail (e.g. Tnn, SPI) – levels not reached more than once every T years.
    Returns the same object type as `annual` with a `return_period` dimension.
    """
    return frequency_analysis(annual, t=list(return_periods), dist="genextreme", mode=mode, freq="YS")


def gev_params(annual):
    """GEV parameters (dparams: c, loc, scale) per grid cell."""
    return fit(annual, dist="genextreme")


# --------------------------------------------------------------------------- trend
TREND_TEST = "hamed_rao"      # modified Mann-Kendall test (Hamed & Rao 1998): variance corrected for autocorrelation


def _mk(arr, test=TREND_TEST):
    import pymannkendall as pmk
    if np.isnan(arr).any():
        return np.nan, np.nan, np.nan
    if np.ptp(arr) == 0:                              # constant series (e.g. no frost days): no trend
        return 1.0, 0.0, float(arr[0])
    fn = pmk.hamed_rao_modification_test if test == "hamed_rao" else pmk.original_test
    r = fn(arr)
    return r.p, r.slope, r.intercept


def mann_kendall(data: xr.DataArray, dim: str = "time", test: str = TREND_TEST) -> xr.Dataset:
    """Trend significance and Sen's slope / intercept along `dim` (cells with any NaN -> NaN).

    test="hamed_rao" (default): modified Mann-Kendall test of Hamed & Rao (1998), whose variance of S is corrected
    for the serial correlation of the (detrended, ranked) series; test="original": classical Mann-Kendall.
    The slope and intercept (Sen 1968) are the same for both; only the p-value changes.
    """
    p, slope, intercept = xr.apply_ufunc(
        _mk, data, input_core_dims=[[dim]], output_core_dims=[[], [], []],
        vectorize=True, dask="parallelized", kwargs={"test": test},
        output_dtypes=[float, float, float],
    )
    out = xr.Dataset({"p_values": p, "slopes": slope, "intercepts": intercept})
    out.attrs["trend_test"] = ("Hamed & Rao (1998) modified Mann-Kendall test" if test == "hamed_rao"
                               else "Mann-Kendall test") + "; Sen's slope (units per year)"
    return out


def mann_kendall_multi(annual: xr.Dataset, dim: str = "time", test: str = TREND_TEST) -> xr.Dataset:
    """Trend of every variable of a multi-variable indicator (HI, FWI): `<var>_p_values`, ..."""
    parts = []
    for v in annual.data_vars:
        t = mann_kendall(annual[v], dim, test)
        parts.append(t.rename({k: f"{v}_{k}" for k in t.data_vars}))
    out = xr.merge(parts)
    out.attrs["trend_test"] = parts[0].attrs["trend_test"]
    return out


def compute_trend(trend: xr.Dataset) -> xr.Dataset:
    """Evaluate a lazy trend result with a process pool: pymannkendall is pure Python, so dask threads would run
    one cell at a time (GIL).  Workers = SLURM_CPUS_PER_TASK (or the dask default)."""
    import os
    n = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or None
    return trend.load(scheduler="processes", num_workers=n)

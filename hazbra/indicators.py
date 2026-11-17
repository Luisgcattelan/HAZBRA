"""Registry of the 16 HAZBRA climate-hazard indicators.

Every indicator is ONE `Indicator` entry describing
  * which daily variables it needs (CLIMBRA names and BR-DWGD/XAVIER names),
  * the xclim computation  `annual(daily, ctx)`  that turns daily data into the annual (or
    seasonal-annual) series — the same function is used for CLIMBRA and for the observations,
  * how it is processed (GEV return levels or not, special cases).  CLIMBRA indicators are always computed on
    the full 1850-2100 daily series and then cut to the GWL windows.

`pipeline_climbra.py` and `pipeline_obs.py` are generic drivers over this registry; they replace
the 32 near-identical scripts of the original `bashs/` folder (01.* = CLIMBRA, 02.* = observations).
The comments marked "ORIGINAL" document behaviour taken verbatim from those scripts that a reviewer
may want to question; they are collected in docs/CONSOLIDATION_NOTES.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr
import xclim
from xclim.core.calendar import percentile_doy

from .io import reindex_full_daily, seasonal_to_annual_time, to_datetime64_time, to_noleap_daily

YS = "YS"
ALL_OBS = ("XAVIER", "CHIRPS", "MERGE")
XAVIER_ONLY = ("XAVIER",)

# BR-DWGD (XAVIER) variable name -> CLIMBRA name, so one `annual()` serves both products.
OBS_TO_STD = {"Tmax": "tasmax", "Tmin": "tasmin", "RH": "hur", "u2": "sfcWind", "pr": "pr"}


@dataclass
class Ctx:
    """Run context passed to every `annual()` function."""
    calib: slice = slice("1981", "2010")     # calibration period of percentile-based indicators
    mask: bool = True                        # restore NaN outside the domain (see CDD)
    product: str = "CLIMBRA"                 # "CLIMBRA" | "OBS"


def _nan_mask(out, ref: xr.DataArray):
    """NaN wherever the input variable is NaN (ocean / outside domain). Run-length and counting
    functions would otherwise return 0 there."""
    return out.where(~np.isnan(ref.isel(time=10)))


# =========================================================================== annual() functions
def _rx1day(v, ctx):
    return v["pr"].resample(time=YS).max()


def _rx5day(v, ctx):
    """Annual maximum of the 5-day precipitation total (mm), over consecutive days of the series.

    Same computation as xclim `max_n_day_precipitation_amount` (rolling sum with skipna=False, annual maximum of the
    complete windows), written out because the CLIMBRA series have their non-existent model days removed
    (`_prep_model_days`), so the time step is not regular.  The xclim *indicator* RX5day used before discarded
    every year with a single missing day, i.e. every leap year of the 365-day models and every year of the
    360-day models (HadGEM3-GC31-LL, KACE, UKESM1-0-LL).
    """
    pr = v["pr"]
    out = pr.rolling(time=5).sum(skipna=False).resample(time=YS).max(dim="time")
    return out.assign_attrs(units="mm", long_name="Maximum 5-day precipitation amount")


def _txx(v, ctx):
    return v["tasmax"].resample(time=YS).max()


def _tnn(v, ctx):
    return v["tasmin"].resample(time=YS).min()


def _cdd(v, ctx):
    with xclim.set_options(data_validation="log"):
        out = xclim.indices.maximum_consecutive_dry_days(v["pr"], freq=YS)
    # ORIGINAL: only the CLIMBRA script masks the ocean for CDD (observations are not masked).
    return _nan_mask(out, v["pr"]) if ctx.mask else out


def _cwd(v, ctx):
    with xclim.set_options(data_validation="log"):
        return xclim.indices.maximum_consecutive_wet_days(v["pr"], freq=YS)


def _cdd_season(v, ctx):
    """Longest dry spell per meteorological season (DJF, MAM, JJA, SON), dims (season, time)."""
    with xclim.set_options(data_validation="log"):
        q = xclim.indices.maximum_consecutive_dry_days(v["pr"], thresh="1 mm/day", freq="QS-DEC").load()
    out = q.groupby(["time.season", "time.year"]).mean(dim="time")
    return seasonal_to_annual_time(out)


def _frost(v, ctx):
    # Threshold 6 degC (convention used in Brazil): longest spell of Tmin < 6 degC.
    return xclim.indices.maximum_consecutive_frost_days(v["tasmin"], freq=YS, thresh="6.0 degC")


def _prcptot(v, ctx):
    with xclim.set_options(data_validation="log"):
        return xclim.indices.prcptot(v["pr"], freq=YS)


def _pr_season(v, ctx):
    """Mean seasonal precipitation (mm/month ~ daily mean x 30), dims (season, time)."""
    out = v["pr"].groupby(["time.year", "time.season"]).mean(dim="time") * 30
    out = seasonal_to_annual_time(out)
    out.attrs["units"] = "mm/month"
    out.attrs["long_name"] = "Seasonal mean precipitation"
    return out


def _spi_monthly(window: int):
    def f(v, ctx):
        # No calibration period is passed: xclim fits the gamma distribution on the WHOLE input series
        # (1850-2100 for CLIMBRA, the observational record for observations).
        with xclim.set_options(data_validation="log"):
            return xclim.indices.standardized_precipitation_index(v["pr"], window=window, freq="MS")
    return f


def _percentile_doy(da, per, calib):
    cal = da.sel(time=calib)
    p = percentile_doy(cal, per=per, window=5).sel(percentiles=per).drop_vars("percentiles")
    return cal, p


def _wsdi(v, ctx):
    tas = v["tasmax"]
    cal, per90 = _percentile_doy(tas, 90, ctx.calib)
    out = xclim.indices.warm_spell_duration_index(tas, per90)
    return _nan_mask(out, cal)


def _dry_hot(v, ctx):
    tas, pr = v["tasmax"], v["pr"]
    tas_cal, tas_per = _percentile_doy(tas, 75, ctx.calib)
    pr_cal = pr.sel(time=ctx.calib)
    pr_cal = pr_cal.where(pr_cal >= 1)                       # wet days only
    pr_per = percentile_doy(pr_cal, per=25, window=5).sel(percentiles=25).drop_vars("percentiles")
    with xclim.set_options(data_validation="log"):
        out = xclim.indices.warm_and_dry_days(tas=tas, pr=pr, tas_per=tas_per, pr_per=pr_per, freq=YS)
    return _nan_mask(out, tas_cal)


def _count_days_hot(hi: xr.DataArray) -> xr.Dataset:
    """Days per year in each NWS heat-index class (deg C)."""
    cat = {
        "days_caution": (hi >= 27) & (hi < 32),
        "days_extreme_caution": (hi >= 32) & (hi < 41),
        "days_danger": (hi >= 41) & (hi < 54),
        "days_extreme_danger": hi >= 54,
    }
    ds = xr.Dataset({k: m.resample(time=YS).sum("time") for k, m in cat.items()})
    total_hot = cat["days_caution"] | cat["days_extreme_caution"] | cat["days_danger"] | cat["days_extreme_danger"]
    ds["days_hot_total"] = total_hot.resample(time=YS).sum("time")
    ds["days_caution"].attrs.update(long_name="HI 27–32 °C", description="Caution: fatigue possible; cramps with prolonged activity.")
    ds["days_extreme_caution"].attrs.update(long_name="HI 32–41 °C", description="Extreme caution: cramps/exhaustion possible; heat stroke with continued activity.")
    ds["days_danger"].attrs.update(long_name="HI 41–54 °C", description="Danger: cramps/exhaustion likely; heat stroke probable with continued activity.")
    ds["days_extreme_danger"].attrs.update(long_name="HI >54 °C", description="Extreme danger: heat stroke imminent.")
    ds["days_hot_total"].attrs.update(long_name="Total hot days (HI ≥ 27 °C)")
    return ds


def _count_days_fwi(fwi: xr.DataArray) -> xr.Dataset:
    """Days per year in each Fire Weather Index class."""
    cat_low = (fwi >= 0) & (fwi < 5)
    cat_moderate = (fwi >= 5) & (fwi < 12)
    cat_high = (fwi >= 12) & (fwi < 22)
    cat_very_high = (fwi >= 22) & (fwi < 50)
    cat_extreme = fwi >= 50
    ds = xr.Dataset({
        "days_low": cat_low.resample(time=YS).sum("time"),
        "days_moderate": cat_moderate.resample(time=YS).sum("time"),
        "days_high": cat_high.resample(time=YS).sum("time"),
        "days_very_high": cat_very_high.resample(time=YS).sum("time"),
        "days_extreme": cat_extreme.resample(time=YS).sum("time"),
    })
    ds["days_danger_total"] = (cat_moderate | cat_high | cat_very_high | cat_extreme).resample(time=YS).sum("time")
    # ORIGINAL labels say 5–11 / 12–21 / 22–50 whereas the code uses [5,12), [12,22), [22,50), >=50.
    ds["days_low"].attrs.update(long_name="FWI 0–5", description="Low fire danger")
    ds["days_moderate"].attrs.update(long_name="FWI 5–11", description="Moderate fire danger")
    ds["days_high"].attrs.update(long_name="FWI 12–21", description="High fire danger")
    ds["days_very_high"].attrs.update(long_name="FWI 22–50", description="Very high fire danger")
    ds["days_extreme"].attrs.update(long_name="FWI >50", description="Extreme fire danger")
    ds["days_danger_total"].attrs.update(long_name="Total fire danger days (FWI ≥ 5)")
    return ds


def _annual_stats(daily: xr.DataArray, prefix: str, counts: xr.Dataset) -> xr.Dataset:
    """<prefix>_mean, <prefix>_max, class counts, NaN outside the domain."""
    mean = daily.resample(time=YS).mean(dim="time", skipna=True).rename(f"{prefix}_mean")
    mx = daily.resample(time=YS).max(dim="time", skipna=True).rename(f"{prefix}_max")
    out = xr.merge([mean, mx, counts])
    return _nan_mask(out, daily)


def _hi(v, ctx):
    hi = xclim.indices.heat_index(v["tasmax"], v["hur"]).load()
    return _annual_stats(hi, "HI", _count_days_hot(hi))


def _fwi(v, ctx):
    fwi = xclim.indices.cffwis_indices(v["tasmax"], v["pr"], v["sfcWind"], v["hur"], v["tasmax"]["lat"])[-1]
    return _annual_stats(fwi, "FWI", _count_days_fwi(fwi))


# =========================================================================== CLIMBRA preprocessing
def _prep_noleap_interp(*names, limit=1):
    """noleap calendar, continuous daily axis, interpolate gaps of <= `limit` days."""
    def f(daily):
        out = dict(daily)
        for n in names:
            out[n] = to_noleap_daily(daily[n]).interpolate_na(dim="time", method="linear", limit=limit)
        return out
    return f


def _prep_fwi(daily):
    """Fill gaps (<= 10 days), cap RH at 100 %, floor wind at 0.01 m/s before the FWI system."""
    fill = dict(dim="time", method="linear", limit=10)
    out = {k: daily[k].interpolate_na(**fill) for k in ("tasmax", "hur", "sfcWind", "pr")}
    out["hur"] = out["hur"].where(out["hur"] <= 100, 100)
    out["sfcWind"] = out["sfcWind"].where(out["sfcWind"] > 0, 0.01)
    return out


def _prep_full_daily(daily):
    """datetime64 axis + complete daily axis (missing days -> NaN); CLIMBRA files miss single days."""
    return {k: reindex_full_daily(v) for k, v in daily.items()}


def _prep_model_days(daily):
    """datetime64 axis and only the days that exist in the model calendar.

    CLIMBRA files use a Gregorian time axis, but days absent from the model calendar are stored as missing values in
    every grid cell: 29 February for the 365-day models, about 7 days per year for the 360-day models (HadGEM3-GC31-LL,
    KACE, UKESM1-0-LL).  Those days are dropped (the same days for every variable), so that spells, n-day windows and
    day counts run over consecutive model days.  Left in, they cut every dry/wet spell and drop whole years from
    xclim indicators with a missing-value check.
    (2014 is missing in every model: the historical files end in 2013, the scenario files start in 2015.)
    """
    out = {k: to_datetime64_time(v) for k, v in daily.items()}
    exists = None
    for v in out.values():
        e = v.notnull().any(dim=[d for d in v.dims if d != "time"]).compute()
        exists = e if exists is None else (exists | e)
    keep = np.flatnonzero(exists.values)
    return {k: v.isel(time=keep) for k, v in out.items()}


def _prep_model_days_fwi(daily):
    """Model days only, then the FWI input cleaning (gap filling, RH cap, wind floor)."""
    return _prep_fwi(_prep_model_days(daily))


# SPI windows -> annual series (applied to one GWL window of the monthly SPI series)
def _spi6_reduce(spi_window):
    return spi_window.resample(time="YS").min()


def _spi3_reduce(spi_window):
    """Seasonal minimum: quarterly (QS-DEC) minimum of monthly SPI3, dims (season, time)."""
    out = spi_window.resample(time="QS-DEC").min(dim="time").groupby(["time.year", "time.season"]).min()
    out = out.rename({"year": "time"})
    out["time"] = pd.to_datetime(out["time"].values, format="%Y")
    return out


# =========================================================================== registry
@dataclass(frozen=True)
class Indicator:
    name: str
    hazard: str
    climbra_vars: tuple
    obs_vars: tuple                      # BR-DWGD names; precipitation-only indicators also run on CHIRPS/MERGE
    obs_datasets: tuple
    annual: Callable                     # (daily: dict[str, DataArray], Ctx) -> annual DataArray / Dataset
    gev: str | None = None               # "max" | "min" | None (no return-level / params products)
    # ---- CLIMBRA: the indicator is always computed on the full 1850-2100 series, then cut to the GWL windows
    prepare_climbra: Callable | None = _prep_full_daily
    climbra_load_daily: bool = True      # read the daily series into memory (False: lazy dask arrays)
    reduce_window: Callable | None = None      # sub-annual series (SPI) -> annual, applied per GWL window
    save_timeseries_climbra: bool = True
    mask_climbra: bool = True
    # ---- observations
    prepare_obs: Callable | None = None
    seasonal: bool = False               # series period starts one year earlier (DJF), ts trimmed afterwards
    obs_reindex_daily: tuple = ()        # datasets whose daily axis is completed before the computation
    obs_monthly_series: bool = False     # also store the monthly series (SPI3)
    notes: str = ""

    @property
    def multi_variable(self) -> bool:
        return self.name in ("HI", "FWI")

    @property
    def product_var(self) -> str:
        return {"HI": "HI_max", "FWI": "FWI_max"}.get(self.name, self.name)


INDICATORS: dict[str, Indicator] = {i.name: i for i in [
    # ---------------- Rainfall
    Indicator("Rx1day", "Rainfall", ("pr",), ("pr",), ALL_OBS, _rx1day, gev="max", climbra_load_daily=False),
    Indicator("Rx5day", "Rainfall", ("pr",), ("pr",), ALL_OBS, _rx5day, gev="max", climbra_load_daily=False,
              prepare_climbra=_prep_model_days,
              obs_reindex_daily=("MERGE",)),
    Indicator("CWD", "Rainfall", ("pr",), ("pr",), ALL_OBS, _cwd, gev="max", climbra_load_daily=False,
              prepare_climbra=_prep_model_days, mask_climbra=False, notes="no ocean mask for CWD (CDD has one)."),
    Indicator("prcptot", "Rainfall", ("pr",), ("pr",), ALL_OBS, _prcptot,
              prepare_climbra=_prep_noleap_interp("pr")),
    Indicator("pr_season", "Rainfall", ("pr",), ("pr",), ALL_OBS, _pr_season,
              prepare_climbra=_prep_noleap_interp("pr"), seasonal=True),
    # ---------------- Drought
    Indicator("CDD", "Drought", ("pr",), ("pr",), ALL_OBS, _cdd, gev="max", climbra_load_daily=False, mask_climbra=True,
              prepare_climbra=_prep_model_days),
    Indicator("CDD_season", "Drought", ("pr",), ("pr",), ALL_OBS, _cdd_season, gev="max", climbra_load_daily=False,
              prepare_climbra=_prep_model_days, mask_climbra=False, seasonal=True),
    Indicator("SPI3", "Drought", ("pr",), ("pr",), ALL_OBS, _spi_monthly(3), gev="min",
              prepare_climbra=_prep_noleap_interp("pr"), reduce_window=_spi3_reduce,
              save_timeseries_climbra=False, seasonal=True, obs_monthly_series=True),
    Indicator("SPI6", "Drought", ("pr",), ("pr",), ALL_OBS, _spi_monthly(6), gev="min",
              prepare_climbra=_prep_noleap_interp("pr"), reduce_window=_spi6_reduce,
              save_timeseries_climbra=False),
    # ---------------- Heat
    Indicator("TXX", "Heat", ("tasmax",), ("Tmax",), XAVIER_ONLY, _txx, gev="max", climbra_load_daily=False),
    Indicator("TNN", "Heat", ("tasmin",), ("Tmin",), XAVIER_ONLY, _tnn, gev="min", climbra_load_daily=False),
    Indicator("Frost", "Heat", ("tasmin",), ("Tmin",), XAVIER_ONLY, _frost, prepare_climbra=_prep_model_days,
              notes="threshold 6 degC (Brazilian convention)."),
    Indicator("WSDI", "Heat", ("tasmax",), ("Tmax",), XAVIER_ONLY, _wsdi,
              prepare_climbra=_prep_noleap_interp("tasmax")),
    # ---------------- Compound
    Indicator("HI", "Compound", ("tasmax", "hur"), ("Tmax", "RH"), XAVIER_ONLY, _hi, gev="max",
              prepare_climbra=_prep_model_days),
    Indicator("FWI", "Compound", ("tasmax", "hur", "pr", "sfcWind"), ("Tmax", "RH", "pr", "u2"), XAVIER_ONLY, _fwi,
              gev="max", prepare_climbra=_prep_model_days_fwi, prepare_obs=_prep_fwi,
              notes="observations use 2-m wind (u2) as wind input."),
    Indicator("Dry_Hot", "Compound", ("tasmax", "pr"), ("Tmax", "pr"), XAVIER_ONLY, _dry_hot,
              prepare_climbra=_prep_noleap_interp("tasmax", "pr")),
]}


def get(name: str) -> Indicator:
    try:
        return INDICATORS[name]
    except KeyError:
        raise SystemExit(f"Unknown indicator '{name}'. Available: {', '.join(INDICATORS)}")


# Observational periods (inclusive). `series` is the period read from the daily files.
OBS_PERIODS = {
    "XAVIER": dict(series=("1981", "2023"), clim=("1981", "2010"), trend=("1981", "2023")),
    "CHIRPS": dict(series=("1981", "2024"), clim=("1981", "2010"), trend=("1981", "2023")),
    "MERGE":  dict(series=("2001", "2024"), clim=("2001", "2020"), trend=("2001", "2023")),
}

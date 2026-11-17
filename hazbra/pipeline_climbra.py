"""CLIMBRA driver: daily bias-corrected CMIP6 -> per-model GWL products -> ensemble statistics.

Per model and indicator it writes (hazard/indicator/individual_models/):
    <MODEL>_climatology.nc      mean of the annual indicator in each GWL window   (dims: GWL, [season], lat, lon)
    <MODEL>_return_period.nc    GEV return levels, 5-100 yr                       (only indicators with `gev`)
    <MODEL>_params.nc           GEV parameters (c, loc, scale)                    (only indicators with `gev`)
    <MODEL>_timeseries.nc       annual series 1850-2100, SSP2-4.5 and SSP5-8.5    (dims: scenario, time, lat, lon)
and, over all models present on disk (hazard/indicator/ensemble/):
    ensemble_{climatology,return_period,timeseries}.nc    mean / median / p10 / p90 on an `ensemble` dim
Outlier filter of the return levels: flagged pixels in <outliers>/CLIMBRA/<Hazard>/<Indicator>/<MODEL>_outliers.nc
and one row per variable / GWL / return period appended to <outliers>/outliers_log.csv
(statistics: 02_postprocess/05_outlier_summary.py).

The indicator is computed on the full daily series (historical + scenario) and then cut to each GWL window:
"present" = 1981-2010; +1.5/+2/+3 degC = 20-year window of GWL_CLIMBRA.csv padded by 5 years.
All indicator-specific science lives in `indicators.py`; this module only orchestrates.
"""
from __future__ import annotations

import time
from pathlib import Path

import xarray as xr

from .constants import GWLS
from .ensemble import build_ensemble, concat_models
from .gwl import gwl_label, gwl_window, load_gwl_table
from .indicators import Ctx, Indicator
from .io import fix_units, open_climbra, save_nc
from .outliers import append_log, drop_outliers, outlier_paths, save_flags
from .stats import gev_params, gev_return_levels

KINDS = ("climatology", "timeseries", "return_period")


def out_dirs(cfg, ind: Indicator) -> tuple[Path, Path]:
    base = Path(cfg.paths.netcdf_climbra) / ind.hazard / ind.name
    return base / "individual_models", base / "ensemble"


def _name(obj, name: str):
    """Single-variable indicators are stored under the indicator's name."""
    return obj.rename(name) if isinstance(obj, xr.DataArray) else obj


def expected_model_files(ind: Indicator, model: str, out_indiv: Path) -> list[Path]:
    kinds = ["climatology"] + (["return_period", "params"] if ind.gev else []) + (
        ["timeseries"] if ind.save_timeseries_climbra else [])
    return [out_indiv / f"{model}_{k}.nc" for k in kinds]


# ---------------------------------------------------------------------------- GWL products
def _gwl_windows(cfg, ind, model, table, series):
    """Cut the full-series indicator to each GWL window -> (gwl, annual values, climatology)."""
    for gwl in GWLS:
        y0, y1 = gwl_window(table, model, gwl, cfg.calc.gwl_pad_years, tuple(cfg.calc.present_period))
        print(f"  - GWL={gwl}: window {y0}..{y1}")
        s_win = series.sel(time=slice(y0, y1))
        annual = ind.reduce_window(s_win) if ind.reduce_window else s_win
        if annual.time.size == 0:
            print(f"    ! no data in {y0}-{y1} for {model}; skipping this GWL")
            continue
        yield gwl, annual, annual.mean("time")


def _gwl_products(cfg, ind, annuals, stats, context, flags):
    clim_l, rp_l, par_l = [], [], []
    for gwl, annual, clim in annuals:
        lab = gwl_label(gwl)
        clim_l.append(_name(clim, ind.name).expand_dims(GWL=[lab]))
        if ind.gev:
            rp = _name(gev_return_levels(annual, cfg.calc.return_periods, ind.gev), ind.name).expand_dims(GWL=[lab])
            par = _name(gev_params(annual), ind.name).expand_dims(GWL=[lab])
            fl = {}
            rp_l.append(drop_outliers(rp, product="CLIMBRA", stats=stats, context=context, flags=fl))
            for k, v in fl.items():
                flags.setdefault(k, []).append(v)
            par_l.append(par)

    def cat(parts):
        out = xr.concat(parts, dim="GWL").astype("float32")
        out["GWL"].attrs["long_name"] = "Global Warming Level"
        return out

    return cat(clim_l), (cat(rp_l) if rp_l else None), (cat(par_l) if par_l else None)


# ---------------------------------------------------------------------------- one model
def _series(cfg, ind, model, scenario, ctx):
    """Indicator over the full historical + scenario period, or None if a variable is missing."""
    ds = open_climbra(cfg, model, ind.climbra_vars, scenario)
    if ds is None:
        return None
    ds = fix_units(ds)
    if ind.climbra_load_daily:
        ds = ds.load()
    daily = {v: ds[v] for v in ind.climbra_vars}
    if ind.prepare_climbra:
        daily = ind.prepare_climbra(daily)
    return ind.annual(daily, ctx).load()


def process_model(cfg, ind: Indicator, model: str, table, overwrite: bool = False) -> bool:
    out_indiv, _ = out_dirs(cfg, ind)
    out_indiv.mkdir(parents=True, exist_ok=True)
    if not overwrite and all(p.exists() for p in expected_model_files(ind, model, out_indiv)):
        print(f"=== {ind.name} / {model}: outputs exist, skipping (use --overwrite)")
        return False

    t0 = time.time()
    print(f"\n=== {ind.name} / {model}")
    ctx = Ctx(calib=slice(*[str(y) for y in cfg.calc.present_period]), mask=ind.mask_climbra, product="CLIMBRA")

    series585 = _series(cfg, ind, model, "ssp585", ctx)          # historical + SSP5-8.5 (defines the GWLs)
    if series585 is None:
        print(f"  ! {model} lacks one of {ind.climbra_vars}; skipping")
        return False

    stats, flags = [], {}
    clim, rp, params = _gwl_products(cfg, ind, _gwl_windows(cfg, ind, model, table, series585), stats,
                                     {"product": "CLIMBRA", "indicator": ind.name, "source": model}, flags)
    save_nc(clim, out_indiv / f"{model}_climatology.nc", ind.name)
    if rp is not None:
        save_nc(rp, out_indiv / f"{model}_return_period.nc", ind.name)
        save_nc(params, out_indiv / f"{model}_params.nc", ind.name)
        nc_path, log_path = outlier_paths(cfg, "CLIMBRA", ind.hazard, ind.name, model)
        save_flags({k: xr.concat(v, dim="GWL") for k, v in flags.items()}, nc_path)
        append_log(stats, log_path)

    if ind.save_timeseries_climbra:
        ts = [_name(series585, ind.name).astype("float32").expand_dims(scenario=["ssp585"])]
        series245 = _series(cfg, ind, model, "ssp245", ctx)
        if series245 is not None:
            ts.insert(0, _name(series245, ind.name).astype("float32").expand_dims(scenario=["ssp245"]))
        save_nc(xr.concat(ts, dim="scenario"), out_indiv / f"{model}_timeseries.nc", ind.name)

    print(f"  done in {(time.time() - t0) / 60:.1f} min")
    return True


# ---------------------------------------------------------------------------- ensembles
def build_ensembles(cfg, ind: Indicator) -> None:
    """Ensemble statistics over every model file found in individual_models/."""
    out_indiv, out_ens = out_dirs(cfg, ind)
    out_ens.mkdir(parents=True, exist_ok=True)
    qlo, qhi = cfg.calc.ensemble_quantiles
    for kind in KINDS:
        stacked = concat_models(out_indiv, f"_{kind}.nc")
        if stacked is None:
            continue
        print(f"=== {ind.name}: ensemble {kind} over {stacked.sizes['model']} models")
        save_nc(build_ensemble(stacked, "model", qlo, qhi), out_ens / f"ensemble_{kind}.nc")


def run(cfg, ind: Indicator, models=None, overwrite: bool = False, ensemble: bool = True,
        per_model: bool = True) -> None:
    table = load_gwl_table(cfg)
    if models is None:
        from .constants import CLIMBRA_MODELS
        models = list(cfg.test.models) if cfg.test.enabled else CLIMBRA_MODELS
    if per_model:
        for m in models:
            process_model(cfg, ind, m, table, overwrite=overwrite or cfg.calc.overwrite)
    if ensemble:
        build_ensembles(cfg, ind)

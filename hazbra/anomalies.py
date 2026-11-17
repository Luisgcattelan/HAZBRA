"""Post-processing of the CLIMBRA products: anomalies, model agreement and delta-change projections.

Step 1  `build_anomaly_ensembles`   (per indicator)
    <netcdf_climbra>/<Hazard>/<Indicator>/individual_models/<MODEL>_{climatology,return_period}.nc
      -> for every variable and model:   anomaly   = GWL - present
                                         anomaly % = (GWL - present) / present * 100   (NaN where present == 0)
      -> ensemble statistics (mean, median, p10, p90) of both
      -> model agreement: number of models with a positive anomaly where >= 66 % of the models agree on the sign
    written to  .../ensemble/ensemble_anomaly_{climatology,return_period}.nc
    variables: <var>_anom, <var>_anom_perc, <var>_model_agreement

Step 2  `build_delta_from_obs_and_anom`   (per indicator, per observational dataset)
    delta-change projection = observed climatology / return level (regridded to the CLIMBRA grid) + ensemble anomaly,
    clipped at the physical minimum of the indicator (and at 365 days – 92 for CDD_season – for day counts
    and spell lengths); the `present` slice is the
    observation itself.
    written to  .../ensemble/ensemble_delta_<DATASET>_{climatology,return_period}.nc
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from .constants import FUTURE_GWLS, INDICATOR_HAZARD
from .ensemble import agreement_mask, build_ensemble, concat_models
from .io import save_nc
from .outliers import is_capped, threshold_for

# Physical lower bound applied to delta-change values (None: unbounded, e.g. temperatures and SPI).
MIN_VALUE = {
    "Rx1day": 0, "Rx5day": 0, "TXX": None, "TNN": None, "CDD": 0, "CDD_season": 0, "CWD": 0,
    "prcptot": 0, "pr_season": 0, "FWI": 0, "HI": 0, "SPI3": None, "Frost": 0, "WSDI": 0,
    "Dry_Hot": 0, "SPI6": None,
}


def _dirs(cfg, indicator: str) -> tuple[Path, Path]:
    base = Path(cfg.paths.netcdf_climbra) / INDICATOR_HAZARD[indicator] / indicator
    return base / "individual_models", base / "ensemble"


# ===================================================================== step 1: anomalies
def build_anomaly_ensembles(cfg, indicator: str, gwls=FUTURE_GWLS, present_label: str = "present",
                            thr_agree: float = 0.66) -> None:
    out_indiv, out_ens = _dirs(cfg, indicator)
    out_ens.mkdir(parents=True, exist_ok=True)
    qlo, qhi = cfg.calc.ensemble_quantiles

    def process(suffix: str, out_name: str):
        ds_all = concat_models(out_indiv, suffix)
        if ds_all is None:
            print(f"[WARN] {indicator}: no files with suffix '{suffix}' (indicator has no such product)")
            return
        ds_future = ds_all.sel(GWL=list(gwls))
        ds_present = ds_all.sel(GWL=present_label).drop_vars("GWL", errors="ignore")

        ds_anom = ds_future - ds_present
        ds_anom_perc = xr.where(ds_present != 0, (ds_future - ds_present) / ds_present * 100.0, np.nan)

        ds_out = xr.Dataset()
        for v in ds_anom.data_vars:
            ds_out[f"{v}_anom"] = ds_anom[v]
            ds_out[f"{v}_anom_perc"] = ds_anom_perc[v]

        ds_ens = build_ensemble(ds_out, "model", qlo, qhi)
        for v in ds_anom.data_vars:
            ds_ens[f"{v}_model_agreement"] = agreement_mask(ds_out[f"{v}_anom"], thr=thr_agree)

        save_nc(ds_ens, out_ens / out_name)
        print(f"[OK] {out_ens / out_name}")

    process("_return_period.nc", "ensemble_anomaly_return_period.nc")
    process("_climatology.nc", "ensemble_anomaly_climatology.nc")


# ===================================================================== step 2: delta change
def _reference_grids(cfg) -> dict[str, xr.DataArray]:
    """First time step of the precipitation grid of each source (defines the land mask of that grid)."""
    xavier_pr = sorted((Path(cfg.paths.raw_xavier) / "pr").glob("*.nc"))[0]
    return {
        "CHIRPS": xr.open_dataset(cfg.paths.chirps_daily).isel(time=0).pr.load(),
        "XAVIER": xr.open_dataset(xavier_pr).isel(time=0).pr.load(),
        "MODEL": xr.open_dataset(Path(cfg.paths.raw_climbra) / "pr" / "ACCESS-CM2-pr-hist.nc").isel(time=0).pr.load(),
    }


def build_delta_from_obs_and_anom(cfg, indicator: str, grids: dict | None = None,
                                  skip_existing: bool = False) -> None:
    hazard = INDICATOR_HAZARD[indicator]
    minn = MIN_VALUE[indicator]
    obs_dir = Path(cfg.paths.netcdf_obs) / hazard / indicator
    _, model_dir = _dirs(cfg, indicator)
    if not obs_dir.is_dir():
        print(f"[WARN] {indicator}: no observational folder {obs_dir}")
        return
    grids = grids or _reference_grids(cfg)
    grid_model = grids["MODEL"]

    obs_datasets = sorted({f.split("_")[0] for f in os.listdir(obs_dir) if f.endswith(".nc")})
    for obs_dataset in obs_datasets:
        for suffix in ("return_period.nc", "climatology.nc"):
            file_obs = obs_dir / f"{obs_dataset}_{suffix}"
            file_model = model_dir / f"ensemble_anomaly_{suffix}"
            if not (file_obs.exists() and file_model.exists()):
                continue                              # e.g. indicators without return-period products
            out_path = model_dir / f"ensemble_delta_{obs_dataset}_{suffix}"
            if out_path.exists():
                if skip_existing:
                    print(f"Skipping existing file: {out_path}")
                    continue
                out_path.unlink()

            print(f"Processing {indicator}: {obs_dataset} / {suffix}")
            ds_obs = xr.open_dataset(file_obs)
            if obs_dataset in ("CHIRPS", "XAVIER"):        # keep only land cells of the source grid
                grid = grids[obs_dataset]
                if "latitude" in grid.coords and "longitude" in grid.coords:
                    grid = grid.rename({"latitude": "lat", "longitude": "lon"})
                ds_obs = ds_obs.where(~np.isnan(grid))
            ds_model = xr.open_dataset(file_model).where(~np.isnan(grid_model))

            ds_obs_interp = ds_obs.interp(lat=ds_model.lat, lon=ds_model.lon)      # observations -> CLIMBRA grid
            varss = [v for v in ds_obs_interp.data_vars if f"{v}_anom" in ds_model.data_vars]
            if not varss:
                print(f"No matching variables for {obs_dataset} / {suffix}")
                continue

            delta_vars = {}
            for var in varss:
                obs_da = ds_obs_interp[var]
                anom_da = ds_model[f"{var}_anom"]

                def fill(da):                              # fill NaN holes by interpolation, mask to the model land grid
                    da = da.rio.write_nodata(np.nan, inplace=False)
                    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
                    da = da.rio.write_crs("EPSG:4326", inplace=False)
                    return da.rio.interpolate_na().where(~np.isnan(grid_model))

                if "season" in obs_da.coords:              # seasonal indicators: season by season
                    obs_da = xr.concat([fill(obs_da.sel(season=s)) for s in obs_da["season"].values], dim="season")
                else:
                    obs_da = fill(obs_da)

                delta = obs_da + anom_da
                if minn is not None:
                    delta = delta.where(delta >= minn, minn)
                if is_capped(var):                         # day counts cannot exceed the year / season length
                    delta = delta.clip(max=threshold_for(var))
                delta = delta.where(~np.isnan(anom_da))

                future = delta["GWL"].values
                delta_full = delta.reindex(GWL=np.concatenate([["present"], future]))
                if "spatial_ref" in obs_da.coords:
                    obs_da = obs_da.drop_vars("spatial_ref")
                if "spatial_ref" in delta_full.coords:
                    delta_full = delta_full.drop_vars("spatial_ref")
                delta_full.loc[dict(GWL="present")] = obs_da.where(~np.isnan(grid_model))
                delta_vars[var] = delta_full

            ds_delta = xr.Dataset(delta_vars, coords=delta_full.coords)
            ds_delta["GWL"] = ["present"] + list(future)
            ds_delta.to_netcdf(out_path)
            print(f"Saved {out_path}")

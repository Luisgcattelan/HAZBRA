"""Multi-model ensemble statistics and model agreement."""
from __future__ import annotations

import numpy as np
import xarray as xr

ENSEMBLE_STATS = ("mean", "median", "p10", "p90")


def build_ensemble(ds: xr.Dataset, model_dim: str = "model", qlo: float = 0.10, qhi: float = 0.90) -> xr.Dataset:
    """For every variable: mean / median / p10 / p90 across `model_dim`, packed on a new `ensemble` dim."""
    names = ["mean", "median", f"p{int(qlo * 100)}", f"p{int(qhi * 100)}"]
    out = {}
    for v in ds.data_vars:
        da = ds[v]
        mean = da.mean(dim=model_dim, skipna=True)
        median = da.median(dim=model_dim, skipna=True)
        q_lo = da.quantile(qlo, dim=model_dim, skipna=True)
        q_hi = da.quantile(qhi, dim=model_dim, skipna=True)
        q_lo = q_lo.drop_vars("quantile", errors="ignore")
        q_hi = q_hi.drop_vars("quantile", errors="ignore")
        da_ens = xr.concat([mean, median, q_lo, q_hi], dim="ensemble", coords="minimal")
        out[v] = da_ens.assign_coords(ensemble=names).astype("float32")
    return xr.Dataset(out)


def agreement_mask(anom: xr.DataArray, thr: float = 0.66) -> xr.DataArray:
    """Number of models with a positive anomaly, kept only where >= thr of the models agree on the sign.

    NaN where fewer than `thr` of the models agree (or the cell is outside the domain). Used for the
    hatching 'fewer than 66% of models agree on the sign of change'.
    """
    if "model" not in anom.dims:
        raise ValueError("Input DataArray must have a 'model' dimension.")
    n_models = anom.sizes["model"]
    n_pos = (anom > 0).sum("model")
    mask = (n_pos >= thr * n_models) | (n_pos <= (1 - thr) * n_models)
    out = n_pos.where(mask)
    out = out.where(~np.isnan(anom.isel(model=0)))
    out.attrs.update({
        "long_name": "Model agreement count",
        "description": f"Grid cells where >={thr:.0%} of models agree on anomaly sign.",
    })
    return out


def concat_models(folder, suffix: str, chunks="auto") -> xr.Dataset | None:
    """Open every `<model>_<suffix>` file of an `individual_models` folder and stack on a `model` dim."""
    import os
    import numpy as np
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(suffix))
    if not files:
        return None
    models = [os.path.basename(f)[: -len(suffix)].rstrip("_") for f in files]   # '<MODEL>_<kind>.nc' -> MODEL
    stacked = xr.concat([xr.open_dataset(p, chunks=chunks) for p in files], dim="model")
    return stacked.assign_coords(model=np.asarray(models, dtype=str))

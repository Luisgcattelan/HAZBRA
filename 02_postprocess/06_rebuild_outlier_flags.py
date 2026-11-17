#!/usr/bin/env python
"""Rebuild the outlier-filter outputs (flag NetCDFs + rows of outliers_log.csv) for return levels that were
computed before the calculation wrote them – without recomputing any indicator.

The unfiltered return levels are exactly the GEV quantiles of the stored parameters (xclim `frequency_analysis`
= `fit` + `parametric_quantile` with q = 1 - 1/T for maxima, 1/T for minima).  They are recomputed from
`<source>_params.nc`, passed through the same outlier filter as in the calculation, and the flags / counts are
written as the calculation would have written them.  As a consistency check the filtered result is compared with
the stored `<source>_return_period.nc` (printed; small differences come from the float32 storage of the parameters).

    python 02_postprocess/06_rebuild_outlier_flags.py                      # every indicator with return levels
    python 02_postprocess/06_rebuild_outlier_flags.py --indicator Rx1day HI --products CLIMBRA
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from xclim.indices.stats import parametric_quantile

from hazbra import indicators
from hazbra.config import add_common_args, load_config
from hazbra.outliers import append_log, drop_outliers, outlier_paths, save_flags


def raw_return_levels(params: xr.Dataset, return_periods, mode: str) -> xr.Dataset:
    t = np.atleast_1d(np.asarray(return_periods, float))
    q = 1 - 1.0 / t if mode == "max" else 1.0 / t
    out = {}
    for v in params.data_vars:
        if "dparams" not in params[v].dims:
            continue
        rl = parametric_quantile(params[v].astype("float64"), q, "genextreme")
        out[v] = rl.rename({"quantile": "return_period"}).assign_coords(return_period=t.astype(int))
    return xr.Dataset(out)


def rebuild(cfg, ind, product, source, params_file, rp_file, overwrite):
    nc_path, log_path = outlier_paths(cfg, product, ind.hazard, ind.name, source)
    if nc_path.exists() and not overwrite:
        print(f"[skip] {nc_path} exists")
        return
    with xr.open_dataset(params_file) as p:
        params = p.load()
    raw = raw_return_levels(params, cfg.calc.return_periods, ind.gev)
    if "GWL" in raw.dims:
        raw = raw.transpose("GWL", "return_period", ...)
    stats, flags = [], {}
    filtered = drop_outliers(raw, product=product, stats=stats, flags=flags,
                             context={"product": product, "indicator": ind.name, "source": source})
    for r in stats:
        r["source"] = source
    save_flags(flags, nc_path)
    append_log(stats, log_path)

    # consistency check against the stored, filtered return levels
    with xr.open_dataset(rp_file) as stored:
        msgs = []
        for v in filtered.data_vars:
            if v in stored:
                a, b = xr.align(filtered[v], stored[v].load(), join="inner")
                d = np.abs(a.transpose(*b.dims).values - b.values)
                rel = np.nanmax(d / np.maximum(np.abs(b.values), 1.0)) if np.isfinite(d).any() else np.nan
                msgs.append(f"{v}: max rel diff {rel:.1e}")
    n_flag = sum(int((f > 0).sum()) for k, f in flags.items() if k.endswith("_flag"))
    print(f"[done] {product}/{ind.name}/{source}: {n_flag} flagged cells; check vs stored: {'; '.join(msgs)}")


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--indicator", nargs="+", help="default: every indicator with return levels")
    ap.add_argument("--products", nargs="+", default=["CLIMBRA", "OBS"], choices=["CLIMBRA", "OBS"])
    args = ap.parse_args()
    cfg = load_config(test=args.test)
    names = args.indicator or [n for n, i in indicators.INDICATORS.items() if i.gev]
    for name in names:
        ind = indicators.get(name)
        if not ind.gev:
            continue
        if "CLIMBRA" in args.products:
            indiv = Path(cfg.paths.netcdf_climbra) / ind.hazard / ind.name / "individual_models"
            for pf in sorted(indiv.glob("*_params.nc")):
                src = pf.name[: -len("_params.nc")]
                rf = indiv / f"{src}_return_period.nc"
                if rf.exists():
                    rebuild(cfg, ind, "CLIMBRA", src, pf, rf, args.overwrite)
        if "OBS" in args.products:
            obs = Path(cfg.paths.netcdf_obs) / ind.hazard / ind.name
            for pf in sorted(obs.glob("*_params.nc")):
                src = pf.name[: -len("_params.nc")]
                rf = obs / f"{src}_return_period.nc"
                if rf.exists():
                    rebuild(cfg, ind, "OBS", src, pf, rf, args.overwrite)


if __name__ == "__main__":
    main()

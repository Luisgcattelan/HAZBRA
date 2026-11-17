#!/usr/bin/env python
"""How many return-level cells did the outlier filter change?  Tables by product, indicator, variable, GWL, return period.

Two sources of counts:
  default        <outliers>/outliers_log.csv, the consolidated CSV every calculation run appends to (exact counts and
                 proportions of spatial and threshold outliers per filter pass; for the capped day counts the threshold
                 outliers are the cells set to the cap). The latest run per product / indicator / source is used.
  --from-product reconstruction for an existing product (no rerun needed): the unfiltered GEV return levels are
                 recomputed from <..>_params.nc and compared with <..>_return_period.nc; a cell counts as replaced
                 when they differ by more than --rtol.  Spatial vs threshold outliers cannot be told apart this way.

Outputs (cfg.paths.tables and cfg.paths.outliers):
  outliers_by_indicator.csv  per product and indicator: cells checked / replaced, % replaced, % spatial, % threshold,
                             worst field (variable / source / GWL / return period) and its %
  outliers_summary.csv       per product, indicator, variable: the same plus mean / max % over fields
  outliers_detail.csv        one row per source (model / dataset), indicator, variable, GWL, return period
  outliers_summary.md        both tables in Markdown (for the paper's Technical Validation / supplement)

    python 02_postprocess/05_outlier_summary.py
    python 02_postprocess/05_outlier_summary.py --from-product --indicator Rx1day TXX
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from hazbra import indicators
from hazbra.config import add_common_args, load_config
from hazbra.outliers import threshold_for
from hazbra.rp_shift import level

# --------------------------------------------------------------------------- counts written by the pipeline
def from_log(cfg, names) -> pd.DataFrame:
    """Rows of <outliers>/outliers_log.csv, latest run per product / indicator / source only."""
    path = Path(cfg.paths.outliers) / "outliers_log.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["indicator"].isin(names)]
    if df.empty:
        return df
    latest = df.groupby(["product", "indicator", "source"])["run_time"].transform("max")
    df = df[df["run_time"] == latest]
    # one row per field: n_valid from the first pass, changes summed over passes (days_* variables: 2 passes)
    keys = ["product", "indicator", "source", "variable", "GWL", "return_period"]
    agg = df.groupby(keys, sort=False).agg(
        n_valid=("n_valid", "first"), n_spatial=("n_spatial", "sum"), n_threshold=("n_threshold", "sum"),
        n_replaced=("n_replaced", "sum"), max_abs_change=("max_abs_change", "max"), threshold=("threshold", "first"),
        capped=("capped", "first"), run_time=("run_time", "first"))
    return agg.reset_index()


# --------------------------------------------------------------------------- reconstruction from the product
def _compare(params_file, rp_file, ind, product, source, rtol):
    rows = []
    with xr.open_dataset(params_file) as par, xr.open_dataset(rp_file) as rp:
        for v in rp.data_vars:
            if v not in par or threshold_for(v, product) is None:
                continue
            p, r = par[v].load(), rp[v].load()
            c, loc, scale = (p.sel(dparams=k, drop=True) for k in ("c", "loc", "scale"))
            mode = "upper" if ind.gev == "max" else "lower"
            gwls = r["GWL"].values if "GWL" in r.dims else [None]
            for g in gwls:
                rg = r if g is None else r.sel(GWL=g)
                cg, lg, sg = (x if g is None else x.sel(GWL=g) for x in (c, loc, scale))
                for T in rg["return_period"].values:
                    filt = rg.sel(return_period=T)
                    raw = xr.apply_ufunc(lambda a, b, d: level(float(T), a, b, d, mode), cg, lg, sg)
                    raw = raw.transpose(*filt.dims).values
                    fv = filt.values
                    valid = np.isfinite(fv)
                    diff = np.abs(raw - fv)
                    changed = valid & ~(diff <= rtol * np.maximum(np.abs(fv), 1.0))
                    rows.append(dict(product=product, indicator=ind.name, source=source, variable=v,
                                     GWL="obs" if g is None else str(g), return_period=int(T),
                                     n_valid=int(valid.sum()), n_spatial=np.nan, n_threshold=np.nan,
                                     n_replaced=int(changed.sum()), threshold=threshold_for(v, product),
                                     max_abs_change=float(np.nanmax(np.where(changed, diff, np.nan))) if changed.any() else 0.0))
    return rows


def from_product(cfg, names, rtol) -> pd.DataFrame:
    rows = []
    for name in names:
        ind = indicators.get(name)
        if not ind.gev:
            continue
        indiv = Path(cfg.paths.netcdf_climbra) / ind.hazard / ind.name / "individual_models"
        for pf in sorted(indiv.glob("*_params.nc")):
            model = pf.name[: -len("_params.nc")]
            rf = indiv / f"{model}_return_period.nc"
            if rf.exists():
                rows += _compare(pf, rf, ind, "CLIMBRA", model, rtol)
        obs = Path(cfg.paths.netcdf_obs) / ind.hazard / ind.name
        for pf in sorted(obs.glob("*_params.nc")):
            ds = pf.name[: -len("_params.nc")]
            rf = obs / f"{ds}_return_period.nc"
            if rf.exists():
                rows += _compare(pf, rf, ind, "OBS", ds, rtol)
        print(f"[{name}] done")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- summary
def summarise(detail: pd.DataFrame) -> pd.DataFrame:
    d = detail.copy()
    d["pct_replaced"] = 100 * d["n_replaced"] / d["n_valid"].where(d["n_valid"] > 0)
    g = d.groupby(["product", "indicator", "variable"], sort=False)
    out = g.agg(threshold=("threshold", "first"), sources=("source", "nunique"), fields=("n_valid", "size"),
                cells_checked=("n_valid", "sum"), cells_replaced=("n_replaced", "sum"),
                spatial_outliers=("n_spatial", "sum"), threshold_outliers=("n_threshold", "sum"),
                pct_mean=("pct_replaced", "mean"), pct_max=("pct_replaced", "max"),
                max_abs_change=("max_abs_change", "max")).reset_index()
    out["pct_total"] = 100 * out["cells_replaced"] / out["cells_checked"]
    if "spatial_outliers" in out and out["spatial_outliers"].notna().any():
        out["pct_spatial"] = 100 * out["spatial_outliers"] / out["cells_checked"]
        out["pct_threshold"] = 100 * out["threshold_outliers"] / out["cells_checked"]
    # share of fields (source x GWL x return period) in which at least one cell was replaced
    out["fields_with_outliers_pct"] = g.apply(lambda x: 100 * (x["n_replaced"] > 0).mean()).values
    if d["n_spatial"].isna().all():
        out = out.drop(columns=["spatial_outliers", "threshold_outliers"])
    return out


def by_indicator(detail: pd.DataFrame) -> pd.DataFrame:
    """Compact table: one row per product and indicator (all variables, sources, GWLs and return periods)."""
    d = detail.copy()
    g = d.groupby(["product", "indicator"], sort=False)
    out = g.agg(variables=("variable", "nunique"), sources=("source", "nunique"),
                cells_checked=("n_valid", "sum"), cells_replaced=("n_replaced", "sum"),
                max_abs_change=("max_abs_change", "max")).reset_index()
    out["pct_replaced"] = 100 * out["cells_replaced"] / out["cells_checked"]
    if d["n_spatial"].notna().any():
        sp = g["n_spatial"].sum().values
        th = g["n_threshold"].sum().values
        out["pct_spatial"] = 100 * sp / out["cells_checked"]
        out["pct_threshold"] = 100 * th / out["cells_checked"]
    d["pct"] = 100 * d["n_replaced"] / d["n_valid"].where(d["n_valid"] > 0)
    out["pct_worst_field"] = g["pct"].max().values
    worst = d.loc[g["pct"].idxmax().values, ["variable", "source", "GWL", "return_period"]]
    out["worst_field"] = [f"{r.variable} / {r.source} / GWL {r.GWL} / {r.return_period} yr" for r in worst.itertuples()]
    return out


def to_markdown(summary: pd.DataFrame) -> str:
    cols = [c for c in ["product", "indicator", "variable", "threshold", "variables", "sources", "cells_checked",
                        "cells_replaced", "pct_total", "pct_replaced", "pct_spatial", "pct_threshold", "pct_max",
                        "pct_worst_field", "worst_field", "fields_with_outliers_pct", "max_abs_change"] if c in summary]
    s = summary[cols].copy()
    for c in ("pct_total", "pct_replaced", "pct_spatial", "pct_threshold", "pct_max", "pct_worst_field",
              "fields_with_outliers_pct"):
        if c in s:
            s[c] = s[c].map(lambda x: f"{x:.3f}")
    s["max_abs_change"] = s["max_abs_change"].map(lambda x: f"{x:.3g}")
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in s.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--indicator", nargs="+", help="default: every indicator with return levels")
    ap.add_argument("--from-product", action="store_true", help="reconstruct counts from params vs return_period files")
    ap.add_argument("--rtol", type=float, default=1e-3, help="relative tolerance for --from-product (default 1e-3)")
    args = ap.parse_args()
    cfg = load_config(test=args.test)
    names = args.indicator or [n for n, i in indicators.INDICATORS.items() if i.gev]

    detail = from_product(cfg, names, args.rtol) if args.from_product else from_log(cfg, names)
    if detail.empty:
        sys.exit("no outlier information found (run 01_calculation first, or use --from-product)")
    summary = summarise(detail)
    compact = by_indicator(detail)
    suffix = "_from_product" if args.from_product else ""
    for out in {Path(cfg.paths.tables), Path(cfg.paths.outliers)}:
        out.mkdir(parents=True, exist_ok=True)
        detail.to_csv(out / f"outliers_detail{suffix}.csv", index=False)
        summary.to_csv(out / f"outliers_summary{suffix}.csv", index=False)
        compact.to_csv(out / f"outliers_by_indicator{suffix}.csv", index=False)
        (out / f"outliers_summary{suffix}.md").write_text(
            "## By indicator\n\n" + to_markdown(compact) + "\n## By variable\n\n" + to_markdown(summary))
        print(f"written to {out}")
    print(to_markdown(compact))


if __name__ == "__main__":
    main()

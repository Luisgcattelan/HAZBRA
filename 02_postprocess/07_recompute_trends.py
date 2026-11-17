#!/usr/bin/env python
"""Recompute the observational trend files (`<DATASET>_trend.nc`) from the stored annual series.

The trend test is the modified Mann-Kendall test of Hamed & Rao (1998) (hazbra/stats.py, TREND_TEST), with
Sen's slope; the period is the full record of each dataset (XAVIER and CHIRPS 1981-2023, MERGE 2001-2023).
The result is identical to what 01_calculation/run_observations.py writes, but needs only `<DATASET>_timeseries.nc`
(no daily data), so it takes minutes instead of days.

    python 02_postprocess/07_recompute_trends.py                       # every indicator and dataset
    python 02_postprocess/07_recompute_trends.py --indicator Rx1day --datasets CHIRPS
    python 02_postprocess/07_recompute_trends.py --test                # on the test product (_test_output/)
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from hazbra import indicators
from hazbra.config import add_common_args, load_config, setup_dask
from hazbra.indicators import OBS_PERIODS
from hazbra.io import save_nc
from hazbra.pipeline_obs import LAT_CHUNK, out_dir
from hazbra.stats import TREND_TEST, compute_trend, mann_kendall, mann_kendall_multi


def recompute(cfg, ind, dataset: str) -> None:
    folder = out_dir(cfg, ind)
    ts_file = folder / f"{dataset}_timeseries.nc"
    if not ts_file.exists():
        print(f"[skip] {ts_file} not found")
        return
    t0 = time.time()
    per = OBS_PERIODS[dataset]["trend"]
    with xr.open_dataset(ts_file) as ds:
        annual = ds.sel(time=slice(f"{per[0]}-01-01", f"{per[1]}-12-31")).load()
    annual = annual.chunk({"time": -1, "lat": LAT_CHUNK})
    trend = mann_kendall_multi(annual) if ind.multi_variable else mann_kendall(annual[ind.name])
    trend = compute_trend(trend)
    trend.attrs.update(trend_period=f"{per[0]}-{per[1]}", n_years=int(annual.sizes["time"]))
    out = folder / f"{dataset}_trend.nc"
    tmp = out.with_suffix(".tmp.nc")
    save_nc(trend, tmp)
    tmp.replace(out)
    print(f"[done] {ind.name}/{dataset}: {out.name} ({TREND_TEST}, {per[0]}-{per[1]}, "
          f"{annual.sizes['time']} yr) in {time.time() - t0:.0f} s")


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--indicator", nargs="+", default=list(indicators.INDICATORS))
    ap.add_argument("--datasets", nargs="+", default=["XAVIER", "CHIRPS", "MERGE"])
    ap.add_argument("--workers", type=int, help="dask threads (default: SLURM_CPUS_PER_TASK)")
    args = ap.parse_args()
    setup_dask(args.workers)
    cfg = load_config(test=args.test)
    for name in args.indicator:
        ind = indicators.get(name)
        for d in ind.obs_datasets:
            if d in args.datasets:
                recompute(cfg, ind, d)


if __name__ == "__main__":
    main()

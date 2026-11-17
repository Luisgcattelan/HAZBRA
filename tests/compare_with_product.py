#!/usr/bin/env python
"""Regression check: compare a --test run with the existing production product on the same cells.

The test run (1x1 degree box) is recomputed with the refactored code; this script reads the matching
production file, cuts it to the same cells and reports the maximum absolute / relative difference per
variable. Both sides are only read. Typical use:

    python 01_calculation/run_climbra.py --test --indicator Rx1day
    python tests/compare_with_product.py climbra Rx1day
    python tests/compare_with_product.py obs Rx1day
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from hazbra import indicators
from hazbra.config import load_config
from hazbra.constants import CLIMBRA_MODELS


def compare(a: xr.Dataset, b: xr.Dataset, label: str, tol=1e-4) -> bool:
    ok = True
    for v in a.data_vars:
        if v not in b.data_vars:
            print(f"  {label}: variable {v} missing in product")
            ok = False
            continue
        x, y = a[v], b[v]
        y = y.sel(lat=x.lat, lon=x.lon, method="nearest")
        # align on shared labelled coordinates (GWL, time, ...) only
        x, y = xr.align(x, y, join="inner", exclude=["lat", "lon"])
        if x.size == 0:
            print(f"  {label}.{v}: nothing to compare (no common coordinates)")
            ok = False
            continue
        d = np.abs(x.values.astype("float64") - y.values.astype("float64"))
        both = np.isfinite(x.values) & np.isfinite(y.values)
        nan_mismatch = int((np.isfinite(x.values) != np.isfinite(y.values)).sum())
        mx = float(np.nanmax(d[both])) if both.any() else float("nan")
        scale = float(np.nanmax(np.abs(y.values[both]))) if both.any() else float("nan")
        rel = mx / scale if scale else float("nan")
        good = (rel <= tol or mx == 0) and nan_mismatch == 0
        ok &= good
        print(f"  {'OK ' if good else 'DIFF'} {label}.{v}: max|diff|={mx:.3g} (rel {rel:.2e}), NaN mismatches={nan_mismatch}, n={int(both.sum())}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["climbra", "obs"])
    ap.add_argument("indicator")
    ap.add_argument("--tol", type=float, default=1e-4, help="max relative difference (default 1e-4)")
    args = ap.parse_args()

    test, prod = load_config(test=True), load_config(test=False)
    ind = indicators.get(args.indicator)
    allok = True
    if args.stage == "climbra":
        t_dir = test.paths.netcdf_climbra / ind.hazard / ind.name
        p_dir = prod.paths.netcdf_climbra / ind.hazard / ind.name
        pairs = sorted((t_dir / "individual_models").glob("*.nc"))
        pairs = [(f, p_dir / "individual_models" / f.name) for f in pairs]
        pairs += [(t_dir / "ensemble" / f.name, None) for f in sorted((t_dir / "ensemble").glob("*.nc"))]
    else:
        t_dir = test.paths.netcdf_obs / ind.hazard / ind.name
        p_dir = prod.paths.netcdf_obs / ind.hazard / ind.name
        pairs = [(f, p_dir / f.name) for f in sorted(t_dir.glob("*.nc"))]

    for t_file, p_file in pairs:
        if p_file is None:
            print(f"- {t_file.name}: (ensemble of the 2 test models only; not comparable with the 19-model product)")
            continue
        if not p_file.exists():
            print(f"- {t_file.name}: no production file")
            continue
        print(f"- {t_file.name}")
        with xr.open_dataset(t_file) as a, xr.open_dataset(p_file) as b:
            allok &= compare(a.load(), b, t_file.stem, args.tol)
    print("\nRESULT:", "all matched" if allok else "differences found (see above)")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()

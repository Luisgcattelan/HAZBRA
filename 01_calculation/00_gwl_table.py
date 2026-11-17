#!/usr/bin/env python
"""Build GWL_CLIMBRA.csv: the 20-year window of each CLIMBRA model at each global warming level.

Input : GWL_all.csv  (global warming level table of the parent CMIP6 simulations; 5 header lines)
Output: GWL_CLIMBRA.csv  (model, ensemble, exp, warming_level, start_year, end_year) — used by every
        calculation script and reported in Table 1 of the paper.

Selection rules
  * member r1i1p1f1, SSP5-8.5 (HadGEM3-GC31-LL: r1i1p1f3; UKESM1-0-LL: r1i1p1f2 – the only members available)
  * only the 19 models of CLIMBRA; CMIP6 model names are mapped to the CLIMBRA names
  * models that never reach +4 degC get a placeholder +4 row (NaN years) so every model has the same levels
  * the 0.61 degC level is dropped
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hazbra.config import add_common_args, load_config
from hazbra.constants import CLIMBRA_MODELS

# CMIP6 name in GWL_all.csv -> CLIMBRA name
RENAME = {
    "EC-Earth3": "EC-EARTH3", "INM-CM5-0": "INM-CM5", "KACE-1-0-G": "KACE", "KIOST-ESM": "KIOST",
    "MPI-ESM1-2-HR": "MPI-ESM1-2", "MRI-ESM2-0": "MRI-ESM2", "INM-CM4-8": "INM-CM4_8",
}
OTHER_MEMBERS = {"HadGEM3-GC31-LL": "r1i1p1f3", "UKESM1-0-LL": "r1i1p1f2"}


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    args = ap.parse_args()
    cfg = load_config(test=args.test)

    df = pd.read_csv(cfg.paths.gwl_table_raw, skiprows=5)
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df["model"] = df["model"].replace(RENAME)

    ssp585 = df[df["exp"] == "ssp585"]
    out = ssp585[ssp585["ensemble"] == "r1i1p1f1"]
    extra = ssp585[ssp585.apply(lambda r: OTHER_MEMBERS.get(r["model"]) == r["ensemble"], axis=1)]
    out = pd.concat([out, extra])
    out = out[out["model"].isin(CLIMBRA_MODELS)].reset_index(drop=True)

    # models without a +4 degC row: copy their +3 row with unknown years
    models_with_4 = out[out["warming_level"] == 4].model.unique()
    missing_4 = out[~out["model"].isin(models_with_4) & (out["warming_level"] == 3)].copy()
    missing_4["warming_level"] = 4
    missing_4[["start_year", "end_year"]] = np.nan
    out = pd.concat([out, missing_4])
    out = out[out["warming_level"] != 0.61].sort_values(["model", "warming_level"]).reset_index(drop=True)

    dest = Path(cfg.paths.gwl_table)
    if cfg.test.enabled:                       # never overwrite the production table in test mode
        dest = Path(cfg.paths.tables) / "GWL_CLIMBRA_test.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"wrote {dest}  ({out.model.nunique()} models)")


if __name__ == "__main__":
    main()

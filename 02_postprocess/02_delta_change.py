#!/usr/bin/env python
"""Delta-change projections: observations (CHIRPS / MERGE / XAVIER) + CLIMBRA ensemble anomalies.

Source: notebook 03.ANOMALIES_ENSEMBLE.ipynb (section "DELTA").  See hazbra/anomalies.py for the method.
Run after 01_anomalies_ensemble.py and 01_calculation/run_observations.py.

    python 02_postprocess/02_delta_change.py --all
    python 02_postprocess/02_delta_change.py --indicator Rx5day --skip-existing
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from hazbra import anomalies
from hazbra.config import add_common_args, load_config
from hazbra.constants import INDICATOR_HAZARD


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--indicator", nargs="+", choices=list(INDICATOR_HAZARD))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="keep existing ensemble_delta_* files (default: rebuild)")
    args = ap.parse_args()
    cfg = load_config(test=args.test)
    names = (list(cfg.test.indicators) if cfg.test.enabled else list(INDICATOR_HAZARD)) if args.all else args.indicator
    if not names:
        ap.error("give --indicator NAME [NAME ...] or --all")
    grids = anomalies._reference_grids(cfg)
    for name in names:
        print(f"\n=== {name}")
        anomalies.build_delta_from_obs_and_anom(cfg, name, grids=grids, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Anomalies vs the present period, ensemble statistics and model agreement of the CLIMBRA products.

Source: notebook 03.ANOMALIES_ENSEMBLE.ipynb (section "CALC").  See hazbra/anomalies.py for the method.
Run after 01_calculation/run_climbra.py.

    python 02_postprocess/01_anomalies_ensemble.py --all
    python 02_postprocess/01_anomalies_ensemble.py --indicator Rx1day HI
    python 02_postprocess/01_anomalies_ensemble.py --test --all      # works on the output of the test run
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
    ap.add_argument("--agreement", type=float, default=0.66, help="fraction of models that must agree on the sign (default 0.66)")
    args = ap.parse_args()
    cfg = load_config(test=args.test)
    names = (list(cfg.test.indicators) if cfg.test.enabled else list(INDICATOR_HAZARD)) if args.all else args.indicator
    if not names:
        ap.error("give --indicator NAME [NAME ...] or --all")
    for name in names:
        print(f"\n=== {name}")
        anomalies.build_anomaly_ensembles(cfg, name, thr_agree=args.agreement)


if __name__ == "__main__":
    main()

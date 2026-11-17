#!/usr/bin/env python
"""Return-period shift of historical extremes (per model and grid cell, then ensemble).

Source: paper_analysis/climbra_rp_shift.py (30.return_periods_shift.ipynb).  See hazbra/rp_shift.py for the method
and README_return_period_shift.md (written next to the CLIMBRA product) for the file description.
Run after 01_calculation/run_climbra.py (needs the *_params.nc and *_return_period.nc files).

    python 02_postprocess/03_return_period_shift.py --all --n-jobs 8
    python 02_postprocess/03_return_period_shift.py --indicator Rx1day
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from hazbra import rp_shift
from hazbra.config import add_common_args, load_config
from hazbra.constants import INDICATOR_HAZARD


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--indicator", nargs="+", choices=list(INDICATOR_HAZARD))
    ap.add_argument("--all", action="store_true", help="every indicator that has parameter files")
    ap.add_argument("--n-jobs", type=int, default=1, help="models in parallel (processes)")
    ap.add_argument("--no-readme", action="store_true", help="do not rewrite README_return_period_shift.md "
                    "(use when several indicators run in parallel)")
    ap.add_argument("--readme-only", action="store_true", help="only rewrite README_return_period_shift.md")
    args = ap.parse_args()
    cfg = load_config(test=args.test)
    cfg.calc.overwrite = cfg.calc.overwrite or args.overwrite
    rp_shift.configure(cfg)
    if args.readme_only:
        rp_shift.write_readme()
    elif args.all:
        rp_shift.run_all(indicators=list(cfg.test.indicators) if cfg.test.enabled else None, n_jobs=args.n_jobs)
    elif args.indicator:
        for name in args.indicator:
            rp_shift.run(name, n_jobs=args.n_jobs, readme=not args.no_readme)
    else:
        ap.error("give --indicator NAME [NAME ...], --all or --readme-only")


if __name__ == "__main__":
    main()

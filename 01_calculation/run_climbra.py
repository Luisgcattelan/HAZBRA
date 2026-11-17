#!/usr/bin/env python
"""Compute CLIMBRA indicator products (per-model GWL files + ensemble statistics).

Replaces the 16 scripts bashs/01.*.py.

    python 01_calculation/run_climbra.py --indicator Rx1day
    python 01_calculation/run_climbra.py --indicator HI FWI --models ACCESS-ESM1-5 MIROC6
    python 01_calculation/run_climbra.py --all
    python 01_calculation/run_climbra.py --indicator Rx1day --ensemble-only    # re-build ensemble files only
    python 01_calculation/run_climbra.py --test --all                          # tiny box, 2 models, writes to test_root
    python 01_calculation/run_climbra.py --indicator Rx1day --dry-run          # list what would be written
    python 01_calculation/run_climbra.py --indicator HI --list-models          # models that have all variables
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from hazbra import indicators
from hazbra.config import add_common_args, load_config, setup_dask
from hazbra.constants import CLIMBRA_MODELS
from hazbra import pipeline_climbra as pc


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--indicator", nargs="+", help=f"one or more of: {', '.join(indicators.INDICATORS)}")
    ap.add_argument("--all", action="store_true", help="every indicator (with --test: the indicators listed in test.yaml)")
    ap.add_argument("--models", nargs="+", help="subset of CLIMBRA models (default: all 19; with --test: test.yaml)")
    ap.add_argument("--ensemble-only", action="store_true", help="skip the per-model step, rebuild ensemble files")
    ap.add_argument("--skip-ensemble", action="store_true", help="per-model step only")
    ap.add_argument("--workers", type=int, help="number of dask threads (default: SLURM_CPUS_PER_TASK inside a job, else all cores)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--list-models", action="store_true",
                    help="print (one per line) the models that have every variable of the indicator, and exit")
    args = ap.parse_args()

    setup_dask(args.workers)
    cfg = load_config(test=args.test)
    if args.all:
        names = list(cfg.test.indicators) if cfg.test.enabled else list(indicators.INDICATORS)
    elif args.indicator:
        names = args.indicator
    else:
        ap.error("give --indicator NAME [NAME ...] or --all")
    models = args.models or (list(cfg.test.models) if cfg.test.enabled else CLIMBRA_MODELS)

    if args.list_models:
        from hazbra.io import climbra_files
        for name in names:
            ind = indicators.get(name)
            for m in models:
                if all(climbra_files(cfg.paths.raw_climbra, m, v) for v in ind.climbra_vars):
                    print(m)
        return

    for name in names:
        ind = indicators.get(name)
        out_indiv, out_ens = pc.out_dirs(cfg, ind)
        if args.dry_run:
            print(f"{ind.name:11s} gev={ind.gev}  vars={ind.climbra_vars}\n    -> {out_indiv}")
            for m in models:
                print("      ", ", ".join(p.name for p in pc.expected_model_files(ind, m, out_indiv)))
            continue
        pc.run(cfg, ind, models=models, overwrite=args.overwrite,
               ensemble=not args.skip_ensemble, per_model=not args.ensemble_only)


if __name__ == "__main__":
    main()

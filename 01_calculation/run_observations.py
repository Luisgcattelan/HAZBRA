#!/usr/bin/env python
"""Compute observational indicator products (CHIRPS, MERGE, BR-DWGD/XAVIER).

Replaces the 16 scripts bashs/02.*.py.

    python 01_calculation/run_observations.py --indicator Rx1day
    python 01_calculation/run_observations.py --indicator Rx5day CDD --datasets CHIRPS MERGE
    python 01_calculation/run_observations.py --all
    python 01_calculation/run_observations.py --test --all        # 1x1 degree box, XAVIER only, writes to test_root
    python 01_calculation/run_observations.py --indicator HI --dry-run
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from hazbra import indicators
from hazbra import pipeline_obs as po
from hazbra.config import add_common_args, load_config, setup_dask
from hazbra.constants import OBS_DATASETS


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--indicator", nargs="+", help=f"one or more of: {', '.join(indicators.INDICATORS)}")
    ap.add_argument("--all", action="store_true", help="every indicator (with --test: the indicators listed in test.yaml)")
    ap.add_argument("--datasets", nargs="+", choices=OBS_DATASETS,
                    help="default: every dataset the indicator is produced for (precipitation indicators: "
                         "XAVIER, CHIRPS, MERGE; all others: XAVIER)")
    ap.add_argument("--workers", type=int, help="number of dask threads (default: SLURM_CPUS_PER_TASK inside a job, else all cores)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    setup_dask(args.workers)
    cfg = load_config(test=args.test)
    if args.all:
        names = list(cfg.test.indicators) if cfg.test.enabled else list(indicators.INDICATORS)
    elif args.indicator:
        names = args.indicator
    else:
        ap.error("give --indicator NAME [NAME ...] or --all")

    for name in names:
        ind = indicators.get(name)
        datasets = args.datasets or list(ind.obs_datasets)
        if cfg.test.enabled and not args.datasets:
            datasets = [d for d in datasets if d in cfg.test.datasets]
        datasets = [d for d in datasets if d in ind.obs_datasets]
        if args.dry_run:
            out = po.out_dir(cfg, ind)
            print(f"{ind.name:11s} vars={ind.obs_vars} gev={ind.gev}\n    -> {out}")
            for d in datasets:
                print("      ", ", ".join(p.name for p in po.expected_files(ind, d, out)))
            continue
        po.run(cfg, ind, datasets=datasets, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

"""Answers the questions run_pipeline.sh needs (printed one item per line).

    python -m hazbra.plan indicators [--test]          # indicators to run
    python -m hazbra.plan models  INDICATOR [--test]   # CLIMBRA models that have every variable of the indicator
    python -m hazbra.plan datasets INDICATOR [--test]  # observational datasets of the indicator
    python -m hazbra.plan gev INDICATOR                # 'yes' if the indicator has return levels
"""
import argparse

from . import indicators
from .config import load_config
from .constants import CLIMBRA_MODELS
from .io import climbra_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["indicators", "models", "datasets", "gev"])
    ap.add_argument("indicator", nargs="?")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    cfg = load_config(test=a.test)
    if a.what == "indicators":
        print("\n".join(cfg.test.indicators if cfg.test.enabled else indicators.INDICATORS))
        return
    ind = indicators.get(a.indicator)
    if a.what == "models":
        models = cfg.test.models if cfg.test.enabled else CLIMBRA_MODELS
        print("\n".join(m for m in models
                        if all(climbra_files(cfg.paths.raw_climbra, m, v) for v in ind.climbra_vars)))
    elif a.what == "datasets":
        ds = [d for d in ind.obs_datasets if not cfg.test.enabled or d in cfg.test.datasets]
        print("\n".join(ds))
    else:
        print("yes" if ind.gev else "no")


if __name__ == "__main__":
    main()

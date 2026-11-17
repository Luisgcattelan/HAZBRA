"""Configuration loader shared by every HAZBRA script.

    from hazbra.config import load_config
    cfg = load_config(test=False)
    cfg.paths.raw_climbra            # pathlib.Path
    cfg.calc.return_periods          # [5, 10, 25, 50, 100]
    cfg.test.enabled                 # True when running with --test
    cfg.test.box                     # dict(center_lon, center_lat, size_deg) or None

`config/paths.yaml` is the single place with absolute paths.  With ``test=True`` the file
`config/test.yaml` is deep-merged on top, which redirects every *output* path below
``test_root`` and shrinks the run (small box, few models).  A different base file can be
selected with the environment variable ``HAZBRA_CONFIG``.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from types import SimpleNamespace

import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = CODE_ROOT / "config"

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _flat_scalars(d: dict, acc: dict | None = None) -> dict:
    acc = {} if acc is None else acc
    for k, v in d.items():
        if isinstance(v, dict):
            _flat_scalars(v, acc)
        elif isinstance(v, str):
            acc[k] = v
    return acc


def _sub_tree(value, scalars):
    if isinstance(value, str):
        return _PLACEHOLDER.sub(lambda m: scalars.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _sub_tree(v, scalars) for k, v in value.items()}
    return value


def _resolve(d: dict) -> dict:
    """Substitute {name} placeholders (name = any scalar key in the file) until nothing changes."""
    for _ in range(10):
        new = _sub_tree(d, _flat_scalars(d))
        if new == d:
            break
        d = new
    return d


def _namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in d.items()})
    return d


def load_config(test: bool = False, base_file: str | os.PathLike | None = None) -> SimpleNamespace:
    base_file = Path(base_file or os.environ.get("HAZBRA_CONFIG", CONFIG_DIR / "paths.yaml"))
    raw = yaml.safe_load(base_file.read_text())
    raw["code_root"] = str(CODE_ROOT)               # usable as {code_root} in the yaml files
    if test:
        raw = _deep_merge(raw, yaml.safe_load((CONFIG_DIR / "test.yaml").read_text()))
    raw = _resolve(raw)

    cfg = _namespace(raw)
    for k, v in vars(cfg.paths).items():
        setattr(cfg.paths, k, Path(v))
    cfg.code_root = CODE_ROOT
    if cfg.test.enabled:
        cfg.test.box = vars(cfg.test.box)      # plain dict: center_lon, center_lat, size_deg
    else:
        cfg.test.box = None
    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Arguments every runnable script exposes."""
    parser.add_argument("--test", action="store_true",
                        help="use config/test.yaml: tiny box, few models, outputs under test_root")
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute files that already exist (default: skip them)")
    return parser


def subset_box(ds, cfg):
    """Cut an xarray object to the test box (no-op outside test mode). Needs 'lat'/'lon' dims."""
    if not cfg.test.enabled:
        return ds
    b = cfg.test.box
    h = b["size_deg"] / 2
    return ds.sel(lat=slice(b["center_lat"] - h, b["center_lat"] + h),
                  lon=slice(b["center_lon"] - h, b["center_lon"] + h))


def setup_dask(workers: int | None = None) -> None:
    """Threaded dask scheduler with `workers` threads, default = the CPUs SLURM allocated to the job
    (dask would otherwise start one thread per core of the whole node)."""
    import dask
    n = workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or None
    if n:
        dask.config.set(scheduler="threads", num_workers=n)

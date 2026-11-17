"""Global-warming-level (GWL) windows per CLIMBRA model."""
from __future__ import annotations

import pandas as pd

from .constants import GWLS


def load_gwl_table(cfg) -> pd.DataFrame:
    """Columns: model, ensemble, exp, warming_level, start_year, end_year (20-year window centred on
    the year the level is reached in the parent SSP5-8.5 simulation)."""
    return pd.read_csv(cfg.paths.gwl_table)


def gwl_window(table: pd.DataFrame, model: str, gwl, pad: int = 5, present=(1981, 2010)) -> tuple[str, str]:
    """(start_year, end_year) as strings.

    "present" is the fixed reference period.  For numeric levels the 20-year window of the table is
    extended by `pad` years on both sides (-> 30-year sample used for climatologies and GEV fits).
    """
    if gwl == "present":
        return str(present[0]), str(present[1])
    row = table[(table["model"] == model) & (table["warming_level"] == gwl)]
    if row.empty:
        raise ValueError(f"No GWL period found for model={model}, gwl={gwl}")
    start = int(row["start_year"].values[0]) - pad
    end = int(row["end_year"].values[0]) + pad
    return str(start), str(end)


def gwl_label(gwl) -> str:
    """'present', '1.5', '2', '3' – the labels used on the GWL dimension of every product."""
    return str(gwl)


__all__ = ["GWLS", "load_gwl_table", "gwl_window", "gwl_label"]

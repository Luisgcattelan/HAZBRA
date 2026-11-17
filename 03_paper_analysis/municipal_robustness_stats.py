"""Municipal robustness statistics quoted in the paper text.

A municipality is *robust* for a variable and warming level when at least 50% of its valid area
has >= 66% of the models agreeing on the sign of the change, i.e. ``agree_area_frac >= 0.5``.
Both ``agree_area_frac`` and the 0/1 flag ``robust`` are already computed per municipality by
``hazbra/municipal.py`` (columns of the ``*_model_agreement`` rows of the CLIMBRA tables); this
script only summarises them: number and share of robust municipalities (unweighted and weighted by
municipal area) for Brazil and the five macro-regions (first digit of the IBGE code), per
indicator / agreement variable / GWL.

Inputs  (cfg.paths): municipal (CLIMBRA/<Hazard>/<Indicator>/ensemble_anomaly_climatology.parquet,
        geometry/municipalities_2024.gpkg for the municipal areas)
Outputs (cfg.paths.tables): municipal_robustness_stats.csv  (and printed to the screen)

Usage
    python 03_paper_analysis/municipal_robustness_stats.py [--test] [--indicators HI]
                [--variables days_danger days_extreme_danger] [--kind anomaly_climatology]

--test : same computation (reads the production municipal tables read-only); output under the test root.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from hazbra.config import load_config, add_common_args
from hazbra import municipal as mun
from hazbra import plotting as hp

REGION_BY_DIGIT = {'1': 'North', '2': 'Northeast', '3': 'Southeast', '4': 'South', '5': 'Central-West'}


def robustness_table(cfg, indicator, kind='anomaly_climatology', variables=None):
    """Summary of the municipal agreement rows of one indicator."""
    cfg = hp.input_cfg(cfg)           # tables are read from the data product (also in --test)
    df = mun.load_table(indicator, 'CLIMBRA', kind=kind, cfg=cfg)
    agr = df[df['variable'].str.endswith('_model_agreement')]
    if variables:
        keep = {f'{v}_model_agreement' for v in variables}
        agr = agr[agr['variable'].isin(keep)]
    if agr.empty:
        raise ValueError(f'no agreement variables found for {indicator} {kind} {variables or ""}')
    area = mun.load_geometry(cfg)[['CD_MUN', 'area_km2']]
    agr = agr.merge(area, on='CD_MUN', how='left')
    agr['region'] = agr['CD_MUN'].astype(str).str[0].map(REGION_BY_DIGIT)
    agr = agr[np.isfinite(agr['agree_area_frac'])]

    rows = []
    for (var, gwl), sub in agr.groupby(['variable', 'GWL'], sort=False):
        groups = [('Brazil', sub)] + [(r, sub[sub['region'] == r]) for r in REGION_BY_DIGIT.values()]
        for region, s in groups:
            n = len(s)
            rob = s['agree_area_frac'] >= mun.ROBUST_AREA_FRAC
            rows.append(dict(indicator=indicator, file_kind=kind, variable=var, GWL=gwl, region=region,
                             n_municipalities=n, n_robust=int(rob.sum()),
                             pct_robust=100 * rob.mean() if n else np.nan,
                             pct_robust_area_weighted=(100 * s.loc[rob, 'area_km2'].sum() / s['area_km2'].sum()
                                                       if n else np.nan),
                             mean_agree_area_frac=s['agree_area_frac'].mean() if n else np.nan))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_common_args(ap)
    ap.add_argument('--indicators', nargs='+', default=['HI'], metavar='IND')
    ap.add_argument('--variables', nargs='*', default=['days_danger', 'days_extreme_danger'],
                    help="variable stems (without '_model_agreement'); empty = all agreement variables")
    ap.add_argument('--kind', default='anomaly_climatology', help='CLIMBRA file kind (default anomaly_climatology)')
    args = ap.parse_args(argv)

    cfg = load_config(test=args.test)
    out = pd.concat([robustness_table(cfg, ind, args.kind, args.variables or None) for ind in args.indicators],
                    ignore_index=True)
    path = Path(cfg.paths.tables) / 'municipal_robustness_stats.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f'[saved] {path}')
    print(out.round(1).to_string(index=False))


if __name__ == '__main__':
    main()

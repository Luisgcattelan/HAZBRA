"""Paper Figure 5: CLIMBRA vs the IPCC AR6 Interactive Atlas (CMIP6) at global warming level 3 °C.

Rows: Rx5day (% change) and TXx (°C change), both on the Atlas 1° grid (CLIMBRA ensemble median
aggregated conservatively from 0.25° to 1°).  Columns: (a) CLIMBRA median, (b) Atlas median
(hatched where model agreement is low: CLIMBRA < 50% of the cell with >= 66% of models agreeing
on the sign; Atlas < 66% of models agreeing), (c) difference CLIMBRA - Atlas, (d) scatter of the
1° cells against the 1:1 line with area-weighted bias, RMSE and Pearson r.

The regional / Brazil statistics quoted in the text (bias, RMSE, r, sign agreement, robust-area
shares, agreement categories) are written to ``fig05_climbra_vs_atlas_gwl3_regions.csv``.
A second table (used in the text) (``..._regional_means.csv``) lists the regional means of the CLIMBRA
p10 / median / p90 and of the Atlas median.

Inputs  (cfg.paths): ipcc_atlas (Atlas downloads: <folder>/{map.nc, consensus.nc}),
        netcdf_climbra (<Hazard>/<Indicator>/ensemble/ensemble_anomaly_climatology.nc),
        shp_brazil, shp_regions
Outputs (cfg.paths.figures): fig05_climbra_vs_atlas_gwl3.{pdf,png}
        (cfg.paths.tables):  fig05_climbra_vs_atlas_gwl3_regions.csv,
                             fig05_climbra_vs_atlas_gwl3_regional_means.csv

Usage
    python 03_paper_analysis/fig05_ipcc_atlas_comparison.py [--test] [--gwl 3] [--indicators Rx5day TXX]

--test : only the TXx row; outputs below the test root.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from hazbra.config import load_config, add_common_args
from hazbra import plotting as hp
from hazbra import ipcc_atlas as ia

N_LEVELS = {'TXX': 24}          # number of colour classes (default 10)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_common_args(ap)
    hp.add_plot_args(ap)
    ap.add_argument('--gwl', default='3', help='global warming level (default 3)')
    ap.add_argument('--indicators', nargs='+', default=None, metavar='IND',
                    help='rows of the figure (default: Rx5day TXX; --test: TXX)')
    args = ap.parse_args(argv)

    hp.init_backend(args.show)
    cfg = load_config(test=args.test)
    indicators = args.indicators or (['TXX'] if args.test else ['Rx5day', 'TXX'])
    stem = f'fig05_climbra_vs_atlas_gwl{args.gwl}'
    tables = Path(cfg.paths.tables)

    fig, _, results, table = ia.plot_atlas_rows(
        cfg, indicators, gwl=args.gwl, n_levels=N_LEVELS,
        table_path=tables / f'{stem}_regions.csv')
    hp.save_figure(fig, stem, cfg.paths.figures, args.formats, dpi=300)
    hp.finish_figure(fig, args.show)
    print(table.round(1).to_string(index=False))

    # regional means of the CLIMBRA ensemble spread vs the Atlas median (quoted in the text)
    means = []
    for res in results:
        means.append(ia.regional_table(cfg, res['indicator'], res['atlas'].to_frame().T))
    means = pd.concat(means, ignore_index=True)
    out = tables / f'{stem}_regional_means.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    means.to_csv(out, index=False)
    print(f'[saved] {out}')


if __name__ == '__main__':
    main()

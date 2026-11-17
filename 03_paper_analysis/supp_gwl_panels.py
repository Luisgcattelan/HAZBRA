"""Supplementary figure: CLIMBRA ensemble change of 14 indicator variables at one global warming level.

One map per (indicator, variable), each with its own symmetric colour scale (98th percentile of |change|
over Brazil), hatched where fewer than 66% of the models agree on the sign of the change; each panel
shows the area-weighted Brazil mean change and the share of the area with robust agreement.

Source: paper_analysis/02.1.panels.ipynb.

Inputs  (cfg.paths): netcdf_climbra (<Hazard>/<Indicator>/ensemble/ensemble_anomaly_climatology.nc), shp_brazil, shp_regions
Outputs (cfg.paths.figures): supp_gwl_panels_gwl<G>.{pdf,png}
        (cfg.paths.tables):  supp_gwl_panels_gwl<G>.csv  (Brazil mean and robust-area share per panel)

Usage
    python 03_paper_analysis/supp_gwl_panels.py [--test] [--gwl 3 2 1.5] [--ensemble median]

--test : GWL 3 only, first 4 panels; outputs below the test root.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.ticker import MaxNLocator

from hazbra.config import load_config, add_common_args
from hazbra import plotting as hp
from hazbra.plotting import indicators_dict, gkey, PC, UNIT_PRETTY

EXTENT = (-75, -30, -36, 6)
BASELINE = '1981–2010'
BRAZIL = 'dissolved'          # outline used by the original notebook (brazil_dissolved.gpkg)

# (indicator, variable stem, 'perc' | 'abs', label)
DEFAULT_PANELS = [
    ('Rx1day', 'Rx1day', 'perc', 'Rx1day'),
    ('Rx5day', 'Rx5day', 'perc', 'Rx5day'),
    ('prcptot', 'prcptot', 'perc', 'Annual precipitation'),
    ('CWD', 'CWD', 'abs', 'Consecutive wet days'),
    ('CDD', 'CDD', 'abs', 'Consecutive dry days'),
    ('SPI6', 'SPI6', 'abs', 'SPI-6'),
    ('TXX', 'TXX', 'abs', 'TXx'),
    ('TNN', 'TNN', 'abs', 'TNn'),
    ('WSDI', 'WSDI', 'abs', 'Warm-spell duration'),
    ('Frost', 'Frost', 'abs', 'Frost days'),
    ('HI', 'HI_max', 'abs', 'Heat index (max)'),
    ('HI', 'days_danger', 'abs', 'HI danger days'),
    ('FWI', 'days_extreme', 'abs', 'Extreme fire-weather days'),
    ('Dry_Hot', 'Dry_Hot', 'abs', 'Dry–hot days'),
]

# diverging colormaps: positive = wetter (green/blue), hotter (red), more dry/fire (brown/orange)
STEM_CMAPS = {
    'Rx1day': 'BrBG', 'Rx5day': 'BrBG', 'prcptot': 'BrBG', 'pr_season': 'BrBG', 'CWD': 'BrBG',
    'CDD': 'BrBG_r', 'CDD_season': 'BrBG_r', 'SPI3': 'BrBG', 'SPI6': 'BrBG',
    'TXX': 'RdBu_r', 'TNN': 'RdBu_r', 'WSDI': 'RdBu_r', 'Frost': 'RdBu',
    'HI_max': 'RdBu_r', 'HI_mean': 'RdBu_r', 'days_hot_total': 'RdBu_r', 'days_caution': 'RdBu_r',
    'days_extreme_caution': 'RdBu_r', 'days_danger': 'RdBu_r', 'days_extreme_danger': 'RdBu_r',
    'days_extreme': 'BrBG_r', 'Dry_Hot': 'BrBG_r',
}
CATEGORY_CMAPS = {'Rainfall': 'BrBG', 'Heat': 'RdBu_r', 'Drought': 'BrBG_r', 'Compound': 'PuOr_r'}
STEM_UNITS = {'TXX': '°C', 'TNN': '°C', 'HI_max': '°C', 'HI_mean': '°C', 'SPI3': '', 'SPI6': '',
              'CDD': 'days', 'CDD_season': 'days', 'CWD': 'days', 'WSDI': 'days', 'Frost': 'days',
              'Dry_Hot': 'days'}
RC = {'font.family': 'sans-serif', 'font.size': 7, 'axes.titlesize': 7.5, 'savefig.dpi': 300,
      'figure.dpi': 150, 'hatch.linewidth': 0.25, 'hatch.color': '0.25'}


def _fmt(x):
    a = abs(x)
    return f'{x:+.0f}' if a >= 10 else (f'{x:+.1f}' if a >= 1 else f'{x:+.2f}')


def load_panel(cfg, base_dir, indicator, stem, anomaly='abs', gwl='3', ensemble='median',
               kind='climatology', return_period=25, clip_brazil=True):
    """Returns (change DataArray (lat, lon), low-agreement mask (0/1) or None, units)."""
    path = Path(base_dir) / indicators_dict[indicator] / indicator / 'ensemble' / f'ensemble_anomaly_{kind}.nc'
    var = f'{stem}_anom_perc' if anomaly == 'perc' else f'{stem}_anom'
    agr = f'{stem}_model_agreement'
    with xr.open_dataset(path) as ds:
        if var not in ds.data_vars:
            raise KeyError(f'{var} not in {path.name}; anomaly variables: '
                           f'{sorted(v for v in ds.data_vars if "anom" in v)}')
        da = hp.std_latlon(ds[var].load())
        ag = hp.std_latlon(ds[agr].load()) if agr in ds.data_vars else None

    def pick(x):
        if 'GWL' in x.dims:
            x = x.assign_coords(GWL=[gkey(v) for v in x['GWL'].values]).sel(GWL=gkey(gwl))
        if 'ensemble' in x.dims:
            x = x.assign_coords(ensemble=[str(v) for v in x['ensemble'].values]).sel(ensemble=ensemble)
        if 'return_period' in x.dims:
            x = x.sel(return_period=return_period, method='nearest')
        extra = [d for d in x.dims if d not in ('lat', 'lon')]
        return x.isel({d: 0 for d in extra}).transpose('lat', 'lon') if extra else x.transpose('lat', 'lon')

    da = pick(da)
    if clip_brazil:
        da = da.where(hp.brazil_mask(cfg, da, BRAZIL))
    low = None
    if ag is not None:
        ag = pick(ag).reindex_like(da, method='nearest', tolerance=1e-3)
        low = (np.isfinite(da) & ~np.isfinite(ag)).astype(float)
    units = str(da.attrs.get('units', ''))
    u = '%' if anomaly == 'perc' else UNIT_PRETTY.get(units, units or STEM_UNITS.get(stem, ''))
    if anomaly != 'perc' and not u:
        u = STEM_UNITS.get(stem, 'days' if stem.startswith('days') else '')
    return da, low, u


def plot_gwl_panels(cfg, base_dir, panels=None, gwl='3', ensemble='median', kind='climatology',
                    return_period=25, ncols=4, max_width=7.1, gap=0.1, n_levels=10, vlims=None,
                    hatch='////', show_stats=True, extent=EXTENT, title=True):
    """Returns fig, axes, summary table (Brazil-mean change and robust-area share per panel)."""
    panels = panels or DEFAULT_PANELS
    vlims = vlims or {}
    data = []
    for ind, stem, an, lab in panels:
        try:
            data.append((ind, stem, an, lab, *load_panel(cfg, base_dir, ind, stem, an, gwl, ensemble, kind,
                                                         return_period)))
        except (FileNotFoundError, KeyError, OSError) as err:
            print(f'[skip] {ind}/{stem}: {err}')
    if not data:
        raise ValueError('no panels could be loaded')

    extent = list(extent)
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    n = len(data)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    left = right = 0.08
    pw = (max_width - left - right - (ncols - 1) * gap) / ncols
    ph = pw * aspect
    t_h, cb_pad, cb_h, cb_lab = 0.2, 0.05, 0.065, 0.36
    rh = t_h + ph + cb_pad + cb_h + cb_lab
    top = 0.42 if title else 0.12
    W, H = max_width, top + nrows * rh + 0.05

    rows = []
    letters = iter('abcdefghijklmnopqrstuvwxyz')
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(W, H))
        axes = []
        for k, (ind, stem, an, lab, da, low, u) in enumerate(data):
            i, j = divmod(k, ncols)
            x0 = left + j * (pw + gap)
            y0 = H - top - i * rh - t_h - ph
            ax = fig.add_axes([x0 / W, y0 / H, pw / W, ph / H], projection=PC)
            axes.append(ax)
            hp.base_map(ax, cfg, extent, brazil=BRAZIL)

            v = da.values[np.isfinite(da.values)]
            vl = vlims.get(stem) or (float(np.quantile(np.abs(v), 0.98)) if v.size else 1.0)
            vl = vl if vl > 0 else 1.0
            cmap = plt.get_cmap(STEM_CMAPS.get(stem) or CATEGORY_CMAPS.get(indicators_dict[ind], 'RdBu_r'))
            norm = (BoundaryNorm(MaxNLocator(n_levels, symmetric=True).tick_values(-vl, vl), cmap.N, extend='both')
                    if n_levels else Normalize(-vl, vl))
            da.plot.pcolormesh(ax=ax, x='lon', y='lat', transform=PC, cmap=cmap, norm=norm,
                               add_colorbar=False, add_labels=False, rasterized=True, zorder=1)
            ax.set_title('')
            if low is not None and float(low.sum()) > 0:
                hp.add_hatch_da(ax, low, hatch)
            ax.text(0.0, 1.03, f'({next(letters)}) {lab}', transform=ax.transAxes, ha='left',
                    va='bottom', fontsize=7.5, fontweight='bold')

            w = np.cos(np.deg2rad(da['lat'])).broadcast_like(da).where(np.isfinite(da))
            mean = float((w * da).sum() / w.sum())
            rob = float((w * (1 - low)).sum() / w.sum() * 100) if low is not None else np.nan
            if show_stats:
                txt = f'{_fmt(mean)}{"%" if u == "%" else " " + u if u else ""}'
                if np.isfinite(rob):
                    txt += f'\nrobust {rob:.0f}%'
                ax.text(0.97, 0.03, txt, transform=ax.transAxes, ha='right', va='bottom', fontsize=5.5,
                        zorder=6, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
            rows.append(dict(indicator=ind, variable=stem, anomaly=an, units=u, GWL=gkey(gwl),
                             ensemble=ensemble, brazil_mean=mean, robust_area_pct=rob, vlim=vl))

            cax = fig.add_axes([(x0 + 0.08 * pw) / W, (y0 - cb_pad - cb_h) / H, 0.84 * pw / W, cb_h / H])
            cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal', extend='both')
            cb.ax.xaxis.set_major_locator(MaxNLocator(5, symmetric=True))
            cb.ax.tick_params(labelsize=5.5, width=0.4, length=2, pad=1)
            cb.outline.set_linewidth(0.4)
            cb.set_label(f'Δ {u}' if u else 'Δ', fontsize=6, labelpad=1)

        if title:
            what = 'climatology' if kind == 'climatology' else f'{return_period}-year return level'
            fig.text(left / W, (H - 0.08) / H,
                     f'CLIMBRA ensemble {ensemble}: change in {what} at GWL {gkey(gwl)} °C (relative to {BASELINE})',
                     ha='left', va='top', fontsize=8.5, fontweight='bold')
            fig.legend(handles=[mpatches.Patch(facecolor='white', edgecolor='0.25', hatch=hatch * 2, lw=0.4,
                                               label='< 66% of models agree on the sign of change')],
                       loc='upper left', bbox_to_anchor=(left / W - 0.005, 1 - 0.2 / H), frameon=False,
                       fontsize=6.5, handlelength=1.6, handleheight=1.2)
    return fig, axes, pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_common_args(ap)
    hp.add_plot_args(ap)
    ap.add_argument('--gwl', nargs='+', default=['3'], help='global warming level(s) (default 3)')
    ap.add_argument('--ensemble', default='median', choices=['median', 'mean', 'p10', 'p90'])
    args = ap.parse_args(argv)

    hp.init_backend(args.show)
    cfg = load_config(test=args.test)
    base_dir = hp.input_path(cfg, 'netcdf_climbra')
    panels = DEFAULT_PANELS[:4] if args.test else DEFAULT_PANELS
    gwls = ['3'] if args.test else args.gwl
    tabs = Path(cfg.paths.tables)
    tabs.mkdir(parents=True, exist_ok=True)
    for g in gwls:
        stem = f'supp_gwl_panels_gwl{gkey(g)}' + ('' if args.ensemble == 'median' else f'_{args.ensemble}')
        fig, _, table = plot_gwl_panels(cfg, base_dir, panels, gwl=g, ensemble=args.ensemble)
        hp.save_figure(fig, stem, cfg.paths.figures, args.formats)
        hp.finish_figure(fig, args.show)
        table.to_csv(tabs / f'{stem}.csv', index=False)
        print(table.round(2).to_string(index=False))


if __name__ == '__main__':
    main()

"""Paper Figures 6 (TXx) and 7 (Rx5day): CLIMBRA ensemble projected change at GWL 1.5 / 2 / 3 °C,
plus the projection statistics quoted in the text.

Figures 6 / 7: rows = ensemble P10 / median / P90, columns = global warming level; absolute change
in °C for TXx (colour limit ±5), % change for Rx5day (±40); hatched where fewer than 66% of the
models agree on the sign of the change.  Each map shows the area-weighted Brazil mean; the median
row also shows the share of the area with >= 66% sign agreement.

Statistics (CSV, cfg.paths.tables)
  <ind>_synthesis.csv           per region (Brazil + 5 IBGE macro-regions) and GWL: area-weighted
                                P10/median/P90/mean of the change, spread (P90 - P10), regional
                                signal-to-noise (median / spread), share of area with
                                |median / (P90 - P10)| >= 1, share of area with median > 0, share of
                                area with >= 66% sign agreement (total / increase / decrease)
  <ind>_synthesis_compact.csv   "median [p10, p90]" per region x GWL, as used in the text tables
Supplementary figures: maps of the signal-to-noise ratio, median / (P90 - P10), with the same hatching,
for TXx, Rx5day and CDD (colour limit ±1); ``--snr-indicators`` changes the list.

Inputs  (cfg.paths): netcdf_climbra (<Hazard>/<Indicator>/ensemble/ensemble_anomaly_climatology.nc;
        variables <Ind>_anom, <Ind>_anom_perc, <Ind>_model_agreement), south_america_limit, shp_regions
Outputs (cfg.paths.figures): fig06_TXX_projection_matrix.{pdf,png}, fig07_Rx5day_projection_matrix.{pdf,png},
                             supp_signal_to_noise_<ind>.{pdf,png}
        (cfg.paths.tables):  <ind>_synthesis.csv, <ind>_synthesis_compact.csv

Usage
    python 03_paper_analysis/fig06_07_projection_matrix.py [--test] [--snr-indicators TXX Rx5day CDD]

--test : TXx only (figure 6 + its tables, no signal-to-noise maps); outputs below the test root.
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
from matplotlib.ticker import MaxNLocator

from hazbra.config import load_config, add_common_args
from hazbra import plotting as hp
from hazbra.plotting import indicators_dict, gkey as _gkey, PC

EXTENT = (-75, -34, -36, 6)
GWLS = ('1.5', '2', '3')
# figure number, indicator, symmetric colour limit
PROJECTION_FIGURES = [('fig06', 'TXX', 5), ('fig07', 'Rx5day', 40)]
RC = hp.pub_rc('0.25')


# =============================================================================
# Data loading
# =============================================================================
def load_climbra(base_dir, indicator, kind='climatology', anomaly='auto', return_period=25):
    """
    Returns (anomaly[ensemble, GWL, lat, lon], agreement[GWL, lat, lon] or None,
             units, anomaly_type).
    """
    if indicator not in indicators_dict:
        raise KeyError(f'{indicator} not in indicators_dict')
    category = indicators_dict[indicator]
    path = Path(base_dir) / category / indicator / 'ensemble' / f'ensemble_anomaly_{kind}.nc'
    if not path.exists():
        raise FileNotFoundError(path)
    if anomaly == 'auto':
        anomaly = 'perc' if hp.relative_ok(indicator) else 'abs'
    var = f'{indicator}_anom_perc' if anomaly == 'perc' else f'{indicator}_anom'
    agr_var = f'{indicator}_model_agreement'

    with xr.open_dataset(path) as ds:
        if var not in ds:
            raise KeyError(f'{var} not in {path}. Available: {list(ds.data_vars)}')
        anom = ds[var].load()
        agree = ds[agr_var].load() if agr_var in ds else None

    def _sel_rp(da):
        if da is not None and 'return_period' in da.dims:
            da = da.sel(return_period=return_period, method='nearest')
        return da

    anom = hp.std_latlon(_sel_rp(anom)).transpose('ensemble', 'GWL', 'lat', 'lon')
    if agree is not None:
        agree = hp.std_latlon(_sel_rp(agree)).transpose('GWL', 'lat', 'lon')
    units = '%' if anomaly == 'perc' else anom.attrs.get('units', '')
    return anom, agree, hp.UNIT_PRETTY.get(units, units), anomaly


def _agreement_mask(agree, g, ref):
    """0/1 mask of the cells with LOW model agreement at GWL ``g`` (for hatching) and the % of the
    valid area (cells finite in ``ref``) where >= 66% of the models agree on the sign.
    Returns (None, None) if unavailable."""
    if agree is None:
        return None, None
    agk = [_gkey(x) for x in agree['GWL'].values]
    if _gkey(g) not in agk:
        return None, None
    ag = agree.isel(GWL=agk.index(_gkey(g)))
    valid = np.isfinite(ref)
    robust = np.isfinite(ag) & valid
    low = ~np.isfinite(ag) & valid
    w = hp.weights(ref).where(valid)
    frac = float((w * robust).sum() / w.sum() * 100)
    return low.astype(float), frac


def _colorbar_and_legend(fig, norm, cmap, label, left, gw, hatch, threshold, with_hatch):
    cax = fig.add_axes([(left + 0.03 * gw) / fig.get_figwidth(), 0.42 / fig.get_figheight(),
                        0.6 * gw / fig.get_figwidth(), 0.1 / fig.get_figheight()])
    cb = hp.cbar(fig, cax, norm, cmap, 'horizontal', label)
    cb.ax.xaxis.set_major_locator(MaxNLocator(7, symmetric=True))
    if with_hatch:
        patch = mpatches.Patch(facecolor='white', edgecolor='0.25', hatch=hatch * 2,
                               linewidth=0.5, label=f'< {threshold}% of models\nagree on sign')
        fig.legend(handles=[patch], loc='center left', frameon=False, fontsize=7,
                   bbox_to_anchor=((left + 0.7 * gw) / fig.get_figwidth(), 0.47 / fig.get_figheight()),
                   handlelength=1.8, handleheight=1.4)


# =============================================================================
# Figures 6 / 7: projection matrix
# =============================================================================
def plot_projection_matrix(cfg, base_dir, indicator, kind='climatology', anomaly='auto', return_period=25,
                           gwls=GWLS, ensembles=('p10', 'median', 'p90'), extent=EXTENT,
                           hatch='////', agreement_threshold=66, n_levels=10, vlim=None,
                           panel_width=2.0, gap=0.08):
    """
    Rows = ensemble statistics, columns = global warming levels; hatching where < threshold % of the
    models agree on the sign.

    kind    : 'climatology' (paper) or 'return_period'
    anomaly : 'auto' (% for rainfall/drought, absolute otherwise), 'perc' or 'abs'
    vlim    : symmetric colour limit; None = 98th percentile of |anomaly|
    """
    if kind not in ('climatology', 'return_period'):
        raise ValueError("kind must be 'climatology' or 'return_period'")
    anom, agree, units, anomaly = load_climbra(base_dir, indicator, kind, anomaly, return_period)
    extent = list(extent)
    category = indicators_dict[indicator]
    cmap = plt.get_cmap(hp.DIFF_CMAPS_IND.get(indicator, hp.DIFF_CMAPS.get(category, 'RdBu_r')))
    perc = anomaly == 'perc'

    gkeys = [_gkey(g) for g in anom['GWL'].values]
    ekeys = [str(e) for e in anom['ensemble'].values]
    bmask = hp.brazil_mask(cfg, anom.isel(ensemble=0, GWL=0))

    # ---- panels ----
    nr, nc = len(ensembles), len(gwls)
    panels = {}
    for i, e in enumerate(ensembles):
        for j, g in enumerate(gwls):
            if e not in ekeys or _gkey(g) not in gkeys:
                print(f'[missing] ensemble={e}, GWL={g}')
                panels[(i, j)] = None
                continue
            da = anom.isel(ensemble=ekeys.index(e), GWL=gkeys.index(_gkey(g)))
            panels[(i, j)] = da.where(bmask)

    # ---- agreement masks (one per GWL, same for every row) ----
    hatch_masks, agree_frac = {}, {}
    if agree is None:
        print(f'[info] no {indicator}_model_agreement variable; plotting without hatching')
    else:
        for j, g in enumerate(gwls):
            ref = next((panels[(i, j)] for i in range(nr) if panels[(i, j)] is not None), None)
            if ref is None:
                continue
            m, frac = _agreement_mask(agree, g, ref)
            if m is not None:
                hatch_masks[j], agree_frac[j] = m, frac

    # ---- colour scale shared by all panels ----
    if vlim is None:
        vals = [np.abs(p.values[np.isfinite(p.values)]) for p in panels.values() if p is not None]
        vals = np.concatenate(vals) if vals else np.array([])
        vlim = float(np.quantile(vals, 0.98)) if vals.size else 1.0
    norm = hp.make_norm(-vlim, vlim, cmap, n_levels, symmetric=True)

    # ---- fixed layout (inches) ----
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    pw, ph = panel_width, panel_width * aspect
    left, right, top, bottom = 0.4, 0.1, 0.35, 0.8
    gw, gh = nc * pw + (nc - 1) * gap, nr * ph + (nr - 1) * gap
    W, H = left + gw + right, top + gh + bottom

    def rect(i, j):
        x = left + j * (pw + gap)
        y = bottom + gh - (i + 1) * ph - i * gap
        return [x / W, y / H, pw / W, ph / H]

    letters = iter('abcdefghijklmnopqrstuvwxyz')
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(W, H))
        axes = np.empty((nr, nc), dtype=object)

        for (i, j), da in panels.items():
            ax = fig.add_axes(rect(i, j), projection=PC)
            axes[i, j] = ax
            hp.base_map(ax, cfg, extent)

            if da is None:
                ax.text(0.5, 0.5, 'Not available', ha='center', va='center',
                        transform=ax.transAxes, color='0.4', fontsize=8)
            else:
                da.plot.pcolormesh(ax=ax, x='lon', y='lat', transform=PC, cmap=cmap, norm=norm,
                                   add_colorbar=False, add_labels=False, rasterized=True, zorder=1)
                ax.set_title('')
                if j in hatch_masks:
                    hp.add_hatch_da(ax, hatch_masks[j], hatch)
                txt = f'mean {hp.fmt(hp.wmean(da), True)}{"%" if perc else ""}'
                if ensembles[i] == 'median' and j in agree_frac:
                    txt += f'\nagree {agree_frac[j]:.0f}% area'
                hp.stat_box(ax, txt)
            ax.text(0.03, 0.97, f'({next(letters)})', transform=ax.transAxes,
                    ha='left', va='top', fontsize=8, fontweight='bold', zorder=6)

        # ---- row / column labels ----
        for j, g in enumerate(gwls):
            fig.text((left + j * (pw + gap) + pw / 2) / W, (bottom + gh + 0.06) / H,
                     f'GWL {_gkey(g)} °C', ha='center', va='bottom', fontsize=9, fontweight='bold')
        for i, e in enumerate(ensembles):
            fig.text(0.08 / W, (bottom + gh - i * (ph + gap) - ph / 2) / H, hp.ENS_LABELS.get(e, e),
                     rotation=90, ha='left', va='center', fontsize=9, fontweight='bold')

        # ---- colorbar + hatch legend ----
        what = 'climatology' if kind == 'climatology' else f'{return_period}-year return level'
        label = (f'Change in {hp.VAR_LABELS.get(indicator, indicator)} {what}'
                 + (f' ({units})' if units else ''))
        _colorbar_and_legend(fig, norm, cmap, label, left, gw, hatch, agreement_threshold, bool(hatch_masks))
    return fig, axes


# =============================================================================
# Regional synthesis table (median [p10, p90], signal-to-noise, agreement shares)
# =============================================================================
def projection_synthesis_table(cfg, base_dir, indicator, kind='climatology', anomaly='auto', return_period=25,
                               gwls=GWLS, snr_threshold=1.0, min_spread=1e-6, language='en'):
    """
    Synthesis per region (Brazil + IBGE macro-regions) and GWL of the CLIMBRA projected change.

    Columns
      p10 / median / p90 (/ mean)  area-weighted mean of each ensemble-statistic map
      spread                       model range: p90 - p10 (of the regional means)
      snr_regional                 regional signal-to-noise: median / spread
      snr_area_pct                 % of area where the cell-level |median / (p90 - p10)| >= snr_threshold
      increase_area_pct            % of area with median change > 0
      robust_area_pct              % of area where >= 66% of models agree on the sign
      robust_increase_pct          % of area with median > 0 and robust agreement
      robust_decrease_pct          % of area with median < 0 and robust agreement
    All shares are cos(latitude)-area-weighted over the 0.25° cells whose centre is inside the region
    and where the median change is finite.
    """
    anom, agree, units, anomaly = load_climbra(base_dir, indicator, kind, anomaly, return_period)
    gkeys = [_gkey(g) for g in anom['GWL'].values]
    ekeys = [str(e) for e in anom['ensemble'].values]
    lat, lon = anom['lat'].values, anom['lon'].values
    w_all = np.cos(np.deg2rad(np.meshgrid(lon, lat)[1]))
    masks = hp.region_masks(cfg, lat, lon, language=language, brazil='limit')   # Brazil + macro-regions

    agk = [_gkey(g) for g in agree['GWL'].values] if agree is not None else []
    stats = [e for e in ('p10', 'median', 'p90', 'mean') if e in ekeys]
    rows = []
    for g in gwls:
        g = _gkey(g)
        if g not in gkeys:
            print(f'[missing] GWL {g}')
            continue
        f = {e: anom.isel(ensemble=ekeys.index(e), GWL=gkeys.index(g)).values for e in stats}
        med = f['median']
        spread_cell = f['p90'] - f['p10'] if {'p10', 'p90'} <= set(f) else None
        snr_cell = (np.where(spread_cell > min_spread, med / spread_cell, np.nan)
                    if spread_cell is not None else None)
        robust = np.isfinite(agree.isel(GWL=agk.index(g)).values) if g in agk else None

        for name, m in masks.items():
            ok = m & np.isfinite(med)
            if not ok.any():
                continue
            w = w_all[ok]
            ws = w.sum()

            def wmean(x):
                v = x[ok]
                fin = np.isfinite(v)
                return float((w[fin] * v[fin]).sum() / w[fin].sum()) if fin.any() else np.nan

            def share(b):
                return float((w * b[ok]).sum() / ws * 100)

            rec = dict(indicator=indicator, kind=kind, units=units, GWL=g, region=name,
                       n_cells=int(ok.sum()))
            for e in stats:
                rec[e] = wmean(f[e])
            if spread_cell is not None:
                rec['spread'] = rec['p90'] - rec['p10']
                rec['snr_regional'] = rec['median'] / rec['spread'] if abs(rec['spread']) > min_spread else np.nan
                rec['snr_area_pct'] = share(np.nan_to_num(np.abs(snr_cell)) >= snr_threshold)
            rec['increase_area_pct'] = share(med > 0)
            if robust is not None:
                rec['robust_area_pct'] = share(robust)
                rec['robust_increase_pct'] = share(robust & (med > 0))
                rec['robust_decrease_pct'] = share(robust & (med < 0))
            rows.append(rec)
    return pd.DataFrame(rows)


def compact_table(tab):
    """'median [p10, p90]' per region (rows) and GWL (columns)."""
    tab = tab.copy()
    tab['range'] = tab.apply(lambda r: f"{r['median']:+.1f} [{r['p10']:+.1f}, {r['p90']:+.1f}]", axis=1)
    return tab.pivot(index='region', columns='GWL', values='range').loc[tab['region'].unique()]


# =============================================================================
# Supplementary: signal-to-noise maps, median / (p90 - p10)
# =============================================================================
def plot_signal_to_noise(cfg, base_dir, indicator, kind='climatology', anomaly='auto', return_period=25,
                         gwls=GWLS, extent=EXTENT, hatch='////', agreement_threshold=66,
                         snr_threshold=1.0, min_spread=1e-6, n_levels=10, vlim=None,
                         panel_width=2.0, gap=0.08):
    """
    One row, columns = global warming levels.
    Value = median / (p90 - p10): projected change relative to the model spread.
    Each panel reports the Brazil-wide area share with |S/N| >= snr_threshold (cells with
    p90 - p10 <= min_spread are masked) and the share of the area with >= 66% sign agreement.
    """
    if kind not in ('climatology', 'return_period'):
        raise ValueError("kind must be 'climatology' or 'return_period'")
    anom, agree, units, anomaly = load_climbra(base_dir, indicator, kind, anomaly, return_period)
    ekeys = [str(e) for e in anom['ensemble'].values]
    missing = [e for e in ('median', 'p10', 'p90') if e not in ekeys]
    if missing:
        raise KeyError(f'ensemble members {missing} not in file (available: {ekeys})')

    extent = list(extent)
    category = indicators_dict[indicator]
    cmap = plt.get_cmap(hp.DIFF_CMAPS_IND.get(indicator, hp.DIFF_CMAPS.get(category, 'RdBu_r')))
    gkeys = [_gkey(g) for g in anom['GWL'].values]
    bmask = hp.brazil_mask(cfg, anom.isel(ensemble=0, GWL=0))

    def _ens(e):
        return anom.isel(ensemble=ekeys.index(e))

    spread = _ens('p90') - _ens('p10')
    snr_all = (_ens('median') / spread.where(spread > min_spread)).where(bmask)

    # ---- panels + agreement ----
    nc = len(gwls)
    panels, hatch_masks, agree_frac = {}, {}, {}
    if agree is None:
        print(f'[info] no {indicator}_model_agreement variable; plotting without hatching')
    for j, g in enumerate(gwls):
        if _gkey(g) not in gkeys:
            print(f'[missing] GWL={g}')
            panels[j] = None
            continue
        da = snr_all.isel(GWL=gkeys.index(_gkey(g)))
        panels[j] = da
        m, frac = _agreement_mask(agree, g, da)
        if m is not None:
            hatch_masks[j], agree_frac[j] = m, frac

    # ---- colour scale ----
    if vlim is None:
        vals = [np.abs(p.values[np.isfinite(p.values)]) for p in panels.values() if p is not None]
        vals = np.concatenate(vals) if vals else np.array([])
        vlim = float(np.quantile(vals, 0.98)) if vals.size else 1.0
    norm = hp.make_norm(-vlim, vlim, cmap, n_levels, symmetric=True)

    # ---- fixed layout (inches) ----
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    pw, ph = panel_width, panel_width * aspect
    left, right, top, bottom = 0.15, 0.1, 0.35, 0.8
    gw = nc * pw + (nc - 1) * gap
    W, H = left + gw + right, top + ph + bottom

    letters = iter('abcdefghijklmnopqrstuvwxyz')
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(W, H))
        axes = np.empty(nc, dtype=object)

        for j, g in enumerate(gwls):
            x0 = left + j * (pw + gap)
            ax = fig.add_axes([x0 / W, bottom / H, pw / W, ph / H], projection=PC)
            axes[j] = ax
            hp.base_map(ax, cfg, extent)
            da = panels[j]
            if da is None:
                ax.text(0.5, 0.5, 'Not available', ha='center', va='center',
                        transform=ax.transAxes, color='0.4', fontsize=8)
            else:
                da.plot.pcolormesh(ax=ax, x='lon', y='lat', transform=PC, cmap=cmap, norm=norm,
                                   add_colorbar=False, add_labels=False, rasterized=True, zorder=1)
                ax.set_title('')
                if j in hatch_masks:
                    hp.add_hatch_da(ax, hatch_masks[j], hatch)
                valid = np.isfinite(da)
                w = hp.weights(da).where(valid)
                strong = float((w * (np.abs(da) >= snr_threshold)).sum() / w.sum() * 100)
                txt = f'|S/N| ≥ {snr_threshold:g}: {strong:.0f}% area'
                if j in agree_frac:
                    txt += f'\nagree {agree_frac[j]:.0f}% area'
                hp.stat_box(ax, txt)
            ax.text(0.03, 0.97, f'({next(letters)})', transform=ax.transAxes,
                    ha='left', va='top', fontsize=8, fontweight='bold', zorder=6)
            fig.text((x0 + pw / 2) / W, (bottom + ph + 0.06) / H, f'GWL {_gkey(g)} °C',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

        what = 'climatology' if kind == 'climatology' else f'{return_period}-year return level'
        label = (f'Signal-to-noise, {hp.VAR_LABELS.get(indicator, indicator)} {what}:'
                 f'  median / (p90 − p10)')
        _colorbar_and_legend(fig, norm, cmap, label, left, gw, hatch, agreement_threshold, bool(hatch_masks))
    return fig, axes


# =============================================================================
# Driver
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_common_args(ap)
    hp.add_plot_args(ap)
    ap.add_argument('--snr-indicators', nargs='*', default=['TXX', 'Rx5day', 'CDD'], metavar='IND',
                    help='indicators with a supplementary signal-to-noise map (default: TXX Rx5day CDD)')
    args = ap.parse_args(argv)

    hp.init_backend(args.show)
    cfg = load_config(test=args.test)
    base_dir = hp.input_path(cfg, 'netcdf_climbra')
    figs, tabs = Path(cfg.paths.figures), Path(cfg.paths.tables)
    tabs.mkdir(parents=True, exist_ok=True)

    todo = PROJECTION_FIGURES[:1] if args.test else PROJECTION_FIGURES
    for tag, ind, vlim in todo:
        print(f'=== {tag}: {ind} projection matrix ===')
        fig, _ = plot_projection_matrix(cfg, base_dir, ind, vlim=vlim)
        hp.save_figure(fig, f'{tag}_{ind}_projection_matrix', figs, args.formats, dpi=300)
        hp.finish_figure(fig, args.show)

        tab = projection_synthesis_table(cfg, base_dir, ind)
        tab.to_csv(tabs / f'{ind}_synthesis.csv', index=False)
        compact = compact_table(tab)
        compact.to_csv(tabs / f'{ind}_synthesis_compact.csv')
        print(f'[saved] {tabs / f"{ind}_synthesis.csv"} (+ _compact)')
        print(compact.to_string())

    if not args.test:
        for ind in args.snr_indicators:
            print(f'=== signal-to-noise: {ind} ===')
            fig, _ = plot_signal_to_noise(cfg, base_dir, ind, vlim=1)
            hp.save_figure(fig, f'supp_signal_to_noise_{ind}', figs, args.formats, dpi=300)
            hp.finish_figure(fig, args.show)


if __name__ == '__main__':
    main()

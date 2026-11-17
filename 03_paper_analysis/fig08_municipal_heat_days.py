"""Paper Figure 8: change in dangerous-heat days per year, municipal map + city panel.

Dangerous-heat days = days with Heat Index >= 41 °C = ``days_danger + days_extreme_danger``
(Heat Index category thresholds: Danger 41-54 °C, Extreme danger > 54 °C).

(a) map: CLIMBRA ensemble change (default: median) in days per year at GWL 2 °C for every
    municipality (area-weighted mean over the municipality; adaptive buffer for small/coastal
    municipalities, see ``hazbra/municipal.py``), hatched where < 50% of the municipal area has
    >= 66% model agreement on the sign of the change (``agree_area_frac`` of the ``days_danger``
    agreement variable).
(b) city (default Brasília, CD_MUN 5300108): days per year at present (observed, XAVIER/BR-DWGD
    1981-2010 climatology) and projected (present + CLIMBRA change, clipped to 0-365) for the
    P10 / median / P90 of the ensemble at GWL 1.5 / 2 / 3 °C; the median change and the share of the
    municipal area with >= 66% model agreement are printed under each GWL.

Inputs  (cfg.paths): municipal  (parquet tables written by hazbra/municipal.py:
        CLIMBRA/Compound/HI/ensemble_anomaly_climatology.parquet,
        Observations/Compound/HI/XAVIER_climatology.parquet, geometry/municipalities_2024.gpkg,
        geometry/states_2024.gpkg)
Outputs (cfg.paths.figures): fig08_municipal_heat_days.{pdf,png}
        (cfg.paths.tables):  fig08_municipal_heat_days_city.csv  (the values shown in panel b)

Usage
    python 03_paper_analysis/fig08_municipal_heat_days.py [--test] [--cd-mun 5300108] [--gwl 2]

--test : same figure; reads the production municipal tables read-only, writes below the test root.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from hazbra.config import load_config, add_common_args
from hazbra import plotting as hp
from hazbra import municipal as mun

LABELS = {
    ('days_danger', 'days_extreme_danger'): 'Dangerous-heat days (HI ≥ 41 °C)',
    ('days_danger',): 'Danger days (HI 41–54 °C)',
    ('days_extreme_danger',): 'Extreme-danger days (HI > 54 °C)',
    ('days_extreme_caution',): 'Extreme-caution days (HI 32–41 °C)',
    ('days_caution',): 'Caution days (HI 27–32 °C)',
    ('days_hot_total',): 'Hot days (HI ≥ 27 °C)',
}
ENS_COLORS = {'p10': '#FCBBA1', 'median': '#EF3B2C', 'p90': '#99000D'}
ENS_LABELS = {'p10': 'p10', 'median': 'Median', 'p90': 'p90'}
RC = hp.pub_rc('0.3')
MAP_EXTENT = (-74.5, -34.3, -34.2, 5.6)


def _sum_vars(df, names, keys):
    """Sum 'mean' over several variables; keep only rows where all variables are present."""
    parts = [df[df['variable'] == n] for n in names]
    missing = [n for n, p in zip(names, parts) if p.empty]
    if missing:
        avail = sorted(df['variable'].unique())
        raise ValueError(f'{missing} not found; available: {avail}')
    allp = pd.concat(parts)
    out = (allp.groupby(keys, as_index=False)
               .agg(mean=('mean', 'sum'), n=('mean', 'count'), method=('method', 'first')))
    return out[out['n'] == len(names)].drop(columns='n')


def plot_municipal_days(cfg, indicator='HI', variables=('days_danger', 'days_extreme_danger'),
                        cd_mun='5300108', gwl='2', ensemble='median', gwls=('1.5', '2', '3'),
                        ensembles=('p10', 'median', 'p90'), agreement_from=None, obs_dataset='XAVIER',
                        robust_frac=0.5, vlim=None, n_levels=10, label=None, figsize=(7.1, 4.1)):
    """Returns fig, (map axis, city axis), DataFrame with the city values of panel (b)."""
    variables = (variables,) if isinstance(variables, str) else tuple(variables)
    name = label or LABELS.get(variables, ' + '.join(variables))
    agr_stem = agreement_from or variables[0]
    g0, cd = hp.gkey(gwl), str(cd_mun)
    gk = [hp.gkey(g) for g in gwls]
    cfg = hp.input_cfg(cfg)               # municipal tables are read from the data product (also in --test)

    cb = mun.load_table(indicator, 'CLIMBRA', kind='anomaly_climatology', cfg=cfg)
    chg = _sum_vars(cb, [f'{v}_anom' for v in variables], ['CD_MUN', 'GWL', 'ensemble'])
    agr = cb[cb['variable'] == f'{agr_stem}_model_agreement'][['CD_MUN', 'GWL', 'agree_area_frac']]

    geom = mun.load_geometry(cfg)
    geom['CD_MUN'] = geom['CD_MUN'].astype(str)
    states = mun.load_states(cfg)
    states['geometry'] = states.geometry.simplify(0.01)

    city = geom[geom['CD_MUN'] == cd]
    if city.empty:
        raise KeyError(f'CD_MUN {cd} not in the municipal geometry')
    cname, cuf = city['NM_MUN'].iat[0], city['NM_UF'].iat[0]

    # observed present day (sum of variables)
    present = np.nan
    try:
        ob = mun.load_table(indicator, 'Observations', dataset=obs_dataset, kind='climatology', cfg=cfg)
        ob = ob[ob['CD_MUN'] == cd]
        vals = [ob.loc[ob['variable'] == v, 'mean'] for v in variables]
        if all(len(x) for x in vals):
            present = float(sum(x.iat[0] for x in vals))
    except (FileNotFoundError, KeyError, ValueError):
        pass

    # ---- map data and colour scale ----
    m = chg[(chg['GWL'] == g0) & (chg['ensemble'] == ensemble)][['CD_MUN', 'mean']].rename(columns={'mean': 'value'})
    a = agr[agr['GWL'] == g0][['CD_MUN', 'agree_area_frac']]
    geo = geom.merge(m, on='CD_MUN', how='left').merge(a, on='CD_MUN', how='left')
    v = geo['value'].to_numpy(float)
    fin = v[np.isfinite(v)]
    vlim = vlim or (float(np.quantile(np.abs(fin), 0.98)) if fin.size else 1.0) or 1.0
    if fin.size and np.quantile(fin, 0.02) >= 0:                       # (almost) only increases
        cmap = plt.get_cmap('YlOrRd')
        lv = MaxNLocator(n_levels).tick_values(0, vlim)
    else:
        cmap = plt.get_cmap('RdBu_r')
        lv = MaxNLocator(n_levels, symmetric=True).tick_values(-vlim, vlim)
    norm = BoundaryNorm(lv, cmap.N, extend='both')

    # ---- city data ----
    cc = chg[chg['CD_MUN'] == cd]
    ca = agr[agr['CD_MUN'] == cd]
    method = str(cc['method'].iat[0]) if len(cc) else ''
    getv = lambda g, e: (float(cc.loc[(cc['GWL'] == g) & (cc['ensemble'] == e), 'mean'].iat[0])
                         if len(cc.loc[(cc['GWL'] == g) & (cc['ensemble'] == e)]) else np.nan)
    projected = np.isfinite(present)

    with plt.rc_context(RC):
        fig = plt.figure(figsize=figsize, layout='constrained')
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1])
        axm, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

        # ---------- (a) map ----------
        ok = geo['value'].notna()
        if (~ok).any():
            geo[~ok].plot(ax=axm, color='0.88', linewidth=0)
        geo[ok].plot(ax=axm, color=cmap(norm(geo.loc[ok, 'value'].to_numpy())), linewidth=0, rasterized=True)
        low = geo[ok & geo['agree_area_frac'].notna() & (geo['agree_area_frac'] < robust_frac)]
        if len(low):
            low.plot(ax=axm, facecolor='none', edgecolor='0.3', hatch='//////', linewidth=0)
        states.boundary.plot(ax=axm, color='0.25', linewidth=0.35)
        city.boundary.plot(ax=axm, color='k', linewidth=1.1)
        pt = city.geometry.representative_point().iat[0]
        axm.annotate(cname, xy=(pt.x, pt.y), xytext=(pt.x + 4, pt.y + 3.5), fontsize=7, fontweight='bold',
                     arrowprops=dict(arrowstyle='-', lw=0.6, color='k'),
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        axm.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
        axm.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
        axm.set_aspect('equal')
        axm.set_axis_off()
        axm.set_title(f'(a) Change in {name[0].lower() + name[1:]}\nGWL {g0} °C, ensemble {ensemble}',
                      fontsize=8.5, loc='left')
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axm, orientation='horizontal',
                            fraction=0.045, pad=0.02, shrink=0.85, extend='both')
        cbar.set_label('Change (days per year)', fontsize=8)
        cbar.outline.set_linewidth(0.5)
        cbar.ax.tick_params(width=0.5, length=2.5, labelsize=6.5)
        # legend in the empty Pacific corner (south-west), clear of Brazil
        axm.legend(handles=[mpatches.Patch(facecolor='white', edgecolor='0.3', hatch='//////', lw=0.5,
                                           label=f'< {robust_frac:.0%} of area with\n≥ 66% model agreement')],
                   loc='lower left', frameon=False, fontsize=6.3, handlelength=1.6, handleheight=1.3,
                   borderaxespad=0.2)

        # ---------- (b) city ----------
        groups = (['present'] if projected else []) + gk
        x = np.arange(len(groups))
        w = 0.8 / len(ensembles)
        allv = []
        if projected:
            axb.bar(0, present, w * 1.5, color='0.55', zorder=3, label=f'Present ({obs_dataset})')
            axb.text(0, present, f'{present:.0f}', ha='center', va='bottom', fontsize=6.5, zorder=4)
            axb.axhline(present, color='0.4', lw=0.7, ls='--', zorder=2)
            allv.append(present)
        for k, e in enumerate(ensembles):
            ys = []
            for g in gk:
                d = getv(g, e)
                ys.append(np.clip(present + d, 0, 365) if projected else d)
            xs = x[(1 if projected else 0):] + (k - (len(ensembles) - 1) / 2) * w
            axb.bar(xs, ys, w * 0.92, color=ENS_COLORS.get(e, '0.5'), label=ENS_LABELS.get(e, e), zorder=3)
            for xi, yi in zip(xs, ys):
                if np.isfinite(yi):
                    axb.text(xi, yi, f'{yi:.0f}' if projected else f'{yi:+.0f}', ha='center',
                             va='bottom' if yi >= 0 else 'top', fontsize=6, zorder=4)
            allv += ys
        if g0 in gk:
            i0 = groups.index(g0)
            axb.axvspan(i0 - 0.46, i0 + 0.46, color='0.93', zorder=0, lw=0)
        axb.axhline(0, color='0.3', lw=0.6, zorder=2)
        fv = [y for y in allv if np.isfinite(y)]
        if fv:
            lo, hi = min(0, min(fv)), max(0, max(fv))
            axb.set_ylim(lo - 0.1 * (hi - lo) * (lo < 0), hi + 0.25 * ((hi - lo) or 1))

        tick = []
        for g in groups:
            if g == 'present':
                tick.append('Present\n(observed)')
                continue
            md = getv(g, 'median')
            fr = ca.loc[ca['GWL'] == g, 'agree_area_frac']
            fr = float(fr.iat[0]) if len(fr) else np.nan
            t = f'+{g} °C'
            if projected and np.isfinite(md):
                t += f'\nΔ {md:+.0f} d'
            if np.isfinite(fr):
                t += f'\nagree {fr:.0%}'
            tick.append(t)
        axb.set_xticks(x)
        axb.set_xticklabels(tick, fontsize=6.5)
        axb.set_ylabel('Days per year' if projected else 'Change (days per year)')
        for s_ in ('top', 'right'):
            axb.spines[s_].set_visible(False)
        axb.legend(loc='upper left', frameon=False, fontsize=6.5, ncol=1,
                   title='CLIMBRA projection (present + Δ)' if projected else 'CLIMBRA change',
                   title_fontsize=6.5)
        note = '' if method in ('', 'polygon', 'nan') else f' – {method.replace("buffer", "buffer ")}'
        sub = f'\n{name}' if len(name) < 45 else ''
        axb.set_title(f'(b) {cname} ({cuf}){note}{sub}', fontsize=8.5, loc='left')

    info = pd.DataFrame([dict(CD_MUN=cd, municipality=cname, state=cuf, GWL=g, ensemble=e,
                              present_observed=present, change=getv(g, e),
                              projected=np.clip(present + getv(g, e), 0, 365) if projected else np.nan,
                              agree_area_frac=(float(ca.loc[ca['GWL'] == g, 'agree_area_frac'].iat[0])
                                               if (ca['GWL'] == g).any() else np.nan))
                         for g in gk for e in ensembles])
    info.attrs.update(municipality=cname, state=cuf, present=present, variables=variables)
    return fig, (axm, axb), info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_common_args(ap)
    hp.add_plot_args(ap)
    ap.add_argument('--cd-mun', default='5300108', help='municipality of panel (b) (IBGE code; default Brasília)')
    ap.add_argument('--gwl', default='2', help='warming level of the map (default 2)')
    args = ap.parse_args(argv)

    hp.init_backend(args.show)
    cfg = load_config(test=args.test)
    fig, _, info = plot_municipal_days(cfg, cd_mun=args.cd_mun, gwl=args.gwl)
    hp.save_figure(fig, 'fig08_municipal_heat_days', cfg.paths.figures, args.formats, dpi=300)
    hp.finish_figure(fig, args.show)

    out = Path(cfg.paths.tables) / 'fig08_municipal_heat_days_city.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    info.to_csv(out, index=False)
    print(f'[saved] {out}')
    print(info.round(2).to_string(index=False))


if __name__ == '__main__':
    main()

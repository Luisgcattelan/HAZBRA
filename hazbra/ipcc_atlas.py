"""CLIMBRA ensemble vs IPCC AR6 WGI Interactive Atlas (CMIP6, 1° grid) -- used for paper Fig. 5.

Analyses kept (see ``03_paper_analysis/fig05_ipcc_atlas_comparison.py``)
  (1) conservative aggregation of CLIMBRA (0.25°) to the Atlas 1° grid (exact area overlaps,
      cos-latitude weights, valid-data fraction; cells with < MIN_COVER valid area are dropped)
  (2) pattern comparison per indicator / GWL: CLIMBRA, Atlas, difference, scatter, with
      area-weighted bias, RMSE and pattern correlation
  (3) agreement comparison: CLIMBRA robust (>= ROBUST_AREA_FRAC of the 1° cell with >= 66% model
      agreement) vs Atlas robust (consensus-based sign agreement >= AGREE_THRESHOLD), 5 categories
  (4) regional statistics (Brazil + IBGE macro-regions): bias, RMSE, r, sign agreement, robust-area
      shares and the five agreement categories (``atlas_regional_stats``), and regional means of the
      CLIMBRA p10/median/p90 vs the Atlas median (``regional_table``)

Atlas input folders (as downloaded from the Interactive Atlas), e.g.
  <cfg.paths.ipcc_atlas>/CMIP6 - Maximum 5-day precipitation (RX5day) Change % - Warming 3°C SSP5-8.5
      (rel. to 1981-2010) - Annual (33 models)/{map.nc, consensus.nc}

Functions take ``cfg`` (from ``hazbra.config.load_config``) for every path; nothing is read at import.

Pruned from the original ``ipcc_atlas_comparison.py`` (not used for Fig. 5): the per-indicator driver
``run`` / ``run_all`` and its figures ``plot_pattern``, ``plot_agreement``, ``plot_regional``; the
raw-CMIP6 decomposition ``decompose`` and model-list overlap ``model_overlap`` (analysis 6, plus
``CLIMBRA_MODELS``); the 0.25° comparison grid option (``grid='0.25'``, ``to_grid_nearest``); the
``sequential_one_sign``, ``levels`` and ``cmaps`` options of ``plot_atlas_rows``.
"""
from __future__ import annotations

import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from . import plotting as hp
from .plotting import indicators_dict, gkey as _gkey, std_latlon as _std_latlon

# =============================================================================
# Configuration
# =============================================================================
BOX = (-76.0, -32.0, -36.0, 7.0)          # lon_min, lon_max, lat_min, lat_max (crop of the global files)
MIN_COVER = 0.5                            # min valid CLIMBRA area fraction of a 1° cell
AGREE_THRESHOLD = 0.66                     # sign agreement threshold (CLIMBRA and Atlas)
ROBUST_AREA_FRAC = 0.5                     # CLIMBRA 1° cell robust if >= 50% of its area agrees
CONSENSUS_MODE = 'auto'                    # 'auto', 'fraction_positive' or 'agree_with_sign'
STATISTIC = 'median'                       # CLIMBRA ensemble statistic compared with the Atlas map
GRID_TAG = '1deg'                          # comparison grid: CLIMBRA aggregated conservatively to 1°

# Atlas index name (as in folder names) -> CLIMBRA indicator
ATLAS_TO_CLIMBRA = {
    'TXX': 'TXX', 'TNN': 'TNN', 'RX1DAY': 'Rx1day', 'RX5DAY': 'Rx5day', 'CDD': 'CDD', 'CWD': 'CWD',
    'SPI6': 'SPI6', 'SPI3': 'SPI3', 'PR': 'prcptot', 'PRCPTOT': 'prcptot', 'WSDI': 'WSDI',
}
VAR_ROW_LABELS = {'Rx5day': 'Rx5day', 'Rx1day': 'Rx1day', 'TXX': 'TXx', 'TNN': 'TNn', 'CDD': 'CDD',
                  'CWD': 'CWD', 'SPI6': 'SPI-6', 'prcptot': 'PRCPTOT', 'WSDI': 'WSDI'}

FOLDER_RE = re.compile(
    r'^(?P<project>\S+) - (?P<long>.+?) \((?P<index>[^)]+)\) Change (?P<units>.+?) - '
    r'Warming (?P<gwl>[\d.]+)\s*°?C (?P<scenario>SSP[\d.\-]+) \(rel\. to (?P<baseline>\d{4}-\d{4})\) - '
    r'(?P<season>.+?) \((?P<n_models>\d+) models\)$')

CAT_LABELS = {1: 'Robust in both, same sign', 2: 'Robust in both, opposite sign',
              3: 'Robust in CLIMBRA only', 4: 'Robust in Atlas only', 5: 'Robust in neither'}


def _atlas_dir(cfg):
    return hp.input_path(cfg, 'ipcc_atlas')


def _climbra_dir(cfg):
    return hp.input_path(cfg, 'netcdf_climbra')


# =============================================================================
# Helpers
# =============================================================================
def _crop(da):
    return da.sel(lon=slice(BOX[0], BOX[1]), lat=slice(BOX[2], BOX[3]))


def _edges(c):
    c = np.asarray(c, float)
    mid = (c[:-1] + c[1:]) / 2
    return np.r_[c[0] - (mid[0] - c[0]), mid, c[-1] + (c[-1] - mid[-1])]


def _overlap(fine_e, coarse_e):
    """(n_coarse, n_fine) overlap lengths between 1-D cell intervals."""
    lo = np.maximum(coarse_e[:-1, None], fine_e[None, :-1])
    hi = np.minimum(coarse_e[1:, None], fine_e[None, 1:])
    return np.clip(hi - lo, 0, None)


def _wstats(c, a, w):
    """Area-weighted bias (c - a), RMSE, correlation and means."""
    ok = np.isfinite(c) & np.isfinite(a) & np.isfinite(w) & (w > 0)
    if ok.sum() < 3:
        return dict(n_cells=int(ok.sum()), bias=np.nan, rmse=np.nan, r=np.nan,
                    mean_climbra=np.nan, mean_atlas=np.nan)
    c, a, w = c[ok], a[ok], w[ok]
    ws = w.sum()
    mc, ma = (w * c).sum() / ws, (w * a).sum() / ws
    d = c - a
    r = (w * (c - mc) * (a - ma)).sum() / np.sqrt((w * (c - mc) ** 2).sum() * (w * (a - ma) ** 2).sum())
    return dict(n_cells=int(ok.sum()), bias=float((w * d).sum() / ws),
                rmse=float(np.sqrt((w * d ** 2).sum() / ws)), r=float(r),
                mean_climbra=float(mc), mean_atlas=float(ma))


def _region_masks(cfg, lat, lon):
    """Brazil (dissolved 2024 mesh) + IBGE macro-region masks on a lat/lon grid (source names)."""
    return hp.region_masks(cfg, lat, lon, language='source', brazil='dissolved')


def _outline(cfg, ax):
    hp.get_regions(cfg).boundary.plot(ax=ax, color='0.25', linewidth=0.4)
    ax.set_xlim(-74.5, -34.3)
    ax.set_ylim(-34.2, 5.6)
    ax.set_aspect('equal')
    ax.set_axis_off()


# =============================================================================
# Atlas files
# =============================================================================
def parse_atlas_folder(name):
    m = FOLDER_RE.match(name.strip())
    if not m:
        return None
    d = m.groupdict()
    d['index_key'] = d['index'].upper()
    d['gwl'] = _gkey(d['gwl'])
    d['n_models'] = int(d['n_models'])
    d['relative'] = d['units'].strip() == '%'
    d['indicator'] = ATLAS_TO_CLIMBRA.get(d['index_key'])
    return d


def list_atlas(cfg):
    """Table of the Atlas folders found (and parsed) under ``cfg.paths.ipcc_atlas``."""
    rows = []
    for p in sorted(glob.glob(str(_atlas_dir(cfg) / '*'))):
        p = Path(p)
        if not p.is_dir():
            continue
        d = parse_atlas_folder(p.name)
        if d is None:
            print(f'[atlas] could not parse folder name: {p.name}')
            continue
        d['path'] = str(p)
        d['has_map'] = (p / 'map.nc').exists()
        d['has_consensus'] = (p / 'consensus.nc').exists()
        rows.append(d)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[['index', 'indicator', 'gwl', 'scenario', 'baseline', 'season', 'units', 'relative', 'n_models',
                 'has_map', 'has_consensus', 'path']]
    return df


def _first_var(ds):
    v = [k for k in ds.data_vars if {'lat', 'lon', 'latitude', 'longitude'} & set(ds[k].dims)]
    if not v:
        raise ValueError(f'no gridded variable in {list(ds.data_vars)}')
    return v[0]


def load_atlas(path):
    """Returns (map DataArray, consensus DataArray or None, attrs dict) cropped to Brazil."""
    path = Path(path)
    with xr.open_dataset(path / 'map.nc') as ds:
        var = _first_var(ds)
        m = _crop(_std_latlon(ds[var].load())).squeeze(drop=True)
        attrs = dict(ds.attrs, **ds[var].attrs, variable=var)
    c = None
    cp = path / 'consensus.nc'
    if cp.exists():
        with xr.open_dataset(cp) as ds:
            c = _crop(_std_latlon(ds[_first_var(ds)].load())).squeeze(drop=True)
        c = c.reindex_like(m, method='nearest', tolerance=1e-3)
    return m, c, attrs


def atlas_agreement(cons, val, mode=None):
    """
    Sign agreement (0.5–1) from the Atlas consensus.
    'fraction_positive': consensus = fraction of models with an increase -> agreement = max(c, 1-c)
    'agree_with_sign'  : consensus = fraction agreeing with the ensemble sign -> agreement = c
    'auto' decides from cells where the Atlas change is negative.
    """
    mode = mode or CONSENSUS_MODE
    c, v = np.asarray(cons, float), np.asarray(val, float)
    if mode == 'auto':
        neg = np.isfinite(c) & np.isfinite(v) & (v < 0)
        if neg.sum() >= 5:
            mode = 'fraction_positive' if np.mean(c[neg] < 0.5) > 0.5 else 'agree_with_sign'
        else:
            mode = 'agree_with_sign' if np.nanmin(c) >= 0.5 - 1e-9 else 'fraction_positive'
        print(f'[atlas] consensus interpreted as "{mode}"')
    agree = np.maximum(c, 1 - c) if mode == 'fraction_positive' else c
    return agree, mode


def _atlas_row(cfg, indicator, gwl, relative=None):
    """Atlas file for an indicator and GWL. relative=False -> absolute change (e.g. mm),
    True -> % change, None -> absolute if downloaded, otherwise %."""
    atl = list_atlas(cfg)
    rows = atl[(atl['indicator'] == indicator) & (atl['gwl'] == _gkey(gwl))] if not atl.empty else atl
    if rows.empty:
        raise FileNotFoundError(f'no Atlas file for {indicator} at GWL {gwl}')
    if relative is not None:
        sel = rows[rows['relative'] == bool(relative)]
        if sel.empty:
            kind = '% change' if relative else 'absolute change'
            raise FileNotFoundError(
                f'no Atlas {kind} file for {indicator} at GWL {gwl}; available: {list(rows["units"])}. '
                f'Download the "{"Change %" if relative else "Change"}" map from the Interactive Atlas.')
        rows = sel
    elif (~rows['relative']).any():
        rows = rows[~rows['relative']]
    r = rows.iloc[0]
    print(f'[atlas] {indicator} GWL {gwl}: using change in {r["units"]} ({r["n_models"]} models)')
    return r


# =============================================================================
# CLIMBRA
# =============================================================================
def load_climbra(cfg, indicator, relative, kind='climatology'):
    """Anomaly (ensemble, GWL, lat, lon) and agreement (GWL, lat, lon) on the 0.25° grid."""
    cat = indicators_dict[indicator]
    path = _climbra_dir(cfg) / cat / indicator / 'ensemble' / f'ensemble_anomaly_{kind}.nc'
    var = f'{indicator}_anom_perc' if relative else f'{indicator}_anom'
    agr = f'{indicator}_model_agreement'
    with xr.open_dataset(path) as ds:
        if var not in ds:
            raise KeyError(f'{var} not in {path}: {list(ds.data_vars)}')
        an = _std_latlon(ds[var].load())
        ag = _std_latlon(ds[agr].load()) if agr in ds else None
    an = an.assign_coords(GWL=[_gkey(g) for g in an['GWL'].values],
                          ensemble=[str(e) for e in an['ensemble'].values]).transpose('ensemble', 'GWL', 'lat', 'lon')
    if ag is not None:
        ag = ag.assign_coords(GWL=[_gkey(g) for g in ag['GWL'].values]).transpose('GWL', 'lat', 'lon')
    return an, ag


# =============================================================================
# (1) Conservative aggregation 0.25° -> 1°
# =============================================================================
def conservative_aggregate(da, lat_c, lon_c, min_cover=None):
    """
    Area-conservative mean of da (..., lat, lon) on the target cell centres.
    Returns (mean, cover) where cover = valid fine area / target cell area.
    """
    min_cover = MIN_COVER if min_cover is None else min_cover
    lat_f, lon_f = da['lat'].values, da['lon'].values
    le_f, oe_f = _edges(lat_f), _edges(lon_f)
    le_c, oe_c = _edges(lat_c), _edges(lon_c)
    Olat = _overlap(le_f, le_c) * np.cos(np.deg2rad(lat_f))[None, :]
    Olon = _overlap(oe_f, oe_c)
    X = da.values
    V = np.isfinite(X)
    X0 = np.where(V, X, 0.0)
    num = np.einsum('Ii,...ij,Jj->...IJ', Olat, X0, Olon)
    den = np.einsum('Ii,...ij,Jj->...IJ', Olat, V.astype(float), Olon)
    area_c = (np.diff(le_c) * np.cos(np.deg2rad(lat_c)))[:, None] * np.diff(oe_c)[None, :]
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(den > 0, num / den, np.nan)
        cover = den / area_c
    mean = np.where(cover >= min_cover, mean, np.nan)
    dims = da.dims[:-2] + ('lat', 'lon')
    coords = {d: da[d] for d in da.dims[:-2]}
    coords.update(lat=lat_c, lon=lon_c)
    return (xr.DataArray(mean, dims=dims, coords=coords, attrs=da.attrs),
            xr.DataArray(cover, dims=dims, coords=coords))


# =============================================================================
# Comparison core
# =============================================================================
def compare(cfg, indicator, atlas_row, statistic=None):
    """
    All quantities for one Atlas file (one GWL), on the Atlas 1° grid (CLIMBRA aggregated
    conservatively to the Atlas grid). Returns a dict.
    """
    statistic = statistic or STATISTIC
    amap, acons, aattrs = load_atlas(atlas_row['path'])
    an, ag = load_climbra(cfg, indicator, atlas_row['relative'])
    g = atlas_row['gwl']
    if g not in an['GWL'].values:
        raise KeyError(f'GWL {g} not in CLIMBRA ({list(an["GWL"].values)})')
    fine = an.sel(GWL=g)                                          # (ensemble, lat, lon)
    a_agree, mode = (atlas_agreement(acons.values, amap.values) if acons is not None else (None, None))
    if a_agree is not None:
        a_agree = xr.DataArray(a_agree, coords=amap.coords, dims=amap.dims)

    lat_c, lon_c = amap['lat'].values, amap['lon'].values
    agg, cover = conservative_aggregate(fine, lat_c, lon_c)
    c1 = agg.sel(ensemble=statistic)
    cov = cover.sel(ensemble=statistic).clip(0, 1)
    a1 = amap.where(np.isfinite(c1))
    w = np.cos(np.deg2rad(a1['lat'])).broadcast_like(a1) * cov

    res = dict(indicator=indicator, gwl=g, atlas=atlas_row, attrs=aattrs, grid=GRID_TAG,
               climbra_fine=fine.sel(ensemble=statistic),
               climbra_cmp=c1, atlas_cmp=a1, cover=cov, weights=w,
               stats=_wstats(c1.values, a1.values, w.values))
    if ag is not None and a_agree is not None:
        valid_f = np.isfinite(fine.sel(ensemble=statistic))
        agree_f = (np.isfinite(ag.sel(GWL=g)) & valid_f).astype(float).where(valid_f)
        frac, _ = conservative_aggregate(agree_f, lat_c, lon_c)
        res.update(climbra_agree_frac=frac, atlas_agree=a_agree, consensus_mode=mode)
    return res


def agreement_categories(res):
    """5-category map: 1 both robust same sign, 2 both robust opposite sign, 3 CLIMBRA only,
    4 Atlas only, 5 neither. NaN outside the common domain."""
    c1, a1 = res['climbra_cmp'].values, res['atlas_cmp'].values
    cr = res['climbra_agree_frac'].values >= ROBUST_AREA_FRAC
    ar = res['atlas_agree'].values >= AGREE_THRESHOLD
    ok = np.isfinite(c1) & np.isfinite(a1) & np.isfinite(res['climbra_agree_frac'].values)
    same = np.sign(c1) == np.sign(a1)
    cat = np.full(c1.shape, np.nan)
    cat[ok & cr & ar & same] = 1
    cat[ok & cr & ar & ~same] = 2
    cat[ok & cr & ~ar] = 3
    cat[ok & ~cr & ar] = 4
    cat[ok & ~cr & ~ar] = 5
    w = res['weights'].values
    shares = {k: float(np.nansum(np.where(cat == k, w, 0)) / np.nansum(np.where(ok, w, 0)) * 100)
              for k in range(1, 6)}
    return xr.DataArray(cat, coords=res['climbra_cmp'].coords, dims=res['climbra_cmp'].dims), shares


def regional_table(cfg, indicator, atlas_rows, statistic=None):
    """
    Regional means (Brazil + IBGE macro-regions) on the 0.25° grid: CLIMBRA p10/median/p90 and
    the Atlas 1° field assigned piecewise-constant to the 0.25° cells (equivalent to weighting
    each 1° cell by its overlap with the region).
    """
    statistic = statistic or STATISTIC
    rows = []
    for _, ar in atlas_rows.iterrows():
        an, _ = load_climbra(cfg, indicator, ar['relative'])
        amap, _, _ = load_atlas(ar['path'])
        g = ar['gwl']
        if g not in an['GWL'].values:
            continue
        fine = an.sel(GWL=g)
        lat_f, lon_f = fine['lat'].values, fine['lon'].values
        a_f = amap.sel(lat=xr.DataArray(lat_f, dims='lat'), lon=xr.DataArray(lon_f, dims='lon'),
                       method='nearest').values
        masks = _region_masks(cfg, lat_f, lon_f)
        w = np.cos(np.deg2rad(lat_f))[:, None] * np.ones((1, lon_f.size))
        valid = np.isfinite(fine.sel(ensemble=statistic).values) & np.isfinite(a_f)
        for reg, m in masks.items():
            ok = m & valid
            if not ok.any():
                continue
            ww = w[ok]
            rec = dict(indicator=indicator, region=reg, GWL=g, scenario=ar['scenario'],
                       atlas_n_models=ar['n_models'],
                       atlas=float((ww * a_f[ok]).sum() / ww.sum()))
            for e in ('p10', statistic, 'p90'):
                if e in fine['ensemble'].values:
                    x = fine.sel(ensemble=e).values
                    oke = ok & np.isfinite(x)
                    rec[f'climbra_{e}'] = float((w[oke] * x[oke]).sum() / w[oke].sum())
            rec['diff'] = rec[f'climbra_{statistic}'] - rec['atlas']
            rec['atlas_within_p10_p90'] = (rec.get('climbra_p10', -np.inf) <= rec['atlas'] <=
                                           rec.get('climbra_p90', np.inf))
            rows.append(rec)
    return pd.DataFrame(rows)


# =============================================================================
# Regional statistics
# =============================================================================
def _units_label(res):
    return '%' if res['atlas']['relative'] else res['atlas']['units'].replace('deg C', '°C')


def _region_fractions(cfg, res):
    """{region: fraction (0–1) of each comparison-grid cell inside the region}."""
    fine = res['climbra_fine']
    masks = _region_masks(cfg, fine['lat'].values, fine['lon'].values)
    c = res['climbra_cmp']
    out = {}
    for reg, m in masks.items():
        md = xr.DataArray(m.astype(float), dims=('lat', 'lon'),
                          coords={'lat': fine['lat'], 'lon': fine['lon']})
        frac, _ = conservative_aggregate(md, c['lat'].values, c['lon'].values, min_cover=0.0)
        out[reg] = np.nan_to_num(frac.values)
    return out


def atlas_regional_stats(cfg, results, savepath=None):
    """
    Table per indicator x region (Brazil + IBGE macro-regions) from compare() results:
    area means, bias, RMSE, pattern r, sign agreement of the median change, robust-area shares
    and the five agreement categories (% of the region's area).
    """
    results = results if isinstance(results, (list, tuple)) else [results]
    rows = []
    for res in results:
        c, a = res['climbra_cmp'].values, res['atlas_cmp'].values
        w0 = res['weights'].values
        has_agree = 'climbra_agree_frac' in res
        if has_agree:
            cat, _ = agreement_categories(res)
            cat = cat.values
            cr = res['climbra_agree_frac'].values >= ROBUST_AREA_FRAC
            ar = res['atlas_agree'].values >= AGREE_THRESHOLD
        for reg, f in _region_fractions(cfg, res).items():
            w = w0 * f
            ok = np.isfinite(c) & np.isfinite(a) & np.isfinite(w) & (w > 0)
            st = _wstats(c, a, w)
            rec = dict(indicator=res['indicator'], GWL=res['gwl'], grid=res.get('grid', GRID_TAG),
                       scenario=res['atlas']['scenario'], atlas_n_models=res['atlas']['n_models'],
                       region=reg, **st)
            ws = w[ok].sum()
            rec['sign_agree_pct'] = float((w[ok] * (np.sign(c[ok]) == np.sign(a[ok]))).sum() / ws * 100) if ws else np.nan
            if has_agree:
                okc = ok & np.isfinite(cat)
                wc = w[okc].sum()
                rec['robust_climbra_pct'] = float((w[okc] * cr[okc]).sum() / wc * 100) if wc else np.nan
                rec['robust_atlas_pct'] = float((w[okc] * ar[okc]).sum() / wc * 100) if wc else np.nan
                for k, lab in CAT_LABELS.items():
                    rec[lab] = float((w[okc] * (cat[okc] == k)).sum() / wc * 100) if wc else np.nan
                rec['robustness_match_pct'] = (rec[CAT_LABELS[1]] + rec[CAT_LABELS[5]]
                                               if wc else np.nan)
            rows.append(rec)
    df = pd.DataFrame(rows)
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(savepath, index=False)
    return df


# =============================================================================
# Figure 5
# =============================================================================
def plot_atlas_rows(cfg, indicators=('Rx5day', 'TXX'), gwl='3', statistic=None, relative=None,
                    vlims=None, dlims=None, n_levels=None, hatch='////', panel_width=1.72, gap=0.1,
                    title=False, table_path=None):
    """
    One row per indicator: (1) CLIMBRA, (2) AR6 Atlas, (3) CLIMBRA − Atlas, (4) scatter of cells.
    Maps 1–2 are hatched where model agreement is low (CLIMBRA: < ROBUST_AREA_FRAC of the cell with
    >= 66% agreement; Atlas: consensus-based sign agreement < AGREE_THRESHOLD).
    n_levels: number of colour classes per indicator, e.g. {'TXX': 24} (default 10).
    Returns fig, axes, results (list of compare() dicts), regional statistics table.
    """
    vlims, dlims = vlims or {}, dlims or {}
    n_levels = n_levels or {}
    rel = relative if isinstance(relative, dict) else {ind: relative for ind in indicators}
    results = [compare(cfg, ind, _atlas_row(cfg, ind, gwl, rel.get(ind)), statistic=statistic)
               for ind in indicators]

    nr = len(results)
    aspect = (-34.2 - 5.6) / (-74.5 - (-34.3))          # extent used by _outline (lat span / lon span)
    pw = panel_width
    ph = pw * abs(aspect)
    sw = ph                                             # scatter panel: square
    left, right = 0.45, 0.1
    top = 0.55 if title else 0.32
    t_h, cb_pad, cb_h, cb_lab = 0.22, 0.06, 0.08, 0.38
    rh = t_h + ph + cb_pad + cb_h + cb_lab
    gw = 3 * pw + 2 * gap + gap * 3.5 + sw
    W, H = left + gw + right, top + nr * rh + 0.05
    glab = '1°'
    letters = iter('abcdefghijklmnopqrstuvwxyz')

    rc = hp.pub_rc('0.25', **{'axes.titlesize': 8.5})
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(W, H))
        axes = np.empty((nr, 4), dtype=object)
        for i, res in enumerate(results):
            ind, u = res['indicator'], _units_label(res)
            c1, a1 = res['climbra_cmp'], res['atlas_cmp']
            d = c1 - a1
            both = np.r_[c1.values[np.isfinite(c1.values)], a1.values[np.isfinite(a1.values)]]
            vl = vlims.get(ind) or float(np.quantile(np.abs(both), 0.98))
            cmap = plt.get_cmap(hp.DIFF_CMAPS.get(indicators_dict[ind], 'RdBu_r'))
            lv = MaxNLocator(n_levels.get(ind, 10), symmetric=True).tick_values(-vl, vl)
            norm = BoundaryNorm(lv, cmap.N, extend='both')
            dl = dlims.get(ind) or float(np.nanquantile(np.abs(d.values), 0.98))
            dnorm = BoundaryNorm(MaxNLocator(10, symmetric=True).tick_values(-dl, dl), 256, extend='both')

            ytop = H - top - i * rh
            y0 = ytop - t_h - ph
            xs = [left, left + pw + gap, left + 2 * (pw + gap)]
            xsc = left + 3 * pw + 2 * gap + gap * 3.5
            lows = [None, None]
            if 'climbra_agree_frac' in res:
                lows = [(res['climbra_agree_frac'].values < ROBUST_AREA_FRAC) & np.isfinite(c1.values),
                        (res['atlas_agree'].values < AGREE_THRESHOLD) & np.isfinite(a1.values)]
            titles = [f'CLIMBRA ({statistic or STATISTIC}, {glab})',
                      f'AR6 Atlas ({res["atlas"]["n_models"]} models)', 'CLIMBRA − Atlas']
            for k, (da, nm, low) in enumerate(((c1, norm, lows[0]), (a1, norm, lows[1]), (d, dnorm, None))):
                ax = fig.add_axes([xs[k] / W, y0 / H, pw / W, ph / H])
                axes[i, k] = ax
                da.plot.pcolormesh(ax=ax, x='lon', y='lat', cmap=cmap if k < 2 else 'PuOr_r', norm=nm,
                                   add_colorbar=False, add_labels=False, rasterized=True)
                if low is not None and low.any():
                    hp.add_hatch(ax, da['lon'].values, da['lat'].values, low, hatch, '0.25', zorder=3)
                _outline(cfg, ax)
                ax.set_title(f'({next(letters)}) {titles[k]}', loc='left', fontsize=7.5)
            # colorbars: shared for CLIMBRA + Atlas, separate for the difference
            cax = fig.add_axes([(xs[0] + 0.1 * pw) / W, (y0 - cb_pad - cb_h) / H, (2 * pw + gap - 0.2 * pw) / W, cb_h / H])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal',
                              extend='both')
            cb.set_label(f'Change in {ind} at GWL {res["gwl"]} °C ({u})', fontsize=7)
            cb.ax.tick_params(labelsize=6.5, width=0.5, length=2)
            b = np.asarray(norm.boundaries)
            if b.size > 13:                                  # many classes: label every second boundary
                z = np.where(np.isclose(b, 0))[0]
                cb.set_ticks(b[(z[0] % 2 if z.size else 0)::2])          # keep 0 among the labels
                cb.ax.set_xticklabels([f'{t:g}' for t in cb.get_ticks()])
            cax2 = fig.add_axes([(xs[2] + 0.08 * pw) / W, (y0 - cb_pad - cb_h) / H, 0.84 * pw / W, cb_h / H])
            cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=dnorm, cmap='PuOr_r'), cax=cax2,
                               orientation='horizontal', extend='both')
            cb2.set_label(f'Difference ({u})', fontsize=7)
            cb2.ax.xaxis.set_major_locator(MaxNLocator(5, symmetric=True))
            cb2.ax.tick_params(labelsize=6.5, width=0.5, length=2)

            # scatter
            ax = fig.add_axes([xsc / W, y0 / H, sw / W, ph / H])
            axes[i, 3] = ax
            x, y, w = a1.values.ravel(), c1.values.ravel(), res['weights'].values.ravel()
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 4000:
                ax.hexbin(x[ok], y[ok], C=w[ok], reduce_C_function=np.sum, gridsize=40, bins='log',
                          cmap='Greys', mincnt=1, linewidths=0)
            else:
                ax.scatter(x[ok], y[ok], s=5 * w[ok] / np.nanmax(w[ok]), color='0.25', alpha=0.6, lw=0)
            lo, hi = float(min(x[ok].min(), y[ok].min())), float(max(x[ok].max(), y[ok].max()))
            pad = 0.05 * (hi - lo)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='tab:red', lw=0.7)
            ax.axhline(0, color='0.7', lw=0.4)
            ax.axvline(0, color='0.7', lw=0.4)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlabel(f'Atlas ({u})', fontsize=7)
            ax.set_ylabel(f'CLIMBRA ({u})', fontsize=7)
            ax.tick_params(labelsize=6.5, width=0.5, length=2)
            for s_ in ('top', 'right'):
                ax.spines[s_].set_visible(False)
            st = res['stats']
            extra = ''
            if 'climbra_agree_frac' in res:
                _, sh = agreement_categories(res)
                extra = f'\nrobust same sign {sh[1]:.0f}%'
            ax.text(0.04, 0.97, f'bias {st["bias"]:+.2f}\nRMSE {st["rmse"]:.2f}\nr = {st["r"]:.2f}{extra}',
                    transform=ax.transAxes, ha='left', va='top', fontsize=6)
            ax.set_title(f'({next(letters)}) cells ({glab})', loc='left', fontsize=7.5)

            # row label
            fig.text(0.1 / W, (y0 + ph / 2) / H, VAR_ROW_LABELS.get(ind, ind), rotation=90, ha='left',
                     va='center', fontsize=9, fontweight='bold')

        g = results[0]['gwl']
        if title:
            fig.text(left / W, (H - 0.08) / H,
                     f'CLIMBRA vs IPCC AR6 Interactive Atlas (CMIP6, {results[0]["atlas"]["scenario"]}): '
                     f'change at GWL {g} °C relative to {results[0]["atlas"]["baseline"]}',
                     ha='left', va='top', fontsize=8.5, fontweight='bold')
        fig.legend(handles=[mpatches.Patch(facecolor='white', edgecolor='0.25', hatch=hatch * 2, lw=0.4,
                                           label=f'Low agreement: CLIMBRA < {ROBUST_AREA_FRAC:.0%} of cell with '
                                                 f'≥ {AGREE_THRESHOLD:.0%} of models agreeing on sign; '
                                                 f'Atlas < {AGREE_THRESHOLD:.0%} of models agreeing')],
                   loc='upper left', bbox_to_anchor=(left / W - 0.005, 1 - (0.24 if title else 0.02) / H),
                   frameon=False, fontsize=6.3, handlelength=1.6, handleheight=1.1)

    table = atlas_regional_stats(cfg, results, savepath=table_path)
    return fig, axes, results, table

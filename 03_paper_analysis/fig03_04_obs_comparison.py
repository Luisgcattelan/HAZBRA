"""Paper Figures 3 and 4: pairwise comparison of the observational data sets (CHIRPS, MERGE, XAVIER).

Fig 3: Rx5day climatology.   Fig 4: 25-year return level of CDD.

Each figure is a 3 x 3 matrix of the data sets: diagonal = map of the data set (with the
area-weighted Brazil mean), upper triangle = difference maps (row - column), lower triangle =
density scatter (row vs. column) with the 1:1 line.  Area-weighted (cos latitude) bias, RMSE and
Pearson correlation are printed in the panels.  All data sets are regridded to the finest grid
(0.05°, nearest neighbour), masked to Brazil and to the cells valid in all data sets.

The companion CSV tables hold the same metrics (plus MAE, relative bias, MAPE) for Brazil and the
five IBGE macro-regions, lower triangle of the matrix (row - column).

Inputs  (cfg.paths): netcdf_obs  (<Hazard>/<Indicator>/<DATASET>_{climatology,return_period}.nc),
        south_america_limit, shp_regions
Outputs (cfg.paths.figures): fig03_Rx5day_climatology_pairwise.{pdf,png},
                             fig04_CDD_RL25_pairwise.{pdf,png}
        (cfg.paths.tables):  fig03_Rx5day_climatology_pairwise_stats.csv,
                             fig04_CDD_RL25_pairwise_stats.csv

Usage
    python 03_paper_analysis/fig03_04_obs_comparison.py [--test] [--only fig03|fig04]

--test : Fig 3 only (Rx5day); outputs below the test root.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from hazbra.config import load_config, add_common_args
from hazbra import plotting as hp
from hazbra.plotting import indicators_dict, PC

DATASETS = ['CHIRPS', 'MERGE', 'XAVIER']      # panel order of the figures
EXTENT = (-75, -34, -36, 6)


# ---------------------------------------------------------------------------
# Loading and regridding
# ---------------------------------------------------------------------------
def _load(indicator, dataset, kind, base_dir):
    """Main variable of ``<DATASET>_<kind>.nc`` (lon in -180..180) or None if the file is missing."""
    path = Path(base_dir) / indicators_dict[indicator] / indicator / f'{dataset}_{kind}.nc'
    if not path.exists():
        print(f'[missing] {path}')
        return None
    with xr.open_dataset(path) as ds:
        var = indicator if indicator in ds.data_vars else list(ds.data_vars)[0]
        return hp.standardise_lon(ds[var].load())


def _load_field(indicator, dataset, kind, return_period, extent, base_dir, buffer=1.0):
    """Load one 2-D field with standard lat/lon names, ascending coords, cropped to extent + buffer."""
    fname = 'climatology' if kind == 'climatology' else 'return_period'
    da = _load(indicator, dataset, fname, base_dir)
    if da is None:
        return None
    if kind != 'climatology':
        da = da.sel(return_period=return_period, method='nearest')
    lat, lon = hp.latlon_names(da)
    da = da.rename({k: v for k, v in {lat: 'lat', lon: 'lon'}.items() if k != v})
    units = da.attrs.get('units', '')
    da = da.squeeze(drop=True).reset_coords(drop=True).sortby('lat').sortby('lon')
    da = da.sel(lon=slice(extent[0] - buffer, extent[1] + buffer),
                lat=slice(extent[2] - buffer, extent[3] + buffer))
    da.attrs['units'] = units
    return da.transpose('lat', 'lon')


def _grid_res(da):
    return float(np.abs(np.diff(da['lon'].values)).mean())


def _regrid_to_common(fields, target='finest'):
    """Nearest-neighbour regrid of all fields onto the finest / coarsest / named dataset grid."""
    avail = {k: v for k, v in fields.items() if v is not None}
    res = {k: _grid_res(v) for k, v in avail.items()}
    if target == 'finest':
        ref = min(res, key=res.get)
    elif target == 'coarsest':
        ref = max(res, key=res.get)
    elif target in avail:
        ref = target
    else:
        raise ValueError(f'target must be finest, coarsest or one of {list(avail)}')
    tgt = avail[ref]
    out = {}
    for k, v in fields.items():
        if v is None or k == ref:
            out[k] = v
        else:
            out[k] = v.interp(lat=tgt['lat'].values, lon=tgt['lon'].values, method='nearest')
    return out, ref, res


def _prepare_fields(cfg, indicator, kind, return_period, datasets, base_dir, target,
                    clip_brazil, common_mask, extent, verbose=True):
    """Load, regrid to a common grid (nearest neighbour) and mask all datasets."""
    raw = {d: _load_field(indicator, d, kind, return_period, extent, base_dir) for d in datasets}
    if all(v is None for v in raw.values()):
        raise FileNotFoundError(f'No files found for {indicator} ({kind})')
    units = next((v.attrs.get('units', '') for v in raw.values() if v is not None), '')
    fields, ref, res = _regrid_to_common(raw, target)
    if verbose:
        print('native resolution (deg): ' + ', '.join(f'{k}={v:.3f}' for k, v in res.items())
              + f'  ->  common grid: {ref}')
    grid = fields[ref]
    mask = xr.ones_like(grid, dtype=bool)
    if clip_brazil:
        mask = mask & hp.brazil_mask(cfg, grid)
    if common_mask:
        for v in fields.values():
            if v is not None:
                mask = mask & np.isfinite(v)
    fields = {k: (v.where(mask) if v is not None else None) for k, v in fields.items()}
    return fields, units, ref


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _wstats(a, b):
    """Area-weighted bias (a - b), RMSE and pattern correlation (DataArrays on the same grid)."""
    valid = np.isfinite(a) & np.isfinite(b)
    w = hp.weights(a).where(valid)
    ws = w.sum()
    a_, b_ = a.where(valid), b.where(valid)
    d = a_ - b_
    bias = (w * d).sum() / ws
    rmse = np.sqrt((w * d ** 2).sum() / ws)
    am, bm = (w * a_).sum() / ws, (w * b_).sum() / ws
    cov = (w * (a_ - am) * (b_ - bm)).sum() / ws
    r = cov / np.sqrt(((w * (a_ - am) ** 2).sum() / ws) * ((w * (b_ - bm) ** 2).sum() / ws))
    return float(bias), float(rmse), float(r)


def _pair_metrics(a, b, w):
    """Area-weighted metrics of a (row) vs b (column, reference). Arrays of equal shape."""
    ok = np.isfinite(a) & np.isfinite(b) & (w > 0)
    keys = ['bias', 'rel_bias', 'MAE', 'RMSE', 'MAPE', 'r']
    if ok.sum() < 10:
        return {**dict.fromkeys(keys, np.nan), 'n_cells': int(ok.sum())}
    a, b, w = a[ok], b[ok], w[ok]
    ws = w.sum()
    d = a - b
    am, bm = (w * a).sum() / ws, (w * b).sum() / ws
    bias = (w * d).sum() / ws
    nz = np.abs(b) > 1e-6
    mape = 100 * (w[nz] * np.abs(d[nz] / b[nz])).sum() / w[nz].sum() if nz.any() else np.nan
    r = (w * (a - am) * (b - bm)).sum() / np.sqrt((w * (a - am) ** 2).sum() * (w * (b - bm) ** 2).sum())
    return {'bias': bias,
            'rel_bias': 100 * bias / bm if abs(bm) > 1e-6 else np.nan,
            'MAE': (w * np.abs(d)).sum() / ws,
            'RMSE': np.sqrt((w * d ** 2).sum() / ws),
            'MAPE': mape, 'r': r, 'n_cells': int(ok.sum())}


def pairwise_region_stats(cfg, indicator, base_dir, kind='climatology', return_period=25,
                          include_brazil=True, datasets=DATASETS, target='finest', extent=EXTENT):
    """
    Long-format DataFrame of pairwise metrics (row - column, lower triangle) per region.
    Same regridding and masking as plot_pairwise.
    """
    fields, units, ref = _prepare_fields(cfg, indicator, kind, return_period, datasets, base_dir,
                                         target, True, True, list(extent))
    grid = fields[ref]
    w = hp.weights(grid).values
    masks = {'Brazil': np.ones_like(w, dtype=bool)} if include_brazil else {}
    masks.update(hp.region_masks(cfg, grid['lat'].values, grid['lon'].values, include_brazil=False))

    rows = []
    for reg, m in masks.items():
        wm = np.where(m, w, 0.0)
        for i, a in enumerate(datasets):
            for j, b in enumerate(datasets):
                if i <= j or fields[a] is None or fields[b] is None:
                    continue
                rows.append({'indicator': indicator, 'kind': kind,
                             'return_period': return_period if kind == 'return_period' else np.nan,
                             'region': reg, 'row': a, 'col': b, 'pair': f'{a} − {b}',
                             **_pair_metrics(fields[a].values, fields[b].values, wm)})
    df = pd.DataFrame(rows)
    df.attrs['units'] = units
    return df


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_pairwise(cfg, indicator, base_dir, kind='climatology', return_period=25, datasets=DATASETS,
                  target='finest', clip_brazil=True, common_mask=True, extent=EXTENT,
                  n_levels=None, show_stats=True, panel_labels=True, panel_width=2.0, gap=0.1,
                  show_regions=True, region_kw=None):
    """
    Panel (i, j) = row dataset - column dataset: diagonal = field, upper = difference,
    lower = density scatter (row vs column). panel_width / gap in inches. Returns fig, axes, stats.
    """
    if indicator not in indicators_dict:
        raise KeyError(f'{indicator} not in indicators_dict')
    if kind not in ('climatology', 'return_period'):
        raise ValueError("kind must be 'climatology' or 'return_period'")
    category = indicators_dict[indicator]
    extent = list(extent)

    fields, units, ref = _prepare_fields(cfg, indicator, kind, return_period, datasets, base_dir,
                                         target, clip_brazil, common_mask, extent)

    # ---- differences and stats ----
    n = len(datasets)
    diff, stats = {}, {}
    for i, a in enumerate(datasets):
        for j, b in enumerate(datasets):
            if i == j or fields[a] is None or fields[b] is None:
                continue
            diff[(i, j)] = fields[a] - fields[b]
            stats[(a, b)] = _wstats(fields[a], fields[b])

    def _abs_q(arrs, q=0.98):
        v = [np.abs(x.values[np.isfinite(x.values)]) for x in arrs]
        v = np.concatenate(v) if v else np.array([])
        return float(np.quantile(v, q)) if v.size else 1.0

    f_cmap = plt.get_cmap(hp.INDICATOR_CMAPS.get(indicator, hp.CATEGORY_CMAPS.get(category, 'viridis')))
    d_cmap = plt.get_cmap(hp.DIFF_CMAPS_IND.get(indicator, hp.DIFF_CMAPS.get(category, 'RdBu_r')))
    fvmin, fvmax = hp.limits(list(fields.values()), indicator in hp.DIVERGING)
    f_norm = hp.make_norm(fvmin, fvmax, f_cmap, n_levels)
    dlim = _abs_q(diff.values())
    d_norm = hp.make_norm(-dlim, dlim, d_cmap, n_levels, symmetric=True)

    allv = np.concatenate([v.values[np.isfinite(v.values)] for v in fields.values() if v is not None])
    lo, hi = np.quantile(allv, [0.001, 0.999])

    # ---- fixed layout (inches): every panel same size, with the map's aspect ratio ----
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    pw, ph = panel_width, panel_width * aspect
    left, right, top, bottom = 0.75, 0.95, 0.35, 0.95
    gw, gh = n * pw + (n - 1) * gap, n * ph + (n - 1) * gap
    W, H = left + gw + right, top + gh + bottom

    def rect(i, j):
        x = left + j * (pw + gap)
        y = bottom + gh - (i + 1) * ph - i * gap
        return [x / W, y / H, pw / W, ph / H]

    letters = iter('abcdefghijklmnopqrstuvwxyz')
    with plt.rc_context(hp.PUB_RC):
        fig = plt.figure(figsize=(W, H))
        axes = np.empty((n, n), dtype=object)
        used = {'field': False, 'diff': False}

        for i, a in enumerate(datasets):
            for j, b in enumerate(datasets):
                role = 'field' if i == j else ('diff' if i < j else 'scatter')
                A, B = fields[a], fields[b]

                if role == 'scatter':
                    ax = fig.add_axes(rect(i, j))
                    if A is None or B is None:
                        ax.text(0.5, 0.5, 'Not available', ha='center', va='center',
                                transform=ax.transAxes, color='0.4', fontsize=8)
                    else:
                        ok = np.isfinite(A.values) & np.isfinite(B.values)
                        ax.hexbin(B.values[ok], A.values[ok], gridsize=45, mincnt=1, bins='log',
                                  cmap='Greys', extent=(lo, hi, lo, hi), linewidths=0,
                                  rasterized=True)
                        ax.plot([lo, hi], [lo, hi], color='tab:red', lw=0.6)
                        if show_stats:
                            bias, rmse, r = stats[(a, b)]
                            hp.stat_box(ax, f'bias {hp.fmt(bias, True)}\nRMSE {hp.fmt(rmse)}\nr = {r:.2f}',
                                        loc='tl')
                    ax.set_xlim(lo, hi)
                    ax.set_ylim(lo, hi)
                    ax.xaxis.set_major_locator(MaxNLocator(4))
                    ax.yaxis.set_major_locator(MaxNLocator(4))
                    ax.tick_params(labelbottom=(i == n - 1), labelleft=(j == 0),
                                   labelsize=6, width=0.5, length=2)
                    for s in ax.spines.values():
                        s.set_linewidth(0.5)

                else:
                    ax = fig.add_axes(rect(i, j), projection=PC)
                    hp.base_map(ax, cfg, extent, clip_brazil, show_regions, region_kw)
                    data = {'field': A, 'diff': diff.get((i, j))}[role]
                    norm, cmap = {'field': (f_norm, f_cmap), 'diff': (d_norm, d_cmap)}[role]
                    if data is None:
                        ax.text(0.5, 0.5, 'Not available', ha='center', va='center',
                                transform=ax.transAxes, color='0.4', fontsize=8)
                    else:
                        data.plot.pcolormesh(ax=ax, x='lon', y='lat', transform=PC, cmap=cmap,
                                             norm=norm, add_colorbar=False, add_labels=False,
                                             rasterized=True, zorder=1)
                        ax.set_title('')
                        if show_stats:
                            if role == 'field':
                                hp.stat_box(ax, f'mean {hp.fmt(hp.wmean(data))}')
                            else:
                                bias, rmse, r = stats[(a, b)]
                                hp.stat_box(ax, f'bias {hp.fmt(bias, True)}\nRMSE {hp.fmt(rmse)}\nr = {r:.2f}')
                    used[role] = True

                axes[i, j] = ax
                if panel_labels:
                    ax.text(0.03, 0.97, f'({next(letters)})', transform=ax.transAxes,
                            ha='left', va='top', fontsize=8, fontweight='bold', zorder=6)

        # ---- row / column labels at figure level ----
        for k, name in enumerate(datasets):
            fig.text((left + k * (pw + gap) + pw / 2) / W, (bottom + gh + 0.06) / H, name,
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
            fig.text(0.1 / W, (bottom + gh - k * (ph + gap) - ph / 2) / H, name, rotation=90,
                     ha='left', va='center', fontsize=9, fontweight='bold')

        # ---- colorbars in dedicated axes ----
        u = f' ({units})' if units else ''
        if used['diff']:
            cax = fig.add_axes([(left + gw + 0.15) / W, (bottom + 0.075 * gh) / H,
                                0.11 / W, 0.85 * gh / H])
            hp.cbar(fig, cax, d_norm, d_cmap, 'vertical', f'Δ {indicator}, row − column{u}')
        if used['field']:
            bw = 0.6 * gw
            cax = fig.add_axes([(left + (gw - bw) / 2) / W, 0.42 / H, bw / W, 0.1 / H])
            hp.cbar(fig, cax, f_norm, f_cmap, 'horizontal', f'{indicator}{u}')

    return fig, axes, stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
# name, indicator, kind, return period, figure stem
FIGURES = {
    'fig03': dict(indicator='Rx5day', kind='climatology', return_period=25,
                  stem='fig03_Rx5day_climatology_pairwise'),
    'fig04': dict(indicator='CDD', kind='return_period', return_period=25,
                  stem='fig04_CDD_RL25_pairwise'),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    add_common_args(ap)
    hp.add_plot_args(ap)
    ap.add_argument('--only', choices=list(FIGURES), default=None, help='produce a single figure')
    args = ap.parse_args(argv)

    hp.init_backend(args.show)
    cfg = load_config(test=args.test)
    base_dir = hp.input_path(cfg, 'netcdf_obs')
    todo = [args.only] if args.only else (['fig03'] if args.test else list(FIGURES))

    for key in todo:
        spec = FIGURES[key]
        print(f'=== {key}: {spec["indicator"]} {spec["kind"]} ===')
        fig, _, _ = plot_pairwise(cfg, spec['indicator'], base_dir, kind=spec['kind'],
                                  return_period=spec['return_period'])
        hp.save_figure(fig, spec['stem'], cfg.paths.figures, args.formats)
        hp.finish_figure(fig, args.show)

        df = pairwise_region_stats(cfg, spec['indicator'], base_dir, kind=spec['kind'],
                                   return_period=spec['return_period'])
        out = Path(cfg.paths.tables) / f'{spec["stem"]}_stats.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f'[saved] {out}')
        print(df.round(2).to_string(index=False))


if __name__ == '__main__':
    main()

"""Plotting / geometry helpers shared by the paper figure scripts (``03_paper_analysis/``).

Everything that used to be copy-pasted between the figure notebooks lives here once:
publication rcParams, colour-map tables, grid helpers (lat/lon standardisation, Brazil and
macro-region masks, area weights), the cartopy base map, hatching, colour-bar helpers and
the small text formatters used in statistic boxes.

Importing this module never touches the file system: shapefiles are read lazily (and cached)
from ``cfg.paths`` the first time a function needs them, and the matplotlib backend is only
chosen when a script calls :func:`init_backend`.

Geometry sources
----------------
``brazil='limit'``      ``cfg.paths.south_america_limit`` (IBGE 2021 national limit; used for the
                        Brazil mask / outline of the observation and projection figures)
``brazil='dissolved'``  ``cfg.paths.shp_brazil`` (dissolved 2024 mesh; used by the Atlas comparison
                        and the municipal extraction)
The two files describe the same country with slightly different coastlines, so the choice is
kept as an explicit argument to preserve the figures as they were first produced.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import xarray as xr
import geopandas as gpd
import shapely
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .constants import INDICATOR_HAZARD

PC = ccrs.PlateCarree()

# ---------------------------------------------------------------------------
# Constants (colour maps, labels, rcParams)
# ---------------------------------------------------------------------------
indicators_dict = INDICATOR_HAZARD      # indicator -> hazard category (single source: constants.py)

REGION_ORDER = ['N', 'NE', 'CO', 'SE', 'S']
REGION_NAMES_EN = {'N': 'North', 'NE': 'Northeast', 'CO': 'Central-West', 'SE': 'Southeast', 'S': 'South'}
REGION_NAME_COL = 'NM_REGIA'            # region name column of the IBGE macro-region shapefile

CATEGORY_CMAPS = {'Rainfall': 'YlGnBu', 'Heat': 'YlOrRd', 'Drought': 'YlOrBr', 'Compound': 'magma_r'}
INDICATOR_CMAPS = {'SPI3': 'BrBG', 'SPI6': 'BrBG', 'TNN': 'RdYlBu_r', 'Frost': 'Blues'}
# difference / change colour maps: positive = wetter / hotter / drier
DIFF_CMAPS = {'Rainfall': 'BrBG', 'Heat': 'RdBu_r', 'Drought': 'BrBG_r', 'Compound': 'PuOr_r'}
DIFF_CMAPS_IND = {'SPI3': 'BrBG', 'SPI6': 'BrBG', 'Frost': 'RdBu'}
DIVERGING = {'SPI3', 'SPI6'}            # standardised indices (no % change)

VAR_LABELS = {
    'TXX': 'TXx', 'TNN': 'TNn', 'Frost': 'Frost days', 'WSDI': 'WSDI',
    'HI': 'Heat index', 'FWI': 'Fire Weather Index', 'Dry_Hot': 'Dry–hot days',
    'Rx1day': 'Rx1day', 'Rx5day': 'Rx5day', 'CDD': 'CDD', 'CDD_season': 'CDD (season)',
    'CWD': 'CWD', 'prcptot': 'PRCPTOT', 'pr_season': 'Seasonal rainfall',
    'SPI3': 'SPI-3', 'SPI6': 'SPI-6',
}
UNIT_PRETTY = {'degC': '°C', 'deg_C': '°C', 'celsius': '°C', 'Celsius': '°C', 'C': '°C',
               'day': 'days', 'd': 'days', '1': ''}
ENS_LABELS = {'p10': '10th percentile', 'median': 'Median', 'mean': 'Mean', 'p90': '90th percentile'}

PUB_RC = {
    'font.family': 'sans-serif', 'font.size': 8,
    'axes.titlesize': 9, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'savefig.dpi': 300, 'figure.dpi': 150,
}


def pub_rc(hatch_color=None, hatch_linewidth=0.3, **overrides):
    """PUB_RC copy; ``hatch_color`` adds the hatch settings (e.g. '0.25'), ``overrides`` replace keys."""
    rc = dict(PUB_RC)
    if hatch_color is not None:
        rc.update({'hatch.linewidth': hatch_linewidth, 'hatch.color': hatch_color})
    rc.update(overrides)
    return rc


# ---------------------------------------------------------------------------
# Configuration / output helpers
# ---------------------------------------------------------------------------
def init_backend(show: bool = False):
    """Select the non-interactive Agg backend unless ``show`` is requested. Call from ``main()``."""
    if not show:
        mpl.use('Agg')


PRODUCT_KEYS = ('netcdf_climbra', 'netcdf_obs', 'municipal')     # the HAZBRA data product (cfg.paths keys)


def input_cfg(cfg):
    """Configuration to *read* the data product from.

    In ``--test`` mode ``config/test.yaml`` redirects the product paths (``PRODUCT_KEYS``) below the
    test root, where the calculation tests only create tiny-box, few-model files.  The paper
    figures are tested on the real product instead: in test mode this returns the production
    configuration (read-only use), otherwise ``cfg`` itself.  Outputs always use ``cfg``.
    """
    if cfg.test.enabled:
        from .config import load_config
        return load_config(test=False)
    return cfg


def input_path(cfg, key: str) -> Path:
    """``cfg.paths.<key>`` for an *input*; the production location for the data-product keys in test mode."""
    if key in PRODUCT_KEYS:
        return Path(getattr(input_cfg(cfg).paths, key))
    return Path(getattr(cfg.paths, key))


def save_figure(fig, stem: str, out_dir, formats=('pdf', 'png'), **savefig_kw):
    """Save ``fig`` as ``<out_dir>/<stem>.<fmt>`` for every format; returns the written paths.

    Extra keyword arguments (e.g. ``dpi=300``) go to ``fig.savefig``; the default is ``bbox_inches='tight'``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out_dir / f'{stem}.{fmt}'
        fig.savefig(p, **{'bbox_inches': 'tight', **savefig_kw})
        paths.append(p)
        print(f'[saved] {p}')
    return paths


def finish_figure(fig, show: bool = False):
    """``plt.show()`` when requested (interactive use), otherwise release the figure."""
    if show:
        plt.show()
    else:
        plt.close(fig)


def add_plot_args(parser):
    """Arguments shared by all figure scripts (in addition to ``add_common_args``)."""
    parser.add_argument('--show', action='store_true', help='display figures (plt.show) instead of only saving')
    parser.add_argument('--formats', nargs='+', default=['pdf', 'png'], metavar='EXT',
                        help='figure file formats (default: pdf png)')
    return parser


# ---------------------------------------------------------------------------
# Geometry (lazy, cached)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _read_vector(path: str) -> gpd.GeoDataFrame:
    return gpd.read_file(path).to_crs(4326)


def _union(gdf):
    geom = gdf.geometry.union_all() if hasattr(gdf.geometry, 'union_all') else gdf.geometry.unary_union
    shapely.prepare(geom)
    return geom


def _brazil_path(cfg, brazil):
    key = {'limit': 'south_america_limit', 'dissolved': 'shp_brazil'}[brazil]
    return str(input_path(cfg, key))


def brazil_gdf(cfg, brazil='limit') -> gpd.GeoDataFrame:
    return _read_vector(_brazil_path(cfg, brazil))


@functools.lru_cache(maxsize=None)
def _union_cached(path: str):
    return _union(_read_vector(path))


def brazil_geom(cfg, brazil='limit'):
    """Prepared shapely geometry of Brazil (see module docstring for ``brazil``)."""
    return _union_cached(_brazil_path(cfg, brazil))


def get_regions(cfg) -> gpd.GeoDataFrame:
    """IBGE macro-regions ordered N, NE, CO, SE, S."""
    g = _read_vector(str(input_path(cfg, 'shp_regions')))
    if 'SIGLA_RG' in g.columns:
        g = (g.assign(_o=g['SIGLA_RG'].map({s: k for k, s in enumerate(REGION_ORDER)}))
              .sort_values('_o').drop(columns='_o').reset_index(drop=True))
    return g


def get_states(cfg) -> gpd.GeoDataFrame:
    return _read_vector(str(input_path(cfg, 'shp_states')))


def region_masks(cfg, lat, lon, language='source', brazil='limit', include_brazil=True):
    """Boolean masks (cell centre inside polygon) for Brazil and the five macro-regions.

    language='source' -> names from the shapefile (Norte, Nordeste, ...); 'en' -> English names.
    """
    lon2d, lat2d = np.meshgrid(lon, lat)
    masks = {}
    if include_brazil:
        masks['Brazil'] = shapely.contains_xy(brazil_geom(cfg, brazil), lon2d, lat2d)
    reg = get_regions(cfg)
    if language == 'en' and 'SIGLA_RG' in reg.columns:
        names = reg['SIGLA_RG'].map(REGION_NAMES_EN)
    else:
        names = reg[REGION_NAME_COL]
    for name, geom in zip(names, reg.geometry):
        shapely.prepare(geom)
        masks[str(name)] = shapely.contains_xy(geom, lon2d, lat2d)
    return masks


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------
def latlon_names(da):
    lat = next((n for n in ('lat', 'latitude', 'y') if n in da.coords), None)
    lon = next((n for n in ('lon', 'longitude', 'x') if n in da.coords), None)
    if lat is None or lon is None:
        raise ValueError(f'Could not find lat/lon coordinates in {list(da.coords)}')
    return lat, lon


def standardise_lon(da):
    """Convert 0–360 longitudes to -180–180 (sorted)."""
    lat, lon = latlon_names(da)
    if float(da[lon].max()) > 180:
        da = da.assign_coords({lon: ((da[lon] + 180) % 360) - 180}).sortby(lon)
    return da


def std_latlon(da):
    """Rename to lat/lon, -180–180 longitudes, ascending coordinates."""
    lat, lon = latlon_names(da)
    da = da.rename({k: v for k, v in {lat: 'lat', lon: 'lon'}.items() if k != v})
    return standardise_lon(da).sortby('lat').sortby('lon')


def brazil_mask(cfg, da, brazil='limit'):
    """Boolean DataArray: True where the grid-cell centre lies inside Brazil."""
    lon2d, lat2d = np.meshgrid(da['lon'].values, da['lat'].values)
    return xr.DataArray(shapely.contains_xy(brazil_geom(cfg, brazil), lon2d, lat2d),
                        dims=('lat', 'lon'), coords={'lat': da['lat'], 'lon': da['lon']})


def weights(da):
    """cos(latitude) area weights broadcast to the field."""
    return np.cos(np.deg2rad(da['lat'])).broadcast_like(da)


def wmean(x):
    """Area-weighted mean ignoring NaNs."""
    w = weights(x).where(np.isfinite(x))
    return float((x * w).sum() / w.sum())


def gkey(x):
    """Normalise GWL labels: '2', 2, 2.0 -> '2'; '1.5' -> '1.5' (non-numeric labels pass through)."""
    try:
        return f'{float(x):g}'
    except (TypeError, ValueError):
        return str(x)


def relative_ok(indicator):
    """% change is meaningful for rainfall/drought amounts and durations, not for °C or SPI."""
    return indicators_dict.get(indicator) in ('Rainfall', 'Drought') and indicator not in DIVERGING


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def limits(arrays, diverging=False, q=(0.02, 0.98)):
    """Colour limits from the pooled quantiles of several DataArrays (None entries ignored)."""
    vals = [a.values.ravel() for a in arrays if a is not None]
    if not vals:
        return None, None
    vals = np.concatenate(vals)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    vmin, vmax = np.quantile(vals, q)
    if diverging:
        m = max(abs(vmin), abs(vmax))
        return -m, m
    return vmin, vmax


def make_norm(vmin, vmax, cmap, n_levels=None, symmetric=False):
    if vmin is None:
        return Normalize()
    if n_levels:
        levels = MaxNLocator(n_levels, symmetric=symmetric).tick_values(vmin, vmax)
        return BoundaryNorm(levels, cmap.N, extend='both')
    return Normalize(vmin, vmax)


def fmt(x, sign=False):
    f = '.1f' if abs(x) >= 1 else '.2f'
    return format(x, ('+' if sign else '') + f)


def stat_box(ax, text, loc='br'):
    x, ha = (0.97, 'right') if loc == 'br' else (0.03, 'left')
    y, va = (0.03, 'bottom') if loc == 'br' else (0.86, 'top')
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=6, zorder=6,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))


def cbar(fig, cax, norm, cmap, orientation, label):
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation=orientation, extend='both')
    cb.set_label(label, fontsize=8)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(width=0.5, length=2.5, labelsize=7)
    return cb


def base_map(ax, cfg, extent, clip_brazil=True, show_regions=True, region_kw=None, brazil='limit'):
    """Cartopy base map: land/ocean, coastlines, borders, macro-region and Brazil outlines."""
    ax.set_extent(extent, crs=PC)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='0.93', edgecolor='none', zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='white', edgecolor='none', zorder=0)
    ax.coastlines('50m', linewidth=0.4, color='0.2', zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.3, edgecolor='0.4', zorder=3)
    if show_regions:
        kw = dict(facecolor='none', edgecolor='0.15', linewidth=0.45, zorder=4)
        kw.update(region_kw or {})
        ax.add_geometries(get_regions(cfg).geometry, crs=PC, **kw)
    if clip_brazil:
        ax.add_geometries(brazil_gdf(cfg, brazil).geometry, crs=PC, facecolor='none',
                          edgecolor='k', linewidth=0.6, zorder=5)
    ax.spines['geo'].set_linewidth(0.5)


def add_hatch(ax, lon, lat, mask, hatch='////', color='0.25', transform=None, zorder=2):
    """Hatch cells where ``mask`` == 1 (0/1 array on the lon/lat grid).

    ``transform=PC`` on cartopy axes, ``None`` on plain axes.
    """
    kw = {} if transform is None else {'transform': transform}
    cs = ax.contourf(lon, lat, np.asarray(mask, dtype=float), levels=[0.5, 1.5], colors='none',
                     hatches=[hatch], zorder=zorder, **kw)
    try:                                   # matplotlib >= 3.8
        cs.set_edgecolor(color)
        cs.set_linewidth(0)
    except AttributeError:                 # older matplotlib
        for c in cs.collections:
            c.set_edgecolor(color)
            c.set_linewidth(0)
    return cs


def add_hatch_da(ax, mask, hatch='////', color='0.25'):
    """:func:`add_hatch` for a 0/1 DataArray on a cartopy axis."""
    return add_hatch(ax, mask['lon'], mask['lat'], mask.values, hatch, color, transform=PC)

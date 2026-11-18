"""
Municipal-level extraction of gridded climate hazard indicators for Brazil.

For every IBGE municipality (2024 mesh) and every NetCDF file of an indicator
(observations: CHIRPS, MERGE, XAVIER; CLIMBRA: ensemble files), all data variables and
all combinations of their non-spatial dimensions (ensemble, GWL, return_period, ...)
are summarised with area-weighted zonal statistics.

Method (per grid, cached):
  * exact fractional pixel coverage of each municipality polygon (area weights)
  * adaptive 25 km buffer (clipped to Brazil) when the polygon has < MIN_PIXELS valid
    pixel-equivalents or < MIN_VALID_FRAC of its area with valid data
  * nearest valid pixel (<= NEAREST_MAX_KM) as last resort
  * every value records the method used

Paths (all from ``cfg.paths``, resolved lazily by :func:`get_paths`)
  inputs : netcdf_obs / netcdf_climbra  (HAZBRA NetCDF product), shp_municipalities, shp_brazil
  outputs: municipal (= <out_dir> below)

Outputs under <out_dir> (layout identical to the published product):
  geometry/municipalities_2024.gpkg        cleaned geometries (+ buffer layer)
  geometry/removed_parts.csv               lagoons and oceanic islands removed
  geometry/states_2024.gpkg                state outlines (union of the municipalities; used by the maps)
  weights/                                 cached sparse pixel weights per grid
  Observations/<Category>/<Indicator>/<DATASET>_<kind>.parquet
  CLIMBRA/<Category>/<Indicator>/<file stem>.parquet
  maps/<Indicator>_municipalities.gpkg     wide, ready-to-map table (+ _columns.csv)
  logs/checks.csv                          consistency checks and method counts
  README.md                                description of the data (regenerated each run)

Command line (also wrapped by ``02_postprocess/04_municipal_aggregation.py``)
    python -m hazbra.municipal --indicators Rx5day TXX      # selected indicators
    python -m hazbra.municipal                              # all indicators
    python -m hazbra.municipal --test                       # TXX only, written below the test root

``--test`` reads the NetCDF product, the cleaned geometry and the cached pixel weights from the
production locations (read-only) and writes everything else under the test root.

Python usage
    from hazbra import municipal as mun
    mun.run_municipalities(['Rx5day'])       # one indicator (cfg = production config)
    df = mun.load_table('Rx5day', source='CLIMBRA')
"""
from __future__ import annotations

import os
import glob
import time
import datetime
import itertools
import warnings
import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely
from shapely.errors import GEOSException
from scipy import sparse
from scipy.spatial import cKDTree

from .constants import INDICATOR_HAZARD, OBS_DATASETS


# =============================================================================
# Configuration (method parameters; paths come from cfg via get_paths)
# =============================================================================
KEEP_COLS = ['CD_MUN', 'NM_MUN', 'NM_UF']
LAGOON_CODES = {'4300001', '4300002'}      # Lagoa Mirim, Lagoa dos Patos (RS): not municipalities
ISLAND_LON = -34.0                         # polygon parts east of this longitude are removed
EQUAL_AREA_CRS = 'ESRI:102033'             # South America Albers Equal Area Conic

BUFFER_KM = 25.0
NEAREST_MAX_KM = 50.0
MIN_PIXELS = 1.0              # valid pixel-equivalents required inside the polygon
MIN_VALID_FRAC = 0.5          # share of polygon area with valid data required
MINMAX_MIN_FRAC = 0.05        # pixels covering < 5% of their area are ignored for min/max
ROBUST_AREA_FRAC = 0.5        # agreement: robust if >= 50% of area has >= 66% model agreement
P_SIG = 0.05                  # trend significance level
TREND_SLOPE_VAR, TREND_P_VAR = 'slopes', 'p_values'

WIDE_ENSEMBLE = 'median'      # CLIMBRA statistic used in the wide GeoPackage
WIDE_RP = 25                  # CLIMBRA return period used in the wide GeoPackage
WRITE_CSV = False             # also write CSV copies of the long tables

indicators_dict = INDICATOR_HAZARD
DIVERGING = {'SPI3', 'SPI6'}

META_COLS = ['source', 'dataset', 'indicator', 'category', 'file_kind', 'variable', 'units']
STAT_COLS = ['mean', 'std', 'min', 'max', 'n_pixels', 'valid_area_frac', 'method',
             'nearest_km', 'agree_area_frac', 'robust', 'sig_area_frac', 'mean_sig']
R_EARTH = 6371.0
KM_PER_DEG2 = 111.32 ** 2
TEST_INDICATORS = ['TXX']     # indicators processed by --test
TEST_SKIP_KINDS = ('timeseries',)   # file kinds skipped by --test (slowest files, same code path)


def get_paths(cfg) -> SimpleNamespace:
    """All file locations used by this module, as strings, derived from ``cfg.paths``.

    In ``--test`` mode the outputs go below the test root; an indicator's NetCDF files, the cleaned
    geometry, the pixel weights and the tables read by the loaders are taken from the production
    locations (``*_ro`` attributes, read-only, never written) when they do not exist in the test root.
    """
    prod = None
    if cfg.test.enabled:
        from .config import load_config
        prod = load_config(test=False).paths

    out = str(cfg.paths.municipal)
    if prod is not None and os.path.realpath(out) == os.path.realpath(str(prod.municipal)):
        raise RuntimeError('test mode would write into the production municipal tables: ' + out)
    P = SimpleNamespace(
        obs_dir=str(cfg.paths.netcdf_obs), climbra_dir=str(cfg.paths.netcdf_climbra),
        obs_dir_ro=str(prod.netcdf_obs) if prod is not None else None,
        climbra_dir_ro=str(prod.netcdf_climbra) if prod is not None else None,
        mun_file=str(cfg.paths.shp_municipalities), brazil_file=str(cfg.paths.shp_brazil),
        out_dir=out,
        geom_dir=os.path.join(out, 'geometry'), weight_dir=os.path.join(out, 'weights'),
        maps_dir=os.path.join(out, 'maps'), log_dir=os.path.join(out, 'logs'),
        out_dir_ro=str(prod.municipal) if prod is not None else None,
        geom_dir_ro=os.path.join(str(prod.municipal), 'geometry') if prod is not None else None,
        weight_dir_ro=os.path.join(str(prod.municipal), 'weights') if prod is not None else None,
    )
    P.geom_file = os.path.join(P.geom_dir, 'municipalities_2024.gpkg')
    P.states_file = os.path.join(P.geom_dir, 'states_2024.gpkg')
    P.removed_file = os.path.join(P.geom_dir, 'removed_parts.csv')
    P.checks_file = os.path.join(P.log_dir, 'checks.csv')
    P.readme_file = os.path.join(P.out_dir, 'README.md')
    return P


def _readable(P, name):
    """Path of geometry file ``name`` to read: the output location, else (test mode) the read-only
    production copy."""
    rw = os.path.join(P.geom_dir, name)
    if not os.path.exists(rw) and P.geom_dir_ro and os.path.exists(os.path.join(P.geom_dir_ro, name)):
        return os.path.join(P.geom_dir_ro, name)
    return rw


def _default_paths(cfg=None):
    if cfg is None:
        from .config import load_config
        cfg = load_config(test=False)
    return get_paths(cfg)


# =============================================================================
# Small helpers
# =============================================================================
def _skip(path, overwrite):
    if os.path.exists(path) and not overwrite:
        print(f'[skip] exists: {path}')
        return True
    return False


def _std_latlon(da):
    lat = next((n for n in ('lat', 'latitude', 'y') if n in da.coords), None)
    lon = next((n for n in ('lon', 'longitude', 'x') if n in da.coords), None)
    if lat is None or lon is None:
        raise ValueError(f'lat/lon not found in {list(da.coords)}')
    da = da.rename({k: v for k, v in {lat: 'lat', lon: 'lon'}.items() if k != v})
    if float(da['lon'].max()) > 180:
        da = da.assign_coords(lon=((da['lon'] + 180) % 360) - 180)
    return da.sortby('lat').sortby('lon')


def _has_latlon(da):
    names = set(da.dims)
    return bool(names & {'lat', 'latitude', 'y'}) and bool(names & {'lon', 'longitude', 'x'})


def _gkey(x):
    try:
        return f'{float(x):g}'
    except (TypeError, ValueError):
        return str(x)


def _clean_val(v):
    if type(v).__module__.split('.')[0] == 'cftime':          # model calendars (noleap, 360_day, ...)
        try:
            return pd.Timestamp(v.year, v.month, v.day, v.hour, v.minute, v.second)
        except ValueError:                                     # e.g. 30 Feb in a 360-day calendar
            return f'{v.year:04d}-{v.month:02d}-{v.day:02d}'
    if isinstance(v, np.datetime64):
        return pd.Timestamp(v)
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode()
    if isinstance(v, (str, np.str_)):
        return str(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _relative_ok(indicator):
    return indicators_dict.get(indicator) in ('Rainfall', 'Drought') and indicator not in DIVERGING


def _xyz(lat, lon):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def _save_table(df, path, write_csv=WRITE_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    try:
        try:
            df.to_parquet(tmp, index=False)
        except Exception as err:                               # e.g. pyarrow ArrowInvalid / ArrowTypeError
            if 'arrow' not in type(err).__module__.lower() and 'Arrow' not in type(err).__name__:
                raise
            bad = [c for c in df.columns if df[c].dtype == object]
            print(f'[warn] {os.path.basename(path)}: {type(err).__name__}; '
                  f'writing object columns {bad} as text')
            df = df.copy()
            for c in bad:
                df[c] = df[c].astype(str)
            df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except ImportError:
        path = path.replace('.parquet', '.csv')
        df.to_csv(path, index=False)
        print('[warn] pyarrow not available: saved CSV instead of Parquet')
    if write_csv and path.endswith('.parquet'):
        df.to_csv(path.replace('.parquet', '.csv'), index=False)
    return path


def _read_table(path, columns=None):
    return pd.read_parquet(path, columns=columns) if path.endswith('.parquet') else pd.read_csv(
        path, usecols=columns, dtype={'CD_MUN': str})


# =============================================================================
# Geometry
# =============================================================================
GRID_SIZE = 1e-7   # snapping precision (degrees, ~1 cm) used only when an overlay fails


def _polygonal(g):
    """Valid, polygon-only version of a geometry (drops stray lines/points left by make_valid)."""
    if g is None or g.is_empty:
        return g
    if not g.is_valid:
        g = shapely.make_valid(g)
    if g.geom_type in ('Polygon', 'MultiPolygon'):
        return g
    polys = [p for p in shapely.get_parts(g) if p.geom_type in ('Polygon', 'MultiPolygon')]
    if not polys:
        return shapely.Polygon()
    out = _safe_union_all(polys)
    return out if out.is_valid else shapely.make_valid(out)


def _safe_union_all(geoms):
    try:
        return shapely.union_all(geoms)
    except GEOSException:
        g2 = [shapely.make_valid(g) for g in geoms]
        try:
            return shapely.union_all(g2, grid_size=GRID_SIZE)
        except GEOSException:
            return shapely.union_all([shapely.buffer(g, 0) for g in g2], grid_size=GRID_SIZE)


def _safe_overlay(func, a, b):
    """Element-wise shapely overlay (intersection / union) that retries failing pairs after
    repairing and snapping to GRID_SIZE, instead of raising a TopologyException."""
    try:
        return func(a, b)
    except GEOSException:
        pass
    scalar = np.ndim(a) == 0 and np.ndim(b) == 0
    A = np.atleast_1d(np.asarray(a, dtype=object))
    Bv = np.atleast_1d(np.asarray(b, dtype=object))
    bc = np.broadcast(A, Bv)
    out = np.empty(bc.shape, dtype=object)
    n_fixed = 0
    for i, (x, y) in enumerate(bc):
        try:
            out.flat[i] = func(x, y)
        except GEOSException:
            n_fixed += 1
            x2, y2 = _polygonal(x), _polygonal(y)
            try:
                out.flat[i] = func(x2, y2, grid_size=GRID_SIZE)
            except GEOSException:
                out.flat[i] = func(shapely.buffer(x2, 0), shapely.buffer(y2, 0), grid_size=GRID_SIZE)
    if n_fixed:
        print(f'[geometry] {n_fixed} overlay(s) repaired with snapping (grid {GRID_SIZE:g}°)')
    return out[0] if scalar else out


def _brazil_land(P):
    gdf = gpd.read_file(P.brazil_file).to_crs(4326)
    return _polygonal(_safe_union_all([_polygonal(g) for g in gdf.geometry]))


def prepare_geometry(P, overwrite=False):
    """Clean IBGE municipalities: drop lagoons and oceanic islands, build 25 km buffers."""
    buf_layer = f'municipalities_buffer{BUFFER_KM:g}km'
    existing = _readable(P, 'municipalities_2024.gpkg')
    if os.path.exists(existing) and not overwrite:
        gdf = gpd.read_file(existing, layer='municipalities')
        gbuf = gpd.read_file(existing, layer=buf_layer)
        for g in (gdf, gbuf):
            g['CD_MUN'] = g['CD_MUN'].astype(str)
        return gdf, gbuf

    os.makedirs(P.geom_dir, exist_ok=True)
    raw = gpd.read_file(P.mun_file)
    missing = [c for c in KEEP_COLS if c not in raw.columns]
    if missing:
        raise KeyError(f'columns {missing} not in {P.mun_file}')
    gdf = raw[KEEP_COLS + ['geometry']].copy()
    gdf['CD_MUN'] = gdf['CD_MUN'].astype(str)
    gdf = gdf.to_crs(4326)
    n_bad = int((~gdf.geometry.is_valid).sum())
    gdf['geometry'] = [_polygonal(g) for g in gdf.geometry]
    if n_bad:
        print(f'[geometry] repaired {n_bad} invalid municipal geometries')
    order = gdf['CD_MUN'].tolist()
    removed = []

    # lagoons coded as municipalities
    lag = gdf['CD_MUN'].isin(LAGOON_CODES)
    if lag.any():
        a = gdf[lag].to_crs(EQUAL_AREA_CRS).area.values / 1e6
        rp = gdf[lag].geometry.representative_point()
        for (_, r), ar, p in zip(gdf[lag].iterrows(), a, rp):
            removed.append(dict(CD_MUN=r.CD_MUN, NM_MUN=r.NM_MUN, NM_UF=r.NM_UF,
                                reason='lagoon (not a municipality)', lon=p.x, lat=p.y, area_km2=ar))
    gdf = gdf[~lag]

    # oceanic islands: polygon parts east of ISLAND_LON
    parts = gdf.explode(index_parts=False).reset_index(drop=True)
    parts = parts[parts.geometry.geom_type == 'Polygon'].reset_index(drop=True)
    rp = parts.geometry.representative_point()
    isl = rp.x.values > ISLAND_LON
    if isl.any():
        a = parts[isl].to_crs(EQUAL_AREA_CRS).area.values / 1e6
        for (_, r), ar, p in zip(parts[isl].iterrows(), a, rp[isl]):
            removed.append(dict(CD_MUN=r.CD_MUN, NM_MUN=r.NM_MUN, NM_UF=r.NM_UF,
                                reason=f'oceanic island (lon > {ISLAND_LON:g})',
                                lon=p.x, lat=p.y, area_km2=ar))
    kept = parts[~isl]
    # recombine remaining parts per municipality (no union: parts come from a valid
    # geometry and do not overlap, so a MultiPolygon is enough and avoids GEOS overlays)
    by_code = {}
    for cd, g in zip(kept['CD_MUN'], kept.geometry):
        by_code.setdefault(cd, []).append(g)
    attrs = kept.drop_duplicates('CD_MUN').set_index('CD_MUN')[['NM_MUN', 'NM_UF']]
    codes = [c for c in order if c in by_code]
    geoms = [_polygonal(ps[0] if len(ps) == 1 else shapely.MultiPolygon(ps))
             for ps in (by_code[c] for c in codes)]
    gdf2 = gpd.GeoDataFrame({'CD_MUN': codes,
                             'NM_MUN': attrs.loc[codes, 'NM_MUN'].to_numpy(),
                             'NM_UF': attrs.loc[codes, 'NM_UF'].to_numpy()},
                            geometry=geoms, crs=4326)
    present = set(codes)

    gdf2['area_km2'] = gdf2.to_crs(EQUAL_AREA_CRS).area.values / 1e6
    rpt = gdf2.geometry.representative_point()
    gdf2['rep_lon'], gdf2['rep_lat'] = rpt.x.values, rpt.y.values

    rem = pd.DataFrame(removed, columns=['CD_MUN', 'NM_MUN', 'NM_UF', 'reason', 'lon', 'lat', 'area_km2'])
    rem['municipality_removed'] = ~rem['CD_MUN'].isin(present)
    rem.to_csv(P.removed_file, index=False)
    for _, r in rem.iterrows():
        what = 'whole municipality' if r.municipality_removed else 'part'
        print(f'[geometry] removed {what}: {r.NM_MUN} ({r.NM_UF}) – {r.reason}, '
              f'{r.area_km2:.1f} km² at ({r.lon:.2f}, {r.lat:.2f})')

    # buffers clipped to Brazil, always containing the municipality itself
    land = _brazil_land(P)
    shapely.prepare(land)
    bufs = gdf2.to_crs(EQUAL_AREA_CRS).buffer(BUFFER_KM * 1000).to_crs(4326).to_numpy()
    bufs = np.array([_polygonal(b) for b in bufs], dtype=object)
    inside = shapely.contains(land, bufs)
    clipped = bufs.copy()
    if (~inside).any():
        clipped[~inside] = _safe_overlay(shapely.intersection, bufs[~inside], land)
    clipped = _safe_overlay(shapely.union, clipped, gdf2.geometry.to_numpy())
    clipped = np.array([_polygonal(g) for g in clipped], dtype=object)
    gbuf = gpd.GeoDataFrame(gdf2[KEEP_COLS].copy(), geometry=clipped, crs=4326)

    if os.path.exists(P.geom_file):
        os.remove(P.geom_file)
    gdf2.to_file(P.geom_file, layer='municipalities', driver='GPKG')
    gbuf.to_file(P.geom_file, layer=buf_layer, driver='GPKG')
    print(f'[geometry] {len(gdf2)} municipalities -> {P.geom_file}')
    return gdf2, gbuf


def build_states(P, gdf, overwrite=False):
    """State outlines (union of the municipalities per ``NM_UF``) -> geometry/states_2024.gpkg.

    Used by the municipal maps. Skipped when the file already exists (in the output location or,
    in test mode, in the read-only production copy)."""
    existing = _readable(P, 'states_2024.gpkg')
    if os.path.exists(existing) and not overwrite:
        return existing
    rows = [(uf, _polygonal(_safe_union_all([_polygonal(x) for x in grp.geometry])))
            for uf, grp in gdf.groupby('NM_UF')]
    st = gpd.GeoDataFrame({'NM_UF': [r[0] for r in rows]}, geometry=[r[1] for r in rows], crs=4326)
    os.makedirs(P.geom_dir, exist_ok=True)
    st.to_file(P.states_file, driver='GPKG')
    print(f'[geometry] {len(st)} states -> {P.states_file}')
    return P.states_file


def _geo_context(gdf, gbuf, P):
    if not (gdf['CD_MUN'].values == gbuf['CD_MUN'].values).all():
        raise ValueError('municipality and buffer layers are not in the same order')
    return dict(
        paths=P,
        base=pd.DataFrame(gdf[KEEP_COLS]).reset_index(drop=True).astype(str),
        geoms=gdf.geometry.to_numpy(), bufs=gbuf.geometry.to_numpy(),
        area=gdf['area_km2'].to_numpy(dtype=float),
        mun_xyz=_xyz(gdf['rep_lat'].to_numpy(float), gdf['rep_lon'].to_numpy(float)),
        n=len(gdf))


# =============================================================================
# Pixel weights
# =============================================================================
def _cell_edges(c):
    c = np.asarray(c, dtype=float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = (c[:-1] + c[1:]) / 2
    return np.r_[c[0] - (mid[0] - c[0]), mid, c[-1] + (c[-1] - mid[-1])]


def _pixel_area(lat, lon):
    le, oe = _cell_edges(lat), _cell_edges(lon)
    return np.diff(le)[:, None] * np.diff(oe)[None, :] * KM_PER_DEG2 * np.cos(np.deg2rad(lat))[:, None]


def _build_weights(geoms, lat, lon):
    """Sparse (n_geoms x n_pixels) matrix of intersection areas (km²)."""
    le, oe = _cell_edges(lat), _cell_edges(lon)
    nlat, nlon = lat.size, lon.size
    coslat = np.cos(np.deg2rad(lat))
    rows, cols, vals = [], [], []
    for r, g in enumerate(geoms):
        if g is None or g.is_empty:
            continue
        minx, miny, maxx, maxy = g.bounds
        j0 = max(np.searchsorted(oe, minx, 'right') - 1, 0)
        j1 = min(np.searchsorted(oe, maxx, 'left'), nlon)
        i0 = max(np.searchsorted(le, miny, 'right') - 1, 0)
        i1 = min(np.searchsorted(le, maxy, 'left'), nlat)
        if j1 <= j0 or i1 <= i0:
            continue
        ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing='ij')
        ii, jj = ii.ravel(), jj.ravel()
        boxes = shapely.box(oe[jj], le[ii], oe[jj + 1], le[ii + 1])
        g = _polygonal(g)
        if g is None or g.is_empty:
            continue
        shapely.prepare(g)
        inside = shapely.contains(g, boxes)
        touch = ~inside & shapely.intersects(g, boxes)
        a = np.zeros(ii.size)
        a[inside] = shapely.area(boxes[inside])
        if touch.any():
            a[touch] = shapely.area(_safe_overlay(shapely.intersection, boxes[touch], g))
        keep = a > 0
        rows.append(np.full(keep.sum(), r))
        cols.append(ii[keep] * nlon + jj[keep])
        vals.append(a[keep] * KM_PER_DEG2 * coslat[ii[keep]])
    if not vals:
        return sparse.csr_matrix((len(geoms), nlat * nlon))
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(geoms), nlat * nlon))


def _minmax_index(Wf):
    n = Wf.shape[0]
    r = np.repeat(np.arange(n), np.diff(Wf.indptr))
    c, f = Wf.indices, Wf.data
    passes = f >= MINMAX_MIN_FRAC
    has = np.bincount(r[passes], minlength=n) > 0
    use = passes | ~has[r]
    return r[use], c[use]


_WCACHE = {}


def get_weights(lat, lon, geo, overwrite=False):
    P = geo['paths']
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    key = (f'{lat.size}x{lon.size}_lat{lat[0]:.4f}_{lat[-1]:.4f}_lon{lon[0]:.4f}_{lon[-1]:.4f}'
           f'_n{geo["n"]}_b{BUFFER_KM:g}')
    ckey = (P.weight_dir, key)
    if ckey in _WCACHE and not overwrite:
        return _WCACHE[ckey]
    os.makedirs(P.weight_dir, exist_ok=True)
    fp, fb = (os.path.join(P.weight_dir, f'{key}_{s}.npz') for s in ('polygon', 'buffer'))
    if not (os.path.exists(fp) and os.path.exists(fb)) and P.weight_dir_ro:     # test mode: reuse production cache
        fp_ro, fb_ro = (os.path.join(P.weight_dir_ro, f'{key}_{s}.npz') for s in ('polygon', 'buffer'))
        if os.path.exists(fp_ro) and os.path.exists(fb_ro):
            fp, fb = fp_ro, fb_ro
    if os.path.exists(fp) and os.path.exists(fb) and not overwrite:
        Wp, Wb = sparse.load_npz(fp).tocsr(), sparse.load_npz(fb).tocsr()
    else:
        t0 = time.time()
        print(f'[weights] building for grid {lat.size}x{lon.size} (first time only)...')
        Wp = _build_weights(geo['geoms'], lat, lon)
        Wb = _build_weights(geo['bufs'], lat, lon)
        fp, fb = (os.path.join(P.weight_dir, f'{key}_{s}.npz') for s in ('polygon', 'buffer'))
        sparse.save_npz(fp, Wp)
        sparse.save_npz(fb, Wb)
        print(f'[weights] done in {time.time() - t0:.0f} s')
    pix = _pixel_area(lat, lon).ravel()
    inv = sparse.diags(1.0 / pix)
    G = dict(key=key, lat=lat, lon=lon, npix=lat.size * lon.size,
             Wp=Wp, Wb=Wb, Wpf=(Wp @ inv).tocsr(), Wbf=(Wb @ inv).tocsr(),
             tot_p=np.asarray(Wp.sum(axis=1)).ravel(), tot_b=np.asarray(Wb.sum(axis=1)).ravel(),
             colw=np.asarray(Wp.sum(axis=0)).ravel())
    G['mm_p'], G['mm_b'] = _minmax_index(G['Wpf']), _minmax_index(G['Wbf'])
    _WCACHE[ckey] = G
    return G


# =============================================================================
# Zonal statistics
# =============================================================================
def _rowminmax(rc, n_rows, X):
    r, c = rc
    mn = np.full((n_rows, X.shape[0]), np.nan)
    mx = np.full((n_rows, X.shape[0]), np.nan)
    for k in range(X.shape[0]):
        v = X[k, c]
        fin = np.isfinite(v)
        a = np.full(n_rows, np.inf)
        b = np.full(n_rows, -np.inf)
        np.minimum.at(a, r[fin], v[fin])
        np.maximum.at(b, r[fin], v[fin])
        mn[:, k] = np.where(np.isfinite(a), a, np.nan)
        mx[:, k] = np.where(np.isfinite(b), b, np.nan)
    return mn, mx


def _zonal(Wa, Wf, tot, mm, X):
    """Area-weighted stats for X (n_combo, n_pix). Returns dict of (n_mun, n_combo) arrays."""
    valid = np.isfinite(X)
    V = valid.T.astype(np.float64)
    X0 = np.where(valid, X, 0.0).T
    sw = Wa @ V
    s1 = Wa @ X0
    s2 = Wa @ (X0 * X0)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(sw > 0, s1 / sw, np.nan)
        std = np.sqrt(np.clip(np.where(sw > 0, s2 / sw, np.nan) - mean ** 2, 0, None))
        vfrac = np.where(tot[:, None] > 0, sw / tot[:, None], np.nan)
    mn, mx = _rowminmax(mm, Wa.shape[0], X)
    return dict(mean=mean, std=std, min=mn, max=mx, n_pixels=Wf @ V, valid_area_frac=vfrac)


def _select(P, B, n_p, n_b, vfrac_p):
    ok_p = (n_p >= MIN_PIXELS) & (vfrac_p >= MIN_VALID_FRAC)
    use_b = ~ok_p & (n_b > 0)
    use_p = ok_p | (~use_b & (n_p > 0))
    return use_p, use_b


def _field_stats(X, G, geo):
    P = _zonal(G['Wp'], G['Wpf'], G['tot_p'], G['mm_p'], X)
    B = _zonal(G['Wb'], G['Wbf'], G['tot_b'], G['mm_b'], X)
    use_p, use_b = _select(P, B, P['n_pixels'], B['n_pixels'], P['valid_area_frac'])
    out = {k: np.where(use_p, P[k], np.where(use_b, B[k], np.nan)) for k in P}
    method = np.where(use_p, 'polygon', np.where(use_b, f'buffer{BUFFER_KM:g}km', 'none')).astype(object)
    nearest_km = np.full(method.shape, np.nan)

    need = ~(use_p | use_b)
    if need.any():
        lon2d, lat2d = np.meshgrid(G['lon'], G['lat'])
        pix_xyz = _xyz(lat2d.ravel(), lon2d.ravel())
        for k in np.where(need.any(axis=0))[0]:
            valid = np.isfinite(X[k])
            if not valid.any():
                continue
            tree = cKDTree(pix_xyz[valid])
            rows = np.where(need[:, k])[0]
            d, idx = tree.query(geo['mun_xyz'][rows])
            km = 2 * R_EARTH * np.arcsin(np.clip(d / 2, 0, 1))
            ok = km <= NEAREST_MAX_KM
            vals = X[k][valid][idx]
            rr = rows[ok]
            for s in ('mean', 'min', 'max'):
                out[s][rr, k] = vals[ok]
            out['std'][rr, k] = 0.0
            out['n_pixels'][rr, k] = 0.0
            out['valid_area_frac'][rr, k] = 0.0
            method[rr, k] = 'nearest'
            nearest_km[rr, k] = km[ok]
    out['method'] = method
    out['nearest_km'] = nearest_km
    return out, use_p, use_b


def _agreement_stats(X, domain, G):
    """X: agreement variable (NaN where < 66% agree); domain: pixels with valid anomaly data."""
    D = np.broadcast_to(domain[None, :], X.shape)
    A = (np.isfinite(X) & D).astype(np.float64).T
    Dm = D.astype(np.float64).T
    res = {}
    for tag, Wa, Wf, tot in (('p', G['Wp'], G['Wpf'], G['tot_p']), ('b', G['Wb'], G['Wbf'], G['tot_b'])):
        dom = Wa @ Dm
        with np.errstate(invalid='ignore', divide='ignore'):
            res[tag] = dict(frac=np.where(dom > 0, (Wa @ A) / dom, np.nan), n=Wf @ Dm,
                            vfrac=np.where(tot[:, None] > 0, dom / tot[:, None], np.nan))
    use_p, use_b = _select(None, None, res['p']['n'], res['b']['n'], res['p']['vfrac'])
    pick = lambda k: np.where(use_p, res['p'][k], np.where(use_b, res['b'][k], np.nan))
    frac = pick('frac')
    # mean agreement value over agreeing pixels
    Xa = np.where(D, X, np.nan)
    P = _zonal(G['Wp'], G['Wpf'], G['tot_p'], G['mm_p'], Xa)
    B = _zonal(G['Wb'], G['Wbf'], G['tot_b'], G['mm_b'], Xa)
    return dict(
        mean=np.where(use_p, P['mean'], np.where(use_b, B['mean'], np.nan)),
        n_pixels=pick('n'), valid_area_frac=pick('vfrac'),
        method=np.where(use_p, 'polygon', np.where(use_b, f'buffer{BUFFER_KM:g}km', 'none')).astype(object),
        agree_area_frac=frac,
        robust=np.where(np.isfinite(frac), (frac >= ROBUST_AREA_FRAC).astype(float), np.nan))


def _trend_extras(S, Pv, use_p, use_b, G):
    valid = np.isfinite(S) & np.isfinite(Pv)
    sig = valid & (Pv < P_SIG)
    Vt, St = valid.astype(float).T, sig.astype(float).T
    SS = np.where(sig, S, 0.0).T
    out = {}
    for tag, Wa in (('p', G['Wp']), ('b', G['Wb'])):
        v, s = Wa @ Vt, Wa @ St
        with np.errstate(invalid='ignore', divide='ignore'):
            out[tag] = (np.where(v > 0, s / v, np.nan), np.where(s > 0, (Wa @ SS) / s, np.nan))
    frac = np.where(use_p, out['p'][0], np.where(use_b, out['b'][0], np.nan))
    msig = np.where(use_p, out['p'][1], np.where(use_b, out['b'][1], np.nan))
    return frac, msig


# =============================================================================
# File processing
# =============================================================================
def discover_files(P, indicator, sources=('Observations', 'CLIMBRA')):
    cat = indicators_dict[indicator]
    items = []

    def _indir(root, root_ro, *sub):
        d = os.path.join(root, cat, indicator, *sub)
        if not os.path.isdir(d) and root_ro:                  # test mode: production product (read-only)
            d = os.path.join(root_ro, cat, indicator, *sub)
        return d

    if 'Observations' in sources:
        d = _indir(P.obs_dir, P.obs_dir_ro)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if not f.endswith('.nc'):
                    continue
                ds_name = next((x for x in OBS_DATASETS if f.upper().startswith(x.upper() + '_')), None)
                if ds_name is None:
                    continue
                kind = f[len(ds_name) + 1:-3]
                items.append(dict(source='Observations', dataset=ds_name, kind=kind,
                                  path=os.path.join(d, f),
                                  out=os.path.join(P.out_dir, 'Observations', cat, indicator,
                                                   f'{ds_name}_{kind}.parquet')))
    if 'CLIMBRA' in sources:
        d = _indir(P.climbra_dir, P.climbra_dir_ro, 'ensemble')
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if not f.endswith('.nc'):
                    continue
                stem = f[:-3]
                items.append(dict(source='CLIMBRA', dataset='ensemble',
                                  kind=stem.replace('ensemble_', '', 1), path=os.path.join(d, f),
                                  out=os.path.join(P.out_dir, 'CLIMBRA', cat, indicator, f'{stem}.parquet')))
    return items


def _to_frame(res, combos, dims, geo, meta):
    n, nc = geo['n'], len(combos)
    df = pd.concat([geo['base']] * nc, ignore_index=True)
    for k, v in meta.items():
        df[k] = v
    for j, d in enumerate(dims):
        vals = [_clean_val(c[j]) for c in combos]
        if any(isinstance(x, pd.Timestamp) for x in vals):
            if all(isinstance(x, pd.Timestamp) for x in vals):
                col = pd.DatetimeIndex(vals)
            else:                                              # mixed: keep all dates as text
                col = pd.Index([x.strftime('%Y-%m-%d') if isinstance(x, pd.Timestamp) else str(x)
                                for x in vals])
            df[d] = np.repeat(col.to_numpy(), n)
        else:
            df[d] = np.repeat(np.array(vals), n)
    for s in STAT_COLS:
        arr = res.get(s)
        df[s] = arr.T.reshape(-1) if arr is not None else np.nan
    df['method'] = df['method'].astype(str).replace('nan', 'none')
    return df


def _check(res, X, G, geo, item, var):
    x = X[0]
    valid = np.isfinite(x)
    cw = G['colw']
    m = valid & (cw > 0)
    grid_mean = float((cw[m] * x[m]).sum() / cw[m].sum()) if m.any() else np.nan
    mm = res['mean'][:, 0]
    ok = np.isfinite(mm)
    mun_mean = float((geo['area'][ok] * mm[ok]).sum() / geo['area'][ok].sum()) if ok.any() else np.nan
    cnt = pd.Series(res['method'][:, 0]).value_counts()
    return dict(source=item['source'], dataset=item['dataset'], indicator=item['indicator'],
                file_kind=item['kind'], variable=var, grid_mean_brazil=grid_mean,
                municipal_mean_brazil=mun_mean,
                rel_diff_pct=100 * (mun_mean - grid_mean) / abs(grid_mean) if grid_mean else np.nan,
                n_polygon=int(cnt.get('polygon', 0)),
                n_buffer=int(cnt.get(f'buffer{BUFFER_KM:g}km', 0)),
                n_nearest=int(cnt.get('nearest', 0)), n_missing=int(cnt.get('none', 0)),
                processed=datetime.datetime.now().isoformat(timespec='seconds'))


def _companions(agr_var, main_vars):
    """Main variables ordered by how well they match an agreement variable (same stem first),
    used to define the valid-data domain of the agreement fraction."""
    stem = agr_var
    for suf in ('_rp_model_agreement', '_model_agreement', '_agreement'):
        if stem.endswith(suf):
            stem = stem[:-len(suf)]
            break
    preferred = [f'{stem}_probability_ratio', f'{stem}_new_return_period', f'{stem}_anom',
                 f'{stem}_anom_perc', stem]
    first = [m for m in preferred if m in main_vars]
    same = [m for m in main_vars if m.startswith(stem + '_') and m not in first]
    return first + same + [m for m in main_vars if m not in first and m not in same]


def process_file(item, geo, overwrite_weights=False):
    with xr.open_dataset(item['path']) as ds:
        ds = ds.load()
    variables = [v for v in ds.data_vars if _has_latlon(ds[v])]
    agr_vars = [v for v in variables if 'agreement' in v.lower()]
    main_vars = [v for v in variables if v not in agr_vars]
    frames, checks, cache = {}, [], {}
    meta0 = dict(source=item['source'], dataset=item['dataset'], indicator=item['indicator'],
                 category=indicators_dict[item['indicator']], file_kind=item['kind'])

    for v in main_vars + agr_vars:
        da = _std_latlon(ds[v])
        dims = [d for d in da.dims if d not in ('lat', 'lon')]
        da = da.transpose(*dims, 'lat', 'lon')
        G = get_weights(da['lat'].values, da['lon'].values, geo, overwrite_weights)
        X = da.values.reshape(-1, G['npix']).astype(np.float64)
        combos = list(itertools.product(*[da[d].values for d in dims])) if dims else [()]
        meta = dict(meta0, variable=v, units=str(da.attrs.get('units', '')))

        if v in agr_vars:
            domain = None
            for mv in _companions(v, main_vars):
                dm = _std_latlon(ds[mv])
                if dm.sizes['lat'] == G['lat'].size and dm.sizes['lon'] == G['lon'].size:
                    other = [d for d in dm.dims if d not in ('lat', 'lon')]
                    domain = np.isfinite(dm.transpose(*other, 'lat', 'lon').values
                                         ).reshape(-1, G['npix']).any(axis=0)
                    break
            if domain is None:
                domain = np.ones(G['npix'], dtype=bool)
            res = _agreement_stats(X, domain, G)
        else:
            res, use_p, use_b = _field_stats(X, G, geo)
            cache[v] = (X, use_p, use_b, G)
            checks.append(_check(res, X, G, geo, item, v))
        frames[v] = _to_frame(res, combos, dims, geo, meta)

    # trend extras: share of area with significant trend, mean slope over significant pixels
    if TREND_SLOPE_VAR in cache and TREND_P_VAR in cache:
        S, use_p, use_b, G = cache[TREND_SLOPE_VAR]
        Pv = cache[TREND_P_VAR][0]
        if S.shape == Pv.shape:
            frac, msig = _trend_extras(S, Pv, use_p, use_b, G)
            for v in (TREND_SLOPE_VAR, TREND_P_VAR):
                frames[v]['sig_area_frac'] = frac.T.reshape(-1)
            frames[TREND_SLOPE_VAR]['mean_sig'] = msig.T.reshape(-1)

    df = pd.concat(frames.values(), ignore_index=True)
    dim_cols = [c for c in df.columns if c not in KEEP_COLS + META_COLS + STAT_COLS]
    return df[KEEP_COLS + META_COLS + dim_cols + STAT_COLS], checks


def _update_checks(P, new):
    if not new:
        return
    os.makedirs(P.log_dir, exist_ok=True)
    new = pd.DataFrame(new)
    keys = ['source', 'dataset', 'indicator', 'file_kind', 'variable']
    if os.path.exists(P.checks_file):
        old = pd.read_csv(P.checks_file)
        old = old.merge(new[keys], on=keys, how='left', indicator=True)
        old = old[old['_merge'] == 'left_only'].drop(columns='_merge')
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(P.checks_file, index=False)


# =============================================================================
# Wide, ready-to-map GeoPackage
# =============================================================================
def _dim_cols(df):
    return [c for c in df.columns if c not in KEEP_COLS + META_COLS + STAT_COLS]


def _fix_other_dims(sub, keep=()):
    for d in _dim_cols(sub):
        if d in keep:
            continue
        vals = sub[d].dropna().unique()
        if len(vals) > 1:
            sub = sub[sub[d] == vals[0]]
    return sub


def _series(sub, stat='mean'):
    return sub.groupby('CD_MUN', sort=False)[stat].first()


def _main_var(df, indicator):
    vs = list(df['variable'].unique())
    return indicator if indicator in vs else vs[0]


def _wide_rp_shift(df, vs, indicator, add, gtag):
    """Wide columns from ensemble_rp_shift: present-day level, new return period, probability
    ratio (ensemble WIDE_ENSEMBLE) and model agreement for the WIDE_RP-year event."""
    stems = sorted(x[:-len('_new_return_period')] for x in vs if x.endswith('_new_return_period'))
    if indicator in stems:
        stems = [indicator]
    for stem in stems:
        tag = '' if stem == indicator else f'_{stem}'
        hist = df[df['variable'] == f'{stem}_hist_return_level']
        if len(hist):
            if 'ensemble' in hist.columns:
                hist = hist[hist['ensemble'] == WIDE_ENSEMBLE]
            if 'event_rp' in hist.columns:
                hist = hist[hist['event_rp'] == WIDE_RP]
            add(f'CB_histRL{WIDE_RP:g}{tag}', _series(_fix_other_dims(hist, ['ensemble', 'event_rp'])),
                f'CLIMBRA ensemble {WIDE_ENSEMBLE}: present-day {WIDE_RP:g}-year level of {stem}',
                str(hist['units'].iat[0]) if len(hist) else '')
        for suffix, col, stat, units, text in (
                ('_new_return_period', 'newRP', 'mean', 'years',
                 f'return period at the GWL of the present-day {WIDE_RP:g}-year {stem} event'),
                ('_probability_ratio', 'PR', 'mean', '1',
                 f'probability ratio (p_GWL / p_present) of the present-day {WIDE_RP:g}-year {stem} event'),
                ('_rp_model_agreement', 'agreeRP', 'agree_area_frac', 'fraction',
                 f'share of municipal area where >= 66% of models agree on the direction of the '
                 f'frequency change of the {WIDE_RP:g}-year {stem} event')):
            var = f'{stem}{suffix}'
            if var not in vs:
                continue
            sub = df[df['variable'] == var]
            if stat == 'mean' and 'ensemble' in sub.columns and sub['ensemble'].notna().any():
                sub = sub[sub['ensemble'] == WIDE_ENSEMBLE]
            if 'event_rp' in sub.columns:
                sub = sub[sub['event_rp'] == WIDE_RP]
            if 'GWL' not in sub.columns:
                continue
            for g in sub['GWL'].dropna().unique():
                s_ = _fix_other_dims(sub[sub['GWL'] == g], ['GWL', 'ensemble', 'event_rp'])
                who = '' if stat != 'mean' else f'ensemble {WIDE_ENSEMBLE}: '
                add(f'CB_{col}{WIDE_RP:g}{tag}_{gtag(g)}', _series(s_, stat),
                    f'CLIMBRA {who}{text}, GWL {_gkey(g)} °C', units)


def build_wide(P, indicator, gdf, overwrite=False):
    cat = indicators_dict[indicator]
    out = os.path.join(P.maps_dir, f'{indicator}_municipalities.gpkg')
    if _skip(out, overwrite):
        return out
    cols, desc = {}, []

    def add(name, s, text, units=''):
        cols[name] = s
        desc.append(dict(column=name, description=text, units=units))

    def gtag(g):
        return f'{_gkey(g).replace(".", "p")}C'

    for f in sorted(glob.glob(os.path.join(P.out_dir, 'Observations', cat, indicator, '*.*'))):
        if not f.endswith(('.parquet', '.csv')):
            continue
        df = _read_table(f)
        dsn, kind = df['dataset'].iat[0], df['file_kind'].iat[0]
        if kind == 'trend' and TREND_SLOPE_VAR in set(df['variable']):
            sub = _fix_other_dims(df[df['variable'] == TREND_SLOPE_VAR])
            u = sub['units'].iat[0]
            add(f'{dsn}_trend', _series(sub), f'{dsn}: mean trend slope of {indicator}', u)
            add(f'{dsn}_trend_sig', _series(sub, 'sig_area_frac'),
                f'{dsn}: share of municipal area with significant trend (p < {P_SIG})', 'fraction')
            continue
        v = _main_var(df, indicator)
        sub = df[df['variable'] == v]
        u = sub['units'].iat[0]
        if kind == 'climatology':
            add(f'{dsn}_clim', _series(_fix_other_dims(sub)), f'{dsn}: climatology of {indicator}', u)
        elif kind == 'return_period' and 'return_period' in sub.columns:
            for rp in sorted(sub['return_period'].dropna().unique()):
                s = _fix_other_dims(sub[sub['return_period'] == rp], ['return_period'])
                add(f'{dsn}_RL{rp:g}', _series(s), f'{dsn}: {rp:g}-year return level of {indicator}', u)

    perc_v, abs_v, agr_v = f'{indicator}_anom_perc', f'{indicator}_anom', f'{indicator}_model_agreement'
    for f in sorted(glob.glob(os.path.join(P.out_dir, 'CLIMBRA', cat, indicator, '*.*'))):
        if not f.endswith(('.parquet', '.csv')):
            continue
        df = _read_table(f)
        kind = df['file_kind'].iat[0]
        vs = set(df['variable'])
        if 'rp_shift' in kind:
            _wide_rp_shift(df, vs, indicator, add, gtag)
            continue
        if not kind.startswith('anomaly_'):                    # timeseries, delta_*, absolute climatology
            continue
        k = 'RL' if 'return_period' in kind else ('clim' if 'climatology' in kind else kind)
        v = perc_v if (_relative_ok(indicator) and perc_v in vs) else (abs_v if abs_v in vs else None)
        for var, is_agree in ((v, False), (agr_v if agr_v in vs else None, True)):
            if var is None:
                continue
            sub = df[df['variable'] == var]
            if 'ensemble' in sub.columns and sub['ensemble'].notna().any() and not is_agree:
                sub = sub[sub['ensemble'] == WIDE_ENSEMBLE]
            kname = k
            if 'return_period' in sub.columns and sub['return_period'].notna().any():
                sub = sub[sub['return_period'] == WIDE_RP]
                kname = f'{k}{WIDE_RP:g}'
            if 'GWL' not in sub.columns:
                continue
            for g in sub['GWL'].dropna().unique():
                s = _fix_other_dims(sub[sub['GWL'] == g], ['GWL', 'ensemble', 'return_period'])
                what = 'climatology' if k == 'clim' else f'{WIDE_RP:g}-year return level'
                if is_agree:
                    add(f'CB_{kname}_agree_{gtag(g)}', _series(s, 'agree_area_frac'),
                        f'CLIMBRA: share of municipal area where >= 66% of models agree on the sign '
                        f'of the change in {indicator} {what}, GWL {_gkey(g)} °C', 'fraction')
                else:
                    suf = 'pct' if var == perc_v else 'abs'
                    u = '%' if suf == 'pct' else sub['units'].iat[0]
                    add(f'CB_{kname}_{suf}_{gtag(g)}', _series(s),
                        f'CLIMBRA ensemble {WIDE_ENSEMBLE}: change in {indicator} {what}, '
                        f'GWL {_gkey(g)} °C vs present', u)

    if not cols:
        print(f'[wide] nothing to write for {indicator}')
        return None
    wide = pd.DataFrame(cols)
    wide.index.name = 'CD_MUN'
    wide = wide.reset_index()
    wide['CD_MUN'] = wide['CD_MUN'].astype(str)
    g = gdf[KEEP_COLS + ['geometry']].merge(wide, on='CD_MUN', how='left')
    os.makedirs(P.maps_dir, exist_ok=True)
    if os.path.exists(out):
        os.remove(out)
    g.to_file(out, layer=indicator, driver='GPKG')
    pd.DataFrame(desc).to_csv(out.replace('.gpkg', '_columns.csv'), index=False)
    print(f'[wide] {out} ({len(cols)} columns)')
    return out


# =============================================================================
# README
# =============================================================================
def _md_table(df, floatfmt='{:.3g}'):
    if df is None or df.empty:
        return '_none_\n'
    def f(x):
        if isinstance(x, (float, np.floating)):
            if np.isnan(x):
                return ''
            return f'{x:,.0f}' if abs(x) >= 1000 else floatfmt.format(x)
        return str(x)
    lines = ['| ' + ' | '.join(map(str, df.columns)) + ' |',
             '|' + '|'.join(['---'] * len(df.columns)) + '|']
    lines += ['| ' + ' | '.join(f(x) for x in r) + ' |' for r in df.itertuples(index=False)]
    return '\n'.join(lines) + '\n'


def write_readme(P):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    removed_file = _readable(P, 'removed_parts.csv')
    geom_file = _readable(P, 'municipalities_2024.gpkg')
    removed = pd.read_csv(removed_file, dtype={'CD_MUN': str}) if os.path.exists(removed_file) else None
    checks = pd.read_csv(P.checks_file) if os.path.exists(P.checks_file) else None
    n_mun = len(gpd.read_file(geom_file, layer='municipalities', ignore_geometry=True)) \
        if os.path.exists(geom_file) else 'n/a'

    inv = []
    for src in ('Observations', 'CLIMBRA'):
        for f in sorted(glob.glob(os.path.join(P.out_dir, src, '*', '*', '*.*'))):
            if f.endswith(('.parquet', '.csv')):
                inv.append(dict(source=src, file=os.path.relpath(f, P.out_dir)))
    maps = sorted(os.path.relpath(f, P.out_dir) for f in glob.glob(os.path.join(P.maps_dir, '*.gpkg')))

    B = f'buffer{BUFFER_KM:g}km'
    txt = f"""# Municipal-level climate hazard indicators for Brazil

Part of the HAZBRA dataset (see the dataset `README.md` for the full user guide).
Generated by `hazbra/municipal.py` (https://github.com/Luisgcattelan/HAZBRA) — last update {now}.

Gridded climate hazard indicators (observational datasets and the CLIMBRA multi-model
ensemble) aggregated to the {n_mun} municipalities of the IBGE 2024 municipal mesh with
area-weighted zonal statistics.

## Folder structure

```
{os.path.basename(P.out_dir)}/
├── README.md                              this file
├── geometry/
│   ├── municipalities_2024.gpkg           layers: municipalities, municipalities_{B}
│   ├── removed_parts.csv                  lagoons and oceanic islands removed from the mesh
│   └── states_2024.gpkg                   state outlines (union of the municipalities)
├── Observations/<Category>/<Indicator>/<DATASET>_<kind>.parquet
├── CLIMBRA/<Category>/<Indicator>/<ensemble file>.parquet
├── weights/                               pixel-municipality area weights (sparse .npz, one pair per grid;
│                                          only needed to rerun the aggregation)
├── maps/<Indicator>_municipalities.gpkg   (*) wide, ready-to-map table (+ _columns.csv dictionary)
└── logs/checks.csv                        (*) consistency checks and method counts
```

(*) produced by the code but not part of the published dataset (Zenodo): the maps repeat values of the
long tables, and the checks are summarised below.

## Sources

| Source | Input | Files |
|---|---|---|
| Observations | `netcdf/Observational/<Category>/<Indicator>/` | {', '.join(sorted(OBS_DATASETS))}: `*_climatology.nc`, `*_return_period.nc`, `*_params.nc`, `*_trend.nc`, `*_timeseries.nc` |
| CLIMBRA | `netcdf/CLIMBRA/<Category>/<Indicator>/ensemble/` | all ensemble files (e.g. `ensemble_anomaly_climatology.nc`, `ensemble_anomaly_return_period.nc`, `ensemble_rp_shift.nc`) |

Every data variable of every file is processed, and every non-spatial dimension of the
NetCDF (e.g. `ensemble`, `GWL`, `return_period`) becomes a column of the tables, so no
information in the source files is lost.

## Municipal geometries

* Source: IBGE municipal mesh 2024 (`BR_Municipios_2024`), reprojected to WGS84 (EPSG:4326); invalid geometries repaired.
* Columns kept: `{'`, `'.join(KEEP_COLS)}`, plus `area_km2` (equal-area projection {EQUAL_AREA_CRS})
  and `rep_lon`, `rep_lat` (representative point).
* Removed: the lagoon polygons coded as municipalities (codes {', '.join(sorted(LAGOON_CODES))}),
  and all polygon parts east of {ISLAND_LON:g}° longitude (oceanic islands such as Fernando de
  Noronha, Trindade and Martim Vaz, and the São Pedro e São Paulo archipelago). A municipality
  consisting only of such parts is removed completely; otherwise only the island part is dropped.

Removed parts:

{_md_table(removed)}
## Method

1. **Pixel weights.** For each grid, the exact intersection area (km²) between every grid
   cell and every municipality is computed once and stored as a sparse matrix. Partially
   covered cells contribute in proportion to the overlapping area; cells are weighted by
   their true area (cos latitude).
2. **Polygon statistics.** Area-weighted statistics over the municipality polygon, using only
   cells with valid data.
3. **Adaptive buffer.** If the polygon contains fewer than {MIN_PIXELS:g} valid pixel-equivalents or
   less than {MIN_VALID_FRAC:.0%} of its area has valid data (small or coastal municipalities, land-only
   grids), statistics are computed over a {BUFFER_KM:g} km buffer around the municipality,
   clipped to Brazil's land area (so no ocean cells are used) and always containing the
   municipality itself.
4. **Nearest pixel.** If the buffer still has no valid data, the nearest valid cell to the
   municipality's representative point is used, up to {NEAREST_MAX_KM:g} km. Beyond that the value is missing.
5. The `method` column records which of these was used for every value.

Parameters: buffer {BUFFER_KM:g} km, minimum {MIN_PIXELS:g} valid pixel-equivalents, minimum valid
area {MIN_VALID_FRAC:.0%}, nearest-pixel limit {NEAREST_MAX_KM:g} km, cells covering < {MINMAX_MIN_FRAC:.0%} of their
area ignored for min/max, trend significance p < {P_SIG}, robust agreement ≥ {ROBUST_AREA_FRAC:.0%} of area.

## Long tables (`Observations/`, `CLIMBRA/`)

One row per municipality × variable × combination of the file's dimensions.

| Column | Description |
|---|---|
| `CD_MUN`, `NM_MUN`, `NM_UF` | IBGE municipality code (string), name and state |
| `source` | `Observations` or `CLIMBRA` |
| `dataset` | CHIRPS, MERGE, XAVIER, or `ensemble` |
| `indicator`, `category` | indicator name (e.g. Rx5day) and hazard category |
| `file_kind` | source file type: `climatology`, `return_period`, `params`, `trend`, `timeseries` (observations); `climatology`, `return_period`, `timeseries`, `anomaly_climatology`, `anomaly_return_period`, `rp_shift`, `delta_<DATASET>_climatology`, `delta_<DATASET>_return_period` (CLIMBRA) |
| `variable`, `units` | NetCDF variable name and its units attribute |
| *dimension columns* | one column per NetCDF dimension, e.g. `ensemble`, `GWL`, `return_period` |
| `mean` | area-weighted mean over valid cells |
| `std` | area-weighted standard deviation across cells |
| `min`, `max` | minimum / maximum cell value (cells covering ≥ {MINMAX_MIN_FRAC:.0%} of their area) |
| `n_pixels` | valid pixel-equivalents used (sum of covered cell fractions) |
| `valid_area_frac` | share of the polygon (or buffer) area with valid data |
| `method` | `polygon`, `{B}`, `nearest` or `none` |
| `nearest_km` | distance to the nearest cell (method `nearest` only) |
| `agree_area_frac` | agreement variables only: share of the area where ≥ 66% of models agree on the sign of change |
| `robust` | agreement variables only: 1 if `agree_area_frac` ≥ {ROBUST_AREA_FRAC:g}, else 0 |
| `sig_area_frac` | trend files only: share of the area with p < {P_SIG} |
| `mean_sig` | trend files only (`slopes`): mean slope over significant cells |

**Agreement variables** (`*_model_agreement`) are NaN where fewer than 66% of models agree on
the sign of change. Their `agree_area_frac` is computed relative to the area with valid
anomaly data; `mean` is the mean of the agreement variable over agreeing cells.

**CLIMBRA percentage anomalies** (`*_anom_perc`) are averaged cell by cell (mean of the
per-cell % change), which can differ slightly from the % change of the municipal mean.

## Wide GeoPackages (`maps/`)

One row per municipality with geometry and a selection of key columns, ready for GIS:

* `<DATASET>_clim`, `<DATASET>_RL<T>`: observed climatology and T-year return levels (mean)
* `<DATASET>_trend`, `<DATASET>_trend_sig`: mean trend slope and share of area with significant trend
* `CB_clim_<pct|abs>_<GWL>C`, `CB_RL{WIDE_RP:g}_<pct|abs>_<GWL>C`: CLIMBRA ensemble {WIDE_ENSEMBLE} change in
  climatology / {WIDE_RP:g}-year return level at each GWL (e.g. `2C`, `1p5C` = 1.5 °C)
* `CB_clim_agree_<GWL>C`, `CB_RL{WIDE_RP:g}_agree_<GWL>C`: share of area with ≥ 66% model agreement
* `CB_histRL{WIDE_RP:g}[_<var>]`, `CB_newRP{WIDE_RP:g}[_<var>]_<GWL>C`, `CB_PR{WIDE_RP:g}[_<var>]_<GWL>C`,
  `CB_agreeRP{WIDE_RP:g}[_<var>]_<GWL>C`: return-period shift of the present-day {WIDE_RP:g}-year event
  (from `ensemble_rp_shift.nc`): present-day level, new return period (years), probability ratio
  (ensemble {WIDE_ENSEMBLE}) and share of area with ≥ 66% model agreement on the direction of change.
  `_<var>` is added when the file holds several variables (e.g. HI, FWI).

Each GeoPackage has a `_columns.csv` file describing every column and its units.

Files: {', '.join(f'`{m}`' for m in maps) if maps else '_none yet_'}

## Processing summary

Method counts refer to the first slice of each variable (`n_buffer` counts buffered
municipalities, `n_missing` those without data). `rel_diff_pct` compares the
area-weighted Brazil mean of the municipal values with the grid mean over Brazil (a
consistency check; small differences come from buffers and missing municipalities).

{_md_table(checks[['source', 'dataset', 'indicator', 'file_kind', 'variable', 'rel_diff_pct',
                   'n_polygon', 'n_buffer', 'n_nearest', 'n_missing']] if checks is not None else None)}
Output files ({len(inv)}):

{_md_table(pd.DataFrame(inv)) if inv else '_none yet_'}
## Reading the data

```python
import pandas as pd, geopandas as gpd
df = pd.read_parquet('municipalities/CLIMBRA/Rainfall/Rx5day/ensemble_anomaly_climatology.parquet')
sub = df[(df.variable == 'Rx5day_anom_perc') & (df.ensemble == 'median') & (df.GWL == '2')]
geo = gpd.read_file('municipalities/geometry/municipalities_2024.gpkg', layer='municipalities')
m = geo.merge(sub[['CD_MUN', 'mean', 'method']], on='CD_MUN')
```
"""
    os.makedirs(P.out_dir, exist_ok=True)
    with open(P.readme_file, 'w', encoding='utf-8') as fh:
        fh.write(txt)
    print(f'[readme] {P.readme_file}')


# =============================================================================
# Driver and loaders
# =============================================================================
def run_municipalities(indicators=None, sources=('Observations', 'CLIMBRA'), overwrite=False,
                       overwrite_geometry=False, overwrite_weights=False, wide=True, readme=True,
                       write_csv=False, skip_kinds=(), cfg=None):
    """
    Extract municipal statistics for the given indicators (default: all in INDICATOR_HAZARD).
    Existing outputs are skipped unless overwrite=True. Rebuilding the geometry also
    rebuilds the pixel weights. ``skip_kinds``: file kinds (e.g. 'timeseries') to leave out. ``cfg``: from ``hazbra.config.load_config`` (default: production).
    """
    P = _default_paths(cfg)
    if overwrite_geometry:
        overwrite_weights = True
    os.makedirs(P.out_dir, exist_ok=True)
    gdf, gbuf = prepare_geometry(P, overwrite_geometry)
    build_states(P, gdf, overwrite_geometry)
    geo = _geo_context(gdf, gbuf, P)
    indicators = indicators or list(indicators_dict)
    checks = []
    for ind in indicators:
        if ind not in indicators_dict:
            print(f'[skip] {ind} not in indicators_dict')
            continue
        items = [it for it in discover_files(P, ind, sources) if it['kind'] not in skip_kinds]
        if not items:
            print(f'[skip] no input files for {ind}')
            continue
        print(f'=== {ind}: {len(items)} files ===')
        for it in items:
            it['indicator'] = ind
            if _skip(it['out'], overwrite):
                continue
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                df, chk = process_file(it, geo, overwrite_weights)
            path = _save_table(df, it['out'], write_csv)
            checks += chk
            print(f'[done] {os.path.relpath(path, P.out_dir)}: {len(df):,} rows in {time.time() - t0:.0f} s')
        if wide:
            build_wide(P, ind, gdf, overwrite)
    _update_checks(P, checks)
    if readme:
        write_readme(P)
    print('[finished]')


def load_table(indicator, source='CLIMBRA', dataset=None, kind=None, cfg=None, columns=None):
    """Concatenate the long tables of one indicator (optionally filtered by dataset / kind).

    ``kind`` / ``dataset`` select the files by name before reading (CLIMBRA: ``ensemble_<kind>``,
    Observations: ``<DATASET>_<kind>``), so large tables of other kinds (time series) are not loaded.
    ``columns`` restricts the columns read.
    """
    P = _default_paths(cfg)
    cat = indicators_dict[indicator]
    root = P.out_dir
    if P.out_dir_ro and not os.path.isdir(os.path.join(root, source, cat, indicator)):
        root = P.out_dir_ro                      # test mode: fall back to the production tables (read-only)
    files = sorted(glob.glob(os.path.join(root, source, cat, indicator, '*.parquet')) +
                   glob.glob(os.path.join(root, source, cat, indicator, '*.csv')))
    if kind:      # file stems are <DATASET>_<kind> (Observations) or ensemble_<kind> (CLIMBRA)
        files = [f for f in files
                 if os.path.splitext(os.path.basename(f))[0].split('_', 1)[-1] == kind]
    if dataset and source == 'Observations':
        files = [f for f in files if os.path.basename(f).upper().startswith(dataset.upper() + '_')]
    dfs = [_read_table(f, columns) for f in files]
    if not dfs:
        raise FileNotFoundError(f'no tables for {indicator} in {source}')
    df = pd.concat(dfs, ignore_index=True)
    if dataset and 'dataset' in df:
        df = df[df['dataset'] == dataset]
    if kind and 'file_kind' in df:
        df = df[df['file_kind'] == kind]
    return df


def load_geometry(cfg=None):
    P = _default_paths(cfg)
    return gpd.read_file(_readable(P, 'municipalities_2024.gpkg'), layer='municipalities')


def load_states(cfg=None):
    """State outlines (NM_UF, geometry): ``geometry/states_2024.gpkg`` or, if absent, the union of
    the municipalities per state (computed in memory, nothing is written)."""
    P = _default_paths(cfg)
    path = _readable(P, 'states_2024.gpkg')
    if os.path.exists(path):
        return gpd.read_file(path)
    gdf = load_geometry(cfg)
    rows = [(uf, _polygonal(_safe_union_all([_polygonal(x) for x in grp.geometry])))
            for uf, grp in gdf.groupby('NM_UF')]
    return gpd.GeoDataFrame({'NM_UF': [r[0] for r in rows]}, geometry=[r[1] for r in rows], crs=4326)


# =============================================================================
# Command line
# =============================================================================
def main(argv=None):
    from .config import load_config, add_common_args
    ap = argparse.ArgumentParser(description='Aggregate the HAZBRA NetCDF product to IBGE municipalities.')
    add_common_args(ap)
    ap.add_argument('--indicators', nargs='+', default=None, metavar='IND',
                    help=f'indicators to process (default: all; with --test: {" ".join(TEST_INDICATORS)})')
    ap.add_argument('--sources', nargs='+', choices=['Observations', 'CLIMBRA'],
                    default=['Observations', 'CLIMBRA'])
    ap.add_argument('--overwrite-geometry', action='store_true',
                    help='rebuild the cleaned municipal geometry (and therefore the pixel weights)')
    ap.add_argument('--overwrite-weights', action='store_true', help='rebuild the cached pixel weights')
    ap.add_argument('--skip-kinds', nargs='*', default=None, metavar='KIND',
                    help=f'file kinds to skip, e.g. timeseries (default: none; with --test: {" ".join(TEST_SKIP_KINDS)})')
    ap.add_argument('--no-wide', action='store_true', help='skip the wide GeoPackages in maps/')
    ap.add_argument('--no-readme', action='store_true', help='do not regenerate README.md')
    ap.add_argument('--write-csv', action='store_true', help='also write CSV copies of the long tables')
    args = ap.parse_args(argv)

    cfg = load_config(test=args.test)
    indicators = args.indicators or (TEST_INDICATORS if args.test else None)
    run_municipalities(indicators, sources=tuple(args.sources),
                       overwrite=args.overwrite or bool(cfg.calc.overwrite),
                       overwrite_geometry=args.overwrite_geometry,
                       overwrite_weights=args.overwrite_weights,
                       wide=not args.no_wide, readme=not args.no_readme,
                       write_csv=args.write_csv, cfg=cfg,
                       skip_kinds=tuple(args.skip_kinds if args.skip_kinds is not None
                                        else (TEST_SKIP_KINDS if args.test else ())))


if __name__ == '__main__':
    main()

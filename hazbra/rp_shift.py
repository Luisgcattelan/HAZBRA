"""
CLIMBRA return-period shift, pixel level, per model and ensemble.

For each model, variable and grid cell, the GEV fitted to the present period gives the
historical T-year level z_T. The GEV fitted at each global warming level (GWL) gives the new
probability of reaching that level, hence its new return period and the probability ratio
PR = p_GWL / p_present (PR > 1: the event becomes more frequent).

Inputs (per indicator):
    <netcdf_climbra>/<Category>/<Indicator>/individual_models/<MODEL>_params.nc
        every variable with a 'dparams' dimension (c, loc, scale) is processed
    <netcdf_climbra>/<Category>/<Indicator>/individual_models/<MODEL>_return_period.nc
        used once per variable to detect the GEV convention (shape sign, tail)

Outputs:
    individual_models/<MODEL>_rp_shift.nc
    individual_models/rp_shift_conventions.json
    ensemble/ensemble_rp_shift.nc
    <netcdf_climbra>/README_return_period_shift.md

Usage (see 02_postprocess/03_return_period_shift.py):
    from hazbra import config, rp_shift as crs
    crs.configure(config.load_config(test=False))
    crs.run('Rx1day')                 # one indicator
    crs.run_all()                     # every indicator with parameter files
    crs.show_conventions('TNN')       # which GEV convention was detected
"""

import os
import json
import time
import datetime
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xarray as xr
from scipy.stats import genextreme


# =============================================================================
# Configuration
# =============================================================================
CLIMBRA_DIR = None                       # set by configure(cfg) -> cfg.paths.netcdf_climbra
README_FILE = None                       # default: CLIMBRA_DIR/README_return_period_shift.md
OVERWRITE = False
N_JOBS = 1                               # >1: models in parallel (processes)

PRESENT = 'present'                      # GWL label of the reference period
EVENT_RPS = (5, 10, 25, 50, 100)         # historical events tracked (years)
MAX_RP = 1000.0                          # new return periods are capped here
AGREEMENT_THRESHOLD = 0.66               # agreement on direction of frequency change
MIN_COUNT_LEVEL = 1.0                    # count variables (days): z_T below this is masked
CHECK_T = 25                             # return period used for the convention check
CHECK_TOL_PCT = 1.0                      # warn if the best convention still differs by more

# indicators whose extremes are low values (GEV fitted on minima)
LOWER_TAIL_INDICATORS = {'TNN', 'SPI3', 'SPI6'}
# Expected convention: parameters from xclim.indices.stats.fit / frequency_analysis
# (dist="genextreme", scipy convention c = -xi). mode="max" -> level = isf(1/T) ('upper');
# mode="min" -> genextreme fitted directly to the minima, level = ppf(1/T) ('lower').
EXPECTED_C_SIGN = 1
EXPECTED_MODE = {'upper': 'upper', 'lower': 'lower'}
# count-like variables (days per year): mask degenerate low return levels
COUNT_INDICATORS = {'CDD', 'CDD_season', 'CWD', 'Frost', 'WSDI', 'Dry_Hot'}

from .constants import INDICATOR_HAZARD as indicators_dict


def configure(cfg):
    """Point the module at the product folder of a HAZBRA config (production or test)."""
    global CLIMBRA_DIR, OVERWRITE
    CLIMBRA_DIR = str(cfg.paths.netcdf_climbra)
    OVERWRITE = bool(cfg.calc.overwrite)


STAT_Q = [0.1, 0.5, 0.9]
STAT_NAMES = ['p10', 'median', 'p90']
MODES = {
    'upper': 'GEV fitted to maxima; event = value at or above z_T',
    'negated': 'GEV fitted to negated minima (-x); event = value at or below z_T',
    'lower': 'GEV fitted directly to minima; event = value at or below z_T (lower tail of that GEV)',
}


# =============================================================================
# Paths and small helpers
# =============================================================================
def _ind_dir(indicator):
    return os.path.join(CLIMBRA_DIR, indicators_dict[indicator], indicator)


def _model_dir(indicator):
    return os.path.join(_ind_dir(indicator), 'individual_models')


def _ens_path(indicator):
    return os.path.join(_ind_dir(indicator), 'ensemble', 'ensemble_rp_shift.nc')


def _conv_path(indicator):
    return os.path.join(_model_dir(indicator), 'rp_shift_conventions.json')


def _readme_path():
    return README_FILE or os.path.join(CLIMBRA_DIR, 'README_return_period_shift.md')


def _skip(path, overwrite):
    if os.path.exists(path) and not overwrite:
        print(f'[skip] exists: {path}')
        return True
    return False


def _save_nc(ds, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    enc = {v: {'zlib': True, 'complevel': 4} for v in ds.data_vars
           if np.issubdtype(ds[v].dtype, np.number)}
    tmp = path + '.tmp'
    ds.to_netcdf(tmp, encoding=enc)
    os.replace(tmp, path)


def _gkey(x):
    try:
        return f'{float(x):g}'
    except (TypeError, ValueError):
        return str(x)


def _std_latlon(da):
    lat = next((n for n in ('lat', 'latitude', 'y') if n in da.coords), None)
    lon = next((n for n in ('lon', 'longitude', 'x') if n in da.coords), None)
    if lat is None or lon is None:
        raise ValueError(f'lat/lon not found in {list(da.coords)}')
    da = da.rename({k: v for k, v in {lat: 'lat', lon: 'lon'}.items() if k != v})
    if float(da['lon'].max()) > 180:
        da = da.assign_coords(lon=((da['lon'] + 180) % 360) - 180)
    return da.sortby('lat').sortby('lon')


def list_models(indicator):
    d = _model_dir(indicator)
    if not os.path.isdir(d):
        return []
    return sorted(f[:-len('_params.nc')] for f in os.listdir(d) if f.endswith('_params.nc'))


def param_variables(indicator, model):
    with xr.open_dataset(os.path.join(_model_dir(indicator), f'{model}_params.nc')) as ds:
        return [v for v in ds.data_vars if 'dparams' in ds[v].dims]


def _tail(indicator):
    return 'lower' if indicator in LOWER_TAIL_INDICATORS else 'upper'


def _is_count(indicator, var):
    return indicator in COUNT_INDICATORS or var.lower().startswith('days')


# =============================================================================
# GEV (scipy convention after applying the detected sign)
# =============================================================================
def _bad(c, loc, scale):
    return ~np.isfinite(c) | ~np.isfinite(loc) | ~(scale > 0)


def _gev(fn, x, c, loc, scale):
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = getattr(genextreme, fn)(x, c, loc=loc, scale=scale)
    return np.where(_bad(c, loc, scale) | ~np.isfinite(x), np.nan, out)


def level(T, c, loc, scale, mode):
    """T-year level under a tail mode."""
    p = 1.0 / np.asarray(T, float)
    if mode == 'upper':
        return _gev('isf', p, c, loc, scale)
    if mode == 'lower':
        return _gev('ppf', p, c, loc, scale)
    if mode == 'negated':
        return -_gev('isf', p, c, loc, scale)
    raise ValueError(mode)


def event_prob(z, c, loc, scale, mode):
    """Annual probability of the event defined by level z under a tail mode."""
    if mode == 'upper':
        return _gev('sf', z, c, loc, scale)
    if mode == 'lower':
        return _gev('cdf', z, c, loc, scale)
    if mode == 'negated':
        return _gev('sf', -z, c, loc, scale)
    raise ValueError(mode)


def load_params(indicator, model, var):
    """(c, loc, scale) as DataArrays (GWL, lat, lon), GWL labels normalised."""
    with xr.open_dataset(os.path.join(_model_dir(indicator), f'{model}_params.nc')) as ds:
        da = _std_latlon(ds[var].load())
    da = da.assign_coords(GWL=[_gkey(g) for g in da['GWL'].values],
                          dparams=[str(p) for p in da['dparams'].values])
    out = [da.sel(dparams=p, drop=True).transpose('GWL', 'lat', 'lon') for p in ('c', 'loc', 'scale')]
    return tuple(out)


# =============================================================================
# Convention detection
# =============================================================================
def _reference_level(indicator, model, var, T):
    """T-year present-day level from <MODEL>_return_period.nc, or None."""
    path = os.path.join(_model_dir(indicator), f'{model}_return_period.nc')
    if not os.path.exists(path):
        return None, None
    with xr.open_dataset(path) as ds:
        if var not in ds.data_vars:
            return None, None
        ref = _std_latlon(ds[var].load())
    rp_dim = next((d for d in ref.dims if 'return' in d.lower() or d.lower() in ('rp', 'period')), None)
    if rp_dim is None:
        return None, None
    vals = np.asarray(ref[rp_dim].values, float)
    t_used = float(vals[np.argmin(np.abs(vals - T))])
    ref = ref.sel({rp_dim: ref[rp_dim].values[np.argmin(np.abs(vals - T))]})
    if 'GWL' in ref.dims:
        keys = [_gkey(g) for g in ref['GWL'].values]
        if PRESENT not in keys:
            return None, None
        ref = ref.isel(GWL=keys.index(PRESENT))
    other = [d for d in ref.dims if d not in ('lat', 'lon')]
    if other:
        ref = ref.isel({d: 0 for d in other})
    return ref.transpose('lat', 'lon'), t_used


def detect_convention(indicator, var, models, T=CHECK_T):
    """
    Try every (shape sign, tail mode) and keep the one that reproduces the present-day
    T-year level in the model's own return_period file.
    """
    tail = _tail(indicator)
    modes = ['upper'] if tail == 'upper' else ['lower', 'negated', 'upper']
    for m in models:
        ref, t_used = _reference_level(indicator, m, var, T)
        if ref is None:
            continue
        c, loc, scale = load_params(indicator, m, var)
        cp, lp, sp = (x.sel(GWL=PRESENT).reindex_like(ref, method='nearest', tolerance=1e-3).values
                      for x in (c, loc, scale))
        r = ref.values
        cand = {}
        for sign in (1, -1):
            for mode in modes:
                z = level(t_used, sign * cp, lp, sp, mode)
                ok = np.isfinite(z) & np.isfinite(r)
                if ok.sum() < 10:
                    continue
                d = z[ok] - r[ok]
                cand[f'{sign:+d}|{mode}'] = dict(
                    mae=float(np.mean(np.abs(d))),
                    rel_err_pct=float(np.median(np.abs(d) / np.maximum(np.abs(r[ok]), 1e-9)) * 100),
                    n=int(ok.sum()))
        if not cand:
            continue
        # Median relative error (not the mean): the stored return levels were outlier-filtered (interpolated or
        # capped at 365 days), and those few cells dominate a mean error and can select the wrong sign.
        # Ties are resolved in favour of the xclim convention.
        def _rank(k):
            sgn, md = k.split('|')
            exp = int(sgn) == EXPECTED_C_SIGN and md == EXPECTED_MODE[tail]
            return (round(cand[k]['rel_err_pct'], 6), 0 if exp else 1, cand[k]['mae'])
        best = min(cand, key=_rank)
        sign, mode = best.split('|')
        conv = dict(c_sign=int(sign), mode=mode, tail=tail, checked=True, ref_model=m,
                    T_check=t_used, **cand[best], candidates=cand)
        conv['expected'] = (conv['c_sign'] == EXPECTED_C_SIGN and conv['mode'] == EXPECTED_MODE[tail])
        flag = '' if conv['rel_err_pct'] <= CHECK_TOL_PCT else '  <-- WARNING: no exact match'
        if not conv['expected']:
            flag += (f'  <-- NOTE: differs from the xclim expectation (c_sign={EXPECTED_C_SIGN:+d}, '
                     f'mode={EXPECTED_MODE[tail]})')
        print(f'[convention] {indicator}/{var}: c_sign={sign}, mode={mode} '
              f'(median rel. error {conv["rel_err_pct"]:.3f}% vs {m}_return_period.nc){flag}')
        return conv
    default = dict(c_sign=EXPECTED_C_SIGN, mode=EXPECTED_MODE[tail], tail=tail, checked=False,
                   expected=True, ref_model=None, T_check=None, mae=None, rel_err_pct=None)
    print(f'[convention] {indicator}/{var}: no reference file found -> xclim default '
          f'c_sign={EXPECTED_C_SIGN:+d}, mode={default["mode"]} (unchecked)')
    return default


def get_conventions(indicator, variables, models, overwrite=False):
    path = _conv_path(indicator)
    conv = {}
    if os.path.exists(path) and not overwrite:
        with open(path) as fh:
            conv = json.load(fh)
    changed = False
    for v in variables:
        if v not in conv:
            conv[v] = detect_convention(indicator, v, models)
            changed = True
    if changed or not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fh:
            json.dump(conv, fh, indent=2)
    return conv


def show_conventions(indicator):
    with open(_conv_path(indicator)) as fh:
        conv = json.load(fh)
    for v, c in conv.items():
        print(f'{v:28s} c_sign={c["c_sign"]:+d} mode={c["mode"]:8s} checked={c["checked"]} '
              f'rel_err={c.get("rel_err_pct")}')
    return conv


# =============================================================================
# Per-model computation
# =============================================================================
def compute_model(indicator, model, conventions, overwrite=False):
    out = os.path.join(_model_dir(indicator), f'{model}_rp_shift.nc')
    if _skip(out, overwrite):
        return out
    ev = np.asarray(EVENT_RPS, float)
    data, coords, masked = {}, None, {}
    for v in param_variables(indicator, model):
        conv = conventions.get(v)
        if conv is None:
            continue
        c, loc, scale = load_params(indicator, model, v)
        gwl_all = list(c['GWL'].values)
        if PRESENT not in gwl_all:
            print(f'[skip] {model}/{v}: no "{PRESENT}" period')
            continue
        fut = [g for g in gwl_all if g != PRESENT]
        s, mode = conv['c_sign'], conv['mode']
        prm = {g: (s * c.sel(GWL=g).values, loc.sel(GWL=g).values, scale.sel(GWL=g).values)
               for g in gwl_all}
        cp, lp, sp = prm[PRESENT]
        z = level(ev[:, None, None], cp[None], lp[None], sp[None], mode)            # (event, lat, lon)
        if _is_count(indicator, v):
            valid_before = np.isfinite(z)
            z = np.where(z >= MIN_COUNT_LEVEL, z, np.nan)
            masked[v] = float(1 - np.isfinite(z).sum() / max(valid_before.sum(), 1))
        new_rp = np.full((len(fut),) + z.shape, np.nan)
        for i, g in enumerate(fut):
            p = event_prob(z, prm[g][0][None], prm[g][1][None], prm[g][2][None], mode)
            with np.errstate(divide='ignore', invalid='ignore'):
                T = np.where(p > 0, 1.0 / p, np.inf)
            new_rp[i] = np.where(np.isfinite(p), np.clip(T, 1.0, MAX_RP), np.nan)
        pr = ev[None, :, None, None] / new_rp

        f32 = np.float32
        data[f'{v}_hist_return_level'] = xr.DataArray(
            z.astype(f32), dims=('event_rp', 'lat', 'lon'),
            attrs=dict(long_name=f'Present-day T-year level of {v}', tail_mode=mode))
        data[f'{v}_new_return_period'] = xr.DataArray(
            new_rp.astype(f32), dims=('GWL', 'event_rp', 'lat', 'lon'),
            attrs=dict(units='years', long_name=f'Return period at the GWL of the historical T-year {v} event',
                       cap=MAX_RP))
        data[f'{v}_probability_ratio'] = xr.DataArray(
            pr.astype(f32), dims=('GWL', 'event_rp', 'lat', 'lon'),
            attrs=dict(units='1', long_name=f'p_GWL / p_present for the historical T-year {v} event'))
        coords = dict(event_rp=ev, GWL=fut, lat=c['lat'].values, lon=c['lon'].values)

    if not data:
        print(f'[skip] {model}: nothing computed')
        return None
    ds = xr.Dataset(data, coords=coords)
    ds.attrs.update(model=model, indicator=indicator, max_return_period=MAX_RP,
                    min_count_level=MIN_COUNT_LEVEL,
                    conventions=json.dumps({v: {k: conventions[v][k] for k in ('c_sign', 'mode', 'tail', 'checked')}
                                            for v in conventions}),
                    masked_share_count_vars=json.dumps(masked),
                    created=datetime.datetime.now().isoformat(timespec='seconds'))
    _save_nc(ds, out)
    print(f'[done] {indicator}/{model}')
    return out


def _compute_model_job(args):
    indicator, model, conventions, overwrite = args
    return compute_model(indicator, model, conventions, overwrite)


# =============================================================================
# Ensemble
# =============================================================================
def _quantiles(da):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)          # all-NaN cells (masked / outside land)
        q = da.quantile(STAT_Q, dim='model', skipna=True)
    q = q.assign_coords(quantile=STAT_NAMES).rename(quantile='ensemble').astype(np.float32)
    q.attrs = da.attrs
    return q


def build_ensemble(indicator, models, overwrite=False):
    out = _ens_path(indicator)
    if _skip(out, overwrite):
        return out
    files, used = [], []
    for m in models:
        f = os.path.join(_model_dir(indicator), f'{m}_rp_shift.nc')
        if os.path.exists(f):
            files.append(f)
            used.append(m)
        else:
            print(f'[warn] missing per-model file for {m}')
    if not files:
        print(f'[skip] {indicator}: no per-model rp_shift files')
        return None
    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            parts.append(ds.load())
    ds = xr.concat(parts, dim='model', join='outer', data_vars='all', coords='minimal',
                   compat='override', combine_attrs='drop')
    ds = ds.assign_coords(model=np.asarray(used, dtype=str))

    stems = [k[:-len('_hist_return_level')] for k in ds.data_vars if k.endswith('_hist_return_level')]
    out_vars = {}
    for v in stems:
        rp, pr, hz = ds[f'{v}_new_return_period'], ds[f'{v}_probability_ratio'], ds[f'{v}_hist_return_level']
        n = pr.notnull().sum('model')
        nv = n.where(n > 0)
        more = (pr > 1).sum('model') / nv
        less = (pr < 1).sum('model') / nv
        agree = np.maximum(more, less)
        out_vars[f'{v}_hist_return_level'] = _quantiles(hz)
        out_vars[f'{v}_new_return_period'] = _quantiles(rp)
        out_vars[f'{v}_probability_ratio'] = _quantiles(pr)
        out_vars[f'{v}_frac_more_frequent'] = more.astype(np.float32)
        out_vars[f'{v}_rp_model_agreement'] = agree.where(agree >= AGREEMENT_THRESHOLD).astype(np.float32)
        out_vars[f'{v}_n_models'] = n.astype(np.int16)
        out_vars[f'{v}_frac_capped'] = ((rp >= MAX_RP).sum('model') / nv).astype(np.float32)
        out_vars[f'{v}_rp_model_agreement'].attrs['long_name'] = (
            f'Fraction of models agreeing on the direction of the frequency change '
            f'(NaN where < {AGREEMENT_THRESHOLD:.0%})')
        out_vars[f'{v}_frac_more_frequent'].attrs['long_name'] = 'Fraction of models with PR > 1'
    ens = xr.Dataset(out_vars)
    ens.attrs.update(indicator=indicator, models=', '.join(used), n_models=len(used),
                     agreement_threshold=AGREEMENT_THRESHOLD, max_return_period=MAX_RP,
                     statistics='p10, median, p90 across models, per grid cell',
                     created=datetime.datetime.now().isoformat(timespec='seconds'))
    _save_nc(ens, out)
    print(f'[done] ensemble {indicator} ({len(used)} models, {len(stems)} variables)')
    return out


# =============================================================================
# README
# =============================================================================
def write_readme():
    rows = []
    for ind in indicators_dict:
        cp = _conv_path(ind)
        if not os.path.exists(cp):
            continue
        with open(cp) as fh:
            conv = json.load(fh)
        n_models, ens_ok = '', os.path.exists(_ens_path(ind))
        if ens_ok:
            with xr.open_dataset(_ens_path(ind)) as ds:
                n_models = ds.attrs.get('n_models', '')
        for v, c in conv.items():
            rel = c.get('rel_err_pct')
            rows.append(f"| {ind} | `{v}` | {c['tail']} | {c['mode']} | {c['c_sign']:+d} | "
                        f"{'yes' if c['checked'] else 'no'} | "
                        f"{'' if rel is None else f'{rel:.3f}'} | {n_models} | {'yes' if ens_ok else 'no'} |")
    table = '\n'.join(rows) if rows else '| – | – | – | – | – | – | – | – | – |'
    ev = ', '.join(f'{e:g}' for e in EVENT_RPS)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    txt = f"""# CLIMBRA – return-period shift of historical extremes

Generated by `hazbra/rp_shift.py` (02_postprocess/03_return_period_shift.py), last update {now}.

For every CLIMBRA model, variable and grid cell, the GEV distribution fitted to the present
period (`GWL = {PRESENT}`) gives the historical T-year level z_T (T = {ev} years). The GEV
fitted at each global warming level (GWL) gives the new annual probability of reaching z_T,
hence its new return period T' = 1/p and the probability ratio PR = p_GWL / p_present = T / T'.
Everything is computed per model; ensemble statistics are computed afterwards, per grid cell.

* PR > 1 (T' < T): the historical event becomes more frequent; PR < 1: less frequent.
* New return periods are capped at {MAX_RP:g} years (p = 0 under a bounded GEV tail means the
  historical level is no longer reached in that model; it is stored as {MAX_RP:g}).
* Count variables (days per year) are masked where z_T < {MIN_COUNT_LEVEL:g} day (degenerate fits in
  places where the event almost never occurs).

## Files

```
<Category>/<Indicator>/
├── individual_models/
│   ├── <MODEL>_params.nc              input: GEV parameters (c, loc, scale) per GWL
│   ├── <MODEL>_rp_shift.nc            output: per-model return-period shift
│   └── rp_shift_conventions.json      output: GEV parameter convention check per variable (internal)
└── ensemble/
    └── ensemble_rp_shift.nc           output: ensemble statistics and model agreement
```

### Per-model file `<MODEL>_rp_shift.nc`

For every variable `<var>` of the parameter file:

| Variable | Dimensions | Description |
|---|---|---|
| `<var>_hist_return_level` | event_rp, lat, lon | present-day T-year level z_T (units of `<var>`) |
| `<var>_new_return_period` | GWL, event_rp, lat, lon | return period of z_T at the GWL (years, ≤ {MAX_RP:g}) |
| `<var>_probability_ratio` | GWL, event_rp, lat, lon | p_GWL / p_present |

### Ensemble file `ensemble_rp_shift.nc`

`ensemble` = p10, median, p90 across models (computed per grid cell, like the other CLIMBRA
ensemble files).

| Variable | Dimensions | Description |
|---|---|---|
| `<var>_hist_return_level` | ensemble, event_rp, lat, lon | present-day T-year level |
| `<var>_new_return_period` | ensemble, GWL, event_rp, lat, lon | new return period (years) |
| `<var>_probability_ratio` | ensemble, GWL, event_rp, lat, lon | probability ratio |
| `<var>_frac_more_frequent` | GWL, event_rp, lat, lon | fraction of models with PR > 1 |
| `<var>_rp_model_agreement` | GWL, event_rp, lat, lon | fraction of models agreeing on the direction of change; NaN where < {AGREEMENT_THRESHOLD:.0%} (same convention as `*_model_agreement`) |
| `<var>_n_models` | GWL, event_rp, lat, lon | models with a valid value |
| `<var>_frac_capped` | GWL, event_rp, lat, lon | fraction of models whose new return period hit the {MAX_RP:g}-year cap |

Note: percentiles of the return period and of PR are computed independently; the p10 of the
return period corresponds to the p90 of PR (most frequent end).
"""
    with open(_readme_path(), 'w', encoding='utf-8') as fh:
        fh.write(txt)
    print(f'[readme] {_readme_path()}')


# =============================================================================
# Driver
# =============================================================================
def run(indicator, overwrite=None, n_jobs=None, readme=True):
    overwrite = OVERWRITE if overwrite is None else overwrite
    n_jobs = N_JOBS if n_jobs is None else n_jobs
    if indicator not in indicators_dict:
        raise KeyError(f'{indicator} not in indicators_dict')
    models = list_models(indicator)
    if not models:
        print(f'[skip] {indicator}: no *_params.nc in {_model_dir(indicator)}')
        return None
    with xr.open_dataset(os.path.join(_model_dir(indicator), f'{models[0]}_params.nc')) as ds:
        seasonal = 'season' in ds.dims
    if seasonal:      # per-season GEV parameters (CDD_season, SPI3): the shift is not defined for these
        print(f'[skip] {indicator}: seasonal indicator, return-period shift not computed')
        return None
    variables = param_variables(indicator, models[0])
    print(f'=== {indicator}: {len(models)} models, variables: {variables} ===')
    t0 = time.time()
    conv = get_conventions(indicator, variables, models, overwrite)
    jobs = [(indicator, m, conv, overwrite) for m in models]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            list(ex.map(_compute_model_job, jobs))
    else:
        for j in jobs:
            _compute_model_job(j)
    build_ensemble(indicator, models, overwrite)
    if readme:
        write_readme()
    print(f'[finished] {indicator} in {time.time() - t0:.0f} s')
    return _ens_path(indicator)


def run_all(indicators=None, overwrite=None, n_jobs=None):
    for ind in (indicators or list(indicators_dict)):
        try:
            run(ind, overwrite, n_jobs, readme=False)
        except Exception as err:            # keep going with the other indicators
            print(f'[error] {ind}: {type(err).__name__}: {err}')
    write_readme()

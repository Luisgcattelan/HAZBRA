# HAZBRA – processing code (organised for review)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24319623.svg)](https://doi.org/10.5281/zenodo.24319623)

Code that produces the **HAZBRA** climate-hazard dataset for Brazil (16 annual indicators from CHIRPS, MERGE,
BR-DWGD/XAVIER observations and the 19-model CLIMBRA CMIP6 ensemble at +1.5/+2/+3 °C, grid and municipal scale)
and the figures/numbers of the accompanying data paper.

The input data (CLIMBRA, CHIRPS, MERGE, BR-DWGD) are not redistributed here; see the data paper for their sources.
All paths are set in `config/paths.yaml` (they point to the JASMIN workspace where HAZBRA was produced –
edit that file to run elsewhere).

```
config/            paths.yaml (all absolute paths), test.yaml (test configuration), SLURM settings
hazbra/            python package: indicator registry, pipelines, statistics, municipal aggregation, plotting
01_calculation/    daily data  ->  per-model / per-dataset indicator products + ensemble statistics
02_postprocess/    anomalies, model agreement, delta-change projections, return-period shift, municipal tables
03_paper_analysis/ figures and numbers of the paper (only those that appear in the paper)
tests/             end-to-end test run on a small box + regression check against the existing product
outputs/figures/   figures of the paper (PNG)
tools/             zenodo_dataset/: packing + upload of the dataset to Zenodo, dataset user guide
environment.yml    software environment
```

## Pipeline

Everything runs with one command on SLURM, with job dependencies:

```bash
bash run_pipeline.sh                            # production: all 16 indicators
bash run_pipeline.sh --indicators "Rx1day HI"   # selected indicators
bash run_pipeline.sh --dry                      # print the sbatch commands only
bash run_pipeline.sh --test --local             # test configuration, sequentially on the current node
```

```
for every indicator ─┬─ calc: 1 job per CLIMBRA model ──────────┐
                     └─ calc: 1 job per observational dataset ──┤ afterok
                                                                post: ensemble → anomalies + agreement
                                                                      → return-period shift → delta change
all post jobs ─────────────────────────────────────────────────┐ afterok
                                                               final: return-period README, municipal tables,
                                                                      outlier tables [+ paper figures: --figures]
```

Options: `--overwrite` (default: skip existing files), `--figures`, `--gwl-table` (rebuild `GWL_CLIMBRA.csv` first),
`--stages calc,post,final`, `--sources climbra,obs`, `--datasets "CHIRPS MERGE"` (re-run only some calculation jobs). Resources per stage/indicator: `config/slurm_resources.csv`; account, partition, conda env:
`config/slurm.env`; logs: `logs/`.

| step | script (each runs on its own too, `--help`) | writes (under `paths.netcdf_*`) |
|---|---|---|
| 0 | `01_calculation/00_gwl_table.py` | `GWL_CLIMBRA.csv` (Table 1) |
| calc | `01_calculation/run_climbra.py --indicator X [--models M]` | `CLIMBRA/<Hazard>/<X>/individual_models/<M>_{climatology,return_period,params,timeseries}.nc` + outlier flags (see below) |
| calc | `01_calculation/run_observations.py --indicator X [--datasets D]` | `Observational/<Hazard>/<X>/<D>_{timeseries,climatology,return_period,params,trend}.nc` + outlier flags |
| post | `01_calculation/run_climbra.py --indicator X --ensemble-only` | `.../ensemble/ensemble_{climatology,return_period,timeseries}.nc` |
| post | `02_postprocess/01_anomalies_ensemble.py` | `.../ensemble/ensemble_anomaly_*.nc` (anomaly, % anomaly, model agreement) |
| post | `02_postprocess/03_return_period_shift.py` | `.../*_rp_shift.nc`, `ensemble_rp_shift.nc` |
| post | `02_postprocess/02_delta_change.py` | `.../ensemble/ensemble_delta_<DATASET>_*.nc` |
| final | `02_postprocess/04_municipal_aggregation.py` | `HAZBRA/municipalities/` (parquet tables, GeoPackages) |
| – | `02_postprocess/07_recompute_trends.py` | rewrites the observational `*_trend.nc` (Hamed–Rao modified Mann–Kendall, Sen's slope) from the stored annual series |
| final | `02_postprocess/05_outlier_summary.py` | `netcdf/outliers/outliers_{by_indicator,summary,detail}.csv`, `.md` (also in `outputs/tables`) |
| paper | `03_paper_analysis/*.py` | `outputs/figures`, `outputs/tables` |

The 16 indicators (hazard): Rx1day, Rx5day, CWD, prcptot, pr_season (Rainfall) · CDD, CDD_season, SPI3, SPI6 (Drought) ·
TXX, TNN, Frost, WSDI (Heat) · HI, FWI, Dry_Hot (Compound). All of them are defined in one place, `hazbra/indicators.py`.

## Test configuration

Every runnable script accepts `--test`: it merges `config/test.yaml` over `config/paths.yaml`, which

* cuts the data to a 1° × 1° box (−55.5 °E, −10 °N), uses 2 CLIMBRA models (both have all variables) and XAVIER only;
* **writes only below `_test_output/`** (never into `HAZBRA/`); reads the raw data and the production product read-only.

```bash
python 01_calculation/run_climbra.py --test --indicator Rx1day          # a few minutes
python 01_calculation/run_observations.py --test --indicator Rx1day --workers 4
python tests/compare_with_product.py climbra Rx1day                      # compare with the existing product, same cells
bash run_pipeline.sh --test                                              # whole pipeline as small SLURM jobs
bash tests/run_test_pipeline.sh                                          # same, sequentially here + regression check
```

`--dry-run` prints which files a run would write. `--overwrite` recomputes existing files (default: skip).
`--workers N` limits dask threads (use it on shared login nodes).

`tests/compare_with_product.py` compares a test run cell by cell with the existing product.

## Outlier filter outputs

Every calculation run writes, for the return levels it filters (`paths.outliers` = `HAZBRA/netcdf/outliers/`):

* `CLIMBRA/<Hazard>/<Indicator>/<MODEL>_outliers.nc`, `Observational/<Hazard>/<Indicator>/<DATASET>_outliers.nc` –
  per filtered variable `<var>_flag` (0 kept, 1 spatial outlier, 2 beyond the threshold: capped for day counts,
  removed + interpolated otherwise) and `<var>_raw` (the unfiltered value where flagged);
* `outliers_log.csv` – ONE consolidated CSV for all indicators and runs: every run appends one row per source /
  variable / GWL / return period / filter pass, with counts and proportions (`prop_spatial`, `prop_threshold`,
  `prop_replaced`) and a `run_id`;
* the final job (`02_postprocess/05_outlier_summary.py`) writes `outliers_by_indicator.csv`, `outliers_summary.csv`
  (by variable), `outliers_detail.csv` and `outliers_summary.md`, using the latest run of each product / indicator / source.

`02_postprocess/06_rebuild_outlier_flags.py` creates the same flags and log rows for return levels computed before
these outputs existed (from the stored GEV parameters, no recomputation). `05_outlier_summary.py --from-product` gives
a quick estimate directly from params vs return-level files.

Observations are computed in latitude bands of `calc.obs_tile_rows` rows (`config/paths.yaml`; identical results,
bounded memory) and dask uses the CPUs SLURM allocated to the job (`--workers` overrides).

## Environment

`environment.yml` (python 3.13, xarray 2025.12, xclim 0.60, ...). Cluster settings for `run_pipeline.sh`: `config/slurm.env`.

## Licence and citation

Code: MIT licence (`LICENSE`). How to cite: `CITATION.cff`; every GitHub release is archived on Zenodo; cite all versions with doi:10.5281/zenodo.24319623.
Changes between versions: `CHANGELOG.md`. The HAZBRA dataset itself is distributed separately under CC BY 4.0.

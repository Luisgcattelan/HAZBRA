# 03_paper_analysis – paper figures and numbers

Every script is run from the code root as `python 03_paper_analysis/<script>.py [--test] [--overwrite] [--formats pdf png] [--show]`
and writes to `cfg.paths.figures` / `cfg.paths.tables` (see `config/paths.yaml`; below the test root with `--test`).
Shared helpers: `hazbra/plotting.py` (rcParams, colour maps, base map, masks, hatching), `hazbra/ipcc_atlas.py`
(Atlas comparison), `hazbra/municipal.py` (municipal aggregation; the data product read by Fig. 8).

## Paper item -> script -> inputs -> outputs

| Paper item | Script | Inputs (`cfg.paths.<key>`) | Outputs |
|---|---|---|---|
| Fig. 1 map of Brazil (regions, states, municipalities) | `fig01_map_brazil.py` | `shp_brazil`, `shp_regions`, `shp_states`, `municipal` (`geometry/municipalities_2024.gpkg`; fallback `shp_municipalities`) | figures: `fig01_map_brazil.{pdf,png}` |
| Fig. 3 Rx5day climatology, CHIRPS/MERGE/XAVIER pairwise | `fig03_04_obs_comparison.py` | `netcdf_obs` (`Rainfall/Rx5day/<DS>_climatology.nc`), `south_america_limit`, `shp_regions` | figures: `fig03_Rx5day_climatology_pairwise.*`; tables: `..._pairwise_stats.csv` (bias, rel. bias, MAE, RMSE, MAPE, r per region and pair) |
| Fig. 4 CDD 25-year return level, pairwise | `fig03_04_obs_comparison.py` | `netcdf_obs` (`Drought/CDD/<DS>_return_period.nc`) | figures: `fig04_CDD_RL25_pairwise.*`; tables: `..._pairwise_stats.csv` |
| Fig. 5 CLIMBRA vs IPCC AR6 Atlas, GWL 3 (Rx5day %, TXx °C) + regional bias/RMSE/r in the text | `fig05_ipcc_atlas_comparison.py` (+ `hazbra/ipcc_atlas.py`) | `ipcc_atlas` (Atlas folders with `map.nc`, `consensus.nc`), `netcdf_climbra` (`<Hazard>/<Ind>/ensemble/ensemble_anomaly_climatology.nc`), `shp_brazil`, `shp_regions` | figures: `fig05_climbra_vs_atlas_gwl3.*`; tables: `fig05_climbra_vs_atlas_gwl3_regions.csv` (text statistics), `..._regional_means.csv` (CLIMBRA p10/median/p90 vs Atlas, quoted in the text) |
| Fig. 6 TXx and Fig. 7 Rx5day projected change at GWL 1.5/2/3 (P10/median/P90, hatching < 66% agreement) | `fig06_07_projection_matrix.py` | `netcdf_climbra` (same file as above), `south_america_limit`, `shp_regions` | figures: `fig06_TXX_projection_matrix.*`, `fig07_Rx5day_projection_matrix.*` |
| Text: regional median [p10, p90], signal-to-noise (median/(P90-P10)), share of area with \|S/N\|>=1 and with >=66% sign agreement (Brazil + 5 macro-regions, area-weighted) | `fig06_07_projection_matrix.py` | as above | tables: `TXX_synthesis.csv`, `Rx5day_synthesis.csv` (columns `snr_regional`, `snr_area_pct`, `robust_area_pct`, ...), `*_synthesis_compact.csv`; supplementary figures `supp_signal_to_noise_{TXX,Rx5day,CDD}.*` |
| Fig. 8 municipal change in dangerous-heat days (HI >= 41 °C) at GWL 2 + Brasília panel | `fig08_municipal_heat_days.py` | `municipal` (`CLIMBRA/Compound/HI/ensemble_anomaly_climatology.parquet`, `Observations/Compound/HI/XAVIER_climatology.parquet`, `geometry/`) | figures: `fig08_municipal_heat_days.*`; tables: `fig08_municipal_heat_days_city.csv` (values of panel b) |
| Text: municipal robustness (share of municipalities with >= 50% of area at >= 66% agreement) | already computed by `hazbra/municipal.py` (columns `agree_area_frac`, `robust` of the `*_model_agreement` rows); summarised by `municipal_robustness_stats.py` | `municipal` (HI tables, geometry for areas) | tables: `municipal_robustness_stats.csv` |
| Supplementary: CLIMBRA median change of 14 indicator variables at GWL 3 (also 1.5 / 2 with `--gwl`), hatched < 66% agreement | `supp_gwl_panels.py` (from `02.1.panels.ipynb`) | `netcdf_climbra` (`ensemble_anomaly_climatology.nc` of each indicator), `shp_brazil`, `shp_regions` | figures: `supp_gwl_panels_gwl3.*`; tables: `supp_gwl_panels_gwl3.csv` (Brazil mean, robust-area share) |
| Supplementary: signal-to-noise maps (TXx, Rx5day, CDD) | `fig06_07_projection_matrix.py` | see Fig. 6/7 | figures: `supp_signal_to_noise_<ind>.*` |
| Fig. 2 (workflow), Table 1 (GWL years) | not code here (Table 1: `01_calculation`) | | |

The Fig. 8 inputs are produced by `hazbra/municipal.py` (`02_postprocess/04_municipal_aggregation.py`):
`python -m hazbra.municipal [--indicators HI ...] [--test]`.

## Test mode

`python 03_paper_analysis/<script>.py --test` takes about 10-60 s per script and writes below the test root
(`config/test.yaml`).  In test mode the plotting scripts still read the *production* data product (read-only: the test
product only holds a 1 degree box and two models); only the outputs are redirected.  What is reduced:

| Script | `--test` |
|---|---|
| `fig01_map_brazil.py` | without the municipality layer |
| `fig03_04_obs_comparison.py` | Fig. 3 only |
| `fig05_ipcc_atlas_comparison.py` | TXx row only |
| `fig06_07_projection_matrix.py` | Fig. 6 (TXx) and its tables, no signal-to-noise maps |
| `supp_gwl_panels.py` | GWL 3, first 4 panels |
| `fig08_municipal_heat_days.py`, `municipal_robustness_stats.py` | full (cheap) |
| `hazbra.municipal` | TXX only; geometry and cached weights are reused read-only from production |

## Excluded from the paper analysis (not in this folder)

| Notebook / module | Reason |
|---|---|
| `population_exposure.py`, `13.Municipios_EDA_pop` | population exposure is not in the paper |
| `heat_exposure.py` | not in the paper |
| `999.rebanho` (cattle) | not in the paper |
| `12.Municipios_EDA` | exploratory municipal plots, not a paper figure |
| `31.rp_shift_plots`, `climbra_rp_shift.py`, `30.return_periods_shift` | return-period-shift plots are not in the paper |
| `99.Return_periods_aa` | exploratory |
| `00.SETUP` | one-off helper (population download) |
| `10.Municipios` | calls the municipal extraction (post-processing: `hazbra/municipal.py`) |
| trend plots (`plot_trends`), `plot_indicator`, `plot_region_heatmaps`, `plot_single_dataset` (`01.observations`) | not Fig. 3/4 |
| `02.PROJECTIONS`: `plot_change_per_gwl`, `plot_extremes_vs_means_*`, `hazard_change_table`/`plot_hazard_matrix`, PUB_RC variants | not in the paper |
| `14.Municipios_EDA_days`: `plot_days_maps`, `municipal_days` | not Fig. 8 |
| `999.tests*`, `9999.tests2` | scratch |

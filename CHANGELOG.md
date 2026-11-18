# Changelog

Versions follow the GitHub releases; each release is archived on Zenodo with its own DOI
(the Zenodo "concept DOI" always resolves to the latest version).

## [Unreleased]

## [0.1.1] – 2026-11-18
- Fix: the return-period shift selected the GEV shape sign by the mean error against the outlier-filtered
  return levels, which picked the wrong sign for CDD, CWD and several HI/FWI day counts; it now uses the median
  error (ties resolved in favour of the xclim convention). Affected rp_shift products were recomputed.
- Municipal README: relative paths, published-file notes; README generators without caveat / GEV-convention sections.
- New `tools/zenodo_dataset/`: packing, upload to Zenodo (draft only) and the dataset user guide.
- Paper figures regenerated with the final product (calendar-gap rerun of CDD, CWD, CDD_season and Frost).

## [0.1.0] – 2026-11-17
First public version of the code organised for the review of the HAZBRA data paper (draft 1).

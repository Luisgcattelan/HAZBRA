#!/bin/bash
# End-to-end TEST of the whole pipeline on a 1 x 1 degree box, 2 CLIMBRA models and XAVIER only, run sequentially
# on the current node (run it inside an interactive SLURM allocation, or use `bash run_pipeline.sh --test`).
# Reads the raw data / production files (read-only) and writes ONLY below _test_output/.
#
# Afterwards each test indicator is compared with the production product on the same cells.  Expected differences:
#   * *_return_period.nc: the outlier filter is domain-dependent (a 4x4 box flags other cells than the full grid)
#   * indicators whose method changed after the first review (docs/CONSOLIDATION_NOTES.md, table A-E)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash run_pipeline.sh --test --local --overwrite "$@"
set +u; source "$HOME/miniforge3/bin/activate" geoenv13; set -u
export PYTHONWARNINGS=ignore
for ind in $(python -m hazbra.plan indicators --test); do
  python tests/compare_with_product.py climbra "$ind" || true
  python tests/compare_with_product.py obs "$ind"     || true
done
echo "### done – outputs in _test_output/"

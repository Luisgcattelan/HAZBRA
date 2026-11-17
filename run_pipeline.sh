#!/bin/bash
# HAZBRA – full processing pipeline on SLURM, with job dependencies.
#
#   per indicator:
#     calc   one job per CLIMBRA model  (01_calculation/run_climbra.py --models M --skip-ensemble)
#            one job per obs. dataset   (01_calculation/run_observations.py --datasets D)
#     post   after ALL calc jobs of the indicator succeeded:
#            ensemble statistics -> anomalies + model agreement -> return-period shift -> delta change
#   final    after ALL post jobs succeeded:
#            return-period README, municipal aggregation, outlier flags of earlier runs (06, skips existing),
#            outlier tables [+ paper figures with --figures]
#
# Usage
#   bash run_pipeline.sh                              # everything, production (skips files that exist)
#   bash run_pipeline.sh --indicators "Rx1day HI"     # selected indicators (final stage only for those)
#   bash run_pipeline.sh --test                       # test configuration (config/test.yaml), small jobs
#   bash run_pipeline.sh --test --local               # test configuration, run here sequentially (no SLURM)
#   bash run_pipeline.sh --dry                        # print the sbatch commands, submit nothing
#   options: --overwrite  --figures  --gwl-table (rebuild GWL_CLIMBRA.csv first)  --stages calc,post,final
#            --sources climbra,obs (calc jobs to submit)  --datasets "CHIRPS MERGE" (observational datasets)
#   re-run only some observational jobs, then their post jobs and the final job:
#     bash run_pipeline.sh --overwrite --sources obs --datasets "CHIRPS MERGE" --indicators "SPI3 SPI6"
#
# Resources: config/slurm_resources.csv ; account/partition/qos/conda env: config/slurm.env
# Logs: logs/<job name>_<job id>.{out,err}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source config/slurm.env

TEST=""; LOCAL=""; DRY=""; OVERWRITE=""; FIGURES=""; GWL=""; INDS=""; STAGES="calc,post,final"; SOURCES="climbra,obs"; DATASETS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --test) TEST="--test" ;;
    --local) LOCAL=1 ;;
    --dry) DRY=1 ;;
    --overwrite) OVERWRITE="--overwrite" ;;
    --figures) FIGURES=1 ;;
    --gwl-table) GWL=1 ;;
    --indicators) INDS="$2"; shift ;;
    --stages) STAGES="$2"; shift ;;
    --sources) SOURCES="$2"; shift ;;
    --datasets) DATASETS="$2"; shift ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
  shift
done
has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }
has_source() { [[ ",$SOURCES," == *",$1,"* ]]; }
want_dataset() { [[ -z "$DATASETS" || " $DATASETS " == *" $1 "* ]]; }

set +u; source "$CONDA_ACTIVATE" "$CONDA_ENV"; set -u        # planning calls below (and --local runs)
export PYTHONWARNINGS=ignore
PY="python"
plan() { $PY -m hazbra.plan "$@" $TEST; }
[[ -z "$INDS" ]] && INDS="$(plan indicators | tr '\n' ' ')"
LOGDIR="$ROOT/logs"; mkdir -p "$LOGDIR"

# ---------------------------------------------------------------- helpers
resources() {  # stage indicator -> "time cpus mem"
  local stage="$1" ind="$2"
  [[ -n "$TEST" ]] && stage=test && ind="*"
  awk -F, -v s="$stage" -v i="$ind" '!/^#/ && $1==s && ($2==i || $2=="*") {print $3, $4, $5; exit}' config/slurm_resources.csv
}

FAKE_IDS="$LOGDIR/.dry_ids"; [[ -n "$DRY" ]] && echo 1000 > "$FAKE_IDS"
submit() {  # name stage indicator dependency-list command...   -> prints job id
  local name="$1" stage="$2" ind="$3" deps="$4"; shift 4
  local cmd="$*"
  if [[ -n "$LOCAL" ]]; then
    echo ">>> $name: $cmd" >&2
    bash -c "$cmd" >&2
    echo "local"; return
  fi
  read -r tim cpus mem <<< "$(resources "$stage" "$ind")"
  local dep=()
  [[ -n "$deps" ]] && dep=(--dependency="afterok:${deps// /:}" --kill-on-invalid-dep=yes)
  local sb=(sbatch --parsable --job-name="hz_${name}" --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION"
            --qos="$SLURM_QOS" --time="$tim" --ntasks=1 --cpus-per-task="$cpus" --mem="$mem"
            --output="$LOGDIR/${name}_%j.out" --error="$LOGDIR/${name}_%j.err" "${dep[@]}")
  # The job script is passed on stdin with a bash shebang: `sbatch --wrap` would run it with /bin/sh (bash in POSIX
  # mode), where conda's GDAL activation script fails ("_gdal2tiles.py: not a valid identifier").
  local script="#!/bin/bash
set -eo pipefail
source $CONDA_ACTIVATE $CONDA_ENV
cd $ROOT
export PYTHONWARNINGS=ignore
$cmd"
  if [[ -n "$DRY" ]]; then
    printf '%q ' "${sb[@]}" >&2; echo >&2; echo "$script" | sed 's/^/    | /' >&2
    local id=$(( $(cat "$FAKE_IDS") + 1 )); echo "$id" > "$FAKE_IDS"; echo "$id"
  else
    "${sb[@]}" <<< "$script"
  fi
}

# ---------------------------------------------------------------- stage 0: GWL table
GWL_JOB=""
if [[ -n "$GWL" ]]; then
  GWL_JOB=$(submit gwl_table post "*" "" "$PY 01_calculation/00_gwl_table.py $TEST")
  [[ "$GWL_JOB" == local ]] && GWL_JOB=""
fi

# ---------------------------------------------------------------- per indicator
POST_JOBS=""
for IND in $INDS; do
  CALC_JOBS=""
  if has_stage calc; then
    has_source climbra && for M in $(plan models "$IND"); do
      j=$(submit "climbra_${IND}_${M}" climbra "$IND" "$GWL_JOB" \
          "$PY 01_calculation/run_climbra.py --indicator $IND --models $M --skip-ensemble $TEST $OVERWRITE")
      [[ "$j" != local ]] && CALC_JOBS="$CALC_JOBS $j"
    done
    has_source obs && for D in $(plan datasets "$IND"); do
      want_dataset "$D" || continue
      j=$(submit "obs_${IND}_${D}" obs "$IND" "" \
          "$PY 01_calculation/run_observations.py --indicator $IND --datasets $D $TEST $OVERWRITE")
      [[ "$j" != local ]] && CALC_JOBS="$CALC_JOBS $j"
    done
  fi
  j=""
  if has_stage post; then
    RP=""
    [[ "$(plan gev "$IND")" == yes ]] && RP=" && $PY 02_postprocess/03_return_period_shift.py --indicator $IND --no-readme $TEST $OVERWRITE"
    j=$(submit "post_${IND}" post "$IND" "$(echo $CALC_JOBS)" \
        "$PY 01_calculation/run_climbra.py --indicator $IND --ensemble-only $TEST \
         && $PY 02_postprocess/01_anomalies_ensemble.py --indicator $IND $TEST${RP} \
         && $PY 02_postprocess/02_delta_change.py --indicator $IND $TEST")
    [[ "$j" != local ]] && POST_JOBS="$POST_JOBS $j"
  fi
  echo "[$IND] submitted: calc=$(echo $CALC_JOBS | wc -w) jobs, post=${j:-none}" >&2
done

# ---------------------------------------------------------------- final
if has_stage final; then
  FINAL="$PY 02_postprocess/03_return_period_shift.py --readme-only $TEST \
    && $PY 02_postprocess/04_municipal_aggregation.py --indicators $INDS $TEST $OVERWRITE \
    && $PY 02_postprocess/06_rebuild_outlier_flags.py --indicator $(for i in $INDS; do [[ "$(plan gev "$i")" == yes ]] && echo -n "$i "; done) $TEST \
    && $PY 02_postprocess/05_outlier_summary.py --indicator $(for i in $INDS; do [[ "$(plan gev "$i")" == yes ]] && echo -n "$i "; done) $TEST"
  if [[ -n "$FIGURES" ]]; then
    for f in 03_paper_analysis/fig*.py 03_paper_analysis/supp*.py 03_paper_analysis/municipal_robustness_stats.py; do
      [[ -e "$f" ]] && FINAL="$FINAL && $PY $f $TEST"
    done
  fi
  j=$(submit final final "*" "$(echo $POST_JOBS)" "$FINAL")
  echo "[final] submitted: $j" >&2
fi

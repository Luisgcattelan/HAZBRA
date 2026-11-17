#!/bin/bash
# Wait for the SLURM packing job, then upload the staging folder to the Zenodo DRAFT (never publishes).
#   nohup bash tools/zenodo_dataset/upload_after_pack.sh <pack_jobid> > logs/zenodo_upload.log 2>&1 &
JOB=$1
cd "$(dirname "$0")/../.."
source ~/miniforge3/bin/activate geoenv13
while :; do
    s=$(sacct -j "$JOB" -X -n -o State | tr -d ' ')
    case "$s" in
        COMPLETED) break ;;
        FAILED*|CANCELLED*|TIMEOUT|OUT_OF_ME*|NODE_FAIL) echo "$(date) pack job $JOB ended: $s – nothing uploaded"; exit 1 ;;
    esac
    sleep 600
done
echo "$(date) pack job $JOB completed – uploading"
python -u tools/zenodo_dataset/zenodo_upload.py upload && python tools/zenodo_dataset/zenodo_upload.py status
echo "$(date) done"

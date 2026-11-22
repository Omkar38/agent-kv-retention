#!/usr/bin/env bash
# Download a public model into scratch/results.
# Run only where SeaWulf permits large downloads; otherwise download elsewhere
# and copy the full model directory to MODEL_DIR before submitting the GPU job.
set -euo pipefail

BASE=/gpfs/scratch/hanglu
RESULTS="$BASE/results"
export HF_HOME="$RESULTS/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export PIP_CACHE_DIR="$RESULTS/pip_cache"
export TMPDIR="$RESULTS/tmp"
export MODEL_REPO="Qwen/Qwen2.5-7B"
export MODEL_DIR="$RESULTS/models/qwen2.5-7b"
mkdir -p "$RESULTS"/{hf_cache,pip_cache,tmp,models,logs,runs} "$MODEL_DIR"

module load anaconda/3
eval "$(conda shell.bash hook)"
conda activate kv-week9

python - <<'PY'
import os
from huggingface_hub import snapshot_download
repo = os.environ["MODEL_REPO"]
dest = os.environ["MODEL_DIR"]
print(f"Downloading {repo} into {dest}")
snapshot_download(repo_id=repo, local_dir=dest, max_workers=4)
print(f"Model ready at {dest}")
PY

test -f "$MODEL_DIR/config.json" || {
  echo "ERROR: model config.json missing in $MODEL_DIR" >&2
  exit 1
}
echo "Model files stored in $MODEL_DIR"
echo "Next: sbatch seawulf/week9_a100.slurm"

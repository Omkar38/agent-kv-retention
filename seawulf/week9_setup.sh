#!/usr/bin/env bash
# Run on SeaWulf Milan login: bash -l seawulf/week9_setup.sh
# Conda environment lives in $HOME; experiment caches/results go to scratch.
set -euo pipefail

BASE=/gpfs/scratch/hanglu
RESULTS="$BASE/results"
ENV_NAME=kv-week9
mkdir -p "$RESULTS"/{logs,models,runs,hf_cache,pip_cache,tmp}
export HF_HOME="$RESULTS/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export PIP_CACHE_DIR="$RESULTS/pip_cache"
export TMPDIR="$RESULTS/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

cd "$BASE/agent-kv-retention"
module load anaconda/3
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Reusing Conda environment: $ENV_NAME"
else
  conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"
python -m pip install -r week4/requirements.txt
python -m pip install torch transformers accelerate huggingface_hub
python -m pip check
python -m pip freeze > "$RESULTS/week9_pip_freeze.txt"

python - <<'PY'
import sys, torch, transformers
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__, "CUDA build:", torch.version.cuda)
print("Transformers:", transformers.__version__)
print("CUDA available on login node:", torch.cuda.is_available(), "(False expected)")
PY

echo "Environment ready: $ENV_NAME"
echo "Package manifest: $RESULTS/week9_pip_freeze.txt"
echo "Next: bash -l seawulf/week9_download_model.sh"

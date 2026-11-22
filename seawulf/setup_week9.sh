#!/usr/bin/env bash
# Prepare Week 9 Python environment on SeaWulf's MILAN login node.
# All installed packages and caches remain under /gpfs/scratch/hanglu/results.
set -euo pipefail

ROOT="/gpfs/scratch/hanglu"
RESULTS="$ROOT/results"
REPO="$ROOT/agent-kv-retention"
VENV="$RESULTS/venv"

mkdir -p "$RESULTS/pip_cache" "$RESULTS/hf_cache" "$RESULTS/torch_cache" "$RESULTS/tmp" "$RESULTS/cache"
export PIP_CACHE_DIR="$RESULTS/pip_cache"
export HF_HOME="$RESULTS/hf_cache"
export TORCH_HOME="$RESULTS/torch_cache"
export TMPDIR="$RESULTS/tmp"
export XDG_CACHE_HOME="$RESULTS/cache"

if [[ ! -f "$REPO/week4/requirements.txt" ]]; then
    echo "ERROR: clone week9-cost-calibration into $REPO first." >&2
    exit 1
fi

# If this module name is unavailable, inspect 'module avail python' and use
# the same available Python module in both setup and the batch script.
module load python/3.11.2

python3 --version
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip wheel setuptools
"$VENV/bin/python" -m pip install -r "$REPO/week4/requirements.txt"

# GPU-capable Linux PyTorch wheel. Driver requirements must be satisfied.
"$VENV/bin/python" -m pip install \
    "torch==2.4.1" \
    "transformers>=4.44,<5" \
    "accelerate>=0.34" \
    "huggingface_hub>=0.25"

cd "$REPO"
"$VENV/bin/python" -m unittest discover -s tests -p 'test_calibration.py' -v
"$VENV/bin/python" -m pip freeze > "$RESULTS/week9_setup_packages.txt"

echo
echo "Setup complete. Python: $VENV/bin/python"
echo "All generated files/caches: $RESULTS"
echo "Next: bash $REPO/seawulf/stage_week9_model.sh"

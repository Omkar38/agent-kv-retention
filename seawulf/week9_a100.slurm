#!/bin/bash -l
#SBATCH --job-name=week9_kv_prefill
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/hanglu/results/week9_%j.out
#SBATCH --error=/gpfs/scratch/hanglu/results/week9_%j.err

# Important: mkdir -p /gpfs/scratch/hanglu/results BEFORE sbatch.
# All runtime results, logs, caches and model files remain within results.
set -euo pipefail

ROOT="/gpfs/scratch/hanglu"
RESULTS="$ROOT/results"
REPO="$ROOT/agent-kv-retention"
PY="$RESULTS/venv/bin/python"
MODEL_PATH="$RESULTS/models/Qwen2.5-7B"
RUN_DIR="$RESULTS/week9_$SLURM_JOB_ID"

mkdir -p "$RUN_DIR/tmp" "$RESULTS/cuda_cache" "$RESULTS/hf_cache" "$RESULTS/torch_cache"
export HF_HOME="$RESULTS/hf_cache"
export TORCH_HOME="$RESULTS/torch_cache"
export XDG_CACHE_HOME="$RESULTS/cache"
export PIP_CACHE_DIR="$RESULTS/pip_cache"
export CUDA_CACHE_PATH="$RESULTS/cuda_cache"
export TMPDIR="$RUN_DIR/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

module load python/3.11.2
module load cuda120/toolkit/12.0

if [[ ! -x "$PY" ]]; then
    echo "ERROR: missing virtual environment $PY. Run seawulf/setup_week9.sh first." >&2
    exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "ERROR: missing local model $MODEL_PATH. Run seawulf/stage_week9_model.sh first." >&2
    exit 1
fi

cd "$REPO"
git rev-parse HEAD > "$RUN_DIR/git_commit.txt"
"$PY" -m pip freeze > "$RUN_DIR/python_packages.txt"
nvidia-smi -q > "$RUN_DIR/nvidia_smi.txt"
"$PY" - <<'PY'
import torch
print("PyTorch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot access an allocated CUDA GPU.")
print("GPU:", torch.cuda.get_device_name(0))
PY

"$PY" -m unittest discover -s tests -p 'test_calibration.py' -v

echo "Running real prefill measurements in $RUN_DIR"
"$PY" -m experiments.benchmark_prefill \
    --model-id "$MODEL_PATH" \
    --dtype float16 \
    --context-lengths 1024,2048,4096,8192 \
    --warmup 2 \
    --repeats 5 \
    --output "$RUN_DIR/prefill_measurements.csv"

echo "Summarizing throughput and calculating assumed-cost scenarios"
"$PY" -m experiments.calibrate_costs \
    --measurements "$RUN_DIR/prefill_measurements.csv" \
    --out-dir "$RUN_DIR" \
    --contexts 1024,2048,4096,8192 \
    --budgets 300000,600000,1200000 \
    --c0-values 0.1,0.2,0.4,0.8

echo "Week 9 run complete: $RUN_DIR"
echo "Main files: $RUN_DIR/prefill_measurements.csv and $RUN_DIR/break_even_scenarios.csv"

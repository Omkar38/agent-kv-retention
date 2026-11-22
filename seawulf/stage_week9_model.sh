#!/usr/bin/env bash
# Download an open-weight 7B model into scratch/results BEFORE the GPU job.
# This may require many GB of disk/network. Do not run on a compute node.
set -euo pipefail

ROOT="/gpfs/scratch/hanglu"
RESULTS="$ROOT/results"
PY="$RESULTS/venv/bin/python"
MODEL_ID="Qwen/Qwen2.5-7B"
MODEL_DIR="$RESULTS/models/Qwen2.5-7B"

if [[ ! -x "$PY" ]]; then
    echo "ERROR: run seawulf/setup_week9.sh first." >&2
    exit 1
fi

mkdir -p "$RESULTS/models" "$RESULTS/hf_cache"
export HF_HOME="$RESULTS/hf_cache"
export XDG_CACHE_HOME="$RESULTS/cache"

# Download on the login node only when allowed by SeaWulf network policy.
# To use a separately staged model instead, put its HF config and weights
# under MODEL_DIR and skip this script.
"$PY" - "$MODEL_ID" "$MODEL_DIR" "$RESULTS" <<'PY'
import json
import pathlib
import sys
from huggingface_hub import snapshot_download

model_id, model_dir, results = sys.argv[1:]
dest = pathlib.Path(model_dir)
dest.mkdir(parents=True, exist_ok=True)
local_path = snapshot_download(repo_id=model_id, local_dir=str(dest))
if not (dest / "config.json").is_file():
    raise RuntimeError("Downloaded model is missing config.json")
manifest = {
    "source_model_id": model_id,
    "local_model_dir": str(dest),
    "downloaded_or_verified_location": str(local_path),
    "note": "Prefill throughput measured on this model is not necessarily Llama-3 throughput.",
}
(pathlib.Path(results) / "week9_model_source.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
print("Model staged:", dest)
PY

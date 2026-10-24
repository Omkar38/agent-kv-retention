#!/usr/bin/env bash
set -euo pipefail
python3 week4_handoff.py --input analysis/dataset/pause_handoff.parquet --output-dir results/week4_handoff --seed 560 --max-train 80000 --iterations 75 --break-even-s 1.0

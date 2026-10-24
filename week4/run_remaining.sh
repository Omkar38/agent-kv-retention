#!/usr/bin/env bash
set -euo pipefail
python3 -m week4.extended_analysis \
  --dataset analysis/dataset/pause_handoff.parquet \
  --predictions results/week4_handoff/predictions_test.csv \
  --out results/week4_extended \
  --seed 560 --bootstrap 2000 --max-train 80000 --iterations 75
python3 -m unittest discover -s week4/tests -p 'test_*.py' -v

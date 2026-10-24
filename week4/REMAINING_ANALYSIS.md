# Week 4 — Remaining Research Analysis

The sole canonical dataset is analysis/dataset/pause_handoff.parquet (357,161 rows, 33 columns). No original records or columns are modified.

## One-command reproducibility from the repository root

    python3 -m pip install -r week4/requirements.txt
    bash week4/run_week4.sh
    bash week4/run_remaining.sh

A GitHub Actions workflow is included to perform the same experiments using the dataset checked into Hangting_week4. The generated local predictions and training-only empirical tool CDF source are intentionally not committed; aggregate results and the final report belong in results/week4_extended/.

## Tasks completed by extended_analysis.py

1. Dataset eligibility audit: offline label and timing categories, missing / negative / zero / positive pause lengths. Zero-duration observations are reported, not automatically declared corrupted.
2. Feature timing audit: source round versus result-time metadata and feature restricted ablations with provider/model/turn and optional context size. These do not establish exact inference-stop timestamps.
3. Held-out tail errors: >10s, >60s, and the longest global decile; Claude/Codex and tool subgroups.
4. Threshold sensitivity: false-retain and false-evict rates, near-threshold flips for B/c = 0.1, 0.3, 1, 3, 10, 60 seconds.
5. Quantile P10/P50/P90 and 80% interval coverage by group and conditional long-pause coverage.
6. Paired session bootstrap: 95% confidence intervals for learned-versus-provider/tool median improvements, resampling sessions rather than individual pauses.
7. Reproducibility: SHA256 hashes, package/runtime provenance, fixed seeds and a research report.

Outputs are written under results/week4_extended/:
- target_eligibility.csv, feature_audit.csv
- tail_error_analysis.csv, group_error_extended.csv
- threshold_analysis.csv, calibration_by_group.csv
- session_bootstrap.csv, feature_ablation.csv
- experiment_manifest.json and WEEK4_FINAL_REPORT.md

Important: these analyses use trace-derived tool-pause estimates, not measured GPU cache idleness. The retrospective tool-pause selector is never used as a predictor input. The thresholds are illustrative, NOT calibrated GPU cost ratios or retention algorithms. The next Weeks 6–7 deliver the single-request guarantee and matched-trace, matched-memory comparisons with Continuum-style TTL.

Run all tests with: python -m unittest discover -s week4/tests -p 'test_*.py' -v

# Week 4 — Duration prediction and error analysis

Dataset: `analysis/dataset/pause_handoff.parquet` inherited from `Hangting_week3`. Preserve the original 357,161-row, 33-column dataset; do not overwrite it.

## Run (repository root)

```bash
python3 -m pip install -r week4/requirements.txt
python3 week4_handoff.py --input analysis/dataset/pause_handoff.parquet --output-dir results/week4_handoff --seed 560 --max-train 80000 --iterations 75 --break-even-s 1.0
python3 -m unittest discover -s week4/tests -p 'test_week4_handoff.py' -v
```

The supervised evaluation uses only `analysis_track == "tool_pause_prediction"` with positive finite `pause_s`, preserving inherited session-level train/validation/test assignment. This retrospective filter is not an online feature. The predictor uses a whitelist of provisional decision-time features, never outcomes, timestamps, session IDs, or labels. Real GPU KV-cache residency has not been measured.

Models: global median, provider/tool median, log-duration Ridge, histogram gradient boosting q10/q50/q90; select a point model using validation log1p-MAE. Results: `metrics.json`, `analysis.md`, `predictions_test.csv`, `predictions_validation.csv`, `continuum_train_cdf_reference.csv.gz`, `group_error_analysis.csv`, plots, and joblib models.

The provided Week 4 results summary in `week4/REAL_TRACE_SUMMARY.md` was copied from a user-run report, not independently re-executed in this chat. For fully reproducible figures and detailed output, run the command above on the actual Parquet file. Predictions and train-only per-tool distributions are intended for a later matched-trace Continuum-style TTL comparison. The threshold `B/c=1.0s` is illustrative, not hardware calibrated, nor a new policy or proof.
## October 27 methodology correction

**Statistical inference:** The earlier headline improvement was event-weighted, where large sessions contribute more. The new `session_bootstrap.csv` reports BOTH event-weighted and equal-session-weighted gains and confidence intervals. Do not use the word *significant* without specifying weighting and checking the corresponding interval. Five sessions represent about 43% of test events (as reported by the reviewer); validate that claim against the refreshed session counts before quoting it as a new measurement.

**Cost baselines:** `threshold_analysis.csv` now includes always-keep and always-evict strategies. Its costs are single-pause normalized toy costs, not GPU benchmarks or the λ-policy.

**Threshold classifiers:** The main predictor now trains separate `P(T>B)` gradient-boosted classifiers at `B = 0.1, 0.3, 1, 3, 10, 60 seconds` using the inherited training partition. It exports probabilities on the validation/test prediction CSVs and Brier/AUC diagnostics into `metrics.json`. These classifiers are not yet proved optimal or calibrated for a real GPU scheduler.

**Simulator data:** `results/week4_handoff/simulator_all_tool_pauses.csv.gz` preserves the original nonnegative finite tool-pause cohort INCLUDING all 2,825 zero-duration events (offline replay only).

**Online feature and command-text caveat:** No first-word-of-command feature was added, because the 33-column handoff does not contain command text or timestamp-certified command dispatch features. Tool names, context size, and tool input size still need independent observation-time verification. A model that never predicts long `exec_command` pauses cannot be fixed reliably by inventing command text or using post-outcome details.

**Outputs:** The extended pipeline also emits a tool-name-only ablation. The regenerated aggregate report includes both weighting definitions. Keep the branch's older committed summaries distinct from the newly generated reports until the CI run completes.

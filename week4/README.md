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
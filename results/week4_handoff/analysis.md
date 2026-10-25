# Group 9 — Week 4 duration prediction

**STATUS:** `real_trace_handoff_evaluation`
**Dataset:** `analysis/dataset/pause_handoff.parquet`
**Offline tool-pause cohort:** 311,764 / 357,161 original records
**Session groups:** {'train': 2920, 'val': 362, 'test': 384}
**Validation-selected model:** `hgb_q50`

| Model | Test MAE (s) | Test log1p-MAE | Underprediction | Threshold-flip rate |
|---|---:|---:|---:|---:|
| global_median | 16.8294 | 0.7530 | 55.8% | 38.2% |
| provider_tool_median | 16.4105 | 0.6062 | 53.3% | 29.2% |
| log_ridge | 16.4945 | 0.6556 | 33.9% | 31.4% |
| hgb_q50 | 16.1958 | 0.5441 | 51.9% | 26.4% |

Empirical P10/P50/P90 coverage: 10.8%, 48.1%, 91.2%.
Central 80% interval coverage: 80.4%.

**Important:** B/c=1.0 seconds is an illustrative, uncalibrated sensitivity threshold.
Track filtering uses retrospective data only to define an offline supervised cohort; it is never an online model input.
The data are trace-derived pause estimates, not GPU-resident cache inactivity measurements.
The simplified threshold diagnostic is neither the new λ-policy nor the Continuum algorithm.
Predictions and train-only empirical-CDF reference records are exported for later matched-trace policy comparisons.

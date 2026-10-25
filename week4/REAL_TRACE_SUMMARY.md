# Group 9 — Week 4 duration prediction

**Status:** Real-trace results supplied by project member; not independently reproduced here.
**Input:** `analysis/dataset/pause_handoff.parquet`
**Offline tool-pause cohort:** 311,764 / 357,161 source records
**Session groups:** train 2,920; validation 362; test 384
**Validation-selected model:** `hgb_q50`

| Model | Test MAE (s) | Test log1p-MAE | Underprediction | Threshold-flip rate |
|---|---:|---:|---:|---:|
| global_median | 16.8294 | 0.7530 | 55.8% | 38.2% |
| provider_tool_median | 16.4105 | 0.6062 | 53.3% | 29.2% |
| log_ridge | 16.4945 | 0.6556 | 33.9% | 31.4% |
| hgb_q50 | 16.1958 | 0.5441 | 51.9% | 26.4% |

Empirical P10/P50/P90 coverage: 10.8%, 48.1%, 91.2%. Central 80% interval coverage: 80.4%.

**Limitations:** `B/c=1.0s` is an illustrative, uncalibrated diagnostic threshold. The dataset represents trace-derived pause estimates, not GPU cache idle-time measurements. This diagnostic is neither the learning-augmented λ policy nor the Continuum algorithm. Offline cohort classification may depend on retrospective information and is not a valid online feature. Future policy comparisons must use identical held-out traces and memory constraints. The complete measured JSON, raw prediction CSVs and figures were not supplied in the chat; regenerate them by running the code.

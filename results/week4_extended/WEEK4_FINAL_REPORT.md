# Week 4 Final Research Report — Group 9

## Scope / provenance

- Canonical dataset: 357,161 records, 33 columns; supervised observations: 311,764; held-out test records: 47,773.
- Model inputs exclude target, original cause, timing outcomes, session ID and analysis-track labels. Session IDs are for splitting and uncertainty estimates only.
- Trace-derived tool pauses are proxies, not measured GPU KV-cache residency, and do not themselves establish latency improvements.

## Real-trace duration prediction

- Test log1p-MAE: provider/tool median 0.6062, HGB Q50 0.5441.
- Test MAE: provider/tool median 16.410s, HGB Q50 16.196s.
- Event-weighted session-cluster bootstrap gain in log1p-MAE: 0.0621, 95% interval [0.0340, 0.0953].
- Equal-session-weighted bootstrap gain in log1p-MAE: 0.0064, 95% interval [-0.0218, 0.0326].
- **Inference depends on the estimand:** large sessions dominate event-weighted results; do NOT label the model universally significantly better. Review the equal-session interval before making a significance claim.

## Eligibility and future-information restrictions

- The supervised-track records with excluded target outcomes: zero=2,825, negative=0, missing/nonfinite=0. Zero-duration pauses are not automatically invalid; inspect their logging provenance.
- Tool lists and counts may include interactions not known at the first pause; the original features are candidate signals, not formally certified online signals.
- Restricted-feature ablations include a tool-name-only variant; candidate tool/context fields remain provisionally timed. Tool-name value is not verified as online observable.
- The retrospective offline cohort selector cannot be used in the online policy. Older Codex and multi-tool timing ambiguity must remain disclosed.

## Tail, calibration, and decision sensitivity

- Tool durations are strongly heavy-tailed; see tail_error_analysis.csv for >10s, >60s, top decile, provider-specific error and underprediction.
- Held-out P10/P50/P90 empirical coverage: 10.8% / 48.1% / 91.2%. Central 80% interval coverage: 80.4%.
- Marginally calibrated quantiles need not be calibrated in the long tail or by tool; see calibration_by_group.csv.
- threshold_analysis.csv includes immediate-evict and always-keep cost reference points alongside predictor decisions for six illustrative B/c values.
- B/c settings are NOT calibrated, and these diagnostics are NOT the learning-augmented λ-policy, a Continuum-style TTL, or a consistency/robustness guarantee.

## Reproduction and Week 6 handoff

- Dataset SHA256: 2c0d0f2cd825073dce3f03392a3b30497cad4f209539262c35a4a4e02cbf1d5c
- Test prediction SHA256: 40eaa3c1c07ad1ddb1b1c89ac382be0cc01cb084292b7a14c51f499c967096cd
- Seed 560; resamples 2000; training cap 80000; boosting iterations 75.
- Keep local test predictions and the train-only duration/CDF reference for the later matched-trace, matched-capacity Continuum experiment.
- Week 6 aims to develop and prove a simplified single-request consistency/robustness guarantee; the result is not yet established. Multi-request load-dependent costs belong to simulation.
- Remaining work is future policy development and workload calibration, not an assertion that the current predictor is already an optimal KV retention policy.

# Week 4 Report — Duration Prediction and Error Analysis

**Project:** Learning-Augmented KV-Cache Retention for Tool-Using LLM Agents  
**Course/team:** CSE 542 / AMS 560, Group 9, Fall 2026  
**Branch:** [`Hangting_week4`](https://github.com/Omkar38/agent-kv-retention/tree/Hangting_week4)  
**Scope:** Week 4 — duration prediction, evaluation, and error analysis  
**Dataset:** [`analysis/dataset/pause_handoff.parquet`](../analysis/dataset/pause_handoff.parquet) (TraceLab-derived Version 2 handoff)  
**Status:** Week 4 implementation, extended analysis, report generation, and CI validation completed as of October 24, 2026. The cache-retention algorithm and simulator are future work.

This document is the **branch-level guide** to the folders, scripts, datasets, tests, results, and present conclusions. For the short numerical research summary, see [`results/week4_extended/WEEK4_FINAL_REPORT.md`](../results/week4_extended/WEEK4_FINAL_REPORT.md). For machine-readable statistics, use the CSV/JSON results referenced below.

## October 27 correction — equal-session weighting and policy caveats

**The earlier claim of a statistically significant general HGB advantage was too broad.** The original bootstrap resampled entire sessions, but calculated an **event-weighted gain**, so large sessions contributed proportionally more. A session bootstrap alone does *not* mean each session has equal weight. The revised analysis outputs both event-weighted and **equal-session-weighted** estimates, confidence intervals, and flags for whether the interval excludes zero. Based on the author's equal-session reanalysis, that gain is near zero and not statistically distinguishable from zero. Do not cite the earlier positive event-weighted interval as evidence of improvement for a typical session.

The revised cost table adds **always-keep** and **always-evict** reference strategies, all still under the uncalibrated single-pause cost model. The predictor now exports training-only **P(T > B)** classifiers at six thresholds, as additional decision-facing signals; no λ-policy benefit is yet established. The tool-name-only feature ablation now separates tool identity from other tool/context metadata. An offline simulator export preserves all finite nonnegative tool-pause records, including zero-duration pauses.

A command-first-word feature **has not** been introduced because raw command text and proof of its decision-time availability are absent from the 33-column handoff. Context and tool-input-size availability is likewise unresolved. These are explicit limitations, not completed fixes. The updated full analysis results must be regenerated with the current workflow before quoting its latest estimates.

## 1. Objective and what was accomplished

The Week 4 question is **whether the duration of a tool-related agent pause can be predicted from information potentially available when retention begins**, and how prediction errors relate to the future GPU KV-cache keep/evict decision.

We processed the Week 3 TraceLab-derived handoff using its existing session-level train/validation/test partitions, developed point and quantile duration predictors, compared them with simple historical baselines, and evaluated performance on held-out agent sessions. We also studied long-tail errors, per-provider/tool differences, threshold-flip decisions, feature availability, calibration, and statistical uncertainty.

The real-trace evaluation **does not** establish real GPU cache residency, reduced inference latency, reduced GPU memory consumption, or a consistency/robustness theorem. These require a defined retention policy, a simulator with matched memory constraints, and (optionally) instrumented inference-server measurements in later weeks.

## 2. Repository layout and purpose of every relevant file

The branch inherits the original Week 3 work under `analysis/` and adds the Week 4 predictor, extended analysis, results, and automation. Paths below are **relative to the repository root**. The list describes the files present on this branch when this report was written.

### 2.1 Root and automation

| Path | Role |
|---|---|
| [`Readme.md`](../Readme.md) | Project-wide motivation, simplified ski-rental cost model, literature gap, division of work, intended algorithms/simulator, and experimental roadmap. Some status language predates the completed Week 4 experiments. |
| [`.gitignore`](../.gitignore) | Existing Git ignore rules (currently only `.DS_Store`); large local models/prediction exports should be kept out of commits deliberately. |
| [`week4_handoff.py`](../week4_handoff.py) | **Primary Week 4 entry point.** Loads the authoritative Parquet, constructs the supervised tool-pause cohort, verifies inherited session splits, normalizes candidate features, trains the baseline/Ridge/gradient-boosted models, chooses a point model on validation log1p-MAE, scores held-out test sessions, makes plots, and exports the local prediction/CDF handoff for later policy comparison. Also contains a small synthetic test fixture. |
| [`.github/workflows/week4_extended.yml`](../.github/workflows/week4_extended.yml) | GitHub Actions workflow: install Python dependencies; run both test modules; recompute real-data predictions; run extended analysis; archive aggregate outputs; and commit aggregate research files when branch state permits. It avoids force-pushing over concurrent teammate commits. |

### 2.2 `analysis/` — inherited Week 3 trace preparation and validation

**Purpose:** Provide one categorized, auditable pause table and supporting checks. These files are foundational inputs/references, **not** the new Week 4 prediction algorithms.

| Path | Purpose and output |
|---|---|
| [`analysis/dataset/README.md`](../analysis/dataset/README.md) | Authoritative explanation of the 33-column handoff schema: 15 original fields, 18 added fields; targets, QA/record categories, candidate feature meanings, and split rules. |
| [`analysis/dataset/pause_handoff.parquet`](../analysis/dataset/pause_handoff.parquet) | **Canonical Week 4 input**, 357,161 original rows and 33 columns; includes retrospective pause labels, candidate decision-time features, logging/timing categories, and deterministic session-based splits. Original source records remain represented in this file. |
| [`analysis/trace_facts/README.md`](../analysis/trace_facts/README.md) | Instructions and caveats for descriptive TraceLab tool-latency/session statistics. |
| [`analysis/trace_facts/trace_statistics.py`](../analysis/trace_facts/trace_statistics.py) | Exploratory tool-call latency, session summaries, long-tail thresholds, and available-prediction-field diagnostics; these are not measured GPU cache idle times. |
| [`analysis/validation/README.md`](../analysis/validation/README.md) | Guide to the Week 3 validation scripts and regeneration of the single Version 2 handoff. |
| [`analysis/validation/RUNTIME_MEASUREMENT_PLAN.md`](../analysis/validation/RUNTIME_MEASUREMENT_PLAN.md) | Proposed vLLM/SGLang instrumentation needed to distinguish external waits, queueing, actual GPU-resident KV cache intervals, and modeled simulator costs. This is a plan, not completed GPU validation. |
| [`analysis/validation/WEEK3_VALIDATION.md`](../analysis/validation/WEEK3_VALIDATION.md) | Historical diagnostic findings on trace timing, overlapping calls, session boundaries, and potential Codex tool-attribution leakage. Explains why pause durations must be called proxies. |
| [`analysis/validation/WEEK4_HANDOFF.md`](../analysis/validation/WEEK4_HANDOFF.md) | Explicit Week 3 → Week 4 contract: single lossless handoff, 33 columns, source identity preserved, retrospective fields barred from online features, and outstanding timestamp limitations. |
| [`analysis/validation/build_pause_handoff.py`](../analysis/validation/build_pause_handoff.py) | Rebuilds the categorized feature-engineered Parquet from the upstream Version 2 source, checks row/column preservation, and assigns deterministic provider/session splits. This is not run during every Week 4 training pass because the checked-in handoff already exists. |
| [`analysis/validation/build_tool_dataset.py`](../analysis/validation/build_tool_dataset.py) | Independent exploratory extraction of individual tool intervals, overlapping tool-activity episodes, candidate waits, and signal alignment. These exploratory intermediate artifacts are not substitutes for the checked-in handoff. |
| [`analysis/validation/check_handoff_bias.py`](../analysis/validation/check_handoff_bias.py) | Descriptive checks of categories/coverage and duration distributions by provider, log format, and analysis track. |
| [`analysis/validation/audit_v2_pause_events.py`](../analysis/validation/audit_v2_pause_events.py) | Compares Version 2 pause estimates with runner-reported tool durations and event timestamps; flags older-Codex discrepancies without rewriting the source. |
| [`analysis/validation/inspect_cases.py`](../analysis/validation/inspect_cases.py) | Reproduces and inspects selected anomalous timestamp/attribution cases. |
| [`analysis/validation/validate_week3.py`](../analysis/validation/validate_week3.py) | Independent tool-latency and overlap consistency checks, including earlier trace-boundary investigations. |

### 2.3 `week4/` — Week 4 scripts, instructions, and tests

| Path | Role |
|---|---|
| [`week4/README.md`](README.md) | Initial Week 4 setup, supervised-cohort definition, core model-running command, and output description. |
| [`week4/REAL_TRACE_SUMMARY.md`](REAL_TRACE_SUMMARY.md) | Historical, concise first-run metric summary supplied after running the real trace. Use the committed `results/` files for full/final values; its introductory provenance wording reflects the earlier state of the project. |
| [`week4/REMAINING_ANALYSIS.md`](REMAINING_ANALYSIS.md) | Instructions and specification for the extended audits, tail/threshold studies, calibration, bootstrap, and final-research-report outputs. |
| **`week4/report.md`** | **This file:** complete explanation of the branch's purpose, code, generated outputs, findings, limitations, and Week 6 handoff. |
| [`week4/requirements.txt`](requirements.txt) | Python dependencies: NumPy, pandas, PyArrow, scikit-learn, Matplotlib, and joblib. Uses lower-bound version requirements; actual versions are logged in the experiment manifest. |
| [`week4/run_week4.sh`](run_week4.sh) | Shell command to run the primary real-data predictor with seed 560, up to 80,000 sampled training rows, 75 boosting iterations, and illustrative `B/c = 1.0` second. |
| [`week4/extended_analysis.py`](extended_analysis.py) | **Extended evaluation program.** Joins predictions back to the correct held-out source rows, audits excluded labels and tool-metadata availability, retrains restricted-feature ablations, evaluates tail errors, threshold false-retain/false-evict decisions and quantile coverage, performs paired session bootstrap, and generates aggregate results plus the final scientific report. |
| [`week4/run_remaining.sh`](run_remaining.sh) | Runs the extended analysis against local `results/week4_handoff/predictions_test.csv` and the canonical Parquet, then runs all Week 4 tests. It assumes the primary predictor has already produced its local test prediction file. |
| [`week4/tests/test_week4_handoff.py`](tests/test_week4_handoff.py) | Five synthetic-fixture unit tests for feature whitelist safety, disjoint session splits, invalid split detection, retrospective cohort selection, and tool-name normalization. |
| [`week4/tests/test_extended_analysis.py`](tests/test_extended_analysis.py) | Seven additional synthetic-fixture tests for target eligibility, threshold flip decomposition, tail groups, calibration grouping, bootstrap output, restricted-feature ablations, and prediction/source-row alignment. |

**What the tests establish:** The checked-in testing suite passed 12 tests in the successful GitHub Actions run. Tests check software behavior and data-handling invariants; they do **not** independently prove that all trace fields are observable at the server's actual retention decision.

### 2.4 `results/week4_handoff/` — primary real-data prediction results

These are **committed aggregate outputs** from `week4_handoff.py`.

| File | Contents and purpose |
|---|---|
| [`analysis.md`](../results/week4_handoff/analysis.md) | Human-readable initial summary of four point-model test metrics, chosen model, calibration, and caveats. |
| [`metrics.json`](../results/week4_handoff/metrics.json) | Main machine-readable source of truth for split/cohort counts, selected model, validation/test error metrics, quantile calibration, duration percentiles, and illustrative threshold diagnostics. |
| [`group_error_analysis.csv`](../results/week4_handoff/group_error_analysis.csv) | First grouped error audit by provider, logging format, tool topology, feature metadata category, and tool signature. |
| [`prediction_scatter.png`](../results/week4_handoff/prediction_scatter.png) | Held-out observed versus predicted duration visualization on a log1p scale. |
| [`quantile_calibration.png`](../results/week4_handoff/quantile_calibration.png) | Overall nominal versus observed quantile-coverage plot for P10/P50/P90. |

Additional **local generated files**, which are not committed on this branch, may be created by the core run:

| Local path | Use |
|---|---|
| `results/week4_handoff/predictions_test.csv` | Held-out event IDs, observed durations, point predictions, and quantiles; used for extended analysis and later matched-trace policies. Test labels are evaluation-only, not predictor inputs. |
| `results/week4_handoff/predictions_validation.csv` | Validation-session model predictions and associated observations. |
| `results/week4_handoff/continuum_train_cdf_reference.csv.gz` | **Train-only** provider/tool duration observations for constructing later Continuum-style empirical duration distributions without using held-out test outcomes. |
| `results/week4_handoff/fitted_models.joblib` | Serialized trained Ridge and boosted quantile estimators for local reproducibility. Load only from trusted sources. |

### 2.5 `results/week4_extended/` — final extended real-data analyses

These files were generated by `week4/extended_analysis.py` and committed to the branch.

| File | Contents and use |
|---|---|
| [`WEEK4_FINAL_REPORT.md`](../results/week4_extended/WEEK4_FINAL_REPORT.md) | Short completed research narrative summarizing scope, numeric model comparison, bootstrap inference, exclusions, modeling caveats, and future policy handoff. |
| [`experiment_manifest.json`](../results/week4_extended/experiment_manifest.json) | Input/prediction SHA-256 digests, dataset sizes, random seed, bootstrap replicates, training cap, boosting iterations, Python and library versions, and Git commit provenance. |
| [`target_eligibility.csv`](../results/week4_extended/target_eligibility.csv) | Counts source observations by offline analysis track, provider, log format, timing category, and positive/zero/negative/missing target state. |
| [`feature_audit.csv`](../results/week4_extended/feature_audit.csv) | Counts split/provider/log-format groups, feature-availability category, and tool topology; flags metadata that may be known only after tool completion. |
| [`feature_ablation.csv`](../results/week4_extended/feature_ablation.csv) | Held-out metrics for boosted models using only provider/model, then turn index, then provisional context features, **without tool metadata**. |
| [`tail_error_analysis.csv`](../results/week4_extended/tail_error_analysis.csv) | Global and per-provider model errors on all pauses, >10s, >60s, global top decile, short pauses, and the 10–60s band. |
| [`group_error_extended.csv`](../results/week4_extended/group_error_extended.csv) | Extended model errors/underprediction and long-pause prevalence for provider, format, tool topology, feature category, and tool signatures. |
| [`threshold_analysis.csv`](../results/week4_extended/threshold_analysis.csv) | Compares learned model with provider/tool median at six illustrative `B/c` horizons, including false-retain, false-evict, overall/near-threshold flips, and a simplified one-pause cost diagnostic. |
| [`calibration_by_group.csv`](../results/week4_extended/calibration_by_group.csv) | Marginal and grouped P10/P50/P90 coverage, 80% prediction interval coverage, interval width, and conditional interval coverage for long pauses. |
| [`session_bootstrap.csv`](../results/week4_extended/session_bootstrap.csv) | Paired session-level resampling (2,000 replicates): model improvements and 95% percentile confidence intervals, globally and by provider, for MAE/log-MAE and selected threshold-flip rates. |

## 3. Data, cohort and modeling methodology

### Data provenance and eligibility

The canonical file contains **357,161 records and 33 fields**. The offline tracks include:

| Analysis track | Source records |
|---|---:|
| Tool-pause prediction | 314,589 |
| Human-wait prediction | 33,374 |
| Mixed-wait prediction | 3,534 |
| Boundary/source review | 5,664 |
| **All records** | **357,161** |

Week 4 selects the `tool_pause_prediction` track **only for offline supervised evaluation** and then uses finite, strictly positive `pause_s` values. Exactly **2,825 otherwise eligible tool-pause records have zero duration** and are omitted from this positive-duration experiment. This is a documented modeling decision, **not** proof that zero-duration rows are invalid. No missing or negative labels occurred inside this tool-pause track. The original Parquet retains all rows.

The supervised population is **311,764** observations:

| Split | Observations | Distinct session groups |
|---|---:|---:|
| Train | 239,053 | 2,920 |
| Validation | 24,938 | 362 |
| Test | 47,773 | 384 |

The split is inherited from the Week 3 provider/session SHA-256 rule. The model uses up to **80,000 sampled training observations** (seed 560) for training the learned estimators, chooses the point model on **validation log1p-MAE**, and reports performance on the untouched held-out test partition.

### Features, target, and models

The target is observed trace-derived pause duration `pause_s`. For the learned regressors, training uses `log1p(pause_s)`, with predictions transformed back to seconds. The feature whitelist includes provider, model, tool signature (from masked candidate tool-name metadata), candidate tool counts, log context length, log turn index, log tool-input character count, and candidate tool-presence/multiple-tool flags.

The model **does not** ingest the duration target, subsequent pause endpoint, retrospective cause/track/quality category, eventual tool result, split label, or session ID. Session ID is only used for grouping, leakage checks, and bootstrap resampling. Nonetheless, some *candidate* tool and context metadata have not been proven to exist at the exact inference-server decision instant.

Four point-estimate approaches are compared: global median, provider/tool historical median, log-duration Ridge, and histogram gradient-boosted median (`hgb_q50`). Boosted P10 and P90 regressors also produce duration uncertainty intervals. The primary selected model is `hgb_q50`.

## 4. Measured real-trace results

### 4.1 Overall predictive accuracy

All values below are from held-out **test sessions** (`n = 47,773`), not synthetic fixtures.

| Point predictor | MAE (seconds) ↓ | log1p-MAE ↓ | Underpredicted pauses | Flip rate at illustrative 1s threshold ↓ |
|---|---:|---:|---:|---:|
| Global median | 16.8294 | 0.7530 | 55.8% | 38.17% |
| Provider/tool median | 16.4105 | 0.6062 | 53.3% | 29.21% |
| Log Ridge | 16.4945 | 0.6556 | 33.9% | 31.37% |
| **HGB Q50** | **16.1958** | **0.5441** | 51.9% | **26.39%** |

Relative to the provider/tool median, the selected HGB model improves **log1p-MAE by approximately 10.2%** and raw MAE by approximately **1.3%**. The improvement is stronger in log-duration error than in absolute seconds because very long waits heavily influence raw MAE.

Session-paired bootstrap of **384 held-out sessions**, using 2,000 resamples, estimates a log1p-MAE gain of **0.0621** (positive favors HGB), with **95% percentile interval [0.0340, 0.0953]**. This event-weighted interval does not establish a population-level improvement for an equally weighted typical session, nor a policy-regret theorem or proof of cross-workload generalization.

### 4.2 Heavy tail and differing workloads

Held-out duration percentiles are approximately **P50 0.270 s, P90 10.102 s, P99 126.804 s**. The longest **10% of pauses account for 93.77% of summed observed pause time** in the test cohort.

The HGB model improves the top-decile MAE from **153.30 s** (provider/tool median) to **150.75 s**, yet **91.6%** of those longest-decile pauses are still underpredicted.

For the **1,155 pauses over 60 seconds**, the HGB MAE is **562.54 s**, marginally *worse* than the provider/tool median's **562.22 s**, and HGB underpredicts **98.4%** of these pauses. These are important limitations, not exceptions to discard.

Provider-specific HGB MAE also differs substantially: **Claude 44.70 s** versus **Codex 4.50 s**. This describes these held-out subpopulations; it is not a controlled provider-quality comparison because workloads and tool mixes differ.

### 4.3 Quantile estimates: useful overall, weaker in the tail

Observed aggregate coverage:

| Target | Empirical fraction |
|---|---:|
| P10 (nominal 10%) | 10.8% |
| P50 (nominal 50%) | 48.1% |
| P90 (nominal 90%) | 91.2% |
| Central 80% interval | 80.4% |

However, conditional central-80% interval coverage falls to **54.1% for waits >10 s** and **22.7% for waits >60 s**. Thus apparently good *marginal* calibration does **not** imply reliable upper-tail uncertainty. The code also recorded **126 raw quantile-order crossings** before sorting the three predicted quantiles for reporting.

### 4.4 Decision threshold sensitivity, not a retention-policy experiment

At a hypothetical break-even horizon `b = B/c`, a simplified diagnostic classifies a predicted duration as short/long relative to `b`. It reports classification flips against a duration-aware decision, plus a toy two-action cost comparison. **No GPU resource contention, memory budget, scheduling queue, or actual λ-policy is simulated here.**

| Hypothetical `B/c` | Provider/tool median flip rate | HGB Q50 flip rate |
|---|---:|---:|
| 0.1 s | 27.14% | 27.08% |
| 0.3 s | 35.52% | 28.99% |
| 1 s | 29.21% | 26.39% |
| 3 s | 16.86% | 16.00% |
| 10 s | 10.34% | 9.13% |
| 60 s | **2.39%** | 2.42% |

The advantage depends on the threshold. For example, at **1 s**, HGB improves the *overall* flip rate but has a slightly *worse* flip rate among the 6,168 events near the threshold (**79.04% vs 78.11%**). At 60 s, the provider/tool median has a marginally lower overall flip rate. These comparisons motivate decision-aware metrics instead of assuming lower MAE implies universally better retention.

### 4.5 Feature restrictions and target quality

Restricted-feature boosted models (without tool names/counts/input sizes) achieve these held-out **test log1p-MAEs**:

| Features allowed in the ablation | Test log1p-MAE |
|---|---:|
| Provider + model | 0.7368 |
| Provider + model + turn index | 0.7551 |
| Provider + model + turn index + provisional context size | 0.7604 |
| Full candidate-feature HGB Q50 | **0.5441** |

Tool-related metadata appears valuable for this model/workload, but the comparison is **not** proof that each candidate feature was available at the instant of a real scheduler retention decision. Older Codex result-time metadata is masked in the derived feature fields; some multi-tool totals may also be retrospective. The Week 3 timing audit remains essential context.

## 5. Conclusions that are supported now

1. **Agent pause durations are partially predictable on the held-out TraceLab-derived cohort.** The learned median model improves on global and provider/tool historical baselines, especially on the log-duration scale. The event-weighted result survived paired session-level bootstrap uncertainty analysis, but an equal-session analysis does not support an unconditional significance claim.
2. **Tool identity and related metadata contain useful predictive information in this dataset.** Restricted-feature models perform worse. Whether all such metadata is available online must still be validated against a concrete retention-decision timestamp.
3. **Pause durations are strongly heavy-tailed.** A relatively small set of long pauses dominates total observed waiting time and absolute prediction error.
4. **Quantile estimates are reasonably calibrated in aggregate but poorly cover extreme waits.** Future policies must consider conservative tail handling or distribution-shift failure.
5. **Prediction improvement is decision-dependent.** A lower overall log-MAE does not ensure fewer errors near every `B/c` boundary or under every GPU memory cost.
6. **Week 4 delivered reproducible inputs and diagnostics for policy work, not evidence of system-wide KV-cache savings.** In particular, the teacher-requested theoretical consistency/robustness guarantee and matched-memory comparison with Continuum-style TTL have **not** yet been established; they belong to later project stages.

## 6. Limitations / interpretation rules

- **Trace proxies, not hardware measurements:** `pause_s` estimates wait intervals from coding-agent traces and is not an observed GPU KV-cache idle-residency interval. Parallel tools, command polling, human involvement, round boundaries, and Codex logging conventions can complicate interpretation.
- **Retrospective cohort selection:** `analysis_track == tool_pause_prediction` is used to define an offline analysis cohort, but is **not** an online model feature. A deployed scheduler must determine which situations require a retention decision without using a future-derived label.
- **Candidate feature observability:** Tool metadata and source-derived context/turn features need timestamp-level validation at the policy decision point. Feature ablations document sensitivity, not guaranteed online readiness.
- **Zero durations:** The 2,825 zero-duration tool-pause records were excluded from the strictly-positive training target; the source file preserves them. Changing inclusion criteria is a sensitivity study, not a silent correction.
- **Calibration limitations:** Overall P10/P50/P90 coverage conceals weak conditional coverage in the extreme tail.
- **No cost calibration:** Reported `B/c` settings are hypothetical. Neither the threshold-flip statistic nor the toy cost ratio measures hardware savings, a Continuum reproduction, or robustness in concurrent serving.
- **Finite evaluation domain:** Data come from the selected TraceLab-derived sessions; cross-provider/workload shift and adversarial prediction-failure sweeps remain future experiments.
- **Reproducibility scope:** All aggregate metrics are committed, but event-level prediction exports, trained joblib objects, and training-only CDF records are generated locally/on CI and are not present as ordinary branch files.

## 7. Reproduce Week 4 from the repository root

Use a Python environment with the declared dependencies; the successful CI run used Python 3.11.17 and logged versions in [`experiment_manifest.json`](../results/week4_extended/experiment_manifest.json).

~~~bash
git switch Hangting_week4
python3 -m pip install -r week4/requirements.txt

# 1. Primary learned models and local predictions
bash week4/run_week4.sh

# 2. All extended analyses and unit tests
bash week4/run_remaining.sh

# Optional: run all tests directly
python3 -m unittest discover -s week4/tests -p 'test_*.py' -v
~~~

The second step requires the first step's local `predictions_test.csv`; do not replace it with results fitted using test targets. The core run creates `results/week4_handoff/`; the extended run creates `results/week4_extended/`.

The checked-in GitHub workflow [`week4_extended.yml`](../.github/workflows/week4_extended.yml) also performs an end-to-end real-data run. **All 12 tests and both evaluation stages passed in the verified workflow run** [#40108065062](https://github.com/Omkar38/agent-kv-retention/actions/runs/40108065062). Results are archived as a workflow artifact, and aggregate files are committed to the branch without force-pushing over other contributors' changes.

## 8. Downstream handoff and future work

**Week 5 (buffer/integration):** Freeze the source schema and session splits; confirm which prediction fields are available at an actual retain/evict decision. Keep the zero-duration and older-Codex caveats visible. Maintain the hashed experiment configuration and ensure other teammates can regenerate the outputs.

**Week 6 (algorithm and simulator):** Define the single-request cost `C(τ,T) = c·min(T,τ) + B·1[T>τ]` with `c > 0` and `B > 0`, construct a learning-augmented retention rule with trust parameter `λ`, and prove a consistency/robustness guarantee for the **simplified single-request model**. Integrate the prediction distribution and model outputs into an explicit, trace-driven simulation interface. Handle multiple requests and load-dependent opportunity cost in simulation rather than pretending the fixed-cost proof covers GPU scheduling.

**Week 7 and beyond (fair policy evaluation):** Reconstruct the Continuum-style TTL baseline from its published description, using **training-only** empirical per-tool duration distributions; compare it and the λ-policy on the **same held-out sessions, cost assumptions, memory budgets, and concurrency scenarios**. Introduce biased, shifted, tail-heavy, and adversarial predictions; measure actual simulated memory occupancy, recomputation, latency, throughput, and oracle-relative cost. GPU validation is optional, and any such observations must be labeled local replay measurements.

### Final Week 4 assessment

**Complete for the Week 4 milestone:** dataset handoff reused without mutation; models trained and selected on session-separated data; real test metrics and quantile calibration; extensive tail and threshold diagnostics; eligibility/feature audits; session-bootstrap intervals; automated testing; GitHub results and reproducible execution instructions.

**Not a claim of full project completion:** The validated-retention-policy theorem, Continuum-style matched-capacity evaluation, and load-dependent simulator are separate later deliverables.

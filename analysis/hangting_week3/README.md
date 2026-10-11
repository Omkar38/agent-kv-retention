# Week 3 — TraceLab Processing, Workload Analysis, and Data Validation

**Project:** Learning-Augmented KV-Cache Retention for Tool-Using LLM Agents
**Scope:** The `Hangting_week3` branch includes round-gap characterization, tool latency analysis, interval extraction, overlap-aware tool-activity episodes, data-quality checks, and a handoff for learning-augmented retention research.

The practical entry points for this branch are `week3/`, `analysis/trace_facts/`, and `analysis/hangting_week3/`. This README explains their combined purpose, inputs, usage, and interpretation. [WEEK3_VALIDATION.md](WEEK3_VALIDATION.md) contains the observed counts and anomaly investigations.

## 1. Purpose and significance

An agent alternates between LLM generation and external actions (shell commands, tests, APIs, and human input). While awaiting an external action, retaining a session's KV cache avoids recomputation on resumption, but consumes GPU memory that other requests could use. A retention policy must trade these costs off *before* the next tool-return time is known.

Week 3 turns TraceLab execution records into measurable workload properties: tool-delay distributions, long-tail behavior, context size, differences between Claude and Codex trace structure, and which fields are observable at a proposed decision point. It provides evidence for predicting pause durations in Week 4 and trace-driven retention simulations later in the project.

Two **different** observational units are analyzed. The original round-gap table approximates time between LLM rounds. The tool-activity pipeline measures actual recorded tool-call intervals and unions overlapping calls. They are not interchangeable, and neither has yet been proven identical to a KV-cache idle interval on a running inference server.

## 2. Scripts by responsibility

### `week3/` — round-gap workload characterization

| File | Responsibility |
|---|---|
| `build_pause_table.py` | Build `pauses.parquet` from round transitions, with cause, duration, tool attribution, context proxy and usability flags. Its current next-round matching can attribute later Codex calls to an earlier gap; see the validation report. |
| `look_at_session.py` | Inspect one session's raw round/event ordering to explain unexpected gaps or event types. |
| `plot_pause_cdf.py` | Plot the cumulative distribution of original round-gap durations, including illustrative break-even thresholds. |
| `plot_pause_by_tool.py` | Plot durations by attributed tool and demonstrate a session-held-out, tool-name-only long/short duration rule. Its accuracy is **not** a measured retention-policy cost improvement. |
| `plot_context_size.py` | Plot context-token distribution and illustrative rebuild-time sensitivity; token/s speeds are assumptions, **not** hardware calibrations. |

Additional materials already in `week3/` include `data/pauses.parquet`, figures, and the trace-processing report. Existing documentation may mention `look_at_pauses.py`; that file is not present in this branch.

### `analysis/trace_facts/` — tool and session workload statistics

| File | Responsibility |
|---|---|
| `check_tool_latency.py` | Distribution of individual tool-call wall latencies, including P50, P90, P95, P99 and P99.9; printing order was corrected. |
| `check_session_latency.py` | Per-session aggregates (counts, sums, max and thresholds). Sums double-count overlapping tool intervals and must **not** be described as elapsed session waiting time. |
| `check_tool_thresholds.py` | Shares of valid calls above 5, 10, 30 and 60 seconds by tool type. Counts in this script exclude invalid/null latency records by its WHERE filter. |
| `check_prediction_fields.py` | Sample schema fields and coverage for selected tools using columns present in the released TraceLab schema. Availability in a table alone does **not** establish availability at retention decision time. |

### `analysis/hangting_week3/` — overlap-aware tool extraction and QA

| File | Responsibility |
|---|---|
| `build_tool_dataset.py` | Complete four-stage pipeline: direct tool intervals → merged tool-activity episodes → first-tool candidate features → audited candidate flags. Writes four distinct Parquet artifacts. |
| `validate_week3.py` | Quantitative checks for within-round tool results, timestamp consistency, overlapping calls, event signals inside episodes, boundary/feature completeness, and next-round result-attribution timing. |
| `inspect_cases.py` | Reproducible case studies of a Codex late tool call, late-match origin breakdown, and a Claude negative round-gap example. |
| `WEEK3_VALIDATION.md` | Actual measurements, event-ordering issues, interpretations, and research handoff constraints. |
| `README.md` | This branch-wide execution and dataset guide. |

## 3. Data source and provenance

- Primary raw source: [TraceLab](https://github.com/uw-syfi/TraceLab) coding-agent trace, release **v0.0.1**, database file `trace/syfi_coding_trace.duckdb`. Consult the original release and license terms and cite TraceLab in papers, reports and plots.
- The `week3/` round-gap builder and plots are from the team's earlier Week 3 contribution; `analysis/trace_facts/` holds the team's tool-latency and schema analysis. The overlap-aware pipeline and the cross-workstream validation are documented in `analysis/hangting_week3/`.
- The original derived round-gap dataset and figures are *preserved* rather than rewritten after the Codex attribution audit. Reproduced counts describe that original extraction, not a corrected KV-retention ground truth.
- All measurements in the validation report were obtained from the group's local copy of the released DuckDB file on **October 11, 2026**. Parquet outputs generated locally are not required to be committed to source control. Record the release/version and any preprocessing changes in subsequent experiments.

## 4. Reproduce on macOS / Linux

Requirements: Python 3.10+, `duckdb`, `pandas` and `matplotlib` (plus `numpy` for the existing figures). Use TraceLab's own environment if preferred. Download/prepare its v0.0.1 DuckDB release before running the analysis. Do not replace this database with unverified synthetic records.

The scripts use relative paths, so run all commands **from the TraceLab root**, not from the GitHub repository. Example directory layout:

```text
kv-week3-work/
├── TraceLab/
│   ├── trace/syfi_coding_trace.duckdb
│   └── group_work/
└── agent-kv-retention-hangting/
    ├── week3/
    └── analysis/
        ├── trace_facts/
        └── hangting_week3/
```

```bash
cd ~/kv-week3-work/TraceLab
source .venv/bin/activate        # omit if already active
python -m pip install duckdb pandas matplotlib numpy
mkdir -p group_work
REPO=~/kv-week3-work/agent-kv-retention-hangting
```

**A. Existing round-gap table and figures (optional, useful for comparison):**

The original `week3/` scripts write to `our_work/`, while the new pipeline writes to `group_work/`. On a new checkout, create a separate `our_work` directory and, if needed, copy the resulting `pauses.parquet` into `group_work` for attribution checks:

```bash
mkdir -p our_work
python "$REPO/week3/build_pause_table.py"
python "$REPO/week3/plot_pause_cdf.py"
python "$REPO/week3/plot_pause_by_tool.py"
python "$REPO/week3/plot_context_size.py"
cp our_work/pauses.parquet group_work/pauses.parquet
```

The source-controlled `week3/data/pauses.parquet` can also be used for original round-gap analyses, but verify it corresponds to the same trace release and extraction code.

**B. Build the tool-activity datasets (main pipeline):**

```bash
python "$REPO/analysis/hangting_week3/build_tool_dataset.py"
```

The builder runs all four stages sequentially. You can rerun a stage using `--step intervals`, `--step episodes`, `--step candidates`, or `--step audit` when its inputs are present. Run `--help` for the options. Expect four files in `group_work/` (described below).

**C. Run checks and examine anomalies:**

```bash
python "$REPO/analysis/hangting_week3/validate_week3.py"
python "$REPO/analysis/hangting_week3/inspect_cases.py" --case codex
```

The feature-timing check and late-call case require `group_work/pauses.parquet`; without it, those checks are skipped and explicitly reported as skipped. Other inspection modes are `--case late_calls` and `--case negative_gap`. For a single quantitative check, use `validate_week3.py --step <name>` (see `--help`).

**D. Run the complementary tool statistics:**

```bash
python "$REPO/analysis/trace_facts/check_tool_latency.py"
python "$REPO/analysis/trace_facts/check_session_latency.py"
python "$REPO/analysis/trace_facts/check_tool_thresholds.py"
python "$REPO/analysis/trace_facts/check_prediction_fields.py"
```

Because these are exploratory scripts, outputs are printed directly to the terminal. The figures from step A are stored in `our_work/`. For a reproducibility check, compare counts with [WEEK3_VALIDATION.md](WEEK3_VALIDATION.md), not just whether a script exits successfully.

## 5. Output datasets: semantics, features, and labels

| Data artifact | One row represents | Key fields / interpretation |
|---|---|---|
| `our_work/pauses.parquet` / `group_work/pauses.parquet` | A transition between two LLM rounds | `pause_s`, `cause`, `usable`, `context_tokens` and next-round-associated tool summaries. **Exploratory round-gap record**; Codex attribution and boundary issues remain. |
| `group_work/tool_intervals.parquet` | One raw tool-call record | `tool_start_ts`, `tool_end_ts`, `tool_duration_s`, `reported_tool_duration_s`, `timing_status`. Missing/invalid timestamps are explicitly flagged. |
| `group_work/tool_activity_episodes.parquet` | Union of overlapping timestamp-valid tool calls in a session | `start_ts`, `end_ts`, `n_calls`, `n_rounds`, `tool_activity_duration_s`. Counts and full tool lists reflect the **entire observed episode**. |
| `group_work/tool_episode_candidates.parquet` | An episode linked to its earliest emitted tool call | `candidate_start_ts`, `first_tool_name`, `first_tool_input_chars`, `context_tokens`, `tool_activity_duration_s`, and retrospective episode composition. |
| `group_work/tool_episode_candidates_audited.parquet` | Candidate above plus diagnostic flags | `generation_events_inside`, `user_messages_inside`, `simple_tool_wait_proxy`; all simple and complex cases are retained. |

**Suggested preliminary prediction target:** `tool_activity_duration_s`, explicitly labeled *tool-activity duration*. This is not the wall-clock time to the next LLM decode and is not a verified KV-cache-residency interval.

**Candidate information at first tool emission:** `provider`, `first_tool_name`, `first_tool_input_chars`, and (only after decision-time verification) `model` or `context_tokens`. Session identifiers are suitable for **grouped train/test splits**, not features that should be memorized by the model.

**Never feed these retrospective columns into an online predictor:** `tool_activity_duration_s` (the label), `tool_activity_end_ts`, `observed_n_calls`, `observed_n_rounds`, `observed_tool_names`, `generation_events_inside`, `user_messages_inside`, or `simple_tool_wait_proxy`. Even using a flag for **filtering** a test population requires disclosure because it is derived from future events. Some known-at-source fields may still be unsafe for the actual *KV retention decision*, which occurs after inference completes, not necessarily at first tool issuance.

**Quality and causality notes:** Simultaneous/overlapping tool calls are merged to avoid double-counting tool activity; this does not establish true multi-request GPU behavior. Cases with new user input or generation during an episode are reported, not silently dropped. Do not concatenate round-gap rows and tool-episode rows as if they were independent identical pause samples.

## 6. Key observed results and their limits

Using the v0.0.1 trace, the round-gap builder produced **357,161 candidate transitions**, of which **346,820** were usable under its filters. The direct tool-call analysis yielded **431,905** timestamp-valid tool intervals, merged into **331,400** observed activity episodes: **277,777** met the conservative *single tool, no recorded intervening generation/user input* proxy definition, and **53,623** were more complex. A result-driven next-round matching audit found **63,122** tool-call matches emitted after the putative earlier decision timestamp, predominantly Codex. See the validation report for denominators, exact counts, sessions and caveats.

This means the branch establishes a reproducible **workload characterization and preliminary prediction proxy**, not a verified per-pause TTL simulation workload or measured GPU memory saving. Plot break-even times derived from hypothetical prefill speeds must be recalibrated on real hardware.

## 7. How later weeks can use this work

**Week 4 — pause-duration prediction.** Train preliminary models on directly observed tool-activity durations; split by `session_id` to prevent session leakage. Compare tool-only, argument-size, provider, and verified context features; evaluate error, calibration and long-tail bias separately on simple versus complex episodes. Do not use hindsight episode summaries as predictor features.

**Week 5 — project presentation.** Explain why the tool-latency tail and provider-specific trace structure matter for retention and why an outcome-based feature join can overstate predictability. Present original figures as baseline exploratory analysis and label placeholder hardware costs.

**Weeks 6–7 — policy and simulator.** Define inference-end, pause-start and next-generation-resume events. Recover a separate validated `pauses_v2` with absolute timestamps, causes and exclusions. Use `T` as the actual validated retention interval, then calibrate memory cost `c` and eviction/recompute penalty `B` for `C(τ,T)=c·min(T,τ)+B·1[T>τ]`. Simulate eviction, fixed TTL and Continuum-style TTL under the *same* trace inputs and memory constraints. Prove the single-request consistency/robustness property independently of multi-request queueing effects.

**Weeks 8–10 — stress tests and reporting.** Study prediction bias, distribution shift, heavy-tail resampling and memory contention with clear simple/complex cohort reporting. Evaluate average latency, recomputation, p95/p99 completion time and queue effects after obtaining a valid mapping from traces to inference-server behavior. Preserve source-release and code versions for repeatability.

## 8. Known limitations

The original next-round tool attribution and pause endpoint are unreliable for some Codex traces, and some tool calls occur entirely within a recorded LLM round. Rare negative/duplicate round anomalies also require explicit treatment. Merging tool intervals does not prove the GPU is idle. Current `context_tokens` is a context-size proxy rather than a proven timestamped cache-size measurement. The Week 3 data supports tool workload research now; actual KV-retention-policy claims require a separate inference-stop/resume validation and memory-system calibration.

# Week 3 — TraceLab Processing, Workload Analysis, and Data Validation

**Project:** Learning-Augmented KV-Cache Retention for Tool-Using LLM Agents
**Scope:** The `Hangting_week3` branch includes round-gap characterization, tool latency analysis, interval extraction, overlap-aware tool-activity episodes, data-quality checks, and a handoff for learning-augmented retention research.

This branch contains `analysis/trace_facts/` and `analysis/validation/`. Brian's original [round-gap analysis](https://github.com/Omkar38/agent-kv-retention/tree/Brian/week3) is referenced externally, rather than copied here. This README explains their combined purpose, inputs, usage, and interpretation. [WEEK3_VALIDATION.md](WEEK3_VALIDATION.md) contains the observed counts and anomaly investigations.

## 1. Purpose and significance

An agent alternates between LLM generation and external actions (shell commands, tests, APIs, and human input). While awaiting an external action, retaining a session's KV cache avoids recomputation on resumption, but consumes GPU memory that other requests could use. A retention policy must trade these costs off *before* the next tool-return time is known.

Week 3 turns TraceLab execution records into measurable workload properties: tool-delay distributions, long-tail behavior, context size, differences between Claude and Codex trace structure, and which fields are observable at a proposed decision point. It provides evidence for predicting pause durations in Week 4 and trace-driven retention simulations later in the project.

Two **different** observational units are analyzed. The original round-gap table approximates time between LLM rounds. The tool-activity pipeline measures actual recorded tool-call intervals and unions overlapping calls. They are not interchangeable, and neither has yet been proven identical to a KV-cache idle interval on a running inference server.

## 2. Scripts by responsibility

### [Brian's `week3/`](https://github.com/Omkar38/agent-kv-retention/tree/Brian/week3) — round-gap workload characterization

| File | Responsibility |
|---|---|
| [`build_pause_table.py`](https://github.com/Omkar38/agent-kv-retention/blob/Brian/week3/build_pause_table.py) | Build `pauses.parquet` from round transitions, with cause, duration, tool attribution, context proxy and usability flags. Its current next-round matching can attribute later Codex calls to an earlier gap; see the validation report. |
| [`look_at_session.py`](https://github.com/Omkar38/agent-kv-retention/blob/Brian/week3/look_at_session.py) | Inspect one session's raw round/event ordering to explain unexpected gaps or event types. |
| [`plot_pause_cdf.py`](https://github.com/Omkar38/agent-kv-retention/blob/Brian/week3/plot_pause_cdf.py) | Plot the cumulative distribution of original round-gap durations, including illustrative break-even thresholds. |
| [`plot_pause_by_tool.py`](https://github.com/Omkar38/agent-kv-retention/blob/Brian/week3/plot_pause_by_tool.py) | Plot durations by attributed tool and demonstrate a session-held-out, tool-name-only long/short duration rule. Its accuracy is **not** a measured retention-policy cost improvement. |
| [`plot_context_size.py`](https://github.com/Omkar38/agent-kv-retention/blob/Brian/week3/plot_context_size.py) | Plot context-token distribution and illustrative rebuild-time sensitivity; token/s speeds are assumptions, **not** hardware calibrations. |

Brian's original branch contains [`data/pauses.parquet`](https://github.com/Omkar38/agent-kv-retention/blob/Brian/week3/data/pauses.parquet), the [figures](https://github.com/Omkar38/agent-kv-retention/tree/Brian/week3/figures), and the [original README](https://github.com/Omkar38/agent-kv-retention/blob/Brian/week3/README.md). Two PDFs found in this branch's history remain accessible at commit `63dddf308444f454fd8b35316c58ee9769276fe4`: [Week 3 trace-processing report](https://github.com/Omkar38/agent-kv-retention/blob/63dddf308444f454fd8b35316c58ee9769276fe4/week3/Week3_Trace_Processing_Report.pdf) and [Week 3 progress update](https://github.com/Omkar38/agent-kv-retention/blob/63dddf308444f454fd8b35316c58ee9769276fe4/docs/week3-progress-update.pdf). These PDFs were not on Brian's or David's original branches; the linked copies are historical records. Brian's README also mentions `look_at_pauses.py`, which is absent from the checked branch.

### `analysis/trace_facts/` — tool and session workload statistics

| File | Responsibility |
|---|---|
| `check_tool_latency.py` | Distribution of individual tool-call wall latencies, including P50, P90, P95, P99 and P99.9; printing order was corrected. |
| `check_session_latency.py` | Per-session aggregates (counts, sums, max and thresholds). Sums double-count overlapping tool intervals and must **not** be described as elapsed session waiting time. |
| `check_tool_thresholds.py` | Shares of valid calls above 5, 10, 30 and 60 seconds by tool type. Counts in this script exclude invalid/null latency records by its WHERE filter. |
| `check_prediction_fields.py` | Sample schema fields and coverage for selected tools using columns present in the released TraceLab schema. Availability in a table alone does **not** establish availability at retention decision time. |

### `analysis/validation/` — overlap-aware tool extraction and QA

| File | Responsibility |
|---|---|
| `build_tool_dataset.py` | Six-stage extraction and diagnostic workflow: intervals, overlap-aware episodes, candidates, audit, event alignment, retrospective classification. |
| `build_pause_handoff.py` | **Week 4 bridge:** join Brian's original pause records with Hangting's independently timestamped source-round tool calls; export a quality-labeled pause table and leakage-restricted strict training subset. See [WEEK4_HANDOFF.md](WEEK4_HANDOFF.md). |
| `validate_week3.py` | Quantitative checks for within-round tool results, timestamp consistency, overlapping calls, event signals inside episodes, boundary/feature completeness, and next-round result-attribution timing. |
| `inspect_cases.py` | Reproducible case studies of a Codex late tool call, late-match origin breakdown, and a Claude negative round-gap example. |
| `WEEK3_VALIDATION.md` | Actual measurements, event-ordering issues, interpretations, and research handoff constraints. |
| `README.md` | This branch-wide execution and dataset guide. |

## 3. Data source and provenance

- Primary raw source: [TraceLab](https://github.com/uw-syfi/TraceLab) coding-agent trace, release **v0.0.1**, database file `trace/syfi_coding_trace.duckdb`. Consult the original release and license terms and cite TraceLab in papers, reports and plots.
- Brian's [`week3/` source](https://github.com/Omkar38/agent-kv-retention/tree/Brian/week3) supplies the original round-gap builder, plots, and derived pause data; they are not duplicated here. `analysis/trace_facts/` holds David's tool-latency and schema analysis, with documented corrections. The overlap-aware pipeline and cross-workstream validation are under `analysis/validation/`.
- The original derived round-gap dataset and figures are *preserved* rather than rewritten after the Codex attribution audit. Reproduced counts describe that original extraction, not a corrected KV-retention ground truth.
- All measurements in the validation report were obtained from the group's local copy of the released DuckDB file on **October 11, 2026**. Parquet outputs generated locally are not required to be committed to source control. Record the release/version and any preprocessing changes in subsequent experiments.

## 4. Reproducibility and execution

Use the [TraceLab coding-agent trace](https://github.com/uw-syfi/TraceLab) release v0.0.1 (`syfi_coding_trace.duckdb`) and the `Hangting_week3` checkout of this repository. All analysis scripts expect paths relative to the **TraceLab root**, so the current working directory matters. The original round-gap scripts are referenced from [Brian's branch](https://github.com/Omkar38/agent-kv-retention/tree/Brian/week3); they are not duplicated on `Hangting_week3`.

Dependencies: Python 3.10+, `duckdb`, `pandas`, `numpy`, `matplotlib`, and Git. On Windows, use **PowerShell**, not Bash, for Section 4.2.
### 4.1. Reproduce on macOS / Linux

Requirements: Python 3.10+, `duckdb`, `pandas` and `matplotlib` (plus `numpy` for the existing figures). Use TraceLab's own environment if preferred. Download/prepare its v0.0.1 DuckDB release before running the analysis. Do not replace this database with unverified synthetic records.

The scripts use relative paths, so run all commands **from the TraceLab root**, not from the GitHub repository. Example directory layout:

```text
kv-week3-work/
├── TraceLab/
│   ├── trace/syfi_coding_trace.duckdb
│   └── group_work/
└── agent-kv-retention-hangting/
    └── analysis/
        ├── trace_facts/
        └── validation/
```

```bash
cd ~/kv-week3-work/TraceLab
source .venv/bin/activate        # omit if already active
python -m pip install duckdb pandas matplotlib numpy
mkdir -p group_work
REPO=~/kv-week3-work/agent-kv-retention-hangting
```

**A. Original round-gap table and plots from Brian's branch (optional comparison):**

Brian's source scripts remain on [his branch](https://github.com/Omkar38/agent-kv-retention/tree/Brian/week3), not in this checkout. Retrieve them without switching away from `Hangting_week3`:

```bash
git -C "$REPO" fetch origin Brian
mkdir -p our_work
for name in build_pause_table.py plot_pause_cdf.py plot_pause_by_tool.py plot_context_size.py; do
    git -C "$REPO" show "origin/Brian:week3/$name" > "our_work/$name"
done
python our_work/build_pause_table.py
python our_work/plot_pause_cdf.py
python our_work/plot_pause_by_tool.py
python our_work/plot_context_size.py
cp our_work/pauses.parquet group_work/pauses.parquet
```

Alternatively, to reproduce the original pause-table *audit* without rerunning the figures:

```bash
git -C "$REPO" show origin/Brian:week3/data/pauses.parquet > group_work/pauses.parquet
```

These historical inter-round records have known Codex attribution limitations; they are not a ground-truth KV-idle dataset. The original plots use illustrative cost assumptions.

**B. Build the tool-activity datasets (main pipeline):**

```bash
python "$REPO/analysis/validation/build_tool_dataset.py"
```

The builder runs **six** stages sequentially, including provisional tool-to-generation alignment and retrospective classification (not GPU idleness). You can rerun a stage using `--step intervals`, `--step episodes`, `--step candidates`, `--step audit`, `--step wait_candidates`, or `--step classify`. Run `--help` for options.

**C. Build Brian/Hangting Week 4 pause handoff** (requires `group_work/pauses.parquet` from step A and `tool_intervals.parquet` from step B):

```bash
python "$REPO/analysis/validation/build_pause_handoff.py"
```

This preserves Brian's table, writes `group_work/pauses_v2.parquet`, and creates `group_work/week4_pause_train.parquet` as a conservative **inter-round pause** prediction cohort. See [WEEK4_HANDOFF.md](WEEK4_HANDOFF.md). The original `pauses.parquet` is not overwritten.

**D. Run checks and examine anomalies:**

```bash
python "$REPO/analysis/validation/validate_week3.py"
python "$REPO/analysis/validation/inspect_cases.py" --case codex
```

The feature-timing check and late-call case require `group_work/pauses.parquet`; without it, those checks are skipped and explicitly reported as skipped. Other inspection modes are `--case late_calls` and `--case negative_gap`. For a single quantitative check, use `validate_week3.py --step <name>` (see `--help`).

**E. Run the complementary tool statistics:**

```bash
python "$REPO/analysis/trace_facts/check_tool_latency.py"
python "$REPO/analysis/trace_facts/check_session_latency.py"
python "$REPO/analysis/trace_facts/check_tool_thresholds.py"
python "$REPO/analysis/trace_facts/check_prediction_fields.py"
```

Because these are exploratory scripts, outputs are printed directly to the terminal. The figures from step A are stored in `our_work/`. For a reproducibility check, compare counts with [WEEK3_VALIDATION.md](WEEK3_VALIDATION.md), not just whether a script exits successfully.


### 4.2. Reproduce on Windows (PowerShell)

**A. Set up the paths and Python environment.** The example assumes two sibling directories under your Windows home folder. If starting from scratch, clone [TraceLab](https://github.com/uw-syfi/TraceLab) into `TraceLab` and this repository at branch `Hangting_week3` into `agent-kv-retention-hangting`. Adjust `$ROOT` if you use a different location.

```powershell
$ROOT = Join-Path $HOME 'kv-week3-work'
$TRACE = Join-Path $ROOT 'TraceLab'
$REPO = Join-Path $ROOT 'agent-kv-retention-hangting'
Set-Location $TRACE

New-Item -ItemType Directory -Force -Path 'trace', 'group_work', 'our_work' | Out-Null

# Use TraceLab's local virtual environment when available.
$VENV_PY = Join-Path $TRACE '.venv\Scripts\python.exe'
if (Test-Path $VENV_PY) { $PY = $VENV_PY } else { $PY = 'python' }
& $PY -m pip install duckdb pandas numpy matplotlib
```

Ensure `trace\syfi_coding_trace.duckdb` exists. If it has not been downloaded, retrieve the [TraceLab v0.0.1 release](https://github.com/uw-syfi/TraceLab/releases/tag/v0.0.1):

```powershell
Invoke-WebRequest -Uri 'https://github.com/uw-syfi/TraceLab/releases/download/v0.0.1/syfi_coding_trace.duckdb' `
  -OutFile 'trace\syfi_coding_trace.duckdb'
```

**B. Obtain Brian's original round-gap materials (optional, for comparison).** His source files remain in the [`Brian` branch](https://github.com/Omkar38/agent-kv-retention/tree/Brian/week3). The following **binary-safe** `git archive` method retrieves scripts and the original Parquet file without switching branches. Do not use `git show ... > pauses.parquet` in Windows PowerShell 5.1, because its text redirection can corrupt binary output.

```powershell
git -C $REPO fetch origin Brian
$ZIP = Join-Path (Get-Location) 'group_work\brian_week3.zip'
git -C $REPO archive --format=zip --output="$ZIP" origin/Brian `
  week3/build_pause_table.py `
  week3/plot_pause_cdf.py `
  week3/plot_pause_by_tool.py `
  week3/plot_context_size.py `
  week3/data/pauses.parquet
if ($LASTEXITCODE -ne 0) { throw 'Could not export Brian branch artifacts.' }

Expand-Archive -LiteralPath $ZIP -DestinationPath 'group_work\brian_branch' -Force
Copy-Item 'group_work\brian_branch\week3\*.py' 'our_work\'
Copy-Item 'group_work\brian_branch\week3\data\pauses.parquet' 'group_work\pauses.parquet'
```

To regenerate Brian's round-gap dataset and figures from the downloaded DuckDB instead of reusing the archived Parquet file:

```powershell
& $PY 'our_work\build_pause_table.py'
& $PY 'our_work\plot_pause_cdf.py'
& $PY 'our_work\plot_pause_by_tool.py'
& $PY 'our_work\plot_context_size.py'
Copy-Item 'our_work\pauses.parquet' 'group_work\pauses.parquet' -Force
```

**C. Build the overlap-aware tool-activity dataset.** This runs all six diagnostic stages and writes Parquet outputs to `group_work\`:

```powershell
& $PY (Join-Path $REPO 'analysis\validation\build_tool_dataset.py')
```

To rerun one stage, append `--step intervals`, `--step episodes`, `--step candidates`, `--step audit`, `--step wait_candidates`, or `--step classify`, ensuring required inputs exist.

**C2. Export the Brian/Hangting Week 4 pause handoff:**

```powershell
& $PY (Join-Path $REPO 'analysis\validation\build_pause_handoff.py')
```

Requires `group_work\pauses.parquet` and `group_work\tool_intervals.parquet`. It writes the quality-labeled original-pause bridge and a strict pause-prediction cohort without changing the original table. See [WEEK4_HANDOFF.md](WEEK4_HANDOFF.md).

**D. Run validation and anomaly inspection:**

```powershell
& $PY (Join-Path $REPO 'analysis\validation\validate_week3.py')
& $PY (Join-Path $REPO 'analysis\validation\inspect_cases.py') --case codex
```

The original pause-table attribution check requires `group_work\pauses.parquet`; it will be explicitly skipped when that file is missing. Other case selectors are `--case late_calls` and `--case negative_gap`.

**E. Run the complementary tool and session statistics:**

```powershell
& $PY (Join-Path $REPO 'analysis\trace_facts\check_tool_latency.py')
& $PY (Join-Path $REPO 'analysis\trace_facts\check_session_latency.py')
& $PY (Join-Path $REPO 'analysis\trace_facts\check_tool_thresholds.py')
& $PY (Join-Path $REPO 'analysis\trace_facts\check_prediction_fields.py')
```

**Check reproduction:** Compare the output with [WEEK3_VALIDATION.md](WEEK3_VALIDATION.md): 431,905 valid tool intervals, 331,400 merged episodes, and 277,777 simple tool-wait proxy candidates for the audited TraceLab release. Numerical agreement is an audit checkpoint, not proof that the records are exact GPU KV-cache idle intervals.

## 5. Output datasets: semantics, features, and labels

| Data artifact | One row represents | Key fields / interpretation |
|---|---|---|
| `our_work/pauses.parquet` / `group_work/pauses.parquet` | A transition between two LLM rounds | `pause_s`, `cause`, `usable`, `context_tokens` and next-round-associated tool summaries. **Exploratory round-gap record**; Codex attribution and boundary issues remain. |
| `group_work/tool_intervals.parquet` | One raw tool-call record | `tool_start_ts`, `tool_end_ts`, `tool_duration_s`, `reported_tool_duration_s`, `timing_status`. Missing/invalid timestamps are explicitly flagged. |
| `group_work/tool_activity_episodes.parquet` | Union of overlapping timestamp-valid tool calls in a session | `start_ts`, `end_ts`, `n_calls`, `n_rounds`, `tool_activity_duration_s`. Counts and full tool lists reflect the **entire observed episode**. |
| `group_work/tool_episode_candidates.parquet` | An episode linked to its earliest emitted tool call | `candidate_start_ts`, `first_tool_name`, `first_tool_input_chars`, `context_tokens`, `tool_activity_duration_s`, and retrospective episode composition. |
| `group_work/tool_episode_candidates_audited.parquet` | Candidate above plus diagnostic flags | `generation_events_inside`, `user_messages_inside`, `simple_tool_wait_proxy`; all simple and complex cases are retained. |
| `group_work/agent_wait_candidates.parquet` | Tool-activity episodes aligned to adjacent generation/user signals | Retrospective event timing, not actual GPU idleness. |
| `group_work/agent_waits_classified.parquet` | Retrospectively grouped episodes | Interaction type, tool topology, quality and continuation lag. |
| `group_work/pauses_v2.parquet` | **Brian's inter-round pause table with Hangting's validation joined** | Preserves all original rows and features, adds `quality_reason`, source-round timing, and conservative `target_pause_s`. |
| `group_work/week4_pause_train.parquet` | **Week 4 strictly screened pause-duration training cohort** | Limited prediction-safe feature projection plus deterministic split by session. See [WEEK4_HANDOFF.md](WEEK4_HANDOFF.md). |

`group_work/agent_wait_candidates.parquet` aligns those candidate episodes to preceding and following recorded generation signals and subsequent user messages. Its `alignment_status`, `prior_generation_signal_ts`, `next_generation_signal_ts`, and signal offsets are retrospective diagnostic fields. They **do not** provide exact inference-stop timestamps, actual GPU scheduling boundaries, or KV-cache residency. `wait_candidates` is for offline audit, not direct retention-policy scoring.

**Suggested preliminary prediction target:** `tool_activity_duration_s`, explicitly labeled *tool-activity duration*. This is not the wall-clock time to the next LLM decode and is not a verified KV-cache-residency interval.

**Candidate information at first tool emission:** `provider`, `first_tool_name`, `first_tool_input_chars`, and (only after decision-time verification) `model` or `context_tokens`. Session identifiers are suitable for **grouped train/test splits**, not features that should be memorized by the model.

**Never feed these retrospective columns into an online predictor:** `tool_activity_duration_s` (the label), `tool_activity_end_ts`, `observed_n_calls`, `observed_n_rounds`, `observed_tool_names`, `generation_events_inside`, `user_messages_inside`, or `simple_tool_wait_proxy`. Even using a flag for **filtering** a test population requires disclosure because it is derived from future events. Some known-at-source fields may still be unsafe for the actual *KV retention decision*, which occurs after inference completes, not necessarily at first tool issuance.

**Quality and causality notes:** Simultaneous/overlapping tool calls are merged to avoid double-counting tool activity; this does not establish true multi-request GPU behavior. Cases with new user input or generation during an episode are reported, not silently dropped. Do not concatenate round-gap rows and tool-episode rows as if they were independent identical pause samples.

## 6. Key observed results and their limits

Using the v0.0.1 trace, the round-gap builder produced **357,161 candidate transitions**, of which **346,820** were usable under its filters. The direct tool-call analysis yielded **431,905** timestamp-valid tool intervals, merged into **331,400** observed activity episodes: **277,777** met the conservative *single tool, no recorded intervening generation/user input* proxy definition, and **53,623** were more complex. A result-driven next-round matching audit found **63,122** tool-call matches emitted after the putative earlier decision timestamp, predominantly Codex. See the validation report for denominators, exact counts, sessions and caveats.

This means the branch establishes a reproducible **workload characterization and preliminary prediction proxy**, not a verified per-pause TTL simulation workload or measured GPU memory saving. Plot break-even times derived from hypothetical prefill speeds must be recalibrated on real hardware.

## 7. How later weeks can use this work

**Week 4 — pause-duration prediction.** Use [the Brian/Hangting quality-labeled pause handoff](WEEK4_HANDOFF.md) as the **primary conservative pause-duration target**, then train separate exploratory models on directly observed tool-activity durations; split by `session_id` to prevent session leakage. Compare tool-only, argument-size, provider, and verified context features; evaluate error, calibration and long-tail bias separately on simple versus complex episodes. Do not use hindsight episode summaries as predictor features.

**Week 5 — project presentation.** Explain why the tool-latency tail and provider-specific trace structure matter for retention and why an outcome-based feature join can overstate predictability. Present original figures as baseline exploratory analysis and label placeholder hardware costs.

**Weeks 6–7 — policy and simulator.** Define inference-end, pause-start and next-generation-resume events. Recover a separate validated `pauses_v2` with absolute timestamps, causes and exclusions. Use `T` as the actual validated retention interval, then calibrate memory cost `c` and eviction/recompute penalty `B` for `C(τ,T)=c·min(T,τ)+B·1[T>τ]`. Simulate eviction, fixed TTL and Continuum-style TTL under the *same* trace inputs and memory constraints. Prove the single-request consistency/robustness property independently of multi-request queueing effects.

**Weeks 8–10 — stress tests and reporting.** Study prediction bias, distribution shift, heavy-tail resampling and memory contention with clear simple/complex cohort reporting. Evaluate average latency, recomputation, p95/p99 completion time and queue effects after obtaining a valid mapping from traces to inference-server behavior. Preserve source-release and code versions for repeatability.

## 8. Known limitations

The original next-round tool attribution and pause endpoint are unreliable for some Codex traces, and some tool calls occur entirely within a recorded LLM round. Rare negative/duplicate round anomalies also require explicit treatment. Merging tool intervals does not prove the GPU is idle. Current `context_tokens` is a context-size proxy rather than a proven timestamped cache-size measurement. The Week 3 data supports tool workload research now; actual KV-retention-policy claims require a separate inference-stop/resume validation and memory-system calibration.

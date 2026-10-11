# Hangting — Week 3 TraceLab Validation

## Purpose

This folder contains an independent audit of the TraceLab-derived pause table and a reproducible pipeline for extracting **tool-execution intervals** and grouping overlapping intervals into **tool-activity episodes**. It is part of the group's project *Learning-Augmented KV-Cache Retention for Tool-Using LLM Agents*.

**Scope:** These records support workload characterization and *preliminary tool-duration prediction*. They are **not yet validated KV-cache idle/retention intervals**. Do not use the episode durations as ground-truth cache idle durations in the final retention-policy simulator without an additional inference-stop/resume audit.

See [WEEK3_VALIDATION.md](WEEK3_VALIDATION.md) for the measurements and limitations.

## Dependencies and working directory

Scripts were developed against the TraceLab release database at:

```text
TraceLab/trace/syfi_coding_trace.duckdb
```

Run them with **TraceLab as the current working directory**, because file paths in the scripts are relative to that directory. Install `duckdb` and `pandas` into your Python environment. Figure scripts may additionally require `matplotlib` and other dependencies already used by the original team scripts.

Example, on the development machine:

```bash
cd ~/kv-week3-work/TraceLab
source .venv/bin/activate   # only if your virtual environment is not already active
mkdir -p group_work
python ~/kv-week3-work/agent-kv-retention-hangting/analysis/hangting_week3/build_tool_intervals.py
```

The raw TraceLab DuckDB database is not included in this GitHub folder. Follow the source project's access and license terms when downloading the trace.

## Main derivation pipeline

Starting from the original DuckDB database:

```text
build_tool_intervals.py
  -> group_work/tool_intervals.parquet
build_tool_episodes.py
  -> group_work/tool_activity_episodes.parquet
build_episode_candidates.py
  -> group_work/tool_episode_candidates.parquet
finalize_week3_candidates.py
  -> group_work/tool_episode_candidates_audited.parquet
```

Run the scripts in that order, with TraceLab as the working directory:

```bash
ROOT=~/kv-week3-work/agent-kv-retention-hangting/analysis/hangting_week3
python "$ROOT/build_tool_intervals.py"
python "$ROOT/build_tool_episodes.py"
python "$ROOT/build_episode_candidates.py"
python "$ROOT/finalize_week3_candidates.py"
```

For checking the original round-gap extraction, provide `group_work/pauses.parquet` as produced by the separately maintained Brian Week 3 extraction. The original pause table and the episode table must **not** be appended together as if they represented equivalent events.

## Audit scripts

| Script | Purpose |
|---|---|
| `check_negative_pauses.py` | Investigates negative round-gap durations. |
| `check_feature_timing.py` | Compares matched tool-call emission timestamps with estimated decision times in the original pause extraction. Requires `group_work/pauses.parquet`. |
| `inspect_late_calls.py` | Breaks late matches down by provider and source round. Requires `group_work/pauses.parquet`. |
| `inspect_codex_timing.py` | Inspects a documented Codex event-ordering example. |
| `audit_within_round_results.py` | Counts tool results appearing after first generation events in the same round. |
| `validate_tool_intervals.py` | Checks recorded duration arithmetic and tool-call overlaps. |
| `audit_episode_generation.py` | Counts generation/user-message signals strictly inside episodes. |
| `check_candidate_boundaries.py` | Checks subsequent within-round output/tool-call signals and feature completeness. |

Some script copies in this directory originated from Brian's Week 3 work (such as `build_pause_table.py` and `plot_*`). The canonical versions remain in the team's `week3/` directory; credit and reference the originals rather than presenting those copies as new implementations.

## Dataset semantics and permitted fields

**`tool_intervals.parquet`:** Each row describes a recorded `tool_calls` interval from `emitted_at` to `result_at`. An interval is a tool-execution observation; it is not necessarily an interval during which a model's KV cache can be safely retained or evicted.

**`tool_activity_episodes.parquet`:** Overlapping valid tool intervals are unioned within each provider/session. Episode count is therefore smaller than tool-call count. Grouping by session does not prove that calls are part of one logical model decision or that no generation occurs between sparse recorded events.

**`tool_episode_candidates.parquet`:** A candidate episode with its first emitted tool's name, argument size, round/model and context-token proxy. The `candidate_start_ts` is **first tool emission**, not a verified GPU inference-stop timestamp. `observed_n_calls`, `observed_n_rounds`, `observed_tool_names` and the final duration are retrospective; they must not be provided as input features for prediction at the episode start.

**`tool_episode_candidates_audited.parquet`:** Adds `generation_events_inside`, `user_messages_inside`, and `simple_tool_wait_proxy`. The latter means one recorded tool call and no recorded reasoning/text/user-message events strictly inside the interval. It is a **proxy classification**, not verification of KV-cache idleness. Non-simple episodes are retained.

**Decision-time leakage caution:** Even a feature available in the source round may not have been observable precisely at the start timestamp. In particular, `input_tokens_total` is currently a context proxy rather than a proven online timestamped measurement. When preparing model inputs, validate features against the chosen decision point. Do not derive prediction features from future tool results or the entire episode's final composition.

## Reproducibility and provenance

- Original `pauses.parquet` and associated plots: developed by Brian; independently reproduced and audited here.
- Tool-latency summaries and original schema checks: developed by David; local fixes to percentile printing and unsupported field names are separate changes under `analysis/trace_facts/`.
- Tool interval / episode extraction, leakage audits, and candidate quality flags: this validation contribution.
- Measurements reported in `WEEK3_VALIDATION.md` were observed in one local TraceLab run on October 11, 2026. Re-run the scripts against the same source trace version before comparing counts.

## Follow-up before retention-policy experiments

1. Define an actual per-pause inference-stop timestamp and the next relevant LLM continuation, including within-round activity.
2. Validate decision-time feature availability, and differentiate human inactivity from tool-caused pauses.
3. Resolve overlapping tool activity, repeated/polling calls, and sessions with inconsistent round ordering.
4. Produce and test a **separate** `pauses_v2.parquet` only after the above checks; then re-run workload figures and compare identical traces and memory constraints across retention policies.

The instructor requested a simple single-request consistency/robustness guarantee and load-dependent multi-request interactions as a simulation extension; the data pipeline should support, not conflate, those distinct evaluations.

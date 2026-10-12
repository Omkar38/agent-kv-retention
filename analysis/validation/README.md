# TraceLab validation and unified Version 2 pause handoff

**Project:** Learning-Augmented KV-Cache Retention for Tool-Using LLM Agents.

The **sole official Week 4 dataset** is [`../dataset/pause_handoff.parquet`](../dataset/pause_handoff.parquet). It preserves all rows and all source columns from the corrected Version 2 pause extractor and adds offline record categories, source-timing checks, feature-availability categories, tool topology, and a deterministic session-grouped split. No separate filtered training file is generated. See [the dataset README](../dataset/README.md) for the schema and complete commands.

## Active scripts

| Script | Role |
|---|---|
| `build_pause_handoff.py` | Generate **one** lossless, categorized dataset from `group_work/pauses.parquet` (Version 2). |
| `check_handoff_bias.py` | Report duration distributions and category coverage by provider, log format, and analysis track. |
| `build_tool_dataset.py` | Independently study timestamped tool intervals, overlap unions, tool episodes, and tool/generation alignments. Exploratory and not an alternative pause ground truth. |
| `validate_week3.py` | Independent runner-vs-wall latency comparison, overlap checks, and historical first-input round-boundary diagnostics; obsolete Version 1 tool-attribution checks were removed (not Version 2 pause acceptance checks). |
| `inspect_cases.py` | Reproduce historical anomalous timestamp examples. |
| `audit_v2_pause_events.py` | Independently compare Version 2 pause lengths against matched tool-call runner durations and result-event timestamps, with a separate older-Codex discrepancy report. Diagnostic evidence; never rewrites the dataset. |

Complementary descriptive tool statistics: [`../trace_facts/trace_statistics.py`](../trace_facts/trace_statistics.py). Historical measurements and caveats: [`WEEK3_VALIDATION.md`](WEEK3_VALIDATION.md). The current dataset handoff: [`WEEK4_HANDOFF.md`](WEEK4_HANDOFF.md).

## Rebuild

Install Python, `duckdb`, and `pandas`. Download the [TraceLab v0.0.1 coding trace](https://github.com/uw-syfi/TraceLab/releases/tag/v0.0.1) to `trace/syfi_coding_trace.duckdb`, and retrieve the **Version 2** `build_pause_table.py` from the upstream extraction workstream into TraceLab's `our_work/` directory. From the TraceLab root:

```bash
# The corrected v2 pause extractor must exist at our_work/build_pause_table.py
python our_work/build_pause_table.py
mkdir -p group_work
cp our_work/pauses.parquet group_work/pauses.parquet

# REPO is the local root of this project checkout
python "$REPO/analysis/validation/build_pause_handoff.py"
python "$REPO/analysis/validation/check_handoff_bias.py"

cp group_work/pause_handoff.parquet "$REPO/analysis/dataset/pause_handoff.parquet"
```

The generator refuses the earlier Version 1 schema and verifies that all Version 2 rows and columns are unchanged. The generated handoff has been regenerated and confirmed; rerun the commands only when intentionally updating the derived features.

### Data interpretation

- Every record is retained even if its **category** indicates `session_end`, `unresolved_continuation`, `boundary_missing`, `negative_duration`, or `source_record_review`.
- Source field `usable` remains unchanged because the input schema is preserved; **new** classification fields use informative categories instead of binary labels.
- Tool, human, and mixed waits and single/multiple tool calls are separated, without discarding rows.
- The old Codex log format is retained; tool metadata that may appear only at result time is masked in derived `feature_*` fields, while the original source columns remain intact.
- `pause_s` and `pause_end_at` are retrospective values. `record_category`, `analysis_track`, and QA categories are **analysis-only** labels, not online prediction features.
- TraceLab inter-round timestamps are not observations of actual inference-server KV-cache residency. Later retention simulations must disclose modeled assumptions.

## Separate exploratory tool analysis

Run selected descriptive checks with:

```bash
python "$REPO/analysis/trace_facts/trace_statistics.py" --step all
```

For tool-activity analysis:

```bash
python "$REPO/analysis/validation/build_tool_dataset.py"
python "$REPO/analysis/validation/validate_week3.py" --step intervals
```

The tool extraction now records `timestamp_tool_duration_s`, `runner_tool_duration_s`, `duration_source`, and an effective `tool_duration_s` (runner internal when available, wall-clock fallback). Collapsed timestamps are flagged and excluded from timestamp-based episode union. Descriptive tool statistics also use runner-first duration. The interval-consistency check compares runner-reported and independently timestamp-derived measurements rather than two forms of the same wall duration.

**Version 2 pause event audit** (requires only the original trace and the unified handoff):

```bash
python "$REPO/analysis/validation/audit_v2_pause_events.py"
```

The older Codex pause duration still uses the upstream v2 timestamp convention and may be undermeasured. This audit reports potential discrepancies; it does not invent corrected pause boundaries. The official dataset is kept lossless. Run these checks on the TraceLab machine and review the old-Codex example rows before treating their pause targets as reliable. The handoff and auxiliary tool-analysis scripts have been executed on the local trace, and the 357,161-row, 33-column unified dataset was confirmed. The optimized Version 2 event audit was subsequently rerun on the local trace and completed in 1.438 seconds; its results are diagnostic, not a replacement for the original pause targets.

The official Week 4 artifact is the single regenerated 357,161-row, 33-column Parquet.

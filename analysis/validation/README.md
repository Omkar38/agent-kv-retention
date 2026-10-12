# TraceLab validation and unified Version 2 pause handoff

**Project:** Learning-Augmented KV-Cache Retention for Tool-Using LLM Agents.

The **sole official Week 4 dataset** is [`../dataset/pause_handoff.parquet`](../dataset/pause_handoff.parquet). It preserves all rows and all source columns from the corrected Version 2 pause extractor and adds offline record categories, source-timing checks, feature-availability categories, tool topology, and a deterministic session-grouped split. No separate filtered training file is generated. See [the dataset README](../dataset/README.md) for the schema and complete commands.

## Active scripts

| Script | Role |
|---|---|
| `build_pause_handoff.py` | Generate **one** lossless, categorized dataset from `group_work/pauses.parquet` (Version 2). |
| `check_handoff_bias.py` | Report duration distributions and category coverage by provider, log format, and analysis track. |
| `build_tool_dataset.py` | Independently study timestamped tool intervals, overlap unions, tool episodes, and tool/generation alignments. Exploratory and not an alternative pause ground truth. |
| `validate_week3.py` | Historical timing/overlap experiments, including first-input round-boundary diagnostics. Some checks reflect the older pause extraction and must **not** be used as current Version 2 acceptance checks. |
| `inspect_cases.py` | Reproduce historical anomalous timestamp examples. |

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

The generator refuses the earlier Version 1 schema and verifies that all Version 2 rows and columns are unchanged. It does **not** directly update the committed Parquet; the regenerated file must be committed and pushed locally.

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

Tool timestamps and per-call duration summaries may differ from runner-reported internal execution time in older formats. They are complementary diagnostics, not a second dataset for Week 4.

The official Week 4 artifact is one Parquet, regenerated with the categorized builder.

## 1. Summary of the Dataset

**File:** `pause_handoff.parquet`

This dataset supports pause-duration prediction, error analysis, and later trace-driven KV-cache retention simulations for tool-using LLM agents. It is derived from the [original Version 2 pause dataset](https://github.com/Omkar38/agent-kv-retention/blob/3dd1760f92c32c5013dd8b8d4f9fdd88200a3330/week3/data/pauses.parquet), built using the [Version 2 extraction script](https://github.com/Omkar38/agent-kv-retention/blob/3dd1760f92c32c5013dd8b8d4f9fdd88200a3330/week3/build_pause_table.py) and the [TraceLab v0.0.1 coding-agent trace](https://github.com/uw-syfi/TraceLab/releases/tag/v0.0.1).

The handoff **preserves all 357,161 original rows and all 15 Version 2 columns unchanged** and adds 18 columns for categorical interpretation, candidate prediction features, regression/classification targets, timing checks, and reproducible dataset partitioning. There are **33 columns in total** after regeneration. No original record is removed, including session-ending records, unresolved continuations, and records with incomplete or inconsistent timestamps.

The source dataset estimates the time between agent generation and its next continuation. These are **trace-derived pause-duration estimates**, not direct measurements of GPU KV-cache residency.

## 2. Features (Columns)

### 2.1 Original Version 2 columns — preserved (15)

| Column | Explanation |
|---|---|
| `provider` | Source provider (`claude` or `codex`). |
| `user` | Identifier of the trace user. |
| `session_id` | Agent session identifier; useful for grouping observations. |
| `round_index` | Round number within an agent session. |
| `model` | Model identifier recorded in the trace. |
| `pause_s` | Estimated pause duration in seconds; the supervised prediction target when a duration is available and appropriate for the selected analysis. |
| `cause` | Original retrospective cause: `tool`, `user`, `tool+user`, `end`, or `unknown`. |
| `usable` | Unchanged boolean field from Version 2 describing its original record-screening result. The new categorical columns provide more detailed interpretations. |
| `context_tokens` | Source estimate of context size in tokens; not a direct measurement of KV-cache bytes. |
| `n_tools` | Number of tools associated with the pause under Version 2's extraction rules. |
| `tool_names` | Names of the associated tools. |
| `tool_input_chars` | Total size of tool inputs in characters. |
| `pause_start_at` | Estimated start timestamp of the pause. |
| `pause_end_at` | Estimated end timestamp of the pause, when available. |
| `log_format` | Logging variant: `claude`, `codex`, or `codex_old`. |

### 2.2 Additional columns — why they are needed (18)

| Column | Explanation and purpose |
|---|---|
| `endpoint_duration_s` | Recomputes duration as `pause_end_at - pause_start_at` in seconds, allowing comparison against `pause_s`. |
| `record_category` | Categorizes the event as `tool_wait`, `human_wait`, `mixed_tool_human_wait`, `session_end`, `unresolved_continuation`, or `other_cause`. Needed to distinguish fundamentally different pause situations without dropping rows. |
| `tool_topology` | Labels `single_tool_call`, `multiple_tool_calls`, `no_tool_call`, or `tool_count_not_recorded`. Needed to study how concurrent or multiple tools affect prediction errors. |
| `timing_category` | Labels `timing_consistent`, `boundary_missing`, `negative_duration`, `duration_missing`, `endpoint_disagreement`, or `source_record_review`. Needed to identify timing and source-record issues explicitly. |
| `feature_category` | Classifies tool metadata as `source_round_tool_metadata`, `source_round_no_tool`, `result_logged_tool_metadata`, `older_codex_no_tool`, or `feature_availability_unknown`. Needed to distinguish recorded metadata from information potentially available when a retention decision is made. |
| `analysis_track` | Assigns `tool_pause_prediction`, `human_wait_prediction`, `mixed_wait_prediction`, `boundary_or_source_review`, or `diagnostic_event`. Needed to organize different evaluation cohorts within one dataset. |
| `feature_n_tools` | Prediction-oriented tool count; masked for older Codex result-time tool metadata because decision-time availability is uncertain. |
| `feature_tool_names` | Prediction-oriented tool names with the same older-Codex masking. |
| `feature_tool_input_chars` | Prediction-oriented tool-input character count with the same older-Codex masking. |
| `feature_log1p_context_tokens` | Log-transformed nonnegative context-token count. Reduces scale skew for Week 4 regression; verify source-field availability at decision time. |
| `feature_log1p_round_index` | Log-transformed nonnegative round index. Captures conversation progression, if available at decision time. |
| `feature_log1p_tool_input_chars` | Log-transformed tool-input size; masked when older Codex tool metadata is uncertain. |
| `feature_has_tools` | Whether source-round tool count is positive, masked where its availability is uncertain. |
| `feature_multiple_tools` | Whether more than one tool was recorded, masked where its availability is uncertain. |
| `target_log1p_pause_s` | Log-transformed observed duration, suitable as a regression target; never an input feature. |
| `label_pause_gt_10s` | Retrospective long-pause label for durations over 10 s, for classification/error analysis. |
| `label_pause_gt_60s` | Retrospective long-pause label for durations over 60 s, for tail-error analysis. |
| `split` | Deterministic session-level partition: `train`, `val`, or `test`. Needed to evaluate predictions on sessions not seen during training. |

**Prediction precaution:** `pause_s`, `target_log1p_pause_s`, and `label_*` are targets, never input features. The event outcome (`cause`, `record_category`, `analysis_track`), timing checks, pause endpoints, source-screening labels, and `split` must **not** be supplied as online predictor inputs. `session_id` is for grouping, not ordinary prediction. The `feature_*` columns are candidate features. Source values are provisionally accepted for Week 4; decision-time field availability must still be checked for a deployable online policy. Any Codex timing discrepancies are deferred to future analysis. Every row stays in the dataset.

## 3. How to Read the Dataset

Run the following Python code **from the repository root**. Install `pandas` and `pyarrow` if necessary.

```python
import pandas as pd

df = pd.read_parquet("analysis/dataset/pause_handoff.parquet")

# Inspect the complete dataset and its schema.
print("Rows:", len(df))
print("Columns:", len(df.columns))
print(df.dtypes)
print(df.head())

# Explore different record and timing categories.
print(df["record_category"].value_counts(dropna=False))
print(df["timing_category"].value_counts(dropna=False))
print(df["analysis_track"].value_counts(dropna=False))

# Select a cohort for supervised tool-pause prediction.
tool_pauses = df[df["analysis_track"] == "tool_pause_prediction"]

train = tool_pauses[tool_pauses["split"] == "train"]
val = tool_pauses[tool_pauses["split"] == "val"]
test = tool_pauses[tool_pauses["split"] == "test"]

# Use candidate prediction features; keep the target separate.
feature_cols = [
    "provider",
    "feature_n_tools",
    "feature_tool_names",
    "feature_tool_input_chars",
    "feature_log1p_context_tokens",
    "feature_log1p_round_index",
    "feature_log1p_tool_input_chars",
    "feature_has_tools",
    "feature_multiple_tools",
]
X_train = train[feature_cols]
y_train = train["pause_s"]

print("Train:", len(train), "Validation:", len(val), "Test:", len(test))
```

The `analysis_track` filter is an **offline cohort selection** for evaluation, not an input available to an online retention policy. To analyze human or mixed waits, select the corresponding `analysis_track`. Compare logging formats separately when reporting results. Records with missing boundaries or unresolved continuations remain in the full dataset for diagnostics rather than being silently deleted.

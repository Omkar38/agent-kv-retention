# Unified Version 2 pause dataset

**Official dataset:** `pause_handoff.parquet` — one source-preserving dataset for Week 4 prediction and later cache-retention simulation.

The handoff reads the Version 2 pause table from the upstream Version 2 pause extractor, preserves **every source row and every source column**, and adds independent categorical diagnostics and a deterministic session split. It does not rewrite `pause_s`, `cause`, `n_tools`, `tool_names`, `tool_input_chars`, `context_tokens`, the pause endpoints, or `log_format`.

The source contains 357,161 records; the last tested Version 2 input reported 351,497 records meeting upstream timestamp/linkage criteria and 5,664 requiring other interpretations. **All 357,161 are retained here.** The inherited binary source field `usable` remains untouched solely because the original schema must not be altered. Our analysis uses the more descriptive categories below instead.

## Added columns

| Column | Meaning |
|---|---|
| `record_category` | `tool_wait`, `human_wait`, `mixed_tool_human_wait`, `session_end`, `unresolved_continuation`, `other_cause` |
| `timing_category` | `timing_consistent`, `boundary_missing`, `negative_duration`, `duration_missing`, `endpoint_disagreement`, `source_record_review` |
| `tool_topology` | `single_tool_call`, `multiple_tool_calls`, `no_tool_call`, `tool_count_not_recorded` |
| `feature_category` | `source_round_tool_metadata`, `source_round_no_tool`, `result_logged_tool_metadata`, `older_codex_no_tool`, `feature_availability_unknown` |
| `analysis_track` | `tool_pause_prediction`, `human_wait_prediction`, `mixed_wait_prediction`, `old_codex_tool_timing_study`, `boundary_or_source_review`, `diagnostic_event` |
| `endpoint_duration_s`, `qa_endpoint_duration_s` | Independent arithmetic of Version 2 supplied pause endpoints; **not** independently measured GPU idleness |
| `feature_n_tools`, `feature_tool_names`, `feature_tool_input_chars` | Masked candidate predictor features; old Codex tool metadata is masked because decision-time visibility is uncertain |
| `split` | Stable SHA-256 provider/session assignment: train, val, test |

**Future-information warning:** `pause_s` is the target. `record_category`, `analysis_track`, `timing_category`, `cause`, `split`, result timestamps, and QA fields are **not predictors**; several depend on outcomes. Older Codex observations stay in the dataset, with original source metadata intact, but uncertain model-input features masked. Any application-specific predictor must independently justify feature observability.

## Rebuild the one official dataset

From the Mac terminal:

```bash
cd /path/to/project-checkout
git switch Hangting_week3
git pull --ff-only origin Hangting_week3


cd ~/kv-week3-work/TraceLab
REPO=/path/to/project-checkout
mkdir -p our_work group_work
# Obtain the current Version 2 extractor from the upstream workstream as our_work/build_pause_table.py
python our_work/build_pause_table.py
cp our_work/pauses.parquet group_work/pauses.parquet
python "$REPO/analysis/validation/build_pause_handoff.py"
python "$REPO/analysis/validation/check_handoff_bias.py"
cp group_work/pause_handoff.parquet "$REPO/analysis/dataset/pause_handoff.parquet"

git -C "$REPO" add analysis/dataset/pause_handoff.parquet
git -C "$REPO" commit -m "Regenerate unified categorized Version 2 pause handoff"
git -C "$REPO" push origin Hangting_week3
```

The already committed `pause_handoff.parquet` was produced before these new categories were added. **Regenerate and push the same filename before use.** Do not generate or commit a second model dataset.

Week 4 subsets are made **in memory** using `analysis_track` and `split`, e.g. tool training records: `analysis_track == 'tool_pause_prediction'` and `split == 'train'`. Evaluate human and mixed waits separately. Retain boundary-review and unresolved records for data-quality analysis; not every record defines a measured duration appropriate as a supervised label.

**Limits:** These are TraceLab inter-round pause estimates rather than observed GPU cache-residency intervals. Cite the [TraceLab release](https://github.com/uw-syfi/TraceLab) and note old Codex time-logging limitations.

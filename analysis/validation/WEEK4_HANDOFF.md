# Week 4 handoff — Brian's pauses + Hangting's validation

**Purpose:** Produce a defensible, **quality-labeled inter-round pause-duration proxy** for Week 4 without overwriting Brian's original `pauses.parquet`.

- **Original workload:** Brian's `week3/build_pause_table.py` and `group_work/pauses.parquet` (357,161 original records; 346,820 flagged usable).
- **Independent audit:** Hangting's `analysis/validation/` timing and tool-call checks.
- **Integration:** `analysis/validation/build_pause_handoff.py` links original rows to source-round tool emissions and later matching results. It retains all original rows, records reasons for exclusion, and exports a conservative training view.

## Why this is needed

Brian defines `pause_s` between one round's final model-output event and the next round's last input event. That is conceptually closer to the KV-retention waiting interval than tool duration alone. However, attributes such as `n_tools`, `tool_names`, and `tool_input_chars` were derived by matching **next-round results**. Our audit found 63,122 **matched call records** whose emissions occurred after the earlier decision timestamp, including a documented Codex user-inactivity case. These are **matched records, not distinct affected pauses**.

The new handoff preserves Brian's target definition but uses emitted **current-round tool information**, verified against its later result for offline quality classification. This reduces future-information leakage in training columns; it does **not** reconstruct exact GPU inference-stop events.

## Run

From the TraceLab root, with `trace/syfi_coding_trace.duckdb`, `group_work/pauses.parquet`, and `group_work/tool_intervals.parquet` already present:

```bash
cd ~/kv-week3-work/TraceLab
REPO=~/kv-week3-work/agent-kv-retention-hangting
python "$REPO/analysis/validation/build_pause_handoff.py"
```

Windows PowerShell (inside TraceLab directory):

```powershell
$REPO = Join-Path $HOME 'kv-week3-work\agent-kv-retention-hangting'
python (Join-Path $REPO 'analysis\validation\build_pause_handoff.py')
```

Use `--trace-root`, `--brian-pauses`, or `--tool-intervals` to override paths. To create prerequisites from existing tools, run Brian's original `build_pause_table.py` and copy `our_work/pauses.parquet` to `group_work/pauses.parquet`; run `build_tool_dataset.py --step intervals` to regenerate intervals. Refer to [README.md](README.md) for full macOS/Linux and PowerShell setups.

## Outputs and meanings

| Path (under TraceLab `group_work/`) | Meaning |
|---|---|
| `pauses.parquet` | Brian's untouched original table |
| `pauses_v2.parquet` | One row per original record, original fields plus reconstructed boundary proxies, source-call matching, timestamps, and `quality_reason` |
| `week4_pause_train.parquet` | Conservative, **single-tool** pause-duration training subset, restricted feature projection and deterministic session split |
| `tool_activity_episodes.parquet` | Secondary external-duration analysis from Hangting; **not** the pause target |

`pauses_v2.parquet` retains `pause_s`, `cause`, `usable`, `n_tools`, `tool_names`, and `tool_input_chars` **for auditing**, not as prediction inputs. It includes explicit aliases `brian_pause_s`, `brian_cause`, `brian_usable`, and provenance diagnostics. `corrected_gap_proxy_s` independently recomputes Brian's original endpoint convention, rather than claiming an exact corrected server idle boundary.

Only records with `quality_reason='strict_single_tool_pause_proxy'` receive a non-null `target_pause_s` in `pauses_v2.parquet`.

## Strict single-tool selection

The handoff requires: an original usable record; an unambiguous current and next round; ordered boundary timestamps; next-round inputs preceding or coinciding with next recorded generation; no next-round user message; exactly one valid tool emitted in the source round by the estimated stop timestamp; exactly one matching next-round tool result; non-missing first tool name/input size; and agreement with Brian's pause target within 1 ms. Other rows remain **with their specific quality reason**.

This is a high-precision **proxy** subset, not all valid pauses. In particular, interleaved Codex cases, human waits, overlapping/multiple tools, and uncertain chronology may be excluded. **Quality uses hindsight only to choose/evaluate a cohort; never provide these future-derived flags or result information to an online predictor.** Exclusion can create selection bias, so analyze the excluded cohort separately and do not report strict-only results as whole-workload results.

## Exact Week 4 training columns

| Column | Use |
|---|---|
| `provider` | Candidate predictor; also evaluate provider shift |
| `session_id` | Split/group ID only, **not** a feature |
| `round_index` | Optional ablation once decision-time availability is confirmed |
| `split` | Stable SHA-256-based 80/10/10 train/validation/test assignment **by session**; not an input |
| `model` | Optional ablation, subject to observability |
| `context_tokens_proxy` | Optional ablation; not measured cache bytes and timing needs verification |
| `tool_name_at_decision` | Primary prediction feature (emitted in the source round) |
| `tool_input_chars_at_decision` | Primary prediction feature |
| `target_pause_s` | Training target only, NEVER a prediction input |

Recommended first baseline: tool-only or per-tool median predictor, followed by a small nonlinear model. Evaluate MAE, long-pause classification recall, tail errors, and shifts, holding out sessions. Keep labels and predictions keyed by provider/session/round for downstream simulation, but make sure the retention policy sees only the prediction and verified decision-time features.

## Acceptance tests and limitations

The builder checks total-row preservation, no duplicate strict keys, and nonnegative, non-null strict targets. It prints quality counts by provider, session-split coverage and mismatches between the original and recomputed gap proxy.

**The new strict-cohort counts have not yet been measured:** run the builder against TraceLab v0.0.1 before reporting them. Do not treat the script's ability to compile or its successful GitHub upload as a full data validation.

**Key distinction:** `target_pause_s` is an *inter-round trace waiting proxy*, not the tool's duration, not true scheduler queueing and not observed physical GPU KV-cache residency. The eventual simulator must state its mapping from trace-proxy time to the simplified single-request model. Compare our policy, eviction, fixed TTL, and Continuum-style TTL against identical workloads and memory budgets, and prove a separate single-request consistency/robustness guarantee.
